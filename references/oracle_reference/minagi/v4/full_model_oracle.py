from __future__ import annotations

"""Exact full-model replay harness for RC10 qualification.

Unlike ``reference_executor`` (which holds captured layer inputs fixed), this
module re-runs the *entire* model on the newly assembled token history.  It is
therefore the oracle for context-conditioned hidden-state drift.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import json

import torch

from .oracle import TensorDivergence, tensor_divergence


class AssemblyCaseKind(str, Enum):
    PREFIX_CONTROL = "prefix_control"
    DELETION = "deletion"
    SPARSE_RETRIEVAL = "sparse_retrieval"
    REORDER = "reorder"
    LONG_JUMP = "long_jump"
    MANY_FRAGMENTS = "many_fragments"
    CHANGED_PREDECESSOR = "changed_predecessor"
    DUPLICATE_RETRIEVAL = "duplicate_retrieval"


@dataclass(frozen=True)
class TokenBlock:
    block_id: str
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class AssemblyCase:
    name: str
    kind: AssemblyCaseKind
    block_ids: tuple[str, ...]
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class ExactModelResult:
    tokens: tuple[int, ...]
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ModelDivergence:
    first_token_logit_kl: float
    top1_agree: bool
    prompt_top1_agreement: float
    layer_hidden: tuple[TensorDivergence, ...]


@dataclass(frozen=True)
class QualificationRecord:
    case_name: str
    case_kind: str
    token_count: int
    divergence: ModelDivergence

    def jsonable(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "case_kind": self.case_kind,
            "token_count": self.token_count,
            "divergence": {
                "first_token_logit_kl": self.divergence.first_token_logit_kl,
                "top1_agree": self.divergence.top1_agree,
                "prompt_top1_agreement": self.divergence.prompt_top1_agreement,
                "layer_hidden": [asdict(x) for x in self.divergence.layer_hidden],
            },
        }


def assemble(blocks: Mapping[str, TokenBlock], order: Sequence[str], *, name: str, kind: AssemblyCaseKind) -> AssemblyCase:
    ids = tuple(map(str, order))
    tokens = tuple(t for block_id in ids for t in blocks[block_id].tokens)
    return AssemblyCase(str(name), kind, ids, tokens)


def standard_cases(blocks: Sequence[TokenBlock]) -> tuple[AssemblyCase, ...]:
    if len(blocks) < 7:
        raise ValueError("standard adversarial case set requires at least seven blocks")
    m = {b.block_id: b for b in blocks}
    ids = [b.block_id for b in blocks]
    cases = [
        assemble(m, ids[:4], name="prefix-control", kind=AssemblyCaseKind.PREFIX_CONTROL),
        assemble(m, [ids[0], ids[3]], name="deletion", kind=AssemblyCaseKind.DELETION),
        assemble(m, [ids[0], ids[3], ids[5], ids[6]], name="sparse-retrieval", kind=AssemblyCaseKind.SPARSE_RETRIEVAL),
        assemble(m, [ids[2], ids[0], ids[3]], name="reorder", kind=AssemblyCaseKind.REORDER),
        assemble(m, [ids[0], ids[len(ids)//2], ids[-1]], name="long-jump", kind=AssemblyCaseKind.LONG_JUMP),
        assemble(m, [ids[i] for i in range(0, len(ids), 2)], name="many-fragments", kind=AssemblyCaseKind.MANY_FRAGMENTS),
        assemble(m, [ids[1], ids[3]], name="changed-predecessor", kind=AssemblyCaseKind.CHANGED_PREDECESSOR),
        assemble(m, [ids[0], ids[1], ids[1], ids[2]], name="duplicate-retrieval", kind=AssemblyCaseKind.DUPLICATE_RETRIEVAL),
    ]
    return tuple(cases)


class ExactFullModelOracle:
    def __init__(self, model: torch.nn.Module, *, device: torch.device | str | None = None):
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device

    @torch.no_grad()
    def run(self, tokens: Sequence[int]) -> ExactModelResult:
        ids = torch.tensor([list(map(int, tokens))], dtype=torch.long, device=self.device)
        out = self.model(input_ids=ids, use_cache=False, output_hidden_states=True)
        logits = out.logits.detach().float().cpu()
        hidden = tuple(x.detach().float().cpu() for x in (getattr(out, "hidden_states", None) or ()))
        return ExactModelResult(tuple(map(int, tokens)), logits, hidden)


def _kl(target_logits: torch.Tensor, approx_logits: torch.Tensor) -> float:
    p = torch.log_softmax(target_logits.float(), dim=-1)
    q = torch.log_softmax(approx_logits.float(), dim=-1)
    return float(torch.sum(p.exp() * (p - q), dim=-1).mean())


def compare_model_results(exact: ExactModelResult, accelerated: ExactModelResult) -> ModelDivergence:
    if exact.tokens != accelerated.tokens:
        raise ValueError("exact and accelerated results must describe the same logical token history")
    if exact.logits.shape != accelerated.logits.shape:
        raise ValueError("logit shape mismatch")
    # "first token" means the first predicted token after the assembled prompt,
    # i.e. the final prompt position's next-token distribution.
    e_last = exact.logits[:, -1]
    a_last = accelerated.logits[:, -1]
    kl = _kl(e_last, a_last)
    top1 = bool(torch.equal(e_last.argmax(-1), a_last.argmax(-1)))
    e_ids = exact.logits.argmax(-1)
    a_ids = accelerated.logits.argmax(-1)
    prompt_agree = float((e_ids == a_ids).float().mean())
    if len(exact.hidden_states) != len(accelerated.hidden_states):
        raise ValueError("hidden-state depth mismatch")
    layers = tuple(tensor_divergence(a, e) for a, e in zip(accelerated.hidden_states, exact.hidden_states))
    return ModelDivergence(kl, top1, prompt_agree, layers)


class FullModelQualificationHarness:
    """Compare an accelerated RC10 runner against exact whole-model replay."""
    def __init__(self, oracle: ExactFullModelOracle, accelerated_runner: Callable[[AssemblyCase], ExactModelResult]):
        self.oracle = oracle
        self.accelerated_runner = accelerated_runner

    def run_case(self, case: AssemblyCase) -> QualificationRecord:
        exact = self.oracle.run(case.tokens)
        accelerated = self.accelerated_runner(case)
        return QualificationRecord(case.name, case.kind.value, len(case.tokens), compare_model_results(exact, accelerated))

    def run(self, cases: Sequence[AssemblyCase]) -> tuple[QualificationRecord, ...]:
        return tuple(self.run_case(case) for case in cases)

    @staticmethod
    def write_jsonl(path: str | Path, records: Sequence[QualificationRecord]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.jsonable(), sort_keys=True) + "\n")


__all__ = [
    "AssemblyCaseKind", "TokenBlock", "AssemblyCase", "ExactModelResult",
    "ModelDivergence", "QualificationRecord", "assemble", "standard_cases",
    "ExactFullModelOracle", "compare_model_results", "FullModelQualificationHarness",
]
