from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

from egai.common.canonical import digest, validate_digest
from minagi.v14.experiment_v145 import FrozenArmRecordV145, FrozenBaselineGateV145, FrozenBaselineReportV145


@dataclass(frozen=True)
class SealedPairResult:
    task_digest: str
    fresh_task_receipt_digest: str
    a0: FrozenArmRecordV145
    a1: FrozenArmRecordV145
    schema: str = "mini-agi-v15.1-sealed-pair-result-v1"

    def __post_init__(self) -> None:
        validate_digest(self.task_digest)
        validate_digest(self.fresh_task_receipt_digest)
        if self.a0.task_digest != self.task_digest or self.a1.task_digest != self.task_digest:
            raise ValueError("paired records are not bound to the same sealed task")

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class SealedLearningExperiment:
    pairs: tuple[SealedPairResult, ...]
    baseline_report: FrozenBaselineReportV145
    schema: str = "mini-agi-v15.1-sealed-learning-experiment-v1"

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("sealed learning experiment requires task pairs")
        if self.baseline_report.decision not in {"PASS", "BLOCK"}:
            raise ValueError("invalid baseline decision")

    @property
    def digest(self) -> str:
        return digest(self)


class FreshTaskPairedEvaluator:
    """Reveal each hidden task once inside the evaluator and run A0/A1 on it.

    The learner receives only the signed consumption receipt and aggregate
    experiment output. The same revealed task is used for both arms, which keeps
    the comparison paired without exposing it before candidate freeze.
    """

    def __init__(self, *, vault, metadata, cas, gate: FrozenBaselineGateV145,
                 consumer_id: str = "sealed-a0-a1-evaluator"):
        self.vault = vault
        self.metadata = metadata
        self.cas = cas
        self.gate = gate
        self.consumer_id = str(consumer_id)

    @staticmethod
    def _default_score(output: str, expected: str) -> float:
        return 1.0 if str(output) == str(expected) else 0.0

    def evaluate(self, *, commitments: Sequence, foundation_model_digest: str,
                 production_identity_digest: str,
                 a0: Callable[[str, str], str], a1: Callable[[str, str], str],
                 score_fn: Callable[[str, str], float] | None = None) -> SealedLearningExperiment:
        validate_digest(foundation_model_digest)
        validate_digest(production_identity_digest)
        score_fn = score_fn or self._default_score
        pairs = []
        for commitment in commitments:
            lease = self.metadata.lease(task_id=commitment.task_id, consumer_id=self.consumer_id)
            task, receipt = self.vault.consume(lease, metadata=self.metadata)
            rd = self.cas.put_json(asdict(receipt))
            if rd != receipt.digest:
                raise RuntimeError("fresh-task receipt CAS digest mismatch")
            required = {"family_id", "ring", "input", "expected"}
            if set(task) < required:
                raise ValueError("sealed task is missing required fields")
            family = str(task["family_id"])
            ring = str(task["ring"])
            inp = str(task["input"])
            expected = str(task["expected"])
            out0 = str(a0(family, inp))
            out1 = str(a1(family, inp))
            s0 = float(score_fn(out0, expected))
            s1 = float(score_fn(out1, expected))
            for score in (s0, s1):
                if not 0.0 <= score <= 1.0:
                    raise ValueError("score must be in [0,1]")
            a0r = FrozenArmRecordV145(receipt.task_digest, family, ring, foundation_model_digest,
                                      production_identity_digest, "A0", s0)
            a1r = FrozenArmRecordV145(receipt.task_digest, family, ring, foundation_model_digest,
                                      production_identity_digest, "A1", s1)
            pairs.append(SealedPairResult(receipt.task_digest, receipt.digest, a0r, a1r))
            self.metadata.close(commitment.task_id)
        report = self.gate.evaluate(tuple(p.a0 for p in pairs), tuple(p.a1 for p in pairs))
        exp = SealedLearningExperiment(tuple(pairs), report)
        self.cas.put_json(asdict(exp))
        return exp
