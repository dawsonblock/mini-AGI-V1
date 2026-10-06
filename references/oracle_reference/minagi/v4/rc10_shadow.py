from __future__ import annotations

"""Shadow qualification for the executable RC10 state planes."""

from dataclasses import dataclass, asdict
from typing import Mapping, Sequence, Any

import torch

from .oracle import TensorDivergence, tensor_divergence
from .pic import RelocatedFAPIC, append_segments, relocate
from .rc10 import RC10BlockCache, RiskFeatures
from .rc10_execution import CanonicalRC10Block, RC10StatePlaneExecutor, ReplayPlaneResult


@dataclass(frozen=True)
class AttentionPlaneDivergence:
    layer: int
    key: TensorDivergence
    value: TensorDivergence


@dataclass(frozen=True)
class RC10ShadowResult:
    logical_token_count: int
    reused_blocks: int
    replayed_blocks: int
    recurrent: tuple[tuple[int, TensorDivergence], ...]
    full_attention: tuple[AttentionPlaneDivergence, ...]
    convolution: tuple[tuple[int, TensorDivergence], ...]
    execution_records: tuple[dict[str, Any], ...]

    def jsonable(self) -> dict[str, Any]:
        return {
            "logical_token_count": self.logical_token_count,
            "reused_blocks": self.reused_blocks,
            "replayed_blocks": self.replayed_blocks,
            "recurrent": [
                {"layer": layer, **asdict(div)} for layer, div in self.recurrent
            ],
            "full_attention": [
                {"layer": x.layer, "key": asdict(x.key), "value": asdict(x.value)}
                for x in self.full_attention
            ],
            "convolution": [
                {"layer": layer, **asdict(div)} for layer, div in self.convolution
            ],
            "execution_records": list(self.execution_records),
        }


def _append_exact_attention(
    current: Mapping[int, RelocatedFAPIC],
    fresh,
    start_position: int,
) -> dict[int, RelocatedFAPIC]:
    out = dict(current)
    for layer, seg in fresh.items():
        part = relocate(seg, start_position)
        if int(layer) in out:
            part = append_segments((out[int(layer)], part))
        out[int(layer)] = part
    return out


class RC10ShadowHarness:
    """Compare accelerated RC10 state planes against exact segment replay.

    This harness intentionally does not fabricate logits from state planes.  It
    measures the tensors RC10 actually accelerates and leaves end-to-end logit
    qualification to a runtime that supports safe state injection.
    """

    def __init__(self, executor: RC10StatePlaneExecutor):
        self.executor = executor

    def run(
        self,
        blocks: Sequence[CanonicalRC10Block],
        caches: Mapping[str, RC10BlockCache],
        risks: Mapping[str, RiskFeatures],
    ) -> RC10ShadowResult:
        accelerated = self.executor.execute(blocks, caches, risks)

        prefix: list[int] = []
        exact_recurrent: dict[int, torch.Tensor] = {}
        exact_attention: dict[int, RelocatedFAPIC] = {}
        exact_conv = {}
        for block in blocks:
            exact: ReplayPlaneResult = self.executor.backend.replay_segment(prefix, block)
            exact_recurrent = {int(k): v for k, v in exact.recurrent_states.items()}
            exact_conv = {int(k): v for k, v in exact.conv_states.items()}
            exact_attention = _append_exact_attention(exact_attention, exact.fresh_attention, len(prefix))
            prefix.extend(block.tokens)

        if set(accelerated.recurrent_states) != set(exact_recurrent):
            raise ValueError("accelerated/exact recurrent layer sets differ")
        recurrent = tuple(
            (layer, tensor_divergence(accelerated.recurrent_states[layer], exact_recurrent[layer]))
            for layer in sorted(exact_recurrent)
        )

        if set(accelerated.full_attention) != set(exact_attention):
            raise ValueError("accelerated/exact full-attention layer sets differ")
        fa = []
        for layer in sorted(exact_attention):
            a, e = accelerated.full_attention[layer], exact_attention[layer]
            if a.logical_positions != e.logical_positions:
                raise ValueError("accelerated/exact attention histories differ")
            fa.append(AttentionPlaneDivergence(
                layer,
                tensor_divergence(a.key, e.key),
                tensor_divergence(a.value, e.value),
            ))

        if set(accelerated.conv_states) != set(exact_conv):
            raise ValueError("accelerated/exact convolution layer sets differ")
        conv = tuple(
            (layer, tensor_divergence(accelerated.conv_states[layer].history, exact_conv[layer].history))
            for layer in sorted(exact_conv)
        )

        reused = sum(r.path == "reuse" for r in accelerated.records)
        return RC10ShadowResult(
            len(accelerated.logical_tokens), reused, accelerated.replayed_blocks,
            recurrent, tuple(fa), conv, tuple(asdict(r) for r in accelerated.records),
        )


__all__ = ["AttentionPlaneDivergence", "RC10ShadowResult", "RC10ShadowHarness"]
