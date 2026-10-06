from __future__ import annotations

"""RC10 v4.3 state-plane execution engine.

The engine is deliberately model-agnostic.  A backend is responsible for
replaying a *fresh* boundary (or a complete segment) under the current logical
prefix.  RC10 then applies only a cache variant compiled for exactly that seam
width.  This makes the seam-before-reuse invariant executable rather than just
an assembly plan.

This module does not claim that the resulting tensors can be injected into every
vendor model.  It owns the coherent state-plane semantics and qualification
record; model/runtime-specific injection remains a separate backend concern.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol, Sequence

import torch

from .generation import EffectiveModelGeneration
from .conv_boundary import ConvBoundaryState
from .pic import FAPICSegment, RelocatedFAPIC, append_segments, relocate
from .rc10 import RC10BlockCache, RepairDecision, RiskFeatures, RiskRouter
from .rc10_compiler import token_digest


@dataclass(frozen=True)
class CanonicalRC10Block:
    block_id: str
    tokens: tuple[int, ...]

    @classmethod
    def build(cls, block_id: str, tokens: Sequence[int]) -> "CanonicalRC10Block":
        return cls(str(block_id), tuple(map(int, tokens)))

    @property
    def digest(self) -> str:
        return token_digest(self.tokens)


@dataclass(frozen=True)
class ReplayPlaneResult:
    """Fresh model-derived state for material that was actually replayed.

    ``recurrent_states`` are the complete recurrent states after the logical
    prefix plus replayed material. ``fresh_attention`` contains only the K/V for
    the replayed material, already conditioned on that prefix.
    """

    recurrent_states: Mapping[int, torch.Tensor]
    fresh_attention: Mapping[int, FAPICSegment] = field(default_factory=dict)
    conv_states: Mapping[int, ConvBoundaryState] = field(default_factory=dict)


class RC10ReplayBackend(Protocol):
    """Model-specific exact-replay provider used by the generic RC10 engine."""

    @property
    def full_attention_layers(self) -> tuple[int, ...]: ...

    @property
    def convolution_layers(self) -> tuple[int, ...]: ...

    def replay_boundary(
        self,
        prefix_tokens: Sequence[int],
        block: CanonicalRC10Block,
        seam_width: int,
    ) -> ReplayPlaneResult: ...

    def replay_segment(
        self,
        prefix_tokens: Sequence[int],
        block: CanonicalRC10Block,
    ) -> ReplayPlaneResult: ...

    def replay_selected(self, tokens: Sequence[int]) -> ReplayPlaneResult: ...


class ExecutionPath(str, Enum):
    REUSE = "reuse"
    SEGMENT_REPLAY = "segment_replay"
    EXACT_REPLAY = "exact_selected_replay"


@dataclass(frozen=True)
class BlockExecutionRecord:
    block_id: str
    decision: str
    path: str
    seam_width: int
    risk_probability: float
    prefix_tokens_before: int
    block_tokens: int


@dataclass(frozen=True)
class RC10ExecutionResult:
    logical_tokens: tuple[int, ...]
    recurrent_states: Mapping[int, torch.Tensor]
    full_attention: Mapping[int, RelocatedFAPIC]
    conv_states: Mapping[int, ConvBoundaryState]
    records: tuple[BlockExecutionRecord, ...]

    @property
    def replayed_blocks(self) -> int:
        return sum(r.path != ExecutionPath.REUSE.value for r in self.records)


def _fa_mapping(payload) -> dict[int, FAPICSegment]:
    if payload is None:
        return {}
    if isinstance(payload, FAPICSegment):
        # Legacy v4.1/v4.2 single-plane payload.  Layer -1 is intentionally
        # synthetic; a backend that requires named FA layers will therefore
        # cause this payload to fail closed toward segment replay.
        return {-1: payload}
    if isinstance(payload, Mapping):
        out: dict[int, FAPICSegment] = {}
        for k, v in payload.items():
            if not isinstance(v, FAPICSegment):
                raise TypeError("full-attention cache mapping must contain FAPICSegment values")
            out[int(k)] = v
        return out
    raise TypeError("unsupported full-attention cache payload")


def _append_fresh(
    existing: Mapping[int, RelocatedFAPIC],
    fresh: Mapping[int, FAPICSegment],
    start_position: int,
) -> dict[int, RelocatedFAPIC]:
    out = dict(existing)
    for layer, seg in fresh.items():
        part = relocate(seg, start_position)
        if int(layer) in out:
            part = append_segments((out[int(layer)], part))
        out[int(layer)] = part
    return out


def _append_cached_suffix(
    existing: Mapping[int, RelocatedFAPIC],
    cached: Mapping[int, FAPICSegment],
    *,
    start_position: int,
    seam_width: int,
) -> dict[int, RelocatedFAPIC]:
    out = dict(existing)
    for layer, seg in cached.items():
        suffix = relocate(seg, start_position + seam_width, skip=seam_width)
        if not suffix.logical_positions:
            continue
        if int(layer) in out:
            suffix = append_segments((out[int(layer)], suffix))
        out[int(layer)] = suffix
    return out


class RC10StatePlaneExecutor:
    """Execute coherent RC10 state planes with monotonic fallback.

    The risk router is evaluated before each block from caller-supplied features.
    Reuse is accepted only when:
      * canonical token count/digest match the compiled artifact;
      * effective model generation/cache ABI match;
      * the exact requested seam variant exists;
      * every full-attention layer required by the backend has a compiled plane.

    Otherwise execution falls toward exact segment replay.  No stale/mismatched
    cache artifact is used opportunistically.
    """

    def __init__(
        self,
        generation: EffectiveModelGeneration,
        router: RiskRouter,
        backend: RC10ReplayBackend,
    ):
        self.generation = generation
        self.router = router
        self.backend = backend

    def _validate_block(self, canonical: CanonicalRC10Block, cache: RC10BlockCache) -> None:
        if canonical.block_id != cache.block_id:
            raise ValueError("canonical/cache block id mismatch")
        if len(canonical.tokens) != cache.token_count:
            raise ValueError("canonical/cache token count mismatch")
        if canonical.digest != cache.token_digest:
            raise ValueError("canonical/cache token digest mismatch")
        cache.compatibility.assert_compatible(self.generation)

    def _can_reuse_attention(self, cache: RC10BlockCache) -> bool:
        required = set(map(int, self.backend.full_attention_layers))
        if not required:
            return True
        compiled = set(_fa_mapping(cache.full_attention_payload))
        return required.issubset(compiled)

    def _can_reuse_conv(self, cache: RC10BlockCache) -> bool:
        required = set(map(int, self.backend.convolution_layers))
        if not required:
            return True
        payload = cache.boundary.trailing_conv_payload
        if not isinstance(payload, Mapping):
            return False
        return required.issubset(set(map(int, payload)))

    def execute(
        self,
        blocks: Sequence[CanonicalRC10Block],
        caches: Mapping[str, RC10BlockCache],
        risks: Mapping[str, RiskFeatures],
    ) -> RC10ExecutionResult:
        prefix: list[int] = []
        recurrent: dict[int, torch.Tensor] = {}
        attention: dict[int, RelocatedFAPIC] = {}
        conv_states: dict[int, ConvBoundaryState] = {}
        records: list[BlockExecutionRecord] = []

        for canonical in blocks:
            if canonical.block_id not in caches:
                # No compiled cache is a normal condition: exact segment replay.
                replay = self.backend.replay_segment(prefix, canonical)
                attention = _append_fresh(attention, replay.fresh_attention, len(prefix))
                recurrent = {int(k): v for k, v in replay.recurrent_states.items()}
                conv_states = {int(k): v for k, v in replay.conv_states.items()}
                records.append(BlockExecutionRecord(
                    canonical.block_id, RepairDecision.SEGMENT_REPLAY.value,
                    ExecutionPath.SEGMENT_REPLAY.value, 0, 1.0,
                    len(prefix), len(canonical.tokens),
                ))
                prefix.extend(canonical.tokens)
                continue

            cache = caches[canonical.block_id]
            self._validate_block(canonical, cache)
            risk = risks.get(canonical.block_id)
            if risk is None:
                raise KeyError(f"missing risk features for block {canonical.block_id}")
            probability = self.router.probability(risk)
            decision = self.router.decide(risk, cache.supported_seams())

            if decision in (RepairDecision.EXACT, RepairDecision.SEGMENT_REPLAY):
                if decision == RepairDecision.EXACT:
                    replay = self.backend.replay_selected(tuple(prefix) + canonical.tokens)
                    # EXACT means all state planes are reset to the exact logical
                    # history, not merely the current segment.
                    attention = _append_fresh({}, replay.fresh_attention, 0)
                    path = ExecutionPath.EXACT_REPLAY
                else:
                    replay = self.backend.replay_segment(prefix, canonical)
                    attention = _append_fresh(attention, replay.fresh_attention, len(prefix))
                    path = ExecutionPath.SEGMENT_REPLAY
                recurrent = {int(k): v for k, v in replay.recurrent_states.items()}
                conv_states = {int(k): v for k, v in replay.conv_states.items()}
                records.append(BlockExecutionRecord(
                    canonical.block_id, decision.value, path.value, 0, probability,
                    len(prefix), len(canonical.tokens),
                ))
                prefix.extend(canonical.tokens)
                continue

            width = {
                RepairDecision.REUSE_8: 8,
                RepairDecision.REUSE_32: 32,
                RepairDecision.REUSE_128: 128,
            }[decision]
            variant = cache.variant(width)
            if variant is None or width >= len(canonical.tokens) or not self._can_reuse_attention(cache) or not self._can_reuse_conv(cache):
                replay = self.backend.replay_segment(prefix, canonical)
                attention = _append_fresh(attention, replay.fresh_attention, len(prefix))
                recurrent = {int(k): v for k, v in replay.recurrent_states.items()}
                conv_states = {int(k): v for k, v in replay.conv_states.items()}
                records.append(BlockExecutionRecord(
                    canonical.block_id, RepairDecision.SEGMENT_REPLAY.value,
                    ExecutionPath.SEGMENT_REPLAY.value, 0, probability,
                    len(prefix), len(canonical.tokens),
                ))
                prefix.extend(canonical.tokens)
                continue

            # RC10 invariant: fresh seam first.
            boundary = self.backend.replay_boundary(prefix, canonical, width)
            recurrent = {int(k): v for k, v in boundary.recurrent_states.items()}
            conv_states = {int(k): v for k, v in boundary.conv_states.items()}
            attention = _append_fresh(attention, boundary.fresh_attention, len(prefix))

            # Then, and only then, apply the exact-width cached suffix.
            by_layer = {int(s.layer): s for s in variant.summaries if s.layer is not None}
            if len(by_layer) != len(variant.summaries):
                raise ValueError("compiled recurrent summaries require explicit layer ids")
            missing = set(by_layer) - set(recurrent)
            if missing:
                raise ValueError(f"fresh boundary did not produce recurrent layers {sorted(missing)}")
            for layer, summary in by_layer.items():
                recurrent[layer] = summary.apply(recurrent[layer])

            attention = _append_cached_suffix(
                attention, _fa_mapping(cache.full_attention_payload),
                start_position=len(prefix), seam_width=width,
            )
            if self.backend.convolution_layers:
                payload = cache.boundary.trailing_conv_payload
                for layer in self.backend.convolution_layers:
                    conv_states[int(layer)] = payload[int(layer)]
            records.append(BlockExecutionRecord(
                canonical.block_id, decision.value, ExecutionPath.REUSE.value,
                width, probability, len(prefix), len(canonical.tokens),
            ))
            prefix.extend(canonical.tokens)

        # Strong coherence check for every full-attention plane the backend says
        # is semantically active.
        for layer in self.backend.full_attention_layers:
            part = attention.get(int(layer))
            if part is None:
                raise ValueError(f"missing final full-attention plane for layer {layer}")
            if len(part.logical_positions) != len(prefix):
                raise ValueError(f"full-attention layer {layer} does not describe the full logical history")
            if part.logical_positions != tuple(range(len(prefix))):
                raise ValueError(f"full-attention layer {layer} logical positions are not contiguous")

        missing_conv = set(map(int, self.backend.convolution_layers)) - set(conv_states)
        if missing_conv:
            raise ValueError(f"missing final convolution planes for layers {sorted(missing_conv)}")

        return RC10ExecutionResult(tuple(prefix), recurrent, attention, conv_states, tuple(records))


__all__ = [
    "CanonicalRC10Block", "ReplayPlaneResult", "RC10ReplayBackend",
    "ExecutionPath", "BlockExecutionRecord", "RC10ExecutionResult",
    "RC10StatePlaneExecutor",
]
