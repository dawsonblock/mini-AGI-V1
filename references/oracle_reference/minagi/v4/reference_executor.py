from __future__ import annotations

"""Numerical RC10 reference executor for qualification.

The executor works over already-captured recurrent layer inputs.  It therefore
answers one narrow question exactly: does seam replay followed by the cached
suffix summary produce the same recurrent state as replaying the selected
captured segment inputs?  Whole-model hidden-state drift is intentionally left
to the exact model replay harness.
"""

from dataclasses import dataclass
from typing import Sequence
import torch

from .gdn_reference import GDNInputs, delta_step
from .oracle import OracleComparison, compare
from .rc10 import AffineSegmentSummary, TransitionOrientation


@dataclass(frozen=True)
class SegmentExecution:
    segment_id: str
    steps: tuple[GDNInputs, ...]
    cached_suffix: AffineSegmentSummary

    @property
    def seam_width(self) -> int:
        return int(self.cached_suffix.seam_width)


@dataclass(frozen=True)
class ReferenceExecutionResult:
    accelerated_state: torch.Tensor
    exact_state: torch.Tensor
    comparison: OracleComparison


def execute_segment(
    state: torch.Tensor,
    segment: SegmentExecution,
    *,
    orientation: TransitionOrientation | None = None,
) -> torch.Tensor:
    orient = orientation or segment.cached_suffix.orientation
    if orient != segment.cached_suffix.orientation:
        raise ValueError("orientation mismatch")
    seam = segment.seam_width
    if seam < 0 or seam >= len(segment.steps):
        raise ValueError("invalid compiled seam")
    out = state
    for step in segment.steps[:seam]:
        out = delta_step(out, step, orientation=orient)
    return segment.cached_suffix.apply(out)


def execute_exact(
    state: torch.Tensor,
    segments: Sequence[SegmentExecution],
    *,
    orientation: TransitionOrientation,
) -> torch.Tensor:
    out = state
    for segment in segments:
        for step in segment.steps:
            out = delta_step(out, step, orientation=orientation)
    return out


def execute_accelerated(
    state: torch.Tensor,
    segments: Sequence[SegmentExecution],
    *,
    orientation: TransitionOrientation,
) -> torch.Tensor:
    out = state
    for segment in segments:
        out = execute_segment(out, segment, orientation=orientation)
    return out


def qualify(
    state: torch.Tensor,
    segments: Sequence[SegmentExecution],
    *,
    orientation: TransitionOrientation = TransitionOrientation.LEFT,
) -> ReferenceExecutionResult:
    accelerated = execute_accelerated(state, segments, orientation=orientation)
    exact = execute_exact(state, segments, orientation=orientation)
    return ReferenceExecutionResult(accelerated, exact, compare(accelerated, exact))
