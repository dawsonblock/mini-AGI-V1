from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence, Any
import math
import numpy as np
import torch
from .generation import CacheCompatibility

class TransitionOrientation(str, Enum):
    LEFT = "left_multiply"
    RIGHT = "right_multiply"

@dataclass(frozen=True)
class AffineSegmentSummary:
    transition: torch.Tensor
    zero_state: torch.Tensor
    orientation: TransitionOrientation = TransitionOrientation.LEFT
    seam_width: int = 0
    layer: int | None = None
    segment_id: str = ""

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        if self.orientation == TransitionOrientation.LEFT:
            return torch.matmul(self.transition, state) + self.zero_state
        return torch.matmul(state, self.transition) + self.zero_state


def compose_summaries(first: AffineSegmentSummary,
                      second: AffineSegmentSummary) -> AffineSegmentSummary:
    """Return a summary equivalent to applying ``first`` then ``second``."""
    if first.orientation != second.orientation:
        raise ValueError("orientation mismatch")
    if first.orientation == TransitionOrientation.LEFT:
        transition = torch.matmul(second.transition, first.transition)
        zero_state = torch.matmul(second.transition, first.zero_state) + second.zero_state
    else:
        transition = torch.matmul(first.transition, second.transition)
        zero_state = torch.matmul(first.zero_state, second.transition) + second.zero_state
    return AffineSegmentSummary(
        transition, zero_state, first.orientation,
        seam_width=0,
        layer=second.layer if second.layer is not None else first.layer,
        segment_id="+".join(x for x in (first.segment_id, second.segment_id) if x),
    )


def summary_from_steps(transitions: Sequence[torch.Tensor],
                       writes: Sequence[torch.Tensor], *,
                       orientation: TransitionOrientation = TransitionOrientation.LEFT,
                       segment_id: str = "") -> AffineSegmentSummary:
    if len(transitions) != len(writes) or not transitions:
        raise ValueError("aligned non-empty steps required")
    d = transitions[0].shape[-1]
    eye = torch.eye(d, dtype=transitions[0].dtype, device=transitions[0].device)
    while eye.ndim < transitions[0].ndim:
        eye = eye.unsqueeze(0)
    transition = eye.expand_as(transitions[0]).clone()
    zero_state = torch.zeros_like(writes[0])
    for t, u in zip(transitions, writes):
        if orientation == TransitionOrientation.LEFT:
            zero_state = torch.matmul(t, zero_state) + u
            transition = torch.matmul(t, transition)
        else:
            zero_state = torch.matmul(zero_state, t) + u
            transition = torch.matmul(transition, t)
    return AffineSegmentSummary(transition, zero_state, orientation,
                                segment_id=segment_id)

@dataclass(frozen=True)
class BoundaryPayload:
    leading_tokens: tuple[int, ...] = ()
    trailing_conv_payload: Any = None
    reference_hidden_digest: str = ""
    max_repair_width: int = 8

@dataclass(frozen=True)
class RecurrentVariant:
    seam_width: int
    summaries: tuple[AffineSegmentSummary, ...]

@dataclass(frozen=True)
class RC10BlockCache:
    block_id: str
    token_count: int
    token_digest: str
    compatibility: CacheCompatibility
    recurrent_variants: tuple[RecurrentVariant, ...] = ()
    full_attention_payload: Any = None
    boundary: BoundaryPayload = BoundaryPayload()
    tier: str = "warm"

    def variant(self, width: int) -> RecurrentVariant | None:
        for variant in self.recurrent_variants:
            if variant.seam_width == int(width):
                return variant
        return None

    def supported_seams(self) -> tuple[int, ...]:
        return tuple(sorted(v.seam_width for v in self.recurrent_variants))

@dataclass(frozen=True)
class RiskFeatures:
    seam_state_delta: float
    hidden_anchor_delta: float
    gate_shift: float
    conv_correction: float
    reorder_distance: float
    segment_count: float
    max_segment_tokens: float
    predecessor_changed: float

    def vector(self) -> np.ndarray:
        return np.asarray([
            1.0,
            self.seam_state_delta,
            self.hidden_anchor_delta,
            self.gate_shift,
            self.conv_correction,
            self.reorder_distance,
            self.segment_count,
            self.max_segment_tokens / 8192.0,
            self.predecessor_changed,
        ], dtype=np.float64)

class RepairDecision(str, Enum):
    REUSE_8 = "reuse_8"
    REUSE_32 = "reuse_32"
    REUSE_128 = "reuse_128"
    SEGMENT_REPLAY = "segment_replay"
    EXACT = "exact_selected_replay"

class RiskRouter:
    """Calibrated risk model with a monotonic fallback ladder.

    The weights are intended to be fit from exact-replay labels. RC10 does not
    hard-code a hand-designed coherence score as a correctness assumption.
    """
    def __init__(self, weights: Sequence[float] | None = None,
                 thresholds: tuple[float, float, float] = (0.15, 0.35, 0.65)):
        self.weights = np.zeros(9, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
        if self.weights.shape != (9,):
            raise ValueError("expected nine logistic weights including bias")
        self.thresholds = tuple(map(float, thresholds))

    def probability(self, features: RiskFeatures) -> float:
        z = float(self.weights @ features.vector())
        z = max(-40.0, min(40.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, features: Sequence[RiskFeatures], labels: Sequence[int], *,
            lr: float = 0.1, steps: int = 600, l2: float = 1e-4):
        if len(features) != len(labels) or not features:
            raise ValueError("aligned non-empty calibration data required")
        x = np.stack([f.vector() for f in features])
        y = np.asarray(labels, dtype=np.float64)
        w = self.weights.copy()
        for _ in range(int(steps)):
            z = np.clip(x @ w, -40, 40)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = (x.T @ (p - y)) / len(y) + l2 * w
            w -= float(lr) * grad
        self.weights = w
        return self

    def decide(self, features: RiskFeatures,
               available_seams: Iterable[int] = (8,)) -> RepairDecision:
        p = self.probability(features)
        seams = set(map(int, available_seams))
        a, b, c = self.thresholds
        if p < a and 8 in seams:
            return RepairDecision.REUSE_8
        if p < b and 32 in seams:
            return RepairDecision.REUSE_32
        if p < c and 128 in seams:
            return RepairDecision.REUSE_128
        if p < 0.9:
            return RepairDecision.SEGMENT_REPLAY
        return RepairDecision.EXACT
