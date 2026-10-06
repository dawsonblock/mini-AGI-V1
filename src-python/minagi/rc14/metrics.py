from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping


def _finite(values: Iterable[float]) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if any(not math.isfinite(v) for v in out):
        raise ValueError("metrics must be finite")
    return out


def forgetting(previous_scores: Iterable[float], current_score: float) -> float:
    prior = _finite(previous_scores)
    current = _finite((current_score,))[0]
    if not prior:
        return 0.0
    return max(0.0, max(prior) - current)


def forward_transfer(history_score: float, frozen_baseline_score: float) -> float:
    a, b = _finite((history_score, frozen_baseline_score))
    return a - b


def backward_transfer(old_after: float, old_before: float) -> float:
    a, b = _finite((old_after, old_before))
    return a - b


def interference_matrix(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    if set(before) != set(after):
        raise ValueError("before/after capability sets must match")
    return {k: _finite((after[k], before[k]))[0] - float(before[k]) for k in sorted(before)}


@dataclass(frozen=True)
class ContinualMetrics:
    fresh_task_gain: float
    mean_forgetting: float
    worst_case_forgetting: float
    forward_transfer: float
    backward_transfer: float
    falsification_survival: float
    security_regressions: int
    unauthorized_writes: int
    memory_growth: float
    compute_cost: float
    latency_delta: float
    reproducibility: float

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if name in {"security_regressions", "unauthorized_writes"}:
                if int(value) < 0:
                    raise ValueError(f"{name} must be non-negative")
            elif not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.falsification_survival <= 1.0:
            raise ValueError("falsification_survival must be in [0,1]")
        if not 0.0 <= self.reproducibility <= 1.0:
            raise ValueError("reproducibility must be in [0,1]")
