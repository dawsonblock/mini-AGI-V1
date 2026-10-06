from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from .canonical import sha256_json


@dataclass(frozen=True)
class ExperimentCheckpoint:
    experience_count: int
    base_model_digest: str
    future_task_success: float
    old_task_success: float
    calibration_error: float
    tokens_per_task: float
    tool_calls_per_task: float
    seconds_per_task: float
    skill_reuse_rate: float
    repeated_error_rate: float

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class FrozenFoundationExperimentSpec:
    base_model_digest: str
    sealed_future_task_digest: str
    checkpoints: tuple[int, ...] = (0, 50, 100, 250, 500, 1000)
    max_retention_drop: float = 0.01
    schema: str = "mini-agi-egai-frozen-foundation-experiment-v2"

    def __post_init__(self) -> None:
        if not self.base_model_digest.startswith("sha256:"):
            raise ValueError("base_model_digest must be sha256")
        if not self.sealed_future_task_digest.startswith("sha256:"):
            raise ValueError("sealed_future_task_digest must be sha256")
        if tuple(sorted(set(self.checkpoints))) != tuple(self.checkpoints) or self.checkpoints[0] != 0:
            raise ValueError("checkpoints must be strictly increasing and begin at zero")


class FrozenFoundationProtocol:
    def __init__(self, spec: FrozenFoundationExperimentSpec):
        self.spec = spec
        self.results: list[ExperimentCheckpoint] = []

    def record(self, checkpoint: ExperimentCheckpoint) -> None:
        if checkpoint.base_model_digest != self.spec.base_model_digest:
            raise RuntimeError("foundation model changed during frozen-foundation experiment")
        expected = self.spec.checkpoints[len(self.results)] if len(self.results) < len(self.spec.checkpoints) else None
        if expected is None or checkpoint.experience_count != expected:
            raise ValueError("checkpoint does not match experiment schedule")
        self.results.append(checkpoint)

    def summary(self) -> dict:
        if not self.results:
            return {"complete": False, "checkpoints": 0}
        first = self.results[0]
        last = self.results[-1]
        return {
            "complete": len(self.results) == len(self.spec.checkpoints),
            "checkpoints": len(self.results),
            "future_task_delta": last.future_task_success - first.future_task_success,
            "retention_delta": last.old_task_success - first.old_task_success,
            "calibration_delta": last.calibration_error - first.calibration_error,
            "frozen_model_verified": all(x.base_model_digest == self.spec.base_model_digest for x in self.results),
            "passes_primary_hypothesis": (
                last.future_task_success > first.future_task_success
                and (first.old_task_success - last.old_task_success) <= self.spec.max_retention_drop
                and all(x.base_model_digest == self.spec.base_model_digest for x in self.results)
            ),
        }


def compute_fte(
    before: float,
    after: float,
    *,
    experience: int,
    compute_cost: float = 0.0,
    capacity_growth: float = 0.0,
    compute_lambda: float = 1.0,
    capacity_mu: float = 1.0,
) -> float:
    denom = max(
        1e-12,
        float(experience) + compute_lambda * max(0.0, compute_cost) + capacity_mu * max(0.0, capacity_growth),
    )
    return (float(after) - float(before)) / denom
