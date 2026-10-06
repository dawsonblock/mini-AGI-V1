from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class ModelTimeClock:
    """Tracks parameter displacement rather than counting optimizer steps."""

    value: float = 0.0

    def advance(self, update_norm: float) -> float:
        update_norm = float(update_norm)
        if not math.isfinite(update_norm) or update_norm < 0.0:
            raise ValueError("update_norm must be finite and non-negative")
        self.value += update_norm
        return self.value


@dataclass(frozen=True)
class ReplayDecision:
    should_replay: bool
    priority: float
    reason: str


@dataclass
class AdaptiveReplayScheduler:
    model_time_interval: float = 1.0
    forgetting_weight: float = 1.0
    importance_weight: float = 0.5
    min_priority: float = 0.5
    _last_replay_model_time: float = 0.0

    def decide(self, *, model_time: float, estimated_forgetting_risk: float, memory_importance: float) -> ReplayDecision:
        vals = (model_time, estimated_forgetting_risk, memory_importance)
        if any(not math.isfinite(float(x)) for x in vals):
            raise ValueError("replay inputs must be finite")
        displacement = max(0.0, float(model_time) - self._last_replay_model_time)
        priority = (
            displacement / max(1e-12, self.model_time_interval)
            + self.forgetting_weight * max(0.0, float(estimated_forgetting_risk))
            + self.importance_weight * max(0.0, float(memory_importance))
        )
        should = displacement >= self.model_time_interval or priority >= self.min_priority
        reason = "parameter displacement / forgetting risk crossed replay threshold" if should else "replay not yet justified"
        return ReplayDecision(should, priority, reason)

    def mark_replayed(self, model_time: float) -> None:
        if float(model_time) < self._last_replay_model_time:
            raise ValueError("model_time cannot move backwards")
        self._last_replay_model_time = float(model_time)
