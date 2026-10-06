from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass
class ModelTimeClock:
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


@dataclass(frozen=True)
class ReplayMemorySignalRC14:
    memory_id: str
    task_family: str
    estimated_forgetting_risk: float
    memory_importance: float
    uncertainty: float = 0.0
    age_model_time: float = 0.0
    interference_risk: float = 0.0
    novelty: float = 0.0

    def __post_init__(self) -> None:
        if not self.memory_id or not self.task_family:
            raise ValueError("memory_id and task_family are required")
        vals = (self.estimated_forgetting_risk, self.memory_importance, self.uncertainty,
                self.age_model_time, self.interference_risk, self.novelty)
        if any(not math.isfinite(float(x)) or float(x) < 0.0 for x in vals):
            raise ValueError("replay memory signals must be finite and non-negative")


@dataclass(frozen=True)
class ReplayBatchRC14:
    memory_ids: tuple[str, ...]
    priorities: tuple[float, ...]
    task_families: tuple[str, ...]
    model_time: float


@dataclass
class AdaptiveReplayScheduler:
    model_time_interval: float = 1.0
    forgetting_weight: float = 1.0
    importance_weight: float = 0.5
    uncertainty_weight: float = 0.5
    age_weight: float = 0.25
    interference_weight: float = 0.75
    novelty_weight: float = 0.20
    min_priority: float = 1.0
    _last_replay_model_time: float = 0.0

    def _priority(self, *, model_time: float, estimated_forgetting_risk: float, memory_importance: float,
                  uncertainty: float = 0.0, age_model_time: float = 0.0, interference_risk: float = 0.0,
                  novelty: float = 0.0) -> float:
        vals = (model_time, estimated_forgetting_risk, memory_importance, uncertainty, age_model_time, interference_risk, novelty)
        if any(not math.isfinite(float(x)) for x in vals):
            raise ValueError("replay inputs must be finite")
        if any(float(x) < 0.0 for x in vals):
            raise ValueError("replay inputs must be non-negative")
        displacement = max(0.0, float(model_time) - self._last_replay_model_time)
        return (
            displacement / max(1e-12, self.model_time_interval)
            + self.forgetting_weight * float(estimated_forgetting_risk)
            + self.importance_weight * float(memory_importance)
            + self.uncertainty_weight * float(uncertainty)
            + self.age_weight * float(age_model_time) / max(1e-12, self.model_time_interval)
            + self.interference_weight * float(interference_risk)
            + self.novelty_weight * float(novelty)
        )

    def decide(self, *, model_time: float, estimated_forgetting_risk: float, memory_importance: float,
               uncertainty: float = 0.0, age_model_time: float = 0.0) -> ReplayDecision:
        priority = self._priority(model_time=model_time, estimated_forgetting_risk=estimated_forgetting_risk,
                                  memory_importance=memory_importance, uncertainty=uncertainty, age_model_time=age_model_time)
        displacement = max(0.0, float(model_time) - self._last_replay_model_time)
        should = displacement >= self.model_time_interval or priority >= self.min_priority
        reason = "model-time/forgetting priority crossed replay threshold" if should else "replay not yet justified"
        return ReplayDecision(should, priority, reason)

    def rank_batch(self, signals: Iterable[ReplayMemorySignalRC14], *, model_time: float, max_items: int,
                   max_per_task_family: int = 2) -> ReplayBatchRC14:
        if max_items < 1 or max_per_task_family < 1:
            raise ValueError("replay batch limits must be positive")
        rows = []
        seen_ids: set[str] = set()
        for signal in signals:
            if signal.memory_id in seen_ids:
                raise ValueError("duplicate replay memory_id")
            seen_ids.add(signal.memory_id)
            p = self._priority(model_time=model_time, estimated_forgetting_risk=signal.estimated_forgetting_risk,
                               memory_importance=signal.memory_importance, uncertainty=signal.uncertainty,
                               age_model_time=signal.age_model_time, interference_risk=signal.interference_risk,
                               novelty=signal.novelty)
            rows.append((p, signal))
        rows.sort(key=lambda x: (-x[0], x[1].task_family, x[1].memory_id))
        counts: dict[str, int] = {}
        chosen: list[tuple[float, ReplayMemorySignalRC14]] = []
        for priority, signal in rows:
            if priority < self.min_priority:
                continue
            if counts.get(signal.task_family, 0) >= max_per_task_family:
                continue
            chosen.append((priority, signal))
            counts[signal.task_family] = counts.get(signal.task_family, 0) + 1
            if len(chosen) >= max_items:
                break
        return ReplayBatchRC14(
            memory_ids=tuple(s.memory_id for _, s in chosen),
            priorities=tuple(float(p) for p, _ in chosen),
            task_families=tuple(s.task_family for _, s in chosen),
            model_time=float(model_time),
        )

    def mark_replayed(self, model_time: float) -> None:
        if not math.isfinite(float(model_time)) or float(model_time) < self._last_replay_model_time:
            raise ValueError("model_time cannot move backwards or be non-finite")
        self._last_replay_model_time = float(model_time)
