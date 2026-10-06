from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import math

@dataclass
class DriftClock:
    model_time: float = 0.0
    since_replay: float = 0.0

    def observe_update(self, delta_norm: float, reference_norm: float = 1.0) -> float:
        step = abs(float(delta_norm)) / max(abs(float(reference_norm)), 1e-12)
        self.model_time += step
        self.since_replay += step
        return self.model_time

    def mark_replay(self):
        self.since_replay = 0.0

@dataclass
class ReplayItem:
    item_id: str
    importance: float = 0.5
    rarity: float = 0.5
    forgetting_risk: float = 0.5
    uncertainty: float = 0.5
    previous_loss: float = 0.0
    memory_strength: float = 0.5
    last_rehearsal: float = 0.0

class AdaptiveReplayScheduler:
    """Model-drift trigger + sample-strength priority.

    This operationalizes the FOREVER/MSSR-style principles without claiming a
    faithful reproduction of either research implementation.
    """
    def __init__(self, *, drift_trigger: float = 0.01):
        self.drift_trigger = float(drift_trigger)

    def should_replay(self, clock: DriftClock) -> bool:
        return clock.since_replay >= self.drift_trigger

    def score(self, item: ReplayItem, model_time: float) -> float:
        age = max(0.0, float(model_time) - float(item.last_rehearsal))
        weak = 1.0 - max(0.0, min(1.0, item.memory_strength))
        return (
            1.5 * item.importance + 1.2 * item.rarity +
            1.5 * item.forgetting_risk + 0.8 * item.uncertainty +
            0.5 * math.log1p(max(0.0, item.previous_loss)) +
            0.4 * math.log1p(age) + 1.1 * weak
        )

    def select(self, items: Iterable[ReplayItem], clock: DriftClock, *, limit: int = 32):
        rows = sorted(
            ((self.score(x, clock.model_time), x) for x in items),
            key=lambda z: (-z[0], z[1].item_id),
        )
        return [x for _, x in rows[:max(0, int(limit))]]

    def rehearse(self, item: ReplayItem, clock: DriftClock, success: float = 1.0):
        item.last_rehearsal = clock.model_time
        success = max(0.0, min(1.0, float(success)))
        item.memory_strength = min(1.0, 0.7 * item.memory_strength + 0.3 * success)
