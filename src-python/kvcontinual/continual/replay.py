from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass
class ReplayItem:
    payload: dict
    importance: float = 0.5
    rarity: float = 0.5
    forgetting_risk: float = 0.5
    previous_loss: float = 0.0

    @property
    def priority(self) -> float:
        return 0.35*self.importance + 0.25*self.rarity + 0.30*self.forgetting_risk + 0.10*min(1.0, self.previous_loss)


class ReplayStore:
    def __init__(self):
        self.items: list[ReplayItem] = []

    def add(self, item: ReplayItem) -> None:
        self.items.append(item)

    def sample(self, n: int, seed: int = 0) -> list[ReplayItem]:
        """Weighted sampling without replacement.

        Replaying the same row multiple times in one small batch is usually an
        accidental loss of coverage, so each stored replay item can appear at
        most once per sample call.
        """
        if not self.items or n <= 0:
            return []
        rng = random.Random(seed)
        pool = list(self.items)
        out: list[ReplayItem] = []
        for _ in range(min(n, len(pool))):
            weights = [max(i.priority, 1e-9) for i in pool]
            total = sum(weights)
            needle = rng.random() * total
            acc = 0.0
            chosen = len(pool) - 1
            for idx, weight in enumerate(weights):
                acc += weight
                if needle <= acc:
                    chosen = idx
                    break
            out.append(pool.pop(chosen))
        return out


class DriftReplayTrigger:
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def should_replay(self, normalized_parameter_drift: float) -> bool:
        return normalized_parameter_drift >= self.threshold
