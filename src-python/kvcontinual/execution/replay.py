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

    def sample(self, n: int, seed: int = 0, *, with_replacement: bool = False) -> list[ReplayItem]:
        """Weighted replay sampling.

        RC11 defaults to weighted sampling *without* replacement so a rehearsal
        batch does not silently duplicate examples while excluding available
        unique memories. Replacement remains available as an explicit policy.
        """
        if not self.items or n <= 0:
            return []
        rng = random.Random(seed)
        k = min(n, len(self.items)) if not with_replacement else n
        weights = [max(i.priority, 1e-6) for i in self.items]
        if with_replacement:
            return rng.choices(self.items, weights=weights, k=k)
        pool = list(self.items)
        w = list(weights)
        out: list[ReplayItem] = []
        for _ in range(k):
            total = sum(w)
            target = rng.random() * total
            acc = 0.0
            idx = len(w) - 1
            for j, weight in enumerate(w):
                acc += weight
                if target <= acc:
                    idx = j
                    break
            out.append(pool.pop(idx))
            w.pop(idx)
        return out


class DriftReplayTrigger:
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def should_replay(self, normalized_parameter_drift: float) -> bool:
        return normalized_parameter_drift >= self.threshold
