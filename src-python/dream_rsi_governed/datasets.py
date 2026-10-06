from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .models import DatasetSplit, ReplayWorld


@dataclass(frozen=True)
class SplitConfig:
    development_pct: int = 60
    selection_pct: int = 20
    holdout_pct: int = 20
    salt: str = "dream-rsi-governed-v1"

    def __post_init__(self):
        if self.development_pct + self.selection_pct + self.holdout_pct != 100:
            raise ValueError("split percentages must sum to 100")


def split_for_world(world: ReplayWorld, config: SplitConfig = SplitConfig()) -> DatasetSplit:
    digest = hashlib.sha256(f"{config.salt}|{world.world_id}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    if bucket < config.development_pct:
        return DatasetSplit.DEVELOPMENT
    if bucket < config.development_pct + config.selection_pct:
        return DatasetSplit.SELECTION
    return DatasetSplit.HOLDOUT


def partition(worlds: list[ReplayWorld], config: SplitConfig = SplitConfig()) -> dict[DatasetSplit, list[ReplayWorld]]:
    out = {s: [] for s in DatasetSplit}
    for w in worlds:
        out[split_for_world(w, config)].append(w)
    return out
