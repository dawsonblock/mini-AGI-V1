from __future__ import annotations

from dataclasses import dataclass, field
from kvcontinual.continual.replay import ReplayItem, ReplayStore


@dataclass
class LearningExample:
    prompt: str
    target: str
    source_episode_id: str
    importance: float = 0.5
    verified: bool = False


@dataclass
class CandidateDataset:
    new_examples: list[LearningExample] = field(default_factory=list)
    replay_examples: list[ReplayItem] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.new_examples) + len(self.replay_examples)


class CandidateDatasetBuilder:
    def __init__(self, replay_store: ReplayStore, new_fraction: float = 0.67):
        if not 0 < new_fraction <= 1:
            raise ValueError("new_fraction must be in (0,1]")
        self.replay_store = replay_store
        self.new_fraction = new_fraction

    def build(self, new_examples: list[LearningExample], seed: int = 0) -> CandidateDataset:
        verified = [x for x in new_examples if x.verified]
        if not verified:
            raise ValueError("no verified learning examples")
        replay_n = int(round(len(verified) * (1 - self.new_fraction) / self.new_fraction))
        replay = self.replay_store.sample(replay_n, seed=seed)
        return CandidateDataset(new_examples=verified, replay_examples=replay)
