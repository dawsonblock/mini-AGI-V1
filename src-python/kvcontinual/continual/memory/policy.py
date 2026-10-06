from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemoryDisposition(str, Enum):
    DISCARD = "DISCARD"
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"
    LEARNING_CANDIDATE = "LEARNING_CANDIDATE"


@dataclass
class MemorySignals:
    novelty: float
    future_utility: float
    confidence: float
    recurrence: float
    importance: float
    redundancy: float


class MemoryWritePolicy:
    def score(self, x: MemorySignals) -> float:
        return (
            0.25 * x.novelty
            + 0.30 * x.future_utility
            + 0.15 * x.confidence
            + 0.15 * x.recurrence
            + 0.15 * x.importance
            - 0.30 * x.redundancy
        )

    def decide(self, x: MemorySignals) -> MemoryDisposition:
        s = self.score(x)
        if s < 0.25:
            return MemoryDisposition.DISCARD
        if s < 0.50:
            return MemoryDisposition.EPISODIC
        if s < 0.75:
            return MemoryDisposition.SEMANTIC
        return MemoryDisposition.LEARNING_CANDIDATE
