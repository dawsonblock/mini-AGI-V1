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
    volatility: float = 0.0
    procedurality: float = 0.0
    behavioral_failure: float = 0.0
    cache_invalidation_cost: float = 0.0


class MemoryWritePolicy:
    """Biases changing facts toward external memory and parameter updates toward stable behavior."""
    def score(self, x: MemorySignals) -> float:
        return (
            0.20 * x.novelty + 0.25 * x.future_utility + 0.15 * x.confidence
            + 0.10 * x.recurrence + 0.15 * x.importance - 0.25 * x.redundancy
        )

    def parameterize_score(self, x: MemorySignals) -> float:
        return (
            0.25 * x.recurrence + 0.25 * x.procedurality + 0.25 * x.behavioral_failure
            + 0.15 * x.future_utility + 0.10 * x.confidence
            - 0.35 * x.volatility - 0.30 * x.cache_invalidation_cost
        )

    def decide(self, x: MemorySignals) -> MemoryDisposition:
        s = self.score(x)
        if s < 0.25:
            return MemoryDisposition.DISCARD
        if s < 0.50:
            return MemoryDisposition.EPISODIC
        if self.parameterize_score(x) >= 0.65:
            return MemoryDisposition.LEARNING_CANDIDATE
        return MemoryDisposition.SEMANTIC
