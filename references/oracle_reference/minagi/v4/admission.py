from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class MemoryAction(str, Enum):
    DISCARD = "discard"
    WORKING = "working_memory"
    EPISODIC = "episodic_memory"
    SEMANTIC_CANDIDATE = "semantic_memory_candidate"
    TRAINING_CANDIDATE = "training_candidate"

@dataclass(frozen=True)
class AdmissionSignals:
    future_utility: float
    novelty: float
    recurrence: float
    reliability: float
    redundancy: float
    uncertainty: float
    volatility: float = 0.5
    procedurality: float = 0.0

    def clipped(self):
        vals = [max(0.0, min(1.0, float(x))) for x in (
            self.future_utility, self.novelty, self.recurrence, self.reliability,
            self.redundancy, self.uncertainty, self.volatility, self.procedurality,
        )]
        return AdmissionSignals(*vals)

@dataclass(frozen=True)
class AdmissionDecision:
    action: MemoryAction
    value: float
    reason: str

class MemoryAdmissionController:
    """Interpretable baseline gate before any learned write controller.

    The coefficients are explicitly a baseline to beat, not a research claim.
    A learned controller should only replace this if evaluation shows material
    improvement on retention, precision, and storage cost.
    """
    def __init__(self, *, discard_below: float = 0.20,
                 semantic_above: float = 0.55, training_above: float = 0.72):
        self.discard_below = float(discard_below)
        self.semantic_above = float(semantic_above)
        self.training_above = float(training_above)

    def score(self, signals: AdmissionSignals) -> float:
        s = signals.clipped()
        return (
            0.28 * s.future_utility + 0.18 * s.novelty + 0.14 * s.recurrence +
            0.22 * s.reliability - 0.10 * s.redundancy - 0.08 * s.uncertainty +
            0.06 * s.procedurality
        )

    def decide(self, signals: AdmissionSignals) -> AdmissionDecision:
        s = signals.clipped(); value = self.score(s)
        if value < self.discard_below:
            return AdmissionDecision(MemoryAction.DISCARD, value, "low expected value")
        if s.procedurality >= 0.7 and s.volatility <= 0.35 and value >= self.training_above:
            return AdmissionDecision(MemoryAction.TRAINING_CANDIDATE, value,
                                     "stable procedural evidence with high value")
        if s.reliability >= 0.6 and value >= self.semantic_above:
            return AdmissionDecision(MemoryAction.SEMANTIC_CANDIDATE, value,
                                     "reliable durable information candidate")
        if value >= 0.35:
            return AdmissionDecision(MemoryAction.EPISODIC, value,
                                     "useful but not sufficiently stable for semantic/training promotion")
        return AdmissionDecision(MemoryAction.WORKING, value, "short-horizon relevance")
