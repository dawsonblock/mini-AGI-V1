from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from .memory import CanonicalMemoryStore
from .admission import MemoryAdmissionController, AdmissionSignals, MemoryAction
from .generation import EffectiveModelGeneration
from .cache_registry import NeuralCacheRegistry

@dataclass
class ControlledLearningRuntime:
    """Minimal authority-separated v4 orchestration surface.

    It records experience and creates candidates; it does not directly mutate a
    foundation model. Promotion and cache compilation are separate services.
    """
    memory: CanonicalMemoryStore
    admission: MemoryAdmissionController
    generation: EffectiveModelGeneration
    cache_registry: NeuralCacheRegistry

    @classmethod
    def create(cls, root: str | Path, generation: EffectiveModelGeneration):
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        return cls(
            CanonicalMemoryStore(root / "canonical_memory.sqlite"),
            MemoryAdmissionController(), generation,
            NeuralCacheRegistry(root / "compiled_cache_registry"),
        )

    def observe(self, content: str, *, source: str,
                signals: AdmissionSignals, provenance: dict | None = None):
        decision = self.admission.decide(signals)
        if decision.action == MemoryAction.DISCARD:
            return decision, None
        memory_type = {
            MemoryAction.WORKING: "episodic",
            MemoryAction.EPISODIC: "episodic",
            MemoryAction.SEMANTIC_CANDIDATE: "semantic",
            MemoryAction.TRAINING_CANDIDATE: "procedural",
        }[decision.action]
        state = "candidate" if decision.action in {
            MemoryAction.SEMANTIC_CANDIDATE, MemoryAction.TRAINING_CANDIDATE
        } else "active"
        record = self.memory.append(
            type=memory_type, content=content, source=source,
            confidence=signals.reliability, salience=signals.future_utility,
            usefulness=max(0.0, min(1.0, decision.value)), state=state,
            provenance=provenance,
        )
        return decision, record
