from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kvcontinual import __version__
from kvcontinual.execution.memory import MemoryRecord, MemoryStore, MemorySignals, MemoryWritePolicy

app = FastAPI(title="KVContinual RC10", version=__version__)
memory = MemoryStore(":memory:")
policy = MemoryWritePolicy()


class MemoryIn(BaseModel):
    text: str
    memory_type: str = "episode"
    source_type: str = "user"
    source_id: str = "api"
    confidence: float = 0.5
    novelty: float = 0.5
    future_utility: float = 0.5
    recurrence: float = 0.0
    importance: float = 0.5
    redundancy: float = 0.0
    volatility: float = 0.0
    procedurality: float = 0.0
    behavioral_failure: float = 0.0
    cache_invalidation_cost: float = 0.0
    source_segments: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class QualificationIn(BaseModel):
    candidate_id: str
    target_gain: float
    general_regression: float
    critical_pass: bool
    cache_quality_pass: bool
    cache_warmup_cost: float = 0.0


@app.get("/health")
def health():
    return {"ok": True, "version": __version__}


@app.post("/memory")
def add_memory(x: MemoryIn):
    sig = MemorySignals(
        novelty=x.novelty, future_utility=x.future_utility, confidence=x.confidence,
        recurrence=x.recurrence, importance=x.importance, redundancy=x.redundancy,
        volatility=x.volatility, procedurality=x.procedurality,
        behavioral_failure=x.behavioral_failure, cache_invalidation_cost=x.cache_invalidation_cost,
    )
    disposition = policy.decide(sig)
    if disposition.value == "DISCARD":
        return {"stored": False, "disposition": disposition.value, "score": policy.score(sig)}
    rec = MemoryRecord(
        text=x.text, memory_type=x.memory_type, source_type=x.source_type, source_id=x.source_id,
        confidence=x.confidence, novelty=x.novelty, future_utility=x.future_utility,
        recurrence=x.recurrence, importance=x.importance, redundancy=x.redundancy,
        source_segments=x.source_segments, evidence_ids=x.evidence_ids,
    )
    memory.put(rec)
    return {"stored": True, "id": rec.id, "disposition": disposition.value, "score": policy.score(sig), "parameterize_score": policy.parameterize_score(sig)}


@app.post("/qualification")
def qualify_legacy_disabled(x: QualificationIn):
    raise HTTPException(410, "legacy qualification authority disabled; use minagi v14.1 alpha3 authority pipeline")


@app.post("/promotion/{candidate_id}")
def promote_legacy_disabled(candidate_id: str):
    raise HTTPException(410, "legacy promotion authority disabled; use content-bound alpha3 promotion")
