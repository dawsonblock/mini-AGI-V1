from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from kvcontinual.continual.config import load_runtime_settings
from kvcontinual.continual.memory import MemoryRecord, MemoryStore, MemorySignals, MemoryWritePolicy

settings = load_runtime_settings()
app = FastAPI(title="KVContinual Memory API", version="0.10.1")
memory = MemoryStore(settings.memory_db)
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


@app.get("/health")
def health():
    return {"ok": True, "version": "0.10.1", "memory_db": settings.memory_db}


@app.get("/memory/current")
def current_memory(memory_type: str | None = None):
    return [r.__dict__ for r in memory.list_current(memory_type=memory_type)]


@app.post("/memory")
def add_memory(x: MemoryIn):
    sig = MemorySignals(
        novelty=x.novelty,
        future_utility=x.future_utility,
        confidence=x.confidence,
        recurrence=x.recurrence,
        importance=x.importance,
        redundancy=x.redundancy,
    )
    disposition = policy.decide(sig)
    if disposition.value == "DISCARD":
        return {"stored": False, "disposition": disposition.value, "score": policy.score(sig)}
    rec = MemoryRecord(
        text=x.text,
        memory_type=x.memory_type,
        source_type=x.source_type,
        source_id=x.source_id,
        confidence=x.confidence,
        novelty=x.novelty,
        future_utility=x.future_utility,
        recurrence=x.recurrence,
        importance=x.importance,
        redundancy=x.redundancy,
    )
    memory.put(rec)
    return {"stored": True, "id": rec.id, "disposition": disposition.value, "score": policy.score(sig)}
