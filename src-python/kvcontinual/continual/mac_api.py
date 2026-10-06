from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from kvcontinual.continual.backends import OpenAIBackend
from kvcontinual.continual.adapter_router import AdapterRouter
from kvcontinual.continual.config import load_runtime_settings
from kvcontinual.continual.experience import Episode, EpisodeLog
from kvcontinual.continual.experience_store import ExperienceStore
from kvcontinual.continual.integrations.openai_compatible import OpenAICompatibleClient
from kvcontinual.continual.macos import MacGateway
from kvcontinual.continual.memory import MemoryRecord, MemorySignals, MemoryStore, MemoryWritePolicy
from egai.common.crypto import Ed25519Verifier
from minagi.v14.verification import EpisodeVerificationReceipt, EpisodeVerificationValidator

settings = load_runtime_settings()
model_cfg = settings.model.get("model", {})
endpoint = os.getenv("KVCONTINUAL_MODEL_ENDPOINT", str(model_cfg.get("endpoint", "http://127.0.0.1:8080/v1")))
model_name = os.getenv("KVCONTINUAL_MODEL_NAME", str(model_cfg.get("name", "auto")))
api_key = os.getenv("KVCONTINUAL_MODEL_API_KEY", "local")
backend_kind = os.getenv("KVCONTINUAL_BACKEND_KIND", str(model_cfg.get("backend", "llama.cpp")))
episode_path = os.getenv("KVCONTINUAL_EPISODE_LOG", "data/episodes.jsonl")

memory = MemoryStore(settings.memory_db)
experience = ExperienceStore(settings.experience_db)
policy = MemoryWritePolicy()
episodes = EpisodeLog(episode_path)
client = OpenAICompatibleClient(endpoint, api_key=api_key)
backend = OpenAIBackend(client, backend_kind=backend_kind)
router_path = os.getenv("KVCONTINUAL_ADAPTER_ROUTER", "configs-macos/adapters.json")
adapter_router = AdapterRouter.from_json(router_path)
gateway = MacGateway(client, memory, episodes, model=model_name, experience_store=experience, adapter_router=adapter_router, backend_kind=backend_kind)

app = FastAPI(title="KVContinual macOS Gateway", version="14.0.0-mac")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False


class MemoryIn(BaseModel):
    text: str
    memory_type: str = "episode"
    source_type: str = "user"
    source_id: str = "mac-api"
    confidence: float = 0.5
    novelty: float = 0.5
    future_utility: float = 0.5
    recurrence: float = 0.0
    importance: float = 0.5
    redundancy: float = 0.0
    valid_from: str | None = None
    valid_until: str | None = None


class VerifyEpisodeReceiptIn(BaseModel):
    receipt_id: str
    episode_id: str
    prompt_digest: str
    attempted_output_digest: str
    repaired_output_digest: str
    evidence_root_digest: str
    verifier_id: str
    score: float
    passed: bool
    importance: float = Field(ge=0.0, le=1.0)
    issued_at: float
    verifier_key_id: str
    signature_b64: str
    schema: str = "mini-agi-v14-episode-verification-v1"


def _configured_episode_validator() -> EpisodeVerificationValidator:
    key_hex = os.getenv("MINIAGI_EPISODE_VERIFIER_PUBLIC_KEY_HEX", "")
    key_id = os.getenv("MINIAGI_EPISODE_VERIFIER_KEY_ID", "")
    if not key_hex or not key_id:
        raise HTTPException(503, "signed verification endpoint disabled: trusted verifier key is not configured")
    try:
        raw = bytes.fromhex(key_hex)
        verifier = Ed25519Verifier()
        verifier.register(key_id, raw)
    except Exception as exc:
        raise HTTPException(503, f"invalid trusted verifier configuration: {exc}") from exc
    return EpisodeVerificationValidator(verifier=verifier, trusted_key_ids={key_id})


@app.get("/health")
def health():
    b = backend.health()
    return {
        "ok": b["ok"],
        "version": "14.0.0-mac",
        "platform": "macos-apple-silicon",
        "backend_endpoint": endpoint,
        "backend": b,
        "memory_db": settings.memory_db,
        "experience_db": settings.experience_db,
        "model": model_name,
        "adapter_router_entries": len(adapter_router.adapters),
        "native_cuda_kvmem": False,
        "native_metal_tz": False,
    }


@app.get("/v1/models")
def models():
    try:
        data = client.models()
    except Exception:
        data = [{"id": model_name, "object": "model", "owned_by": "local"}]
    return {"object": "list", "data": data}


@app.get("/backend/capabilities")
def backend_capabilities():
    return {"backend_kind": backend_kind, **backend.capabilities().to_dict()}


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
        verified=False,
        valid_from=x.valid_from,
        valid_until=x.valid_until,
    )
    memory.put(rec)
    return {"stored": True, "id": rec.id, "disposition": disposition.value, "score": policy.score(sig)}


@app.post("/learning/episodes/{episode_id}/verification-receipt")
def install_episode_verification(episode_id: str, x: VerifyEpisodeReceiptIn):
    if x.episode_id != episode_id:
        raise HTTPException(409, "receipt episode_id does not match route")
    validator = _configured_episode_validator()
    try:
        receipt = EpisodeVerificationReceipt(**x.model_dump())
        receipt_digest = experience.install_verification_receipt(receipt, validator)
    except KeyError:
        raise HTTPException(404, "episode not found")
    except (PermissionError, ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"episode_id": episode_id, "verification_receipt_digest": receipt_digest}


@app.post("/learning/episodes/{episode_id}/verify")
def legacy_verify_episode_disabled(episode_id: str):
    raise HTTPException(410, "boolean verification was removed in v14; submit a signed verification receipt")


@app.get("/learning/ready")
def learning_ready(limit: int = 100):
    rows = experience.list_training_ready(limit=min(max(1, limit), 1000))
    return {"count": len(rows), "episodes": [r.__dict__ for r in rows]}


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    extras = dict(request.model_extra or {})
    for key in ("model", "messages", "temperature", "max_tokens", "stream"):
        extras.pop(key, None)

    if request.stream:
        augmented, memories = gateway.prepare_messages(messages)
        payload = {
            "model": request.model or gateway.model,
            "messages": augmented,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            **extras,
        }
        prompt = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        episode = Episode(prompt=str(prompt), response="", retrieved_memories=[m.id for m in memories])
        episodes.append(episode)
        experience.append(episode)
        try:
            return StreamingResponse(client.stream_chat(payload), media_type="text/event-stream")
        except Exception as exc:
            raise HTTPException(502, f"local model backend failed: {exc}") from exc

    try:
        return gateway.chat(
            messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            extra=extras,
            model=request.model,
        )
    except Exception as exc:
        raise HTTPException(502, f"local model backend failed: {exc}") from exc
