from __future__ import annotations

import json
from pathlib import Path

import httpx

from kvcontinual.continual.experience import EpisodeLog
from kvcontinual.continual.integrations.openai_compatible import OpenAICompatibleClient
from kvcontinual.continual.macos.gateway import MacGateway
from kvcontinual.continual.memory import MemoryRecord, MemoryStore


def test_openai_compatible_client_and_memory_gateway(tmp_path: Path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "local-qwen"}]})
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            seen["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Use the new endpoint."}}],
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    client = OpenAICompatibleClient("http://local/v1", client=http)
    assert client.health().ok

    store = MemoryStore(str(tmp_path / "memory.sqlite3"))
    record = MemoryRecord(
        text="The service endpoint moved from port 7000 to port 9000.",
        memory_type="fact",
        source_type="test",
        source_id="m1",
        confidence=0.95,
        utility=0.9,
        importance=0.9,
    )
    store.put(record)
    episodes = EpisodeLog(str(tmp_path / "episodes.jsonl"))
    gateway = MacGateway(client, store, episodes, model="local-qwen")
    response = gateway.chat([{"role": "user", "content": "Which port does the service use now?"}])
    assert response["choices"][0]["message"]["content"] == "Use the new endpoint."
    sent = seen["body"]["messages"]
    assert sent[0]["role"] == "system"
    assert "port 9000" in sent[0]["content"]
    lines = (tmp_path / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == 1
    episode = json.loads(lines[0])
    assert episode["retrieved_memories"] == [record.id]
