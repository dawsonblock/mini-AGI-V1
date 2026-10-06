from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kvcontinual.continual.backends.base import BackendCapabilities
from kvcontinual.continual.integrations.openai_compatible import OpenAICompatibleClient


@dataclass
class OpenAIBackend:
    client: OpenAICompatibleClient
    backend_kind: str = "generic"
    name: str = "openai-compatible"

    def capabilities(self) -> BackendCapabilities:
        kind = self.backend_kind.lower()
        if kind == "llama.cpp":
            return BackendCapabilities(
                streaming=True,
                tools=True,
                lora_load=True,
                lora_hot_switch=True,
                metal_acceleration=True,
            )
        if kind == "mlx":
            return BackendCapabilities(
                streaming=True,
                tools=True,
                lora_load=True,
                lora_hot_switch=False,
                metal_acceleration=True,
            )
        if kind == "vllm":
            return BackendCapabilities(
                streaming=True,
                tools=True,
                lora_load=True,
                lora_hot_switch=True,
            )
        return BackendCapabilities()

    def health(self) -> dict[str, Any]:
        h = self.client.health()
        return {
            "ok": h.ok,
            "status_code": h.status_code,
            "detail": h.detail,
            "backend_kind": self.backend_kind,
            "capabilities": self.capabilities().to_dict(),
        }

    def chat(self, *, model: str, messages: list[dict[str, Any]], temperature: float = 0.2,
             max_tokens: int | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.client.chat(model, messages, temperature=temperature, max_tokens=max_tokens, extra=extra)

    def stream_chat(self, payload: dict[str, Any]):
        return self.client.stream_chat(payload)
