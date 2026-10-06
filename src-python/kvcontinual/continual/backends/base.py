from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Protocol


@dataclass(frozen=True)
class BackendCapabilities:
    streaming: bool = True
    tools: bool = True
    hidden_states: bool = False
    logits: bool = False
    kv_cache_control: bool = False
    recurrent_state: bool = False
    lora_load: bool = False
    lora_hot_switch: bool = False
    exact_replay: bool = False
    metal_acceleration: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


class InferenceBackend(Protocol):
    name: str

    def capabilities(self) -> BackendCapabilities: ...
    def health(self) -> dict[str, Any]: ...
    def chat(self, *, model: str, messages: list[dict[str, Any]], temperature: float = 0.2,
             max_tokens: int | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]: ...
    def stream_chat(self, payload: dict[str, Any]): ...
