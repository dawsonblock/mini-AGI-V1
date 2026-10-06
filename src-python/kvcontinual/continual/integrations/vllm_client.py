from __future__ import annotations

import httpx


class VLLMClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", api_key: str = "local"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def chat(self, model: str, messages: list[dict], temperature: float = 0.2, extra: dict | None = None) -> dict:
        payload = {"model": model, "messages": messages, "temperature": temperature}
        if extra:
            payload.update(extra)
        r = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
