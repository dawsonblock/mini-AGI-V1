from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class BackendHealth:
    ok: bool
    status_code: int | None
    detail: str


class OpenAICompatibleClient:
    """Small dependency-free client for local OpenAI-compatible model servers.

    This is intentionally backend-neutral. It is used by the macOS profile for
    llama.cpp/Metal and MLX servers, and can also talk to vLLM.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "local",
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = client

    @property
    def headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        client = self._client
        if client is not None:
            return client.request(method, f"{self.base_url}{path}", headers=self.headers, **kwargs)
        return httpx.request(
            method,
            f"{self.base_url}{path}",
            headers=self.headers,
            timeout=self.timeout,
            **kwargs,
        )

    def models(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/models")
        response.raise_for_status()
        data = response.json()
        models = data.get("data", []) if isinstance(data, dict) else []
        return [m for m in models if isinstance(m, dict)]

    def health(self) -> BackendHealth:
        try:
            models = self.models()
            return BackendHealth(True, 200, f"ok models={len(models)}")
        except httpx.HTTPStatusError as exc:
            return BackendHealth(False, exc.response.status_code, exc.response.text[:300])
        except Exception as exc:  # network errors belong in a health result
            return BackendHealth(False, None, str(exc))

    def resolve_model(self, configured: str | None) -> str:
        if configured and configured not in {"auto", "default", "local-qwen"}:
            return configured
        models = self.models()
        if not models:
            return configured or "default_model"
        model_id = models[0].get("id")
        return str(model_id) if model_id else (configured or "default_model")

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.resolve_model(model),
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra:
            payload.update(extra)
        response = self._request("POST", "/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("backend returned a non-object response")
        return data

    def stream_chat(self, payload: dict[str, Any]):
        body = dict(payload)
        body["stream"] = True
        body["model"] = self.resolve_model(str(body.get("model") or "auto"))
        if self._client is not None:
            # Mock/test clients do not expose a stable streaming lifetime here;
            # fall back to one buffered response for deterministic tests.
            response = self._client.post(
                f"{self.base_url}/chat/completions", headers=self.headers, json=body
            )
            response.raise_for_status()
            yield response.content
            return
        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=body,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                if chunk:
                    yield chunk
