from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")


def _terms(text: str) -> set[str]:
    return {x.lower() for x in _TOKEN_RE.findall(text) if len(x) > 1}


@dataclass(frozen=True)
class AdapterDescriptor:
    adapter_id: str
    path: str
    description: str
    tags: tuple[str, ...] = ()
    enabled: bool = True
    priority: float = 0.5
    backend_id: int | None = None
    scale: float = 1.0


@dataclass
class AdapterRouter:
    adapters: list[AdapterDescriptor] = field(default_factory=list)
    min_score: float = 0.10

    def route(self, query: str, top_k: int = 1) -> list[tuple[AdapterDescriptor, float]]:
        q = _terms(query)
        scored: list[tuple[AdapterDescriptor, float]] = []
        for a in self.adapters:
            if not a.enabled:
                continue
            text = " ".join((a.description, *a.tags))
            t = _terms(text)
            overlap = 0.0 if not q or not t else len(q & t) / math.sqrt(len(q) * len(t))
            score = 0.85 * overlap + 0.15 * max(0.0, min(1.0, a.priority))
            if score >= self.min_score:
                scored.append((a, score))
        scored.sort(key=lambda x: (x[1], x[0].adapter_id), reverse=True)
        return scored[:max(0, int(top_k))]

    @classmethod
    def from_json(cls, path: str | Path) -> "AdapterRouter":
        p = Path(path)
        if not p.is_file():
            return cls([])
        raw = json.loads(p.read_text())
        if not isinstance(raw, list):
            raise ValueError("adapter router config must be a JSON list")
        adapters = []
        for row in raw:
            adapters.append(AdapterDescriptor(
                adapter_id=str(row["adapter_id"]),
                path=str(row.get("path", "")),
                description=str(row.get("description", "")),
                tags=tuple(str(x) for x in row.get("tags", [])),
                enabled=bool(row.get("enabled", True)),
                priority=float(row.get("priority", 0.5)),
                backend_id=None if row.get("backend_id") is None else int(row["backend_id"]),
                scale=float(row.get("scale", 1.0)),
            ))
        return cls(adapters, min_score=float(0.10))
