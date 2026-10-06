from __future__ import annotations

import copy
import math
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdapterMetadata:
    id: str
    name: str
    d_model: int
    rank: int
    alpha: float
    status: str = "fast"  # fast | candidate | stable | quarantined | retired
    created_at: float = 0.0
    domain: str | None = None
    parent_digest: str | None = None
    examples_seen: int = 0
    score: float = 0.0

    @classmethod
    def new(cls, name: str, d_model: int, rank: int, alpha: float,
            domain: str | None = None, parent_digest: str | None = None):
        return cls(
            id=str(uuid.uuid4()), name=str(name), d_model=int(d_model), rank=int(rank),
            alpha=float(alpha), created_at=time.time(), domain=domain,
            parent_digest=parent_digest,
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AdapterRoute:
    adapter_id: str
    similarity: float
    weight: float


class LowRankAdapter(nn.Module):
    """Small reversible residual adapter.

    A new adapter is an exact no-op because its up projection is initialized to
    zero.  This remains deliberately outside frozen base weights.  It is not a
    claim that final-hidden adapters are as expressive as layer-local LoRA; the
    point of this class is a bounded, reversible first learning substrate.
    """

    def __init__(self, d_model: int, rank: int = 16, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(self.rank, 1)
        self.norm = nn.LayerNorm(self.d_model, elementwise_affine=False)
        self.down = nn.Linear(self.d_model, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.d_model, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def delta(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(F.silu(self.down(self.norm(hidden))))) * self.scale

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.delta(hidden)

    def clone(self) -> "LowRankAdapter":
        out = LowRankAdapter(self.d_model, self.rank, self.alpha,
                             self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.0)
        out.load_state_dict(copy.deepcopy(self.state_dict()))
        return out


class AdapterBank:
    """Expandable reversible skill bank with calibrated abstention.

    Routing is based on frozen representations, but unlike the old v4 router it
    can abstain when the nearest specialist is weak or ambiguous.  Stable IDs
    are independent of routing centroids, so rebuilding the search structure
    does not change adapter identity.
    """

    def __init__(self, d_model: int, rank: int = 16, alpha: float = 16.0,
                 dropout: float = 0.0, top_k: int = 2, max_adapters: int = 64,
                 min_similarity: float = 0.45, min_margin: float = 0.03):
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.top_k = max(1, int(top_k))
        self.max_adapters = max(1, int(max_adapters))
        self.min_similarity = float(min_similarity)
        self.min_margin = max(0.0, float(min_margin))
        self.adapters: dict[str, LowRankAdapter] = {}
        self.meta: dict[str, AdapterMetadata] = {}
        self.centroids: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.adapters)

    @staticmethod
    def representation(hidden: torch.Tensor) -> torch.Tensor:
        h = hidden.detach().float()
        while h.ndim > 1:
            h = h.mean(dim=0)
        return F.normalize(h, dim=-1).cpu()

    def create(self, name: str, *, domain: str | None = None,
               centroid: torch.Tensor | None = None, status: str = "fast",
               parent_digest: str | None = None) -> str:
        if len(self.adapters) >= self.max_adapters:
            raise RuntimeError("adapter bank is at capacity")
        meta = AdapterMetadata.new(name, self.d_model, self.rank, self.alpha,
                                   domain=domain, parent_digest=parent_digest)
        meta.status = str(status)
        aid = meta.id
        self.adapters[aid] = LowRankAdapter(self.d_model, self.rank, self.alpha, self.dropout)
        self.meta[aid] = meta
        if centroid is not None:
            self.centroids[aid] = self.representation(centroid)
        return aid

    def add(self, adapter: LowRankAdapter, metadata: AdapterMetadata,
            centroid: torch.Tensor | None = None) -> None:
        if metadata.d_model != self.d_model:
            raise ValueError("adapter d_model does not match bank")
        self.adapters[metadata.id] = adapter
        self.meta[metadata.id] = metadata
        if centroid is not None:
            c = centroid.detach().float().reshape(-1)
            if c.numel() != self.d_model:
                raise ValueError("centroid dimension does not match bank")
            self.centroids[metadata.id] = F.normalize(c, dim=-1).cpu()

    def update_centroid(self, adapter_id: str, hidden: torch.Tensor,
                        momentum: float = 0.9) -> None:
        if adapter_id not in self.adapters:
            raise KeyError(adapter_id)
        c = self.representation(hidden)
        old = self.centroids.get(adapter_id)
        if old is None:
            self.centroids[adapter_id] = c
        else:
            self.centroids[adapter_id] = F.normalize(
                float(momentum) * old + (1.0 - float(momentum)) * c, dim=-1)

    def route(self, hidden: torch.Tensor, *, statuses: Iterable[str] = ("fast", "stable", "candidate"),
              top_k: int | None = None, allow_ambiguous: bool = False) -> list[AdapterRoute]:
        allowed = set(statuses)
        q = self.representation(hidden)
        scored: list[tuple[str, float]] = []
        for aid, c in self.centroids.items():
            meta = self.meta.get(aid)
            if meta is None or meta.status not in allowed:
                continue
            scored.append((aid, float(torch.dot(q, c))))
        scored.sort(key=lambda x: x[1], reverse=True)
        if not scored or scored[0][1] < self.min_similarity:
            return []
        if (not allow_ambiguous and len(scored) > 1
                and scored[0][1] - scored[1][1] < self.min_margin):
            return []
        selected = scored[:max(1, int(top_k or self.top_k))]
        sims = torch.tensor([s for _, s in selected], dtype=torch.float32)
        ws = F.softmax(sims, dim=0).tolist()
        return [AdapterRoute(aid, sim, float(w)) for (aid, sim), w in zip(selected, ws)]

    def route_or_create(self, hidden: torch.Tensor, name: str = "auto") -> tuple[list[AdapterRoute], str | None]:
        routed = self.route(hidden)
        if routed:
            return routed, None
        if len(self.adapters) >= self.max_adapters:
            return [], None
        aid = self.create(f"{name}-{len(self.adapters):03d}", centroid=hidden)
        return [AdapterRoute(aid, 1.0, 1.0)], aid

    def apply(self, hidden: torch.Tensor, routes: list[AdapterRoute]) -> torch.Tensor:
        if not routes:
            return hidden
        delta = torch.zeros_like(hidden)
        for r in routes:
            adapter = self.adapters[r.adapter_id].to(hidden.device)
            delta = delta + float(r.weight) * adapter.delta(hidden)
        return hidden + delta
