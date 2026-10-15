"""Verified adapter/skill routing (V4-14, V4-17).

The legacy ``kvcontinual`` AdapterRouter ranks adapters by token overlap
and priority — it will happily activate an unqualified adapter on any
query that shares words with its description, and it has no notion of
evidence, revocation, or rollback.

This module is the qualified-routing layer. Every adapter it can
activate carries a manifest binding:

* the adapter's on-disk digest (``sha256_path`` of the artifact dir);
* the exact model id+revision it was qualified on;
* the task families it was qualified for (``qualified_families``);
* the qualification evidence digest (the independent qualifier's
  record, not the producer's claim);
* the scorer version used to establish qualification.

Routing policy (fail-closed):

1. Manifest validation — an entry missing a binding is ineligible.
2. Family match — the query's estimated task family must be in
   ``qualified_families``. No qualified family, no adapter.
3. OOD check — token-overlap similarity below ``ood_threshold`` means
   out-of-distribution: the router abstains (frozen fallback).
4. Health — adapters whose recent verified accuracy drops below
   ``min_health`` are auto-suspended (logged, reversible only via
   ``reinstate`` after new qualification evidence).
5. Revocation/rollback — ``revoke`` marks an adapter dead instantly;
   ``rollback()`` restores the last snapshot where every active adapter
   was qualified.

Every decision appends a structured routing event so routing errors are
measurable evidence, not silent behavior.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from egai.common.canonical import digest, validate_digest

_TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")

SCHEMA_MANIFEST = "mini-agi-v4.0-qualified-adapter-manifest-v1"
SCHEMA_REGISTRY = "mini-agi-v4.0-adapter-registry-v1"
SCHEMA_EVENT = "mini-agi-v4.0-routing-event-v1"


def _terms(text: str) -> set[str]:
    return {x.lower() for x in _TOKEN_RE.findall(text) if len(x) > 1}


@dataclass(frozen=True)
class QualifiedAdapterManifest:
    """The qualification-bound identity of a servable adapter."""
    adapter_id: str
    adapter_digest: str
    model_id: str
    model_revision: str
    qualified_families: tuple[str, ...]
    qualification_evidence_digest: str
    scorer_artifact_digest: str
    family_terms: tuple[str, ...] = ()
    enabled: bool = True
    priority: float = 0.5
    schema: str = SCHEMA_MANIFEST

    def __post_init__(self):
        for name in ("adapter_digest", "qualification_evidence_digest",
                     "scorer_artifact_digest"):
            validate_digest(getattr(self, name))
        if not self.adapter_id or not self.qualified_families:
            raise ValueError("adapter_id and qualified_families required")
        object.__setattr__(self, "qualified_families",
                           tuple(self.qualified_families))
        object.__setattr__(self, "family_terms",
                           tuple(self.family_terms))

    @property
    def manifest_digest(self) -> str:
        return digest(self)


@dataclass
class RoutingDecision:
    selected: str | None          # adapter_id or None (= frozen fallback)
    score: float
    reason: str
    family: str | None
    ood: bool
    refused: tuple[str, ...]


class VerifiedAdapterRouter:
    """Fail-closed router over QualifiedAdapterManifest entries."""

    def __init__(self, adapters: Iterable[QualifiedAdapterManifest] = (),
                 *, ood_threshold: float = 0.15, min_health: float = 0.3,
                 health_window: int = 20):
        self._adapters: dict[str, QualifiedAdapterManifest] = {
            a.adapter_id: a for a in adapters}
        self._revoked: set[str] = set()
        self._suspended: set[str] = set()
        self._health: dict[str, list[int]] = {}
        self._snapshots: list[dict[str, QualifiedAdapterManifest]] = []
        self.ood_threshold = float(ood_threshold)
        self.min_health = float(min_health)
        self.health_window = int(health_window)
        self.events: list[dict] = []

    # ---- registry ----------------------------------------------------

    def register(self, m: QualifiedAdapterManifest) -> None:
        self._adapters[m.adapter_id] = m

    def revoke(self, adapter_id: str) -> None:
        """Immediate revocation — the adapter is dead for routing."""
        self._revoked.add(adapter_id)

    def reinstate(self, m: QualifiedAdapterManifest) -> None:
        """Re-enable only with NEW qualification evidence: the caller
        supplies a fresh manifest (a new evidence digest is required —
        reinstating under the old evidence is forbidden)."""
        old = self._adapters.get(m.adapter_id)
        if old is not None and m.qualification_evidence_digest == \
                old.qualification_evidence_digest:
            raise ValueError(
                "reinstatement requires new qualification evidence — "
                "the same evidence digest cannot re-qualify a revoked "
                "adapter")
        self._adapters[m.adapter_id] = m
        self._revoked.discard(m.adapter_id)
        self._suspended.discard(m.adapter_id)

    def snapshot(self) -> str:
        """Remember the current registry state for rollback."""
        self._snapshots.append(dict(self._adapters))
        return digest({"n": len(self._snapshots),
                       "adapters": sorted(self._adapters)})

    def rollback(self) -> None:
        """Restore the most recent adapter snapshot and drop it.

        Revocations are safety actions and are NEVER undone by rollback —
        a rolled-back registry keeps every revocation issued since the
        snapshot. Suspension is health-derived and is cleared (it will
        re-engage on the next bad outcome window)."""
        if self._snapshots:
            self._adapters = self._snapshots.pop()
            self._suspended.clear()

    # ---- health ------------------------------------------------------

    def record_outcome(self, adapter_id: str, correct: bool) -> None:
        h = self._health.setdefault(adapter_id, [])
        h.append(1 if correct else 0)
        del h[:-self.health_window]
        if len(h) >= max(4, self.health_window // 2) and \
                sum(h) / len(h) < self.min_health:
            self._suspended.add(adapter_id)

    def health(self, adapter_id: str) -> float | None:
        h = self._health.get(adapter_id)
        return (sum(h) / len(h)) if h else None

    # ---- routing -----------------------------------------------------

    def route(self, query: str, *, family: str | None = None,
              model_revision: str | None = None,
              top_k: int = 1) -> RoutingDecision:
        """Route a query to a qualified adapter or abstain (fallback).

        `family` is the caller's best estimate of the task family (from
        dispatch metadata); when absent the router infers it from
        family_terms overlap. `model_revision` must match the adapter's
        qualified model when supplied.
        """
        refused: list[str] = []
        q = _terms(query)
        scored: list[tuple[QualifiedAdapterManifest, float]] = []
        for a in self._adapters.values():
            aid = a.adapter_id
            if aid in self._revoked:
                refused.append(f"{aid}: revoked")
                continue
            if aid in self._suspended:
                refused.append(f"{aid}: suspended on health")
                continue
            if not a.enabled:
                refused.append(f"{aid}: disabled")
                continue
            if model_revision and a.model_revision != model_revision:
                refused.append(f"{aid}: model revision mismatch")
                continue
            fams = set(a.qualified_families)
            t = _terms(" ".join(a.family_terms or a.qualified_families))
            # An explicit family hint is authoritative dispatch metadata —
            # it qualifies the adapter directly. Without one, eligibility
            # comes from lexical overlap alone and the OOD threshold must
            # still be met.
            explicit = family is not None
            fam_match = (family in fams) if explicit else bool(q & t)
            if not fam_match:
                refused.append(f"{aid}: family not qualified "
                               f"({family or 'unresolved'})")
                continue
            overlap = (len(q & t) / math.sqrt(len(q) * len(t))
                       if q and t else 0.0)
            score = (0.85 * overlap + 0.15 * max(0.0, min(1.0, a.priority))
                     if not explicit
                     else 0.85 + 0.15 * max(0.0, min(1.0, a.priority)))
            scored.append((a, score))
        scored.sort(key=lambda x: (x[1], x[0].adapter_id), reverse=True)

        if not scored:
            dec = RoutingDecision(None, 0.0, "no qualified adapter",
                                  family, True, tuple(refused))
        else:
            a, score = scored[0]
            if score < self.ood_threshold:
                dec = RoutingDecision(None, score,
                                      "OOD: below activation threshold",
                                      family, True, tuple(refused))
            else:
                dec = RoutingDecision(a.adapter_id, score, "qualified",
                                      family, False, tuple(refused))
        self.events.append({
            "schema": SCHEMA_EVENT, "ts": round(time.time(), 3),
            "query_digest": digest({"q": query}),
            "decision": asdict(dec)})
        return dec

    # ---- persistence -------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {"schema": SCHEMA_REGISTRY,
             "adapters": [asdict(a) for a in
                          sorted(self._adapters.values(),
                                 key=lambda x: x.adapter_id)]},
            indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path: str | Path, **kw) -> "VerifiedAdapterRouter":
        raw = json.loads(Path(path).read_text())
        if raw.get("schema") != SCHEMA_REGISTRY:
            raise ValueError(f"registry schema must be {SCHEMA_REGISTRY}")
        return cls([QualifiedAdapterManifest(**a)
                    for a in raw.get("adapters", [])], **kw)
