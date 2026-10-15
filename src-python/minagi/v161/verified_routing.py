"""Verified adapter/skill routing (V4-14, V4-17, V4-21, V4-22).

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
* the scorer version used to establish qualification;
* (schema v2) an Ed25519 signature from the qualification authority —
  a manifest without one is a *proposer's claim*, and the router
  refuses it whenever an authorization policy is configured.

Routing policy (fail-closed):

1. Manifest admissibility — when a ``manifest_verifier``/``authority``
   is configured, unsigned or wrongly-signed manifests never reach the
   registry; when an ``artifact_resolver`` is configured, the adapter's
   on-disk bytes must hash to ``adapter_digest`` at registration and at
   every ``from_json`` reload (tamper between save and load re-bars the
   adapter). Ineligible manifests are recorded and surface in every
   routing decision's ``refused`` list.
2. Family match — the query's estimated task family must be in
   ``qualified_families``. An explicit ``family`` hint is trusted
   dispatch metadata; without one, eligibility comes from lexical
   overlap with ``family_terms`` alone and the OOD threshold must still
   be met. Production callers do not have benchmark labels — the
   hinted path and the inferred path are different measurements.
3. OOD check — token-overlap similarity below ``ood_threshold`` means
   out-of-distribution: the router abstains (frozen fallback).
4. Health — adapters whose recent verified accuracy drops below
   ``min_health`` are auto-suspended (logged, reversible only via
   ``reinstate`` after new qualification evidence).
5. Revocation/rollback — ``revoke`` marks an adapter dead instantly;
   ``rollback()`` restores the last snapshot where every active adapter
   was qualified. Revocations are safety actions: rollback never undoes
   them and — since registry schema v2 — ``to_json``/``from_json``
   preserves them across save/reload. A v1 registry file *cannot*
   prove its adapters were not revoked before save; loading one marks
   ``state_complete = False`` and refuses outright under
   ``strict_state=True``.

Every decision appends a structured routing event so routing errors are
measurable evidence, not silent behavior.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import (Ed25519Signer, Ed25519Verifier,
                                SignedEnvelope)
from minagi.v161.authority import AuthorityRegistry
from minagi.v161.runtime_closure3 import sha256_path

_TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")

SCHEMA_MANIFEST_V1 = "mini-agi-v4.0-qualified-adapter-manifest-v1"
SCHEMA_MANIFEST = "mini-agi-v4.0-qualified-adapter-manifest-v2"
SCHEMA_REGISTRY_V1 = "mini-agi-v4.0-adapter-registry-v1"
SCHEMA_REGISTRY = "mini-agi-v4.0-adapter-registry-v2"
SCHEMA_EVENT = "mini-agi-v4.0-routing-event-v1"


def _terms(text: str) -> set[str]:
    return {x.lower() for x in _TOKEN_RE.findall(text) if len(x) > 1}


def lexical_terms(text: str) -> tuple[str, ...]:
    """Public tokenization used for ``family_terms`` — deterministic
    surface vocabulary a manifest carries for inferred routing."""
    return tuple(sorted(_terms(text)))


@dataclass(frozen=True)
class QualifiedAdapterManifest:
    """The qualification-bound identity of a servable adapter.

    ``signer_key_id``/``signature_b64`` form an Ed25519 envelope over
    every other field (the schema string included). An unsigned manifest
    is a claim; a signed one is an authorization — the router enforces
    the difference whenever a verifier or authority is configured.
    """
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
    signer_key_id: str = ""
    signature_b64: str = ""
    schema: str = SCHEMA_MANIFEST

    def __post_init__(self):
        for name in ("adapter_digest", "qualification_evidence_digest",
                     "scorer_artifact_digest"):
            validate_digest(getattr(self, name))
        if not self.adapter_id or not self.qualified_families:
            raise ValueError("adapter_id and qualified_families required")
        if bool(self.signer_key_id) != bool(self.signature_b64):
            raise ValueError(
                "partial signature envelope: signer_key_id and "
                "signature_b64 must be set together")
        object.__setattr__(self, "qualified_families",
                           tuple(self.qualified_families))
        object.__setattr__(self, "family_terms",
                           tuple(self.family_terms))

    @property
    def manifest_digest(self) -> str:
        return digest(self)

    @property
    def signed(self) -> bool:
        return bool(self.signer_key_id)

    def unsigned_body(self) -> dict:
        d = asdict(self)
        d.pop("signer_key_id")
        d.pop("signature_b64")
        return d

    def sign(self, signer: Ed25519Signer) -> "QualifiedAdapterManifest":
        """Return a copy carrying the signer's Ed25519 envelope over the
        unsigned body — the qualification authority's authorization."""
        env = signer.sign(self.unsigned_body())
        return replace(self, signer_key_id=env.key_id,
                       signature_b64=env.signature_b64)

    def verify(self, verifier: Ed25519Verifier) -> bool:
        """True iff the signature envelope verifies against a registered
        public key. Unsigned manifests never verify."""
        if not self.signed:
            return False
        return verifier.verify(
            self.unsigned_body(),
            SignedEnvelope(self.signer_key_id, self.signature_b64))


@dataclass
class RoutingDecision:
    selected: str | None          # adapter_id or None (= frozen fallback)
    score: float
    reason: str
    family: str | None
    ood: bool
    refused: tuple[str, ...]


class VerifiedAdapterRouter:
    """Fail-closed router over QualifiedAdapterManifest entries.

    Authorization policy (all optional, all fail-closed when set):

    * ``manifest_verifier`` — every manifest must carry a valid Ed25519
      signature or it never enters the registry.
    * ``authority`` — an ``AuthorityRegistry``; the manifest's signer
      must additionally be authorized for ``required_role`` at check
      time. Supplying an authority without a verifier builds one from
      the authority's currently-valid keys.
    * ``artifact_resolver`` — ``adapter_id -> artifact path``; the
      adapter's measured digest must equal the manifest's
      ``adapter_digest``. ``artifact_digester`` selects the hash (default
      ``sha256_path`` of the tree; producers binding only weights+config
      pass e.g. ``peft_trainer.adapter_artifact_digest``). Runs at
      ``register``/``reinstate``/``__init__`` and again at every
      ``from_json`` — content tampered between save and reload re-bars
      the adapter.

    With none configured the router is in *provisional* mode — manifests
    are taken at face value and each routing event records that no
    authorization was enforced.
    """

    def __init__(self, adapters: Iterable[QualifiedAdapterManifest] = (),
                 *, ood_threshold: float = 0.15, min_health: float = 0.3,
                 health_window: int = 20,
                 manifest_verifier: Ed25519Verifier | None = None,
                 authority: AuthorityRegistry | None = None,
                 required_role: str = "qualification",
                 artifact_resolver: Callable[[str], str | Path] | None = None,
                 artifact_digester: Callable[[str | Path], str] = sha256_path):
        self._adapters: dict[str, QualifiedAdapterManifest] = {}
        self._ineligible: dict[str, str] = {}
        self._revoked: set[str] = set()
        self._suspended: set[str] = set()
        self._health: dict[str, list[int]] = {}
        self._snapshots: list[dict[str, QualifiedAdapterManifest]] = []
        self.ood_threshold = float(ood_threshold)
        self.min_health = float(min_health)
        self.health_window = int(health_window)
        self.events: list[dict] = []
        self.state_complete = True
        self.registry_warnings: list[str] = []
        self._verifier = (manifest_verifier
                          if manifest_verifier is not None
                          else (authority.verifier() if authority is not None
                                else None))
        self._authority = authority
        self._required_role = str(required_role)
        self._artifact_resolver = artifact_resolver
        self._artifact_digester = artifact_digester
        for a in adapters:
            self.register(a)

    # ---- admissibility ----------------------------------------------

    def _admissibility(self, m: QualifiedAdapterManifest) -> str:
        """'' if the manifest may enter the registry, else the reason."""
        if self._verifier is not None:
            if not m.signed:
                return "manifest unsigned"
            if not m.verify(self._verifier):
                return "signature invalid"
        if self._authority is not None:
            if not m.signed:
                return "manifest unsigned"
            if not self._authority.is_authorized(
                    self._required_role, m.signer_key_id):
                return (f"signer not authorized for role "
                        f"{self._required_role}")
        if self._artifact_resolver is not None:
            try:
                actual = self._artifact_digester(
                    self._artifact_resolver(m.adapter_id))
            except Exception as exc:
                return f"artifact unverifiable ({exc})"
            if actual != m.adapter_digest:
                return "adapter artifact digest mismatch"
        return ""

    # ---- registry ----------------------------------------------------

    def register(self, m: QualifiedAdapterManifest) -> None:
        reason = self._admissibility(m)
        if reason:
            self._ineligible[m.adapter_id] = reason
            self.events.append({
                "schema": SCHEMA_EVENT, "ts": round(time.time(), 3),
                "kind": "registration_refused",
                "adapter_id": m.adapter_id, "reason": reason})
            return
        self._ineligible.pop(m.adapter_id, None)
        self._adapters[m.adapter_id] = m

    def ineligible(self) -> dict[str, str]:
        return dict(self._ineligible)

    def revoke(self, adapter_id: str) -> None:
        """Immediate revocation — the adapter is dead for routing."""
        self._revoked.add(adapter_id)

    def reinstate(self, m: QualifiedAdapterManifest) -> None:
        """Re-enable only with NEW qualification evidence: the caller
        supplies a fresh manifest (a new evidence digest is required —
        reinstating under the old evidence is forbidden). The manifest
        must also pass every configured admissibility check."""
        old = self._adapters.get(m.adapter_id)
        if old is not None and m.qualification_evidence_digest == \
                old.qualification_evidence_digest:
            raise ValueError(
                "reinstatement requires new qualification evidence — "
                "the same evidence digest cannot re-qualify a revoked "
                "adapter")
        self.register(m)
        if m.adapter_id not in self._adapters:
            raise ValueError(
                f"reinstatement refused: "
                f"{self._ineligible.get(m.adapter_id, 'unknown reason')}")
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
        refused: list[str] = [
            f"{aid}: ineligible manifest ({reason})"
            for aid, reason in sorted(self._ineligible.items())]
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
            "controls": {
                "signature_enforced": self._verifier is not None
                or self._authority is not None,
                "authority_role": (self._required_role
                                   if self._authority is not None else None),
                "artifact_verified": self._artifact_resolver is not None,
            },
            "decision": asdict(dec)})
        return dec

    # ---- persistence -------------------------------------------------

    def to_json(self) -> str:
        """Registry schema v2 — adapters plus the safety state a v1 file
        silently dropped (revocations, suspensions, health history)."""
        return json.dumps(
            {"schema": SCHEMA_REGISTRY,
             "adapters": [asdict(a) for a in
                          sorted(self._adapters.values(),
                                 key=lambda x: x.adapter_id)],
             "revoked": sorted(self._revoked),
             "suspended": sorted(self._suspended),
             "health": {k: list(v) for k, v in
                        sorted(self._health.items())}},
            indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path: str | Path, *, strict_state: bool = False,
                  **kw) -> "VerifiedAdapterRouter":
        raw = json.loads(Path(path).read_text())
        schema = raw.get("schema")
        if schema == SCHEMA_REGISTRY_V1:
            # A v1 file never recorded revocation/suspension/health — any
            # of its adapters could have been revoked before save and the
            # file cannot prove otherwise. Load it (manifests re-check
            # admissibility as usual) but mark the state as incomplete,
            # or refuse entirely under strict_state.
            if strict_state:
                raise ValueError(
                    "v1 registries cannot prove their safety state — "
                    "an adapter revoked before save would reload as "
                    "routable; refusing under strict_state")
            r = cls([QualifiedAdapterManifest(**a)
                     for a in raw.get("adapters", [])], **kw)
            r.state_complete = False
            r.registry_warnings.append(
                "registry schema v1 predates revocation/suspension "
                "persistence — a revoked adapter saved under v1 reloads "
                "as routable; this file cannot prove none existed")
            return r
        if schema != SCHEMA_REGISTRY:
            raise ValueError(f"registry schema must be {SCHEMA_REGISTRY}")
        r = cls([QualifiedAdapterManifest(**a)
                 for a in raw.get("adapters", [])], **kw)
        r._revoked = {str(x) for x in raw.get("revoked", [])}
        r._suspended = {str(x) for x in raw.get("suspended", [])}
        r._health = {str(k): [int(x) for x in v]
                     for k, v in raw.get("health", {}).items()}
        return r
