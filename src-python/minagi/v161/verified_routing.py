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

V4-28 — authenticated registry state and activation-time binding:

* Registry schema v3 seals the mutable safety state under an authority
  signature plus a monotonic ``seq`` (see
  ``authenticated_registry.py``). With a ``registry_verifier``,
  ``from_json`` refuses a tampered or unsigned state — deleting a
  revocation from the file can no longer restore access — and a
  snapshot whose ``seq`` predates ``min_seq`` cannot roll safety state
  backwards. An optional signed ``RevocationLog`` is unioned over the
  file's revocation list as the authoritative record.
* Adapter bytes are re-verified at activation: when an
  ``artifact_resolver`` is configured, the selected adapter's on-disk
  digest is re-checked inside ``route`` (``recheck_at_activation``,
  default on) — weights modified between registration and use are
  rejected, not served.
* Model identity is required by default (``require_model_identity``):
  a call without ``model_revision``, or a manifest without one, fails
  closed rather than silently skipping the binding check.
* Externally supplied ``family`` hints are untrusted unless
  ``family_authenticated=True``: an unauthenticated hint narrows the
  candidate set but never bypasses the lexical/OOD evidence gate — a
  forged hint cannot force activation.

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
from minagi.v161.authenticated_registry import (GENESIS_DIGEST,
                                                RegistryEventLog,
                                                RegistryStateAuthority,
                                                RevocationLog,
                                                open_state, seal_state)
from minagi.v161.runtime_closure3 import sha256_path

_TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")

SCHEMA_MANIFEST_V1 = "mini-agi-v4.0-qualified-adapter-manifest-v1"
SCHEMA_MANIFEST = "mini-agi-v4.0-qualified-adapter-manifest-v2"
SCHEMA_REGISTRY_V1 = "mini-agi-v4.0-adapter-registry-v1"
SCHEMA_REGISTRY_V2 = "mini-agi-v4.0-adapter-registry-v2"
SCHEMA_REGISTRY = "mini-agi-v4.0-adapter-registry-v3"
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
    qualification_generation: int = 1
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
                 artifact_digester: Callable[[str | Path], str] = sha256_path,
                 require_model_identity: bool = True,
                 recheck_at_activation: bool = True,
                 state_signer: Ed25519Signer | None = None,
                 revocation_log=None,
                 event_log=None,
                 state_authority=None):
        self._adapters: dict[str, QualifiedAdapterManifest] = {}
        self._ineligible: dict[str, str] = {}
        self._revoked: set[str] = set()
        self._suspended: set[str] = set()
        self._health: dict[str, list[int]] = {}
        self._generations: dict[str, int] = {}
        self._snapshots: list[dict[str, QualifiedAdapterManifest]] = []
        self.ood_threshold = float(ood_threshold)
        self.min_health = float(min_health)
        self.health_window = int(health_window)
        self.events: list[dict] = []
        self.state_complete = True
        self.state_authenticated = False
        self.registry_warnings: list[str] = []
        self._seq = 0
        self._verifier = (manifest_verifier
                          if manifest_verifier is not None
                          else (authority.verifier() if authority is not None
                                else None))
        self._authority = authority
        self._required_role = str(required_role)
        self._artifact_resolver = artifact_resolver
        self._artifact_digester = artifact_digester
        self._require_model_identity = bool(require_model_identity)
        self._recheck_at_activation = bool(recheck_at_activation)
        self._state_signer = state_signer
        self._revocation_log = (revocation_log if revocation_log is None
                                or isinstance(revocation_log, RevocationLog)
                                else RevocationLog(revocation_log))
        self._event_log = (event_log if event_log is None
                           or isinstance(event_log, RegistryEventLog)
                           else RegistryEventLog(event_log))
        self._state_authority = (
            state_authority if state_authority is None
            or isinstance(state_authority, RegistryStateAuthority)
            else RegistryStateAuthority(state_authority))
        # Constructor-time adapter loads are provenance, not fresh
        # authorization events — suppress event emission while loading.
        self._loading = True
        for a in adapters:
            self.register(a)
        self._loading = False

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

    def _emit(self, event_type: str, adapter_id: str, *,
              generation: int = 0, artifact_digest: str = "",
              model_revision: str = "", evidence_digest: str = ""):
        """Append a signed state-transition event, when a log and
        signer are configured and this is not a constructor load."""
        if (self._event_log is None or self._loading
                or self._state_signer is None):
            return None
        return self._event_log.append(
            self._state_signer, event_type, adapter_id,
            qualification_generation=generation,
            artifact_digest=artifact_digest,
            model_revision=model_revision,
            evidence_digest=evidence_digest)

    def register(self, m: QualifiedAdapterManifest) -> None:
        aid = m.adapter_id
        known = aid in self._adapters or aid in self._generations
        if known and m.qualification_generation <= \
                self._generations.get(aid, 0):
            # Re-registration with an already-seen qualification
            # generation is not a new authorization — a revoked or
            # stale adapter cannot re-enter under old evidence.
            self._ineligible[aid] = (
                f"outdated qualification generation "
                f"{m.qualification_generation} (current "
                f"{self._generations.get(aid)})")
            self.events.append({
                "schema": SCHEMA_EVENT, "ts": round(time.time(), 3),
                "kind": "registration_refused",
                "adapter_id": aid,
                "reason": self._ineligible[aid]})
            return
        reason = self._admissibility(m)
        if reason:
            self._ineligible[aid] = reason
            self.events.append({
                "schema": SCHEMA_EVENT, "ts": round(time.time(), 3),
                "kind": "registration_refused",
                "adapter_id": aid, "reason": reason})
            return
        self._ineligible.pop(aid, None)
        self._adapters[aid] = m
        self._generations[aid] = m.qualification_generation
        self._seq += 1
        self._emit(
            "ADAPTER_REGISTERED" if not known
            else ("ADAPTER_REQUALIFIED" if aid in self._revoked
                  else "ADAPTER_QUALIFIED"),
            aid, generation=m.qualification_generation,
            artifact_digest=m.adapter_digest,
            model_revision=m.model_revision,
            evidence_digest=m.qualification_evidence_digest)

    def ineligible(self) -> dict[str, str]:
        return dict(self._ineligible)

    def revoke(self, adapter_id: str) -> None:
        """Immediate revocation — the adapter is dead for routing.

        When the router carries a revocation log and state signer, the
        revocation is also appended to the authoritative log — a
        tampered registry file can no longer undo it."""
        self._revoked.add(adapter_id)
        self._seq += 1
        self._emit("ADAPTER_REVOKED", adapter_id,
                   generation=self._generations.get(adapter_id, 0))
        if self._revocation_log is not None and \
                self._state_signer is not None:
            self._revocation_log.append(self._state_signer, adapter_id,
                                        self._seq)

    def reinstate(self, m: QualifiedAdapterManifest) -> None:
        """Re-enable only with NEW qualification evidence: the caller
        supplies a fresh manifest (a new evidence digest is required —
        reinstating under the old evidence is forbidden) carrying a
        NEW qualification generation. The manifest must also pass
        every configured admissibility check."""
        old = self._adapters.get(m.adapter_id)
        if old is not None and m.qualification_evidence_digest == \
                old.qualification_evidence_digest:
            raise ValueError(
                "reinstatement requires new qualification evidence — "
                "the same evidence digest cannot re-qualify a revoked "
                "adapter")
        self.register(m)
        if self._adapters.get(m.adapter_id) is not m:
            raise ValueError(
                f"reinstatement refused: "
                f"{self._ineligible.get(m.adapter_id, 'unknown reason')}")
        self._revoked.discard(m.adapter_id)
        self._suspended.discard(m.adapter_id)
        self._seq += 1

    def restore(self, adapter_id: str) -> bool:
        """Lift a health suspension. Restoring from REVOCATION is not
        possible here — a revoked adapter requires ADAPTER_REQUALIFIED
        with a new qualification generation (``reinstate``)."""
        if adapter_id in self._revoked:
            return False
        if adapter_id in self._suspended:
            self._suspended.discard(adapter_id)
            self._seq += 1
            self._emit("ADAPTER_RESTORED", adapter_id,
                       generation=self._generations.get(adapter_id, 0))
            return True
        return False

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
            self._seq += 1

    # ---- health ------------------------------------------------------

    def record_outcome(self, adapter_id: str, correct: bool) -> None:
        h = self._health.setdefault(adapter_id, [])
        h.append(1 if correct else 0)
        del h[:-self.health_window]
        self._seq += 1
        if len(h) >= max(4, self.health_window // 2) and \
                sum(h) / len(h) < self.min_health and \
                adapter_id not in self._suspended:
            self._suspended.add(adapter_id)
            self._emit("ADAPTER_SUSPENDED", adapter_id,
                       generation=self._generations.get(adapter_id, 0))

    def health(self, adapter_id: str) -> float | None:
        h = self._health.get(adapter_id)
        return (sum(h) / len(h)) if h else None

    # ---- routing -----------------------------------------------------

    def route(self, query: str, *, family: str | None = None,
              family_authenticated: bool = False,
              model_revision: str | None = None,
              top_k: int = 1) -> RoutingDecision:
        """Route a query to a qualified adapter or abstain (fallback).

        ``family`` is the caller's estimate of the task family.
        Externally supplied hints are UNTRUSTED by default
        (``family_authenticated=False``): a hint narrows the candidate
        set to adapters qualified for that family but never bypasses
        the lexical/OOD evidence gate — a forged hint cannot force an
        activation. Authenticated dispatch metadata
        (``family_authenticated=True``, from a trusted dispatcher)
        retains the direct-qualification path.

        ``model_revision`` binds the routing decision to an actual
        model identity. Under ``require_model_identity`` (default) a
        call without it refuses every adapter — missing identity fails
        closed, it does not skip the check.
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
            gen = self._generations.get(aid)
            if gen is not None and a.qualification_generation < gen:
                # The registry has accepted a newer qualification
                # generation for this adapter — an old manifest must
                # not activate under outdated qualification.
                refused.append(
                    f"{aid}: outdated qualification generation "
                    f"{a.qualification_generation} (registry carries "
                    f"{gen})")
                continue
            if not a.enabled:
                refused.append(f"{aid}: disabled")
                continue
            if self._require_model_identity and not model_revision:
                refused.append(f"{aid}: model revision identity "
                               "required")
                continue
            if not a.model_revision:
                refused.append(f"{aid}: manifest carries no model "
                               "revision")
                continue
            if model_revision and a.model_revision != model_revision:
                refused.append(f"{aid}: model revision mismatch")
                continue
            fams = set(a.qualified_families)
            t = _terms(" ".join(a.family_terms or a.qualified_families))
            explicit = family is not None
            fam_match = (family in fams) if explicit else bool(q & t)
            if fam_match and explicit and not family_authenticated:
                # Untrusted hint: it may narrow WHICH adapters are
                # considered, but eligibility still requires lexical
                # evidence — the hint alone cannot force activation.
                fam_match = bool(q & t)
            if not fam_match:
                refused.append(f"{aid}: family not qualified "
                               f"({family or 'unresolved'})")
                continue
            overlap = (len(q & t) / math.sqrt(len(q) * len(t))
                       if q and t else 0.0)
            score = (0.85 * overlap + 0.15 * max(0.0, min(1.0, a.priority))
                     if not (explicit and family_authenticated)
                     else 0.85 + 0.15 * max(0.0, min(1.0, a.priority)))
            scored.append((a, score))
        scored.sort(key=lambda x: (x[1], x[0].adapter_id), reverse=True)

        dec: RoutingDecision | None = None
        while scored:
            a, score = scored[0]
            if score < self.ood_threshold:
                dec = RoutingDecision(None, score,
                                      "OOD: below activation threshold",
                                      family, True, tuple(refused))
                break
            if self._artifact_resolver is not None and \
                    self._recheck_at_activation:
                # The qualification binding is re-verified at the moment
                # of activation — bytes changed since registration make
                # the adapter ineligible, they are never served.
                reason = ""
                try:
                    actual = self._artifact_digester(
                        self._artifact_resolver(a.adapter_id))
                    if actual != a.adapter_digest:
                        reason = "artifact digest mismatch at activation"
                except Exception as exc:
                    reason = f"artifact unverifiable at activation ({exc})"
                if reason:
                    self._ineligible[a.adapter_id] = reason
                    refused.append(f"{a.adapter_id}: {reason}")
                    self.events.append({
                        "schema": SCHEMA_EVENT, "ts": round(time.time(), 3),
                        "kind": "activation_recheck_failed",
                        "adapter_id": a.adapter_id, "reason": reason})
                    scored.pop(0)
                    continue
            dec = RoutingDecision(a.adapter_id, score, "qualified",
                                  family, False, tuple(refused))
            break
        if dec is None:
            dec = RoutingDecision(None, 0.0, "no qualified adapter",
                                  family, True, tuple(refused))
        self.events.append({
            "schema": SCHEMA_EVENT, "ts": round(time.time(), 3),
            "query_digest": digest({"q": query}),
            "family_hint": family,
            "family_hint_authenticated": bool(family_authenticated),
            "controls": {
                "signature_enforced": self._verifier is not None
                or self._authority is not None,
                "authority_role": (self._required_role
                                   if self._authority is not None else None),
                "artifact_verified": self._artifact_resolver is not None,
                "artifact_rechecked_at_activation":
                    self._artifact_resolver is not None
                    and self._recheck_at_activation,
                "model_identity_required": self._require_model_identity,
            },
            "decision": asdict(dec)})
        return dec

    # ---- persistence -------------------------------------------------

    def to_json(self, signer: Ed25519Signer | None = None) -> str:
        """Registry schema v3 — adapters plus the mutable safety state
        (revocations, suspensions, health history) sealed under a
        monotonic ``seq`` and, when ``signer`` (or the router's
        ``state_signer``) is supplied, an authority signature.

        An unsigned v3 document is a *claim* about state — loadable in
        non-strict mode but flagged ``state_authenticated = False``. A
        signed document is an authorization snapshot: tampering with it
        fails verification at load."""
        signer = signer or self._state_signer
        state = {
            "adapters": [asdict(a) for a in
                         sorted(self._adapters.values(),
                                key=lambda x: x.adapter_id)],
            "revoked": sorted(self._revoked),
            "suspended": sorted(self._suspended),
            "generations": {k: v for k, v in
                            sorted(self._generations.items())},
            "health": {k: list(v) for k, v in
                       sorted(self._health.items())}}
        doc = seal_state(state, self._seq, signer)
        if self._state_authority is not None and signer is not None:
            # Attest the newest accepted state under the separately-held
            # authority — a load can then prove it sees the NEWEST
            # signed generation, not just some signed generation.
            self._state_authority.record_head(
                signer, seq=self._seq, state_digest=digest(state),
                event_head_digest=(self._event_log.last_digest()
                                   if self._event_log is not None
                                   else GENESIS_DIGEST))
        return json.dumps(doc, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path: str | Path, *, strict_state: bool = False,
                  registry_verifier: Ed25519Verifier | None = None,
                  revocation_log=None, min_seq: int = 0,
                  event_log=None, state_authority=None,
                  require_authority: bool = False,
                  event_authority=None,
                  **kw) -> "VerifiedAdapterRouter":
        """Load a registry file, newest schema first.

        * v3 — ``open_state`` enforces the signature whenever a
          ``registry_verifier`` is given (tampered state — e.g. a
          deleted revocation — fails closed) and refuses a ``seq``
          below ``min_seq``. ``strict_state=True`` additionally
          requires the state to be authenticated.
        * v2 — structurally complete but UNauthenticated (unsigned,
          unsequenced): loads non-strict with a warning, refuses under
          ``strict_state``.
        * v1 — cannot prove its safety state at all: loads marked
          ``state_complete = False``, refuses under ``strict_state``.

        ``revocation_log`` (a ``RevocationLog`` or path) is the
        authoritative revocation record: its verified entries are
        unioned over the file's own list, so deleting a revocation from
        a snapshot — signed or not — never restores access.

        v4.0.3 adds two authority layers:

        * ``event_log`` (a ``RegistryEventLog`` or path) — the signed
          state-transition chain. When it verifies, its folded final
          status is AUTHORITATIVE for every adapter it covers: a
          revoke cannot be dropped from history, and a REQUALIFIED
          generation overrides stale doc-level revocations. A broken
          chain is never folded; under ``require_authority`` it fails
          the load outright.
        * ``state_authority`` (a ``RegistryStateAuthority`` or path) —
          a separately-custodied signed head binding
          (seq, state_digest, event-head). The loaded file must match
          the attested head exactly: a stale signed snapshot or a
          truncated ledger is refused. ``require_authority`` makes the
          authority mandatory — production loads fail closed when it
          is missing, unreadable or disagreeing.
        """
        raw = json.loads(Path(path).read_text())
        schema = raw.get("schema")
        log = (revocation_log if revocation_log is None
               or isinstance(revocation_log, RevocationLog)
               else RevocationLog(revocation_log))
        ev_log = (event_log if event_log is None
                  or isinstance(event_log, RegistryEventLog)
                  else RegistryEventLog(event_log))
        authority_store = (
            state_authority if state_authority is None
            or isinstance(state_authority, RegistryStateAuthority)
            else RegistryStateAuthority(state_authority))
        if schema in (SCHEMA_REGISTRY_V1, SCHEMA_REGISTRY_V2) \
                and require_authority:
            raise ValueError(
                f"registry schema {schema!r} cannot satisfy the "
                "authority requirement — it predates signed sequenced "
                "state entirely")
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
        elif schema == SCHEMA_REGISTRY_V2:
            # v2 recorded the safety state but unsigned and unsequenced —
            # a modified file cannot be detected and the file cannot
            # prove it is the newest state.
            if strict_state:
                raise ValueError(
                    "v2 registries are unauthenticated — the safety "
                    "state cannot be verified against tampering; "
                    "refusing under strict_state")
            r = cls([QualifiedAdapterManifest(**a)
                     for a in raw.get("adapters", [])], **kw)
            r._revoked = {str(x) for x in raw.get("revoked", [])}
            r._suspended = {str(x) for x in raw.get("suspended", [])}
            r._health = {str(k): [int(x) for x in v]
                         for k, v in raw.get("health", {}).items()}
            r.registry_warnings.append(
                "registry schema v2 is unauthenticated — revocations "
                "are mutable in the file and there is no sequence to "
                "rule out a stale snapshot")
        else:
            opened = open_state(raw, registry_verifier, min_seq=min_seq)
            state = opened["state"]
            if strict_state and registry_verifier is None:
                raise ValueError(
                    "strict_state requires an authenticated registry — "
                    "supply registry_verifier")
            if require_authority and not opened["authenticated"]:
                raise ValueError(
                    "unsigned registry state cannot satisfy the "
                    "authority requirement")
            ctor_kw = dict(kw)
            if ev_log is not None:
                ctor_kw["event_log"] = ev_log
            if authority_store is not None:
                ctor_kw["state_authority"] = authority_store
            r = cls([QualifiedAdapterManifest(**a)
                     for a in state.get("adapters", [])], **ctor_kw)
            r._revoked = {str(x) for x in state.get("revoked", [])}
            r._suspended = {str(x) for x in state.get("suspended", [])}
            r._generations = {str(k): int(v) for k, v in
                              state.get("generations", {}).items()}
            r._health = {str(k): [int(x) for x in v]
                         for k, v in state.get("health", {}).items()}
            r._seq = opened["seq"]
            r.state_authenticated = opened["authenticated"]
            if not opened["authenticated"]:
                r.registry_warnings.append(
                    "registry state is unsigned — revocation deletions "
                    "cannot be detected; provisional mode only")

            # Durable state authority (v4.0.3): the file must match the
            # separately-custodied attested head exactly — seq AND state
            # digest AND the event log's head. A stale signed snapshot
            # or a ledger swapped for a shorter one is refused.
            if authority_store is not None or require_authority:
                if authority_store is None:
                    raise ValueError(
                        "registry state authority required but not "
                        "configured")
                if registry_verifier is None:
                    raise ValueError(
                        "registry state authority cannot be "
                        "authenticated without registry_verifier")
                ev_head = (ev_log.last_digest()
                           if (ev_log is not None
                               and ev_log.path.exists()) else None)
                reason = authority_store.check(
                    opened["seq"], digest(state),
                    verifier=registry_verifier,
                    event_head_digest=ev_head)
                if reason:
                    raise ValueError(reason)

            # Signed event chain (v4.0.3): when it verifies, its folded
            # final status is authoritative for the adapters it covers —
            # a dropped tail event is a dropped revocation, so a broken
            # chain is never folded and fails closed under
            # require_authority.
            if ev_log is not None:
                if not ev_log.path.exists():
                    msg = ("registry event ledger missing — cannot "
                           "prove current authorization state")
                    if require_authority:
                        raise ValueError(msg)
                    r.registry_warnings.append(msg)
                else:
                    folded, failures = ev_log.effective_state(
                        registry_verifier, event_authority)
                    if failures:
                        if require_authority:
                            raise ValueError(
                                "registry event chain failed "
                                "verification: " + "; ".join(failures))
                        r.registry_warnings.extend(
                            f"event log: {f}" for f in failures)
                    else:
                        for aid, st in folded.items():
                            if st["status"] == "revoked":
                                r._revoked.add(aid)
                                r._suspended.discard(aid)
                            elif st["status"] == "suspended":
                                r._suspended.add(aid)
                                r._revoked.discard(aid)
                            else:
                                r._revoked.discard(aid)
                                r._suspended.discard(aid)
                            r._generations[aid] = max(
                                r._generations.get(aid, 0),
                                st["generation"])
        # The authoritative revocation log overrides whatever the file
        # claims: logged revocations are unioned in, never removed.
        if log is not None:
            logged, failures, log_seq = log.load(registry_verifier)
            r._revoked |= logged
            r.registry_warnings.extend(
                f"revocation log: {f}" for f in failures)
            r._seq = max(r._seq, log_seq)
        return r
