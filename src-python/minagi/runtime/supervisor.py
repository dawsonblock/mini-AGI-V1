"""v16.4.3 serving supervisor — transactional activation over the
authority store (HARDENING_PLAN WP1/WP2/WP3/WP5).

v16.4.2 introduced the lifecycle and the durable JSONL journal.
v16.4.3 closes what was still process-local or caller-controlled:

  * grant consumption is reserved in `AuthorityStore` (SQLite,
    BEGIN IMMEDIATE) — durable across restarts, atomic across threads
    and processes (SEC-201/SEC-207);
  * activation ids are generated inside the trusted service
    (`secrets.token_hex(16)`) — campaign/seed are metadata, never a
    path component (SEC-204);
  * the serving pointer moves in the SAME transaction as the COMMITTED
    intent event — the pointer and the journal cannot diverge;
  * recovery distinguishes durable history from live state: a restart
    begins UNAVAILABLE/RECOVERY_REQUIRED and never reports SERVING for
    a model that is not resident — restoring traffic means re-admitting
    through the same verified path (SEC-202);
  * rollback re-validates revocation freshness and requires a live,
    retained, committed predecessor — durable intent and completion
    events bracket the pointer move (SEC-205 routing half).

Ordering guarantees (unchanged in spirit, now transactional):

  * the authorization event is committed before staging;
  * commit intent + pointer swap are one database transaction;
  * the signed completion event lands after;
  * any post-load failure unloads the candidate;
  * a pointer never names a model that is not resident.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

from egai.common.crypto import Ed25519Signer

from minagi.security.admission_grants import (GrantRefused,
                                              verify_grant)
from minagi.v161.authority import as_utc
from minagi.v161.immutable_snapshot import (MeasuredSnapshot,
                                            verify_snapshot)

from .access_policy import opaque_id
from .activation_state import (ActivationState, IllegalTransition,
                               check_transition)
from .authority_store import (AuthorityStore, AuthorityStoreError,
                              GrantConsumed)


class ServingState(str, Enum):
    """Live-traffic state of this service instance — distinct from the
    historical activation records (SEC-202)."""
    UNAVAILABLE = "UNAVAILABLE"            # no qualified model serving
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"  # durable commit exists but
                                           # its live state is unknown
    PREPARING = "PREPARING"                # reloading + verifying
    READY = "READY"                        # probes passed, not routed
    SERVING = "SERVING"                    # live model receiving traffic
    QUARANTINED = "QUARANTINED"            # withdrawn by authority


class SupervisedBackend(Protocol):
    """The contract a serving backend must satisfy under the
    supervisor: load into a non-serving handle, prove health, be
    unloadable on demand."""
    backend_id: str

    def load(self, snapshot: MeasuredSnapshot):  # pragma: no cover
        ...

    def health_probe(self, handle) -> None:  # pragma: no cover
        ...

    def unload(self, handle) -> None:  # pragma: no cover
        ...


class ActivationRefused(PermissionError):
    """The supervisor refused to move a candidate forward."""


class ActivationError(RuntimeError):
    """A supervised transition failed after authorization."""


class RecoveryRequired(ActivationRefused):
    """A durable committed activation exists but is not live — restore
    it through the verified path instead of trusting the pointer."""


@dataclass
class _Activation:
    activation_id: str
    state: ActivationState
    snapshot: MeasuredSnapshot | None = None
    grant: object | None = None
    backend: object | None = None
    handle: object | None = None


class ServingSupervisor:
    """Transactional activation control for the serving runtime."""

    def __init__(self, store: AuthorityStore, *,
                 runtime_signer: Ed25519Signer,
                 registry, runtime_identity: str = "local-supervisor",
                 min_policy_epoch: int = 0,
                 revocation_epoch_provider=None,
                 backend_manifest_doc=None, backend_modules=(),
                 backend_deps=(),
                 now: datetime | None = None):
        if runtime_signer is None:
            raise ActivationRefused(
                "the supervisor requires a protected runtime signing "
                "identity — unsigned completion records are not evidence")
        self.store = store
        self.signer = runtime_signer
        self.registry = registry
        self.runtime_identity = str(runtime_identity)
        self.min_policy_epoch = int(min_policy_epoch)
        self._revocation_epoch_provider = revocation_epoch_provider
        self.backend_manifest_doc = backend_manifest_doc
        self.backend_modules = tuple(backend_modules)
        self.backend_deps = tuple(backend_deps)
        self._now = now
        self._lock = threading.RLock()
        self._activations: dict[str, _Activation] = {}
        self._live_handles: dict[str, tuple] = {}
        self._active_id: str | None = None
        self._retained: str | None = None
        self._serving_state = ServingState.UNAVAILABLE
        self.router = None  # set by the service (WP8)
        # Boot state: the pointer is history, not liveness. A durable
        # committed pointer yields RECOVERY_REQUIRED; nothing is SERVING
        # until a live handle is re-verified.
        try:
            pointer = self.store.read_pointer()
        except AuthorityStoreError:
            pointer = None
        if pointer is not None and pointer.get("activation_id"):
            self._serving_state = ServingState.RECOVERY_REQUIRED
            self._pending_restore_id = str(pointer["activation_id"])
        else:
            self._pending_restore_id = None

    # --- plumbing ---------------------------------------------------
    def _at(self) -> int:
        return int(as_utc(self._now).timestamp())

    def _get(self, activation_id: str) -> _Activation:
        act = self._activations.get(activation_id)
        if act is None:
            raise ActivationRefused(
                f"unknown activation {activation_id!r} — the supervisor "
                "drives state, it does not infer it")
        return act

    def _transition(self, act: _Activation, dst: ActivationState, *,
                    event_type: str = "transition",
                    detail: dict | None = None,
                    sign: bool = False) -> None:
        check_transition(act.state, dst)
        self.store.append_event(
            activation_id=act.activation_id, event_type=event_type,
            from_state=act.state.value, to_state=dst.value,
            at=self._at(), detail=detail,
            signer=self.signer if sign else None)
        act.state = dst

    @property
    def serving_state(self) -> ServingState:
        return self._serving_state

    @property
    def active_id(self) -> str | None:
        return self._active_id

    def active_pointer(self) -> dict | None:
        try:
            return self.store.read_pointer()
        except AuthorityStoreError:
            return None

    # --- the lifecycle ----------------------------------------------
    def request(self, *, activation_id: str | None = None) -> _Activation:
        """REQUESTED — register a candidate. Activation ids are opaque
        and server-generated (SEC-204); a caller may not choose its own
        storage identity. `activation_id` exists only for journal
        replay/migration and must still pass the opaque-id grammar."""
        if activation_id is None:
            activation_id = opaque_id()
        else:
            from .access_policy import validate_opaque_id
            try:
                validate_opaque_id(activation_id,
                                   what="activation id")
            except Exception as exc:
                raise ActivationRefused(str(exc)) from exc
        with self._lock:
            if activation_id in self._activations:
                raise ActivationRefused(
                    f"activation id {activation_id!r} already exists")
            act = _Activation(activation_id=activation_id,
                              state=ActivationState.REQUESTED)
            self.store.append_event(
                activation_id=activation_id, event_type="requested",
                from_state="", to_state="REQUESTED", at=self._at())
            self._activations[activation_id] = act
            return act

    def authorize(self, activation_id: str, grant_doc) -> _Activation:
        """AUTHORIZED — the supervisor verifies the grant itself and
        reserves it in the transactional store: id, nonce, digest, and
        activation id are unique across restarts and concurrent
        callers (SEC-201/SEC-207)."""
        act = self._get(activation_id)
        try:
            grant = verify_grant(
                grant_doc, self.registry, now=self._now,
                audience_runtime_identity=self.runtime_identity)
        except GrantRefused as exc:
            self.abort(activation_id, reason=f"grant refused: {exc}")
            raise ActivationRefused(f"grant refused: {exc}") from exc
        if grant.policy_epoch < self.min_policy_epoch:
            self.abort(activation_id,
                       reason="grant policy epoch superseded")
            raise ActivationRefused(
                f"grant policy epoch {grant.policy_epoch} is older than "
                f"the operative epoch {self.min_policy_epoch} — "
                "superseded authorization refused")
        with self._lock:
            try:
                self.store.reserve_grant(
                    grant, activation_id=activation_id, at=self._at(),
                    from_state=act.state.value,
                    detail={"grant_id": grant.grant_id,
                            "manifest": grant.runtime_manifest_digest,
                            "backend": grant.backend_id,
                            "policy_epoch": grant.policy_epoch})
            except GrantConsumed as exc:
                self.abort(activation_id, reason="grant replay")
                raise ActivationRefused(
                    "grant replay refused — a grant authorizes exactly "
                    "one activation") from exc
            except AuthorityStoreError as exc:
                self.abort(activation_id,
                           reason=f"authority store: {exc}")
                raise ActivationError(
                    f"the authority store could not reserve the grant: "
                    f"{exc}") from exc
            act.grant = grant
            act.state = ActivationState.AUTHORIZED
            return act

    def stage(self, activation_id: str,
              snapshot: MeasuredSnapshot) -> _Activation:
        """STAGED — the measured snapshot is re-verified before it is
        allowed any further; staging produces measurement, never
        authorization."""
        act = self._get(activation_id)
        if not isinstance(snapshot, MeasuredSnapshot):
            self.abort(activation_id,
                       reason="object is not a MeasuredSnapshot")
            raise ActivationRefused(
                "stage() requires a MeasuredSnapshot produced by "
                "stage_snapshot — a caller-built object with a path() "
                "method is not evidence")
        if act.grant is not None and \
                act.grant.artifact_root_digest != \
                digest_root(snapshot):
            self.abort(activation_id,
                       reason="snapshot digest != grant artifact_root")
            raise ActivationRefused(
                "the staged artifact root does not match the grant's "
                "artifact_root_digest — authorization does not transfer")
        verify_snapshot(snapshot)
        act.snapshot = snapshot
        self._transition(act, ActivationState.STAGED,
                         detail={"artifact_root_digest":
                                 digest_root(snapshot)})
        return act

    def prepare(self, activation_id: str, backend) -> _Activation:
        """PREPARED — the backend loads into a non-serving handle. When
        a backend manifest is configured, the installed implementation
        and dependency closure are re-measured against it first —
        a signature for hf-peft cannot authorize an arbitrary object
        claiming to be hf-peft (SEC-206)."""
        act = self._get(activation_id)
        if act.snapshot is None:
            raise ActivationRefused("prepare requires a staged snapshot")
        if getattr(backend, "backend_id", None) != (
                act.grant.backend_id if act.grant else None):
            self.abort(activation_id, reason="backend != grant backend")
            raise ActivationRefused(
                f"backend {getattr(backend, 'backend_id', None)!r} does "
                "not match the grant's authorized backend")
        if self.backend_manifest_doc is not None:
            try:
                self._verify_backend_identity(backend, act)
            except Exception as exc:  # noqa: BLE001 - fail closed
                self.abort(activation_id,
                           reason=f"backend identity: {exc}")
                raise ActivationRefused(
                    f"backend identity verification failed: {exc}") \
                    from exc
        try:
            handle = backend.load(act.snapshot)
        except Exception as exc:  # noqa: BLE001 - fail closed
            self.abort(activation_id, reason=f"backend load: {exc}")
            raise ActivationError(
                f"backend failed to load the staged snapshot: {exc}") \
                from exc
        act.backend = backend
        act.handle = handle
        self._live_handles[activation_id] = (backend, handle)
        self._transition(act, ActivationState.PREPARED,
                         detail={"backend_id": backend.backend_id})
        return act

    def _verify_backend_identity(self, backend, act) -> None:
        from .backend_manifest import (verify_backend_manifest,
                                       verify_installed_backend)
        manifest = verify_backend_manifest(
            self.backend_manifest_doc, self.registry, now=self._now)
        if manifest.backend_id != backend.backend_id:
            raise ActivationRefused(
                f"backend manifest names {manifest.backend_id!r}, not "
                f"{backend.backend_id!r}")
        if act.grant is not None and act.grant.backend_binary_digest and \
                act.grant.backend_binary_digest != \
                str(self.backend_manifest_doc.get("digest")):
            raise ActivationRefused(
                "grant binds a different backend manifest digest")
        verify_installed_backend(
            manifest, module_names=self.backend_modules,
            dependency_packages=self.backend_deps,
            min_policy_epoch=self.min_policy_epoch)

    def health_check(self, activation_id: str) -> _Activation:
        """READY — health + inference probes must pass before a
        candidate may be committed. A failed probe aborts; the previous
        healthy model was never touched and keeps serving."""
        act = self._get(activation_id)
        backend, handle = self._live_handles.get(
            activation_id, (None, None))
        if backend is None or handle is None:
            raise ActivationRefused(
                "health_check requires a prepared handle")
        try:
            backend.health_probe(handle)
        except Exception as exc:  # noqa: BLE001 - fail closed
            self.abort(activation_id, reason=f"health probe: {exc}")
            raise ActivationError(
                f"health probe failed — candidate aborted, previous "
                f"version remains active: {exc}") from exc
        self._transition(act, ActivationState.READY)
        return act

    def commit_activation(self, activation_id: str,
                          *, expected_previous: str | None = None
                          ) -> _Activation:
        """COMMITTED -> ACTIVE. The commit intent event and the serving
        pointer move in ONE database transaction — a crash cannot leave
        a routed pointer without its commit record. The signed
        completion event lands after; a completion-write failure rolls
        the pointer back."""
        act = self._get(activation_id)
        if expected_previous is not None \
                and self._active_id != expected_previous:
            self.abort(activation_id,
                       reason="expected_previous mismatch")
            raise ActivationRefused(
                f"expected previous activation {expected_previous!r} but "
                f"{self._active_id!r} is active — refusing to swap onto "
                "an unknown base")

        with self._lock:
            check_transition(act.state, ActivationState.COMMITTED)
            previous = self._active_id
            try:
                self.store.commit_with_pointer(
                    activation_id=activation_id, at=self._at(),
                    artifact_root_digest=(
                        digest_root(act.snapshot) if act.snapshot else ""),
                    backend_id=(act.grant.backend_id if act.grant else ""),
                    detail={"kind": "activation_intent"})
            except AuthorityStoreError as exc:
                self.abort(activation_id,
                           reason=f"commit transaction failed: {exc}")
                raise ActivationError(
                    f"commit intent + pointer transaction failed: {exc}") \
                    from exc
            act.state = ActivationState.COMMITTED
            self._active_id = activation_id
            act.state = ActivationState.ACTIVE
            self._serving_state = ServingState.SERVING
            if self.router is not None and act.handle is not None:
                self.router.activate(activation_id, act.backend,
                                     act.handle)

            # signed completion evidence — if it cannot be written the
            # candidate is rolled back, not left active without evidence
            try:
                self.store.append_event(
                    activation_id=activation_id,
                    event_type="activation_completion",
                    from_state="COMMITTED", to_state="ACTIVE",
                    at=self._at(),
                    detail={"kind": "activation_completion",
                            "grant_id": (act.grant.grant_id
                                         if act.grant else ""),
                            "previous": previous or ""},
                    signer=self.signer)
            except AuthorityStoreError as exc:
                self._active_id = previous
                self._serving_state = (ServingState.SERVING if previous
                                       else ServingState.UNAVAILABLE)
                if self.router is not None:
                    if previous is not None and previous in \
                            self._live_handles:
                        pb, ph = self._live_handles[previous]
                        self.router.activate(previous, pb, ph)
                    else:
                        self.router.deactivate(activation_id)
                try:
                    self.store.clear_pointer(
                        at=self._at(),
                        because="completion event failed")
                except AuthorityStoreError:
                    pass
                self.abort(activation_id,
                           reason=f"completion record failed: {exc}")
                raise ActivationError(
                    "the signed activation-completion record could not "
                    "be persisted — the candidate was rolled back rather "
                    "than left serving without evidence") from exc

            if act.grant is not None:
                self.store.set_grant_state(act.grant.grant_id,
                                           "committed")
            # retain one healthy predecessor as the rollback target
            if previous and previous in self._live_handles:
                if self._retained and self._retained != previous:
                    old = self._retained
                    if old in self._live_handles:
                        b, h = self._live_handles.pop(old)
                        try:
                            b.unload(h)
                        except Exception:  # noqa: BLE001 - best effort
                            pass
                    old_act = self._activations.get(old)
                    if old_act is not None and not old_act.state.terminal:
                        self.store.append_event(
                            activation_id=old, event_type="evicted",
                            from_state=old_act.state.value,
                            to_state="ABORTED", at=self._at(),
                            detail={"kind": "retention_evicted",
                                    "by": activation_id})
                        old_act.state = ActivationState.ABORTED
                self._retained = previous
            return act

    # --- failure / rollback paths ------------------------------------
    def abort(self, activation_id: str, *, reason: str = "") -> None:
        """Abort a candidate: unload whatever is loaded and record the
        terminal state. Idempotent — safe to call twice or on a
        candidate that never loaded."""
        with self._lock:
            act = self._activations.get(activation_id)
            if act is None:
                return
            if act.state.terminal:
                return
            pair = self._live_handles.pop(activation_id, None)
            if pair is not None:
                backend, handle = pair
                try:
                    backend.unload(handle)
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
            if self.router is not None:
                self.router.deactivate(activation_id)
            if act.grant is not None:
                self.store.set_grant_state(act.grant.grant_id, "aborted")
            if self._active_id == activation_id:
                self._quarantine_pointer(activation_id, reason)
                try:
                    check_transition(act.state, ActivationState.QUARANTINED)
                    self.store.append_event(
                        activation_id=activation_id,
                        event_type="quarantined",
                        from_state=act.state.value,
                        to_state="QUARANTINED", at=self._at(),
                        detail={"reason": reason}, signer=self.signer)
                    act.state = ActivationState.QUARANTINED
                except IllegalTransition:
                    # COMMITTED -> QUARANTINED is legal; REQUESTED ->
                    # QUARANTINED (active id set but transition grammar
                    # forbids) collapses to ABORTED + pointer repair.
                    self.store.append_event(
                        activation_id=activation_id,
                        event_type="aborted", from_state=act.state.value,
                        to_state="ABORTED", at=self._at(),
                        detail={"reason": reason})
                    act.state = ActivationState.ABORTED
                if act.grant is not None:
                    self.store.set_grant_state(act.grant.grant_id,
                                               "quarantined")
                self._serving_state = ServingState.QUARANTINED
                return
            try:
                check_transition(act.state, ActivationState.ABORTED)
                self.store.append_event(
                    activation_id=activation_id, event_type="aborted",
                    from_state=act.state.value, to_state="ABORTED",
                    at=self._at(), detail={"reason": reason})
                act.state = ActivationState.ABORTED
            except IllegalTransition:
                act.state = ActivationState.ABORTED

    def quarantine_active(self, *, reason: str) -> None:
        """Withdraw the currently active version (e.g. its promotion
        was just revoked) and restore the last committed predecessor
        if one is still held."""
        if self._active_id is None:
            return
        self.abort(self._active_id, reason=reason or "quarantined")

    def rollback(self, activation_id: str | None = None,
                 *, min_revocation_epoch: int | None = None) -> str | None:
        """Restore a retained committed predecessor. The target must be
        live, must still pass the operative revocation epoch, and the
        pointer move is bracketed by durable rollback intent and signed
        completion events (SEC-205/WP8)."""
        with self._lock:
            target = activation_id
            if target is None:
                target = self._retained or self._last_completed_excluding(
                    self._active_id, require_live=True)
            if target is None or target not in self._live_handles:
                raise ActivationRefused(
                    "no retained committed version to roll back to — a "
                    "rollback must re-admit artifacts through admission, "
                    "not improvise")
            tgt_act = self._activations.get(target)
            # Rollback does not bypass security: the predecessor's grant
            # must still satisfy the operative revocation epoch.
            floor = min_revocation_epoch
            if floor is None and self._revocation_epoch_provider is not None:
                floor = int(self._revocation_epoch_provider())
            if floor is not None and tgt_act is not None and \
                    tgt_act.grant is not None and \
                    int(tgt_act.grant.revocation_epoch) < int(floor):
                raise ActivationRefused(
                    f"rollback target's grant was issued against "
                    f"revocation epoch {tgt_act.grant.revocation_epoch}; "
                    f"the operative epoch is {floor} — stale "
                    "authorization refused")
            b, h = self._live_handles[target]
            try:
                b.health_probe(h)
            except Exception as exc:  # noqa: BLE001
                raise ActivationRefused(
                    f"rollback target failed a fresh health probe: {exc}") \
                    from exc

            old = self._active_id
            self.store.append_event(
                activation_id=target, event_type="rollback_intent",
                from_state="ACTIVE", to_state="ACTIVE", at=self._at(),
                detail={"kind": "rollback_intent",
                        "from": old or ""})
            self.store.commit_with_pointer(
                activation_id=target, at=self._at(),
                artifact_root_digest=(
                    digest_root(tgt_act.snapshot)
                    if tgt_act is not None and tgt_act.snapshot else ""),
                backend_id=(tgt_act.grant.backend_id
                            if tgt_act is not None and tgt_act.grant
                            else ""),
                event_type="rollback_pointer",
                from_state="ACTIVE", to_state="ACTIVE",
                detail={"kind": "rollback", "from": old or ""})
            self._active_id = target
            self._serving_state = ServingState.SERVING
            if self.router is not None:
                self.router.activate(target, b, h)
            self.store.append_event(
                activation_id=target, event_type="rollback_completion",
                from_state="ACTIVE", to_state="ACTIVE", at=self._at(),
                detail={"kind": "rollback_completion",
                        "from": old or ""}, signer=self.signer)
            return target

    def _last_completed_excluding(self, exclude: str | None,
                                  *, require_live: bool = False
                                  ) -> str | None:
        """Newest activation that durably completed a commit other than
        `exclude`. With require_live, only a candidate whose model is
        still resident counts — a pointer must never route to a model
        that is not loaded."""
        completed: list[str] = []
        for e in self.store.events():
            if e["to_state"] == "ACTIVE" and \
                    e["event_type"] == "activation_completion" and \
                    e["activation_id"] != exclude:
                completed.append(e["activation_id"])
        for aid in reversed(completed):
            if not require_live or aid in self._live_handles:
                return aid
        return None

    def _quarantine_pointer(self, activation_id: str, reason: str) -> None:
        """Move the serving pointer off a quarantined candidate: restore
        the last committed predecessor *that is still resident*, else
        clear it — a pointer never routes to an unloaded model."""
        fallback = self._last_completed_excluding(
            activation_id, require_live=True)
        try:
            if fallback is not None:
                fb = self._activations.get(fallback)
                self.store.commit_with_pointer(
                    activation_id=fallback, at=self._at(),
                    artifact_root_digest=(
                        digest_root(fb.snapshot)
                        if fb is not None and fb.snapshot else ""),
                    backend_id=(fb.grant.backend_id
                                if fb is not None and fb.grant else ""),
                    detail={"kind": "quarantine_restore",
                            "because": f"quarantine {activation_id}: "
                                       f"{reason}"})
                self._active_id = fallback
                self._serving_state = ServingState.SERVING
                if self.router is not None and fallback in \
                        self._live_handles:
                    rb, rh = self._live_handles[fallback]
                    self.router.activate(fallback, rb, rh)
            else:
                self.store.clear_pointer(
                    at=self._at(),
                    because=f"quarantine {activation_id}: {reason}")
                self._active_id = None
                self._serving_state = ServingState.UNAVAILABLE
        except AuthorityStoreError:
            self._active_id = None
            self._serving_state = ServingState.UNAVAILABLE

    # --- crash recovery ----------------------------------------------
    def recover(self) -> dict:
        """Reconcile the durable event log after a restart. The event
        chain is verified first — tampering, truncation, or a broken
        payload digest fails closed (SEC-205).

        Outcome states:
          * activations stalled before COMMITTED → aborted;
          * COMMITTED intent whose pointer commit never landed → the
            transaction rolled back together, so none can exist — any
            COMMITTED event without a matching pointer-commit is
            quarantined evidence, not silently completed;
          * a pointer naming an unknown activation → cleared;
          * a valid committed pointer → RECOVERY_REQUIRED: historical
            evidence, not liveness. The service must restore through
            the verified path before serving (SEC-202).
        """
        with self._lock:
            report = {"reconciled_completions": [], "aborted": [],
                      "cleared_pointer": False, "serving_state": None,
                      "requires_restoration": None}
            from .journal_v2 import verify_event_log
            events = verify_event_log(self.store, self.registry,
                                      now=self._now)
            pointer = self.store.read_pointer()
            by_id: dict[str, list] = {}
            for e in events:
                by_id.setdefault(e["activation_id"], []).append(e)

            def last_state(aid: str) -> str:
                return by_id[aid][-1]["to_state"] if by_id.get(aid) else ""

            for aid in list(by_id):
                if aid == "__migration__":
                    continue
                state = last_state(aid)
                if state in ("ABORTED", "QUARANTINED", "ACTIVE"):
                    continue
                self.store.append_event(
                    activation_id=aid, event_type="aborted",
                    from_state=state, to_state="ABORTED", at=self._at(),
                    detail={"reason": "recovery: stalled before commit"})
                report["aborted"].append(aid)

            active = str((pointer or {}).get("activation_id") or "")
            if active and active not in by_id:
                self.store.clear_pointer(
                    at=self._at(),
                    because="pointer named an activation with no "
                            "event history")
                report["cleared_pointer"] = True
                active = ""
            if active:
                self._pending_restore_id = active
                self._serving_state = ServingState.RECOVERY_REQUIRED
            else:
                self._pending_restore_id = None
                self._serving_state = ServingState.UNAVAILABLE
            report["serving_state"] = self._serving_state.value
            report["requires_restoration"] = self._pending_restore_id
            return report

    def mark_restored(self, activation_id: str) -> None:
        """Called after a restoration launch commits a live, verified
        model for the activation — the service is SERVING again."""
        with self._lock:
            self._serving_state = ServingState.SERVING
            self._pending_restore_id = None


# Backward-compatible alias: tests and callers imported
# recover_from_journal in v16.4.2.
ServingSupervisor.recover_from_journal = ServingSupervisor.recover


def digest_root(snapshot: MeasuredSnapshot) -> str:
    """The digest that identifies one measured artifact set — what a
    grant's artifact_root_digest binds."""
    from egai.common.canonical import digest
    return digest({name: d for name, d in snapshot.artifact_digests})
