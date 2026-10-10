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

Ordering guarantees (v16.4.4 — two-stage activation, WP-A/SEC-301..303):

  * the authorization event is committed before staging;
  * durable activation intent commits at the next deployment
    generation BEFORE the candidate can accept any traffic —
    authorization is not activation;
  * the router publishes exactly the authorized generation, then the
    supervisor records the observed routing outcome — a truthful,
    signed, post-traffic record;
  * a failed transition reconciles the durable intent to a live
    committed predecessor or a durably UNAVAILABLE state — the
    predecessor pointer is never silently cleared while a model
    serves from memory only;
  * model resources are released only after routing is withdrawn and
    the activation's own request leases have drained — a drain
    timeout retains the backend, it is not unload permission;
  * when the backend runs in a worker process, a drain timeout +
    cancel grace escalates to terminate() — a wedged model is killed
    without stopping the supervisor (RUN-401);
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
                              GenerationConflict, GrantConsumed)


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
        self._deferred_unloads: set[str] = set()
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

    def _durable_state(self, activation_id: str,
                       default: str = "") -> str:
        """The activation's last DURABLE to_state — the authoritative
        from_state for the next journal event. In-memory activation
        state can legally diverge from durable history when reconcile
        events move the deployment record; the journal chain binds the
        durable sequence, not memory."""
        try:
            events = self.store.events(activation_id)
        except AuthorityStoreError:
            return default
        return str(events[-1]["to_state"]) if events else default

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
                            "policy_epoch": grant.policy_epoch,
                            "decision_digest":
                                grant.promotion_decision_digest,
                            "qualification_digest":
                                grant.qualification_digest,
                            "artifact_root_digest":
                                grant.artifact_root_digest,
                            "backend_binary_digest":
                                grant.backend_binary_digest})
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
        if self.router is not None:
            # Register the route entry WITHOUT opening traffic — the
            # router owns (backend, handle) lifetime from here; only an
            # authorized deployment generation may publish it.
            try:
                self.router.prepare_route(activation_id, backend,
                                          handle, at=self._at())
            except Exception as exc:  # noqa: BLE001 - fail closed
                self.abort(activation_id,
                           reason=f"route prepare: {exc}")
                raise ActivationError(
                    f"could not register the prepared route: {exc}") \
                    from exc
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
                          *, expected_previous: str | None = None,
                          expected_generation: int | None = None
                          ) -> _Activation:
        """COMMITTED -> ACTIVE via the recoverable two-stage protocol
        (v16.4.4 WP-A):

          1. ``commit_activation_intent`` — durable authorization at
             the NEXT deployment generation (compare-and-swap on
             ``expected_generation``), BEFORE the candidate can accept
             a single request. The commit event, serving pointer, and
             deployment row commit in ONE transaction.
          2. ``router.publish_route`` — open traffic for exactly the
             authorized generation; a stale generation is refused.
          3. ``record_routing_observation`` — the signed, post-traffic
             evidence that routing actually happened. It is truthful
             because it exists only after routing, and it is never
             presented as pre-traffic authorization.

        A failure between stages reconciles the durable deployment
        intent to a live committed predecessor — or to a durably
        UNAVAILABLE state — instead of leaving the pointer cleared
        while a model serves from memory only."""
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
            dep = self.store.deployment()
            current_gen = int(dep["deployment_generation"]) \
                if dep is not None else 0
            if expected_generation is not None and \
                    int(expected_generation) != current_gen:
                self.abort(activation_id,
                           reason="stale deployment generation")
                raise ActivationRefused(
                    f"expected deployment generation "
                    f"{expected_generation} but the durable generation "
                    f"is {current_gen} — refusing to activate onto a "
                    "superseded transition")
            # stage 1 — durable authorization BEFORE traffic
            try:
                commit = self.store.commit_activation_intent(
                    candidate_id=activation_id,
                    expected_generation=current_gen, at=self._at(),
                    artifact_root_digest=(
                        digest_root(act.snapshot) if act.snapshot else ""),
                    backend_id=(act.grant.backend_id if act.grant else ""),
                    policy_epoch=self.min_policy_epoch,
                    detail={"kind": "activation_intent"})
            except GenerationConflict as exc:
                self.abort(activation_id,
                           reason=f"stale transition: {exc}")
                raise ActivationRefused(str(exc)) from exc
            except AuthorityStoreError as exc:
                self.abort(activation_id,
                           reason=f"commit transaction failed: {exc}")
                raise ActivationError(
                    f"commit intent + pointer transaction failed: {exc}") \
                    from exc
            act.state = ActivationState.COMMITTED
            self._active_id = activation_id
            generation = int(commit["generation"])
            transition_id = str(commit["transition_id"])
            # stage 2 — publish ONLY the already-authorized generation
            try:
                if self.router is not None and act.handle is not None:
                    self.router.publish_route(
                        activation_id, generation=generation,
                        at=self._at())
            except Exception as exc:  # noqa: BLE001 - fail closed
                self._reconcile_deployment(
                    act=act, previous=previous, generation=generation,
                    because=f"route publish failed: {exc}",
                    context="publish")
                self.abort(
                    activation_id,
                    reason=f"routing publication failed: {exc}")
                raise ActivationError(
                    f"routing publication of generation {generation} "
                    f"failed — the deployment was reconciled: {exc}") \
                    from exc
            act.state = ActivationState.ACTIVE
            self._serving_state = ServingState.SERVING
            # stage 3 — the signed OBSERVED routing outcome
            try:
                self.store.record_routing_observation(
                    activation_id=activation_id, generation=generation,
                    transition_id=transition_id, at=self._at(),
                    detail={"kind": "routing_observed",
                            "grant_id": (act.grant.grant_id
                                         if act.grant else ""),
                            "previous": previous or ""},
                    signer=self.signer)
            except AuthorityStoreError as exc:
                # B was published and may already have served traffic:
                # withdraw it, drain it, and reconcile durable intent —
                # never leave it serving without evidence.
                if self.router is not None:
                    self.router.stop_accepting(activation_id)
                self._reconcile_deployment(
                    act=act, previous=previous, generation=generation,
                    because=f"routing observation failed: {exc}",
                    context="observation")
                self.abort(
                    activation_id,
                    reason=f"routing observation record failed: {exc}")
                raise ActivationError(
                    "the signed routing-observation record could not "
                    "be persisted — the candidate was withdrawn and "
                    "the deployment reconciled rather than left "
                    "serving without evidence") from exc

            if act.grant is not None:
                self.store.set_grant_state(act.grant.grant_id,
                                           "committed")
            # retain one healthy predecessor as the rollback target
            if previous and previous in self._live_handles:
                if self._retained and self._retained != previous:
                    old = self._retained
                    if old in self._live_handles:
                        b, h = self._live_handles.pop(old)
                        old_act = self._activations.get(old)
                        self._retire_backend(
                            old_act if old_act is not None
                            else _Activation(old, ActivationState.ABORTED),
                            (b, h))
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

    def _reconcile_deployment(self, *, act: _Activation,
                              previous: str | None, generation: int,
                              because: str, context: str) -> None:
        """After a failed transition stage: CAS the durable deployment
        intent to the live committed predecessor (RESTORED) or to a
        durably UNAVAILABLE state at a NEW generation. The in-memory
        router is then aligned to the durable outcome — the two never
        diverge silently."""
        if self.router is not None:
            self.router.stop_accepting(act.activation_id)
        try:
            if previous and previous in self._live_handles:
                pact = self._activations.get(previous)
                prev_state = self._durable_state(previous, "ACTIVE")
                rec = self.store.reconcile_deployment(
                    expected_generation=generation,
                    desired_id=previous, phase="RESTORED",
                    at=self._at(),
                    event_activation_id=previous,
                    event_type="deployment_restored",
                    event_from_state=prev_state,
                    event_to_state=prev_state,
                    because=f"{context} failure on "
                            f"{act.activation_id}: {because}",
                    artifact_root_digest=(
                        digest_root(pact.snapshot)
                        if pact is not None and pact.snapshot else ""),
                    backend_id=(pact.grant.backend_id
                                if pact is not None and pact.grant
                                else ""),
                    signer=self.signer)
                self._active_id = previous
                self._serving_state = ServingState.SERVING
                if self.router is not None:
                    try:
                        self.router.publish_route(
                            previous,
                            generation=int(rec["generation"]),
                            at=self._at())
                    except Exception:  # noqa: BLE001
                        self.router.stop_accepting(previous)
                        self._active_id = None
                        self._serving_state = ServingState.UNAVAILABLE
            else:
                cand_state = self._durable_state(
                    act.activation_id, "COMMITTED")
                self.store.reconcile_deployment(
                    expected_generation=generation, desired_id="",
                    phase="UNAVAILABLE", at=self._at(),
                    event_activation_id=act.activation_id,
                    event_type="deployment_unavailable",
                    event_from_state=cand_state,
                    event_to_state="QUARANTINED",
                    because=f"{context} failure on "
                            f"{act.activation_id}: {because}",
                    signer=self.signer)
                self._active_id = None
                self._serving_state = ServingState.UNAVAILABLE
        except AuthorityStoreError:
            # Reconciliation itself failed: the durable intent still
            # names the candidate at `generation`. Serve nothing in
            # memory — restart recovery reconciles the durable record.
            self._active_id = None
            self._serving_state = ServingState.UNAVAILABLE

    def _retire_backend(self, act: _Activation, pair) -> dict:
        """SEC-303 ordering for releasing a model: withdraw routing
        FIRST, drain the activation's own leases, cancel cooperatively
        on timeout, and unload only when its in-flight count is zero.
        Resources are retained (and the unload deferred) while leases
        remain — a timeout is never unload permission."""
        activation_id = act.activation_id
        outcome = None
        if self.router is not None:
            if self.router.route_entry(activation_id) is not None:
                outcome = self.router.retire(activation_id)
            else:
                self.router.stop_accepting(activation_id)
        if outcome is None:
            if pair is not None:
                try:
                    pair[0].unload(pair[1])
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
            return {"unloaded": True, "inflight": 0, "terminated": False}
        if outcome.get("terminated"):
            # A wedged model was force-killed — that is a significant
            # operational fact and belongs in the durable journal, not
            # only in the retire outcome dict.
            try:
                durable = self._durable_state(
                    activation_id, act.state.value)
                self.store.append_event(
                    activation_id=activation_id,
                    event_type="backend_terminated",
                    from_state=durable, to_state=durable,
                    at=self._at(),
                    detail={"reason": "request leases outlived the "
                                      "cancel grace — backend worker "
                                      "terminated (RUN-401)",
                            "unloaded": bool(outcome.get("unloaded"))})
            except AuthorityStoreError:
                pass
        if not outcome["unloaded"]:
            self._deferred_unloads.add(activation_id)
            try:
                durable = self._durable_state(
                    activation_id, act.state.value)
                self.store.append_event(
                    activation_id=activation_id,
                    event_type="unload_deferred",
                    from_state=durable, to_state=durable,
                    at=self._at(),
                    detail={"reason": "in-flight leases retain the "
                                      "backend — unload deferred",
                            "inflight": int(outcome.get("inflight", 0))})
            except AuthorityStoreError:
                pass
        else:
            self._deferred_unloads.discard(activation_id)
        return outcome

    def reap(self) -> list[str]:
        """Retry deferred unloads — called opportunistically and at
        shutdown. A model is released only once its leases are gone."""
        still: list[str] = []
        for aid in list(self._deferred_unloads):
            outcome = (self.router.retire(aid)
                       if self.router is not None else
                       {"unloaded": True})
            if not outcome.get("unloaded"):
                still.append(aid)
            else:
                self._deferred_unloads.discard(aid)
        return still

    def shutdown(self, *, timeout: float | None = None) -> dict:
        """Service shutdown: stop all routes, drain, and unload what
        is actually idle. Returns what could not be released."""
        remaining: dict[str, int] = {}
        if self.router is not None:
            for state in self.router.lease_states():
                self.router.stop_accepting(state.activation_id)
            for state in self.router.lease_states():
                out = self.router.retire(state.activation_id,
                                         timeout=timeout)
                if not out["unloaded"]:
                    remaining[state.activation_id] = int(out["inflight"])
            self._deferred_unloads.difference_update(
                set(self._deferred_unloads) - set(remaining))
        return {"unreleased": remaining}

    # --- failure / rollback paths ------------------------------------
    def abort(self, activation_id: str, *, reason: str = "") -> None:
        """Abort a candidate: STOP ROUTING FIRST, drain the
        activation's own request leases, then release model resources
        only when nothing in flight references them (SEC-303).
        Idempotent — safe to call twice or on a candidate that never
        loaded."""
        with self._lock:
            act = self._activations.get(activation_id)
            if act is None:
                return
            if act.state.terminal:
                return
            pair = self._live_handles.pop(activation_id, None)
            self._retire_backend(act, pair)
            if act.grant is not None:
                self.store.set_grant_state(act.grant.grant_id, "aborted")
            durable = self._durable_state(activation_id,
                                          act.state.value)
            if self._active_id == activation_id:
                if durable == "QUARANTINED":
                    act.state = ActivationState.QUARANTINED
                elif durable == "ABORTED":
                    act.state = ActivationState.ABORTED
                else:
                    try:
                        check_transition(
                            ActivationState(durable),
                            ActivationState.QUARANTINED)
                        self.store.append_event(
                            activation_id=activation_id,
                            event_type="quarantined",
                            from_state=durable,
                            to_state="QUARANTINED", at=self._at(),
                            detail={"reason": reason},
                            signer=self.signer)
                        act.state = ActivationState.QUARANTINED
                    except IllegalTransition:
                        # e.g. durable COMMITTED grammar forbids
                        # REQUESTED -> QUARANTINED — collapse to
                        # ABORTED + pointer repair.
                        self.store.append_event(
                            activation_id=activation_id,
                            event_type="aborted", from_state=durable,
                            to_state="ABORTED", at=self._at(),
                            detail={"reason": reason})
                        act.state = ActivationState.ABORTED
                if act.grant is not None:
                    self.store.set_grant_state(act.grant.grant_id,
                                               "quarantined")
                self._serving_state = ServingState.QUARANTINED
                # Reconcile the durable intent AFTER the activation's
                # own terminal event — the per-activation journal chain
                # requires each event's from_state to be the recorded
                # previous to_state.
                self._quarantine_pointer(activation_id, reason,
                                         from_state=act.state.value)
                return
            if durable == "QUARANTINED":
                act.state = ActivationState.QUARANTINED
            elif durable == "ABORTED":
                act.state = ActivationState.ABORTED
            else:
                try:
                    check_transition(ActivationState(durable),
                                     ActivationState.ABORTED)
                    self.store.append_event(
                        activation_id=activation_id,
                        event_type="aborted",
                        from_state=durable, to_state="ABORTED",
                        at=self._at(), detail={"reason": reason})
                except IllegalTransition:
                    pass
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
            dep = self.store.deployment()
            current_gen = int(dep["deployment_generation"]) \
                if dep is not None else 0
            commit = self.store.commit_activation_intent(
                candidate_id=target, expected_generation=current_gen,
                at=self._at(),
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
            generation = int(commit["generation"])
            if self.router is not None:
                try:
                    self.router.publish_route(
                        target, generation=generation, at=self._at())
                except Exception as exc:  # noqa: BLE001 - fail closed
                    self._reconcile_deployment(
                        act=tgt_act, previous=old, generation=generation,
                        because=f"rollback route publish failed: {exc}",
                        context="rollback-publish")
                    raise ActivationError(
                        f"rollback routing publication failed: {exc}") \
                        from exc
            try:
                self.store.record_routing_observation(
                    activation_id=target, generation=generation,
                    transition_id=str(commit["transition_id"]),
                    at=self._at(), event_type="rollback_completion",
                    from_state="ACTIVE", to_state="ACTIVE",
                    detail={"kind": "rollback_completion",
                            "from": old or ""}, signer=self.signer)
            except Exception as exc:  # noqa: BLE001 - fail closed
                self._reconcile_deployment(
                    act=tgt_act, previous=old, generation=generation,
                    because=f"rollback completion failed: {exc}",
                    context="rollback-observation")
                raise ActivationError(
                    f"rollback completion could not be persisted: {exc}") \
                    from exc
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
                    e["event_type"] in ("activation_completion",
                                        "routing_observed",
                                        "rollback_completion",
                                        "deployment_restored") and \
                    e["activation_id"] != exclude:
                completed.append(e["activation_id"])
        for aid in reversed(completed):
            if not require_live or aid in self._live_handles:
                return aid
        return None

    def _quarantine_pointer(self, activation_id: str, reason: str,
                            *, from_state: str = "QUARANTINED"
                            ) -> None:
        """Move the durable deployment intent off a quarantined
        candidate: generation-checked reconcile to the last committed
        predecessor *that is still resident* (RESTORED), else durably
        UNAVAILABLE — a pointer never routes to an unloaded model and
        durable state is never silently emptied while a model serves
        from memory."""
        fallback = self._last_completed_excluding(
            activation_id, require_live=True)
        dep = self.store.deployment()
        gen = int(dep["deployment_generation"]) if dep is not None else 0
        try:
            if fallback is not None:
                fb = self._activations.get(fallback)
                fb_state = self._durable_state(fallback, "ACTIVE")
                rec = self.store.reconcile_deployment(
                    expected_generation=gen, desired_id=fallback,
                    phase="RESTORED", at=self._at(),
                    event_activation_id=fallback,
                    event_type="deployment_restored",
                    event_from_state=fb_state, event_to_state=fb_state,
                    because=f"quarantine {activation_id}: {reason}",
                    artifact_root_digest=(
                        digest_root(fb.snapshot)
                        if fb is not None and fb.snapshot else ""),
                    backend_id=(fb.grant.backend_id
                                if fb is not None and fb.grant else ""),
                    signer=self.signer)
                self._active_id = fallback
                self._serving_state = ServingState.SERVING
                if self.router is not None and fallback in \
                        self._live_handles:
                    rb, rh = self._live_handles[fallback]
                    try:
                        self.router.publish_route(
                            fallback,
                            generation=int(rec["generation"]),
                            at=self._at())
                    except Exception:  # noqa: BLE001
                        self.router.activate(fallback, rb, rh,
                                             at=self._at())
            else:
                self.store.reconcile_deployment(
                    expected_generation=gen, desired_id="",
                    phase="UNAVAILABLE", at=self._at(),
                    event_activation_id=activation_id,
                    event_type="deployment_unavailable",
                    event_from_state=from_state,
                    event_to_state=from_state,
                    because=f"quarantine {activation_id}: {reason}",
                    signer=self.signer)
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
            self.store.verify_admin_chain(self.registry, now=self._now)
            events = verify_event_log(self.store, self.registry,
                                      now=self._now)
            pointer = self.store.read_pointer()
            dep = self.store.deployment()
            if dep is not None:
                desired = str(dep["desired_activation_id"])
                pointed = str((pointer or {}).get("activation_id")
                              or "")
                if desired != pointed:
                    raise AuthorityStoreError(
                        "deployment intent names "
                        f"{desired!r} but the serving pointer names "
                        f"{pointed!r} — the durable record is "
                        "inconsistent and cannot be trusted")
            by_id: dict[str, list] = {}
            for e in events:
                by_id.setdefault(e["activation_id"], []).append(e)

            def last_state(aid: str) -> str:
                return by_id[aid][-1]["to_state"] if by_id.get(aid) else ""

            for aid in list(by_id):
                if aid in ("__migration__", "__deployment__"):
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

    def declare_unavailable(self, *, reason: str) -> None:
        """Recovery determined no eligible model can be restored:
        reconcile the durable intent to a durably UNAVAILABLE state at
        a new generation — never leave a phantom active pointer."""
        with self._lock:
            dep = self.store.deployment()
            gen = int(dep["deployment_generation"]) \
                if dep is not None else 0
            failed = str(dep["desired_activation_id"]) \
                if dep is not None else ""
            durable = self._durable_state(failed) if failed else ""
            to = durable if durable in ("ABORTED", "QUARANTINED") \
                else "QUARANTINED"
            try:
                self.store.reconcile_deployment(
                    expected_generation=gen, desired_id="",
                    phase="UNAVAILABLE", at=self._at(),
                    event_activation_id=failed or "__deployment__",
                    event_type="deployment_unavailable",
                    event_from_state=durable or "",
                    event_to_state=to,
                    because=reason, signer=self.signer)
            except AuthorityStoreError:
                pass
            self._active_id = None
            self._pending_restore_id = None
            self._serving_state = ServingState.UNAVAILABLE


# Backward-compatible alias: tests and callers imported
# recover_from_journal in v16.4.2.
ServingSupervisor.recover_from_journal = ServingSupervisor.recover


def digest_root(snapshot: MeasuredSnapshot) -> str:
    """The digest that identifies one measured artifact set — what a
    grant's artifact_root_digest binds."""
    from egai.common.canonical import digest
    return digest({name: d for name, d in snapshot.artifact_digests})
