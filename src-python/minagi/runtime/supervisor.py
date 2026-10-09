"""v16.4.2 serving supervisor (UPGRADE_PLAN §3.5/§3.6).

The supervisor owns every production model load. It enforces the
activation state machine, journals every transition before it takes
effect, keeps the previous healthy model until a candidate's commit is
durable, and reconciles the journal against the active-version pointer
on recovery.

Ordering guarantees:

  * the activation-intent record is durable BEFORE the traffic pointer
    moves — an intent record is never labelled proof of completed
    activation;
  * the pointer swap is atomic (tmp + fsync + os.replace) — a crash
    mid-commit yields the old pointer or the new one, never a torn one;
  * the signed activation-completion record is written after the swap
    and describes what actually happened;
  * any failure between load and commit unloads the candidate and
    discards its snapshot — nothing unevidenced can serve;
  * the previous healthy model stays resident until the new candidate
    is COMMITTED+ACTIVE, so a failed activation never strands traffic.

The supervisor deliberately does not read caller paths directly: it
verifies the `admission` grant itself, and the backend sees only the
measured snapshot produced by `stage_snapshot`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from egai.common.crypto import Ed25519Signer

from minagi.security.admission_grants import (GrantRefused,
                                              verify_grant)
from minagi.v161.authority import as_utc
from minagi.v161.immutable_snapshot import (MeasuredSnapshot,
                                            verify_snapshot)

from .activation_state import (ActivationState, check_transition)
from .durable_journal import DurableJournal, JournalError


class SupervisedBackend(Protocol):
    """The contract a serving backend must satisfy to run under the
    supervisor: load into a non-serving handle, prove health, and be
    unloadable on demand."""
    backend_id: str

    def load(self, snapshot: MeasuredSnapshot):  # pragma: no cover
        ...

    def health_probe(self, handle) -> None:  # pragma: no cover
        """Raise unless the loaded model answers the probe."""
        ...

    def unload(self, handle) -> None:  # pragma: no cover
        """Release the loaded model. Must tolerate being called twice."""
        ...


class ActivationRefused(PermissionError):
    """The supervisor refused to move a candidate forward."""


class ActivationError(RuntimeError):
    """A supervised transition failed after authorization."""


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

    def __init__(self, journal_dir, *, runtime_signer: Ed25519Signer,
                 registry, runtime_identity: str = "local-supervisor",
                 now: datetime | None = None):
        if runtime_signer is None:
            raise ActivationRefused(
                "the supervisor requires a protected runtime signing "
                "identity — unsigned completion records are not evidence")
        self.journal = DurableJournal(journal_dir)
        self.signer = runtime_signer
        self.registry = registry
        self.runtime_identity = str(runtime_identity)
        self._now = now
        self._activations: dict[str, _Activation] = {}
        self._live_handles: dict[str, object] = {}
        self._active_id: str | None = None
        self._retained: str | None = None
        self._consumed_grants: set[str] = set()
        pointer = self.journal.read_pointer() if \
            self.journal.pointer_path.is_file() else None
        if pointer is not None:
            self._active_id = str(pointer["activation_id"])

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

    def _transition(self, act: _Activation, dst: ActivationState,
                    *, detail: dict | None = None,
                    sign: bool = False) -> None:
        check_transition(act.state, dst)
        self.journal.append(
            activation_id=act.activation_id,
            from_state=act.state.value, to_state=dst.value,
            at=self._at(), detail=detail,
            signer=self.signer if sign else None)
        act.state = dst

    @property
    def active_id(self) -> str | None:
        return self._active_id

    def active_pointer(self) -> dict | None:
        return self.journal.read_pointer() if \
            self.journal.pointer_path.is_file() else None

    # --- the lifecycle ----------------------------------------------
    def request(self, activation_id: str) -> _Activation:
        """REQUESTED — register a candidate. Activation ids are
        write-once so a journal replay cannot alias two candidates."""
        if not activation_id:
            raise ActivationRefused("activation_id required")
        if activation_id in self._activations:
            raise ActivationRefused(
                f"activation id {activation_id!r} already exists")
        act = _Activation(activation_id=activation_id,
                          state=ActivationState.REQUESTED)
        self.journal.append(activation_id=activation_id,
                            from_state="", to_state="REQUESTED",
                            at=self._at())
        self._activations[activation_id] = act
        return act

    def authorize(self, activation_id: str, grant_doc) -> _Activation:
        """AUTHORIZED — the supervisor verifies the grant itself; a
        caller's claim of authorization is not evidence."""
        act = self._get(activation_id)
        try:
            grant = verify_grant(
                grant_doc, self.registry, now=self._now,
                audience_runtime_identity=self.runtime_identity)
        except GrantRefused as exc:
            self.abort(activation_id, reason=f"grant refused: {exc}")
            raise ActivationRefused(f"grant refused: {exc}") from exc
        if grant.grant_id in self._consumed_grants:
            self.abort(activation_id, reason="grant replay")
            raise ActivationRefused(
                "grant id already consumed — a grant authorizes exactly "
                "one activation")
        act.grant = grant
        self._transition(act, ActivationState.AUTHORIZED,
                         detail={"grant_id": grant.grant_id,
                                 "manifest": grant.runtime_manifest_digest,
                                 "backend": grant.backend_id})
        self._consumed_grants.add(grant.grant_id)
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
        """PREPARED — the backend loads into a non-serving handle. A
        load failure aborts: nothing half-loaded can proceed."""
        act = self._get(activation_id)
        if act.snapshot is None:
            raise ActivationRefused("prepare requires a staged snapshot")
        if getattr(backend, "backend_id", None) != (
                act.grant.backend_id if act.grant else None):
            self.abort(activation_id, reason="backend != grant backend")
            raise ActivationRefused(
                f"backend {getattr(backend, 'backend_id', None)!r} does "
                "not match the grant's authorized backend")
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
        """COMMITTED -> ACTIVE. The intent record is durable BEFORE the
        pointer moves; the pointer swap is atomic; the signed completion
        record lands after. A completion-write failure rolls the pointer
        back — nothing unevidenced stays routable."""
        act = self._get(activation_id)
        if expected_previous is not None \
                and self._active_id != expected_previous:
            self.abort(activation_id,
                       reason="expected_previous mismatch")
            raise ActivationRefused(
                f"expected previous activation {expected_previous!r} but "
                f"{self._active_id!r} is active — refusing to swap onto "
                "an unknown base")

        # 1. durable intent — the transition does not exist until the
        #    journal says it does
        self._transition(act, ActivationState.COMMITTED,
                         detail={"kind": "activation_intent"})

        # 2. atomic pointer swap (traffic routing)
        pointer = {"activation_id": activation_id,
                   "artifact_root_digest":
                       digest_root(act.snapshot) if act.snapshot else "",
                   "backend_id": act.grant.backend_id if act.grant else "",
                   "committed_at": self._at()}
        try:
            self.journal.write_pointer(pointer)
        except JournalError as exc:
            self.abort(activation_id,
                       reason=f"pointer write failed: {exc}")
            raise ActivationError(
                f"active-version pointer could not be committed: {exc}") \
                from exc

        previous = self._active_id
        self._active_id = activation_id
        self._transition(act, ActivationState.ACTIVE,
                         detail={"pointer": "swapped",
                                 "previous": previous or ""})

        # 3. signed completion record — evidence of what happened. If
        #    it cannot be written, the activation is rolled back rather
        #    than left active without evidence.
        try:
            self.journal.append(
                activation_id=activation_id, from_state="COMMITTED",
                to_state="ACTIVE", at=self._at(),
                detail={"kind": "activation_completion",
                        "grant_id": (act.grant.grant_id
                                     if act.grant else ""),
                        "artifact_root_digest": pointer[
                            "artifact_root_digest"],
                        "backend_id": pointer["backend_id"],
                        "previous": previous or ""},
                signer=self.signer)
        except JournalError as exc:
            self._active_id = previous
            try:
                if previous is not None:
                    prev_act = self._activations.get(previous)
                    prev_pointer = {"activation_id": previous,
                                    "restored_at": self._at()}
                    if prev_act is not None:
                        prev_pointer["artifact_root_digest"] = \
                            digest_root(prev_act.snapshot) \
                            if prev_act.snapshot else ""
                    self.journal.write_pointer(prev_pointer)
                else:
                    self.journal.write_pointer(
                        {"activation_id": "", "cleared_at": self._at()})
            except JournalError:
                pass  # recovery reconciles whatever landed
            self.abort(activation_id,
                       reason=f"completion record failed: {exc}")
            raise ActivationError(
                "the signed activation-completion record could not be "
                "persisted — the candidate was rolled back rather than "
                "left serving without evidence") from exc

        # The previous version stays resident as the protected
        # deployment target (spec: "maintain the last independently
        # qualified version"). At most one predecessor is retained —
        # deeper history re-admits through the normal path.
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
                    self.journal.append(
                        activation_id=old,
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
        if self._active_id == activation_id:
            # an active candidate being aborted is quarantined — the
            # pointer must not keep routing to a dead model
            self._quarantine_pointer(activation_id, reason)
            self._transition(act, ActivationState.QUARANTINED,
                             detail={"reason": reason}, sign=True)
            return
        self._transition(act, ActivationState.ABORTED,
                         detail={"reason": reason})

    def quarantine_active(self, *, reason: str) -> None:
        """Withdraw the currently active version (e.g. its promotion
        was just revoked) and restore the last committed predecessor
        if one is still held."""
        if self._active_id is None:
            return
        self.abort(self._active_id, reason=reason or "quarantined")

    def rollback(self, activation_id: str | None = None) -> str | None:
        """Restore the retained committed predecessor. The previous
        healthy model is kept resident as the protected deployment
        target, so a pointer swap restores a live backend without
        re-loading. Older history re-admits through admission."""
        target = activation_id
        if target is None:
            target = self._retained or self._last_completed_excluding(
                self._active_id, require_live=True)
        if target is None or target not in self._live_handles:
            raise ActivationRefused(
                "no retained committed version to roll back to — a "
                "rollback must re-admit artifacts through admission, "
                "not improvise")
        prev_pointer = {"activation_id": target,
                        "rolled_back_at": self._at(),
                        "from": self._active_id or ""}
        self.journal.write_pointer(prev_pointer)
        old = self._active_id
        self._active_id = target
        self.journal.append(activation_id=target,
                            from_state="ABORTED", to_state="ACTIVE",
                            at=self._at(),
                            detail={"kind": "rollback",
                                    "from": old or ""},
                            signer=self.signer)
        return target

    def _last_completed_excluding(self, exclude: str | None,
                                  *, require_live: bool = False
                                  ) -> str | None:
        """Newest activation that durably completed a commit other than
        `exclude`. With require_live, only a candidate whose model is
        still resident counts — a pointer must never route to a model
        that is not loaded."""
        completed: list[str] = []
        for r in self.journal.records():
            if r.to_state == "ACTIVE" and r.activation_id != exclude \
                    and r.detail.get("kind") == "activation_completion":
                completed.append(r.activation_id)
        for aid in reversed(completed):
            if not require_live or aid in self._live_handles:
                return aid
        return None

    def _quarantine_pointer(self, activation_id: str, reason: str) -> None:
        """Move the active pointer off a quarantined candidate: restore
        the last committed predecessor *that is still resident*, else
        clear it — a pointer never routes to an unloaded model."""
        fallback = self._last_completed_excluding(
            activation_id, require_live=True)
        try:
            if fallback is not None:
                self.journal.write_pointer(
                    {"activation_id": fallback,
                     "restored_at": self._at(),
                     "because": f"quarantine {activation_id}: {reason}"})
                self._active_id = fallback
            else:
                self.journal.write_pointer(
                    {"activation_id": "", "cleared_at": self._at(),
                     "because": f"quarantine {activation_id}: {reason}"})
                self._active_id = None
        except JournalError:
            self._active_id = None

    # --- crash recovery ----------------------------------------------
    def recover_from_journal(self) -> dict:
        """Reconcile the durable journal with the active pointer after
        a crash. Deterministic rules:

          * pointer names an activation whose journal shows COMMITTED
            intent but no completion → it DID go live; write the
            completion record marked reconciled (evidence gap closed);
          * an activation with COMMITTED intent but the pointer names
            someone else → it never routed; abort it;
          * an activation stalled before COMMITTED → abort it (it could
            not have been routed);
          * a pointer naming an activation with no intent record at all
            → corrupt; clear the pointer to the last completed
            activation (or empty) and refuse to serve the phantom.
        """
        report = {"reconciled_completions": [], "aborted": [],
                  "cleared_pointer": False, "active": None}
        pointer = self.journal.read_pointer() if \
            self.journal.pointer_path.is_file() else None
        records = self.journal.records()
        by_id: dict[str, list] = {}
        for r in records:
            by_id.setdefault(r.activation_id, []).append(r)

        def last_state(aid: str) -> str:
            return by_id[aid][-1].to_state if by_id.get(aid) else ""

        def has(aid: str, state: str, kind: str = "") -> bool:
            return any(r.to_state == state and
                       (not kind or r.detail.get("kind") == kind)
                       for r in by_id.get(aid, ()))

        for aid in list(by_id):
            state = last_state(aid)
            if state in ("ABORTED", "QUARANTINED", "ACTIVE"):
                continue
            if state == "COMMITTED":
                if pointer and pointer.get("activation_id") == aid:
                    # committed AND routed — complete the record
                    self.journal.append(
                        activation_id=aid, from_state="ACTIVE",
                        to_state="ACTIVE", at=self._at(),
                        detail={"kind": "activation_completion",
                                "reconciled": True},
                        signer=self.signer)
                    report["reconciled_completions"].append(aid)
                else:
                    # intent durable but never routed
                    self.journal.append(
                        activation_id=aid, from_state="COMMITTED",
                        to_state="ABORTED", at=self._at(),
                        detail={"reason": "recovery: committed but "
                                          "never routed"})
                    report["aborted"].append(aid)
            else:
                self.journal.append(
                    activation_id=aid, from_state=state,
                    to_state="ABORTED", at=self._at(),
                    detail={"reason": "recovery: stalled before commit"})
                report["aborted"].append(aid)

        if pointer is not None:
            active = str(pointer.get("activation_id") or "")
            if active and active not in by_id:
                fallback = self._last_completed_excluding(active)
                self.journal.write_pointer(
                    {"activation_id": fallback or "",
                     "recovered_at": self._at(),
                     "because": "pointer named an activation with no "
                                "journal history"})
                self._active_id = fallback
                report["cleared_pointer"] = True
            else:
                self._active_id = active or None
        else:
            self._active_id = None
        report["active"] = self._active_id
        return report


def digest_root(snapshot: MeasuredSnapshot) -> str:
    """The digest that identifies one measured artifact set — what a
    grant's artifact_root_digest binds."""
    from egai.common.canonical import digest
    return digest({name: d for name, d in snapshot.artifact_digests})
