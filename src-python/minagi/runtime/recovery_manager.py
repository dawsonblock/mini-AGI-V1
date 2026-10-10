"""v16.4.4 automatic cold-start recovery (WP-F / SEC-307).

Durable intent is not live state. A committed pointer proves which
model SHOULD serve; only a verified live handle may actually serve.
`RecoveryManager` completes the restoration the v16.4.3 recovery
pass deliberately left open:

    start (traffic disabled)
      -> validate durable state (event chain + deployment record)
      -> find the last eligible committed deployment
      -> re-authorize under CURRENT policy + revocation evidence
         (a fresh grant — a consumed admission grant is never reused)
      -> re-measure the staged artifacts (exact bytes, not a name)
      -> load the model without accepting inference
      -> health + memory probes
      -> commit a fresh restoration transition
      -> publish the verified live route

Every step is idempotent: repeating recovery after an interrupted
restoration produces no conflicting generations — the deployment
generation CAS refuses stale transitions — and consumes no extra
grants beyond the fresh restoration grant each attempt requires.

If no eligible model can be restored, the correct result is
UNAVAILABLE — never a phantom "active" pointer.
"""
from __future__ import annotations

from pathlib import Path

from egai.common.canonical import digest

from minagi.security.admission_grants import issue_grant

from .access_policy import contained_child
from .authority_store import AuthorityStoreError
from .supervisor import ActivationError, ActivationRefused, ServingState


class RestorationRefused(PermissionError):
    """A committed deployment cannot be truthfully restored.
    ``permanent`` marks failures no retry can fix (missing or
    mismatched artifacts, unregistered backend) — vs transient
    failures a later recovery attempt may clear."""

    def __init__(self, *args, permanent: bool = False):
        super().__init__(*args)
        self.permanent = bool(permanent)


class RecoveryManager:
    """Cold-start restoration driver for the supervised runtime."""

    def __init__(self, supervisor, *, admission_signer, registry,
                 snapshot_root, backend_factories: dict,
                 revocation_store=None, now=None):
        if admission_signer is None:
            raise RestorationRefused(
                "recovery requires a protected admission signing "
                "identity — restoration grants cannot be unsigned")
        self.supervisor = supervisor
        self.admission_signer = admission_signer
        self.registry = registry
        self.snapshot_root = Path(snapshot_root)
        self.backend_factories = dict(backend_factories)
        self.revocation_store = revocation_store
        self._now = now

    # --- top level ---------------------------------------------------
    def restore(self) -> dict:
        """Reconcile durable state and restore a live serving model —
        or report UNAVAILABLE. Safe to call repeatedly: a supervisor
        that is already SERVING has real live state, so restoration
        is a no-op rather than a duplicate activation."""
        if self.supervisor.serving_state is ServingState.SERVING:
            return {"serving_state": ServingState.SERVING.value,
                    "requires_restoration": None,
                    "restoration": {"status": "already_serving",
                                    "restored_activation": None,
                                    "attempts": []}}
        report = self.supervisor.recover()
        report["restoration"] = {"status": "not_required",
                                 "restored_activation": None,
                                 "attempts": []}
        target = report.get("requires_restoration")
        if not target:
            return report
        pointer = self.supervisor.store.read_pointer() or {}
        candidates = [target] + self._fallback_candidates(target)
        errors: list[str] = []
        for candidate in candidates:
            try:
                restored = self._restore_one(
                    candidate, report=report, pointer=pointer,
                    is_desired=(candidate == target))
            except RestorationRefused as exc:
                errors.append(f"{candidate}: {exc}")
                report["restoration"]["attempts"].append(
                    {"candidate": candidate, "status": "refused",
                     "reason": str(exc),
                     "permanent": bool(exc.permanent)})
                continue
            report["restoration"].update(
                {"status": "restored",
                 "restored_activation": restored,
                 "restored_from": candidate})
            report["serving_state"] = \
                self.supervisor.serving_state.value
            return report
        report["restoration"]["status"] = "unavailable"
        report["restoration"]["errors"] = errors
        if errors and all(
                a.get("permanent") for a in
                report["restoration"]["attempts"]):
            # Every candidate failed permanently — the truthful
            # outcome is a durably UNAVAILABLE deployment, not a
            # pointer that names a model that will never exist.
            self.supervisor.declare_unavailable(
                reason="recovery: no eligible model could be "
                       "restored")
        report["serving_state"] = self.supervisor.serving_state.value
        return report

    # --- one candidate -------------------------------------------------
    def _restore_one(self, candidate: str, *, report: dict,
                     pointer: dict, is_desired: bool
                     ) -> str:
        """Drive a fresh supervised activation that re-admits the
        candidate's measured artifact set — but only after the COMPLETE
        original authority chain re-verifies under current authority
        and revocation evidence (SEC-403). A fresh grant is not a
        substitute for validating the original promotion."""
        from minagi.security.restoration_authority import (
            RestorationAuthorityRefused)
        src = self.snapshot_root / candidate
        if not src.is_dir():
            raise RestorationRefused(
                f"staged snapshot for {candidate} is absent — cannot "
                "restore a model whose artifacts do not exist",
                permanent=True)
        context = self._authorized_context(candidate)
        backend_id = self._authorized_backend(candidate)
        factory = self.backend_factories.get(backend_id)
        if factory is None:
            raise RestorationRefused(
                f"no registered backend factory for {backend_id!r}",
                permanent=True)
        verifier = self._verifier()
        try:
            snapshot = verifier.operative_snapshot()
        except RestorationAuthorityRefused as exc:
            raise RestorationRefused(
                f"current revocation evidence refused: {exc}",
                permanent=True) from exc
        # Re-measure every staged artifact — bytes, not names.
        artifacts, expected = self._remeasure(src)
        artifact_root = digest_root_of(expected)
        if is_desired:
            bound_root = str(pointer.get("artifact_root_digest") or "")
            if bound_root and bound_root != artifact_root:
                raise RestorationRefused(
                    "the staged artifacts do not match the durable "
                    "commit's artifact root — the record does not "
                    "describe these bytes", permanent=True)
        docs = {}
        try:
            docs = self.supervisor.store.authority_docs(candidate)
        except Exception:  # noqa: BLE001 - treated as absent below
            docs = {}
        backend_manifest_digest = ""
        if self.supervisor.backend_manifest_doc is not None:
            backend_manifest_digest = str(
                self.supervisor.backend_manifest_doc.get("digest") or "")
        try:
            auth = verifier.verify(
                candidate_id=candidate,
                authorized_context=context,
                authority_docs=docs,
                measured_digests=expected,
                artifact_root_digest=artifact_root,
                backend_id=backend_id,
                backend_manifest_digest=backend_manifest_digest,
                snapshot=snapshot)
        except RestorationAuthorityRefused as exc:
            raise RestorationRefused(
                f"historical authorization refused: {exc}",
                permanent=True) from exc
        act = self.supervisor.request()
        new_id = act.activation_id
        try:
            grant = issue_grant(
                self.admission_signer,
                decision_digest=auth.decision_digest,
                qualification_digest=auth.qualification_digest,
                runtime_manifest_digest=auth.runtime_manifest_digest,
                artifact_root_digest=auth.artifact_root_digest,
                backend_id=auth.backend_id,
                backend_binary_digest=auth.backend_binary_digest,
                policy_epoch=auth.policy_epoch,
                audience_runtime_identity=
                    self.supervisor.runtime_identity,
                now=self._now,
                revocation_epoch=auth.revocation_epoch)
            self.supervisor.authorize(new_id, grant,
                                      authority_docs=docs)
            staged = self._restage(
                new_id, src, expected,
                manifest_digest=auth.runtime_manifest_digest)
            self.supervisor.stage(new_id, staged)
            backend = factory()
            self.supervisor.prepare(new_id, backend)
            self.supervisor.health_check(new_id)
            self.supervisor.commit_activation(new_id)
        except (ActivationRefused, ActivationError,
                AuthorityStoreError) as exc:
            raise RestorationRefused(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - fail closed
            self.supervisor.abort(new_id, reason=f"restoration: {exc}")
            raise RestorationRefused(
                f"restoration failed: {exc}") from exc
        self.supervisor.mark_restored(new_id)
        try:
            self.supervisor.store.append_event(
                activation_id=new_id, event_type="restoration_completed",
                from_state="ACTIVE", to_state="ACTIVE",
                at=self.supervisor._at(),
                detail={"restored_from": candidate,
                        "decision_digest": auth.decision_digest,
                        "revocation_epoch": auth.revocation_epoch},
                signer=self.supervisor.signer)
        except AuthorityStoreError:
            pass
        return new_id

    # --- helpers ------------------------------------------------------
    def _fallback_candidates(self, exclude: str) -> list[str]:
        """Earlier completed activations whose staged artifacts still
        exist — a previous independently qualified model may restore
        under CURRENT authorization policy."""
        out: list[str] = []
        for e in self.supervisor.store.events():
            if e["event_type"] in ("activation_completion",
                                   "routing_observed",
                                   "rollback_completion") and \
                    e["activation_id"] != exclude and \
                    e["activation_id"] not in out and \
                    (self.snapshot_root / e["activation_id"]).is_dir():
                out.append(e["activation_id"])
        return list(reversed(out))

    def _authorized_context(self, activation_id: str) -> dict:
        for e in reversed(self.supervisor.store.events(activation_id)):
            if e["event_type"] == "authorized":
                return dict(e.get("detail") or {})
        return {}

    def _authorized_backend(self, activation_id: str) -> str:
        return str(self._authorized_context(activation_id)
                   .get("backend") or "hf-peft")

    def _verifier(self):
        """The restoration authority verifier, bound to current
        authority policy: the full original chain must re-verify, and
        the operative revocation snapshot is enforced — a grant is
        issued only against the verified bindings it returns."""
        from minagi.security.restoration_authority import (
            RestorationAuthorizationVerifier)
        return RestorationAuthorizationVerifier(
            self.registry, revocation_store=self.revocation_store,
            now=self._now,
            min_policy_epoch=getattr(self.supervisor,
                                     "min_policy_epoch", 0))

    def _remeasure(self, src: Path):
        from minagi.v161.artifact_closure import close_tree
        artifacts: dict[str, Path] = {}
        expected: dict[str, str] = {}
        for sub in sorted(src.iterdir()):
            if not sub.is_dir() or sub.name.startswith("."):
                continue
            artifacts[sub.name] = sub
            expected[sub.name] = close_tree(sub).digest
        if not artifacts:
            raise RestorationRefused(
                f"staged snapshot {src} contains no artifact trees",
                permanent=True)
        return artifacts, expected

    def _restage(self, new_id: str, src: Path, expected: dict,
                 *, manifest_digest: str):
        """Re-stage the measured artifacts into the new activation's
        snapshot destination — the frozen bytes are re-copied and
        re-verified, never trusted from the old location."""
        from minagi.v161.immutable_snapshot import stage_snapshot
        dest = contained_child(self.snapshot_root, new_id,
                               what="restoration snapshot")
        return stage_snapshot(
            dest, {name: src / name for name in expected},
            expected_digests=expected, manifest_digest=manifest_digest)


def digest_root_of(expected: dict) -> str:
    """The artifact-set digest a grant's artifact_root_digest binds —
    computed from measured per-artifact digests."""
    return digest({name: d for name, d in sorted(expected.items())})


__all__ = ["RecoveryManager", "RestorationRefused", "digest_root_of"]
