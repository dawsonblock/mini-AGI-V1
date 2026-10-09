"""v16.4.2 TrustedRuntimeLauncher — supervised transactional launch.

v16.4.1 made admission mandatory but kept a non-transactional load:
`backend.load()` ran before activation evidence was durable, and a
persistence failure after load raised an error while leaving the
model resident. v16.4.2 (UPGRADE_PLAN §3) makes the launch a
supervised transaction:

  1. resolves the exact signed campaign plan, qualification record,
     promotion decision, and the runtime manifest the decision
     authorizes for this seed;
  2. verifies every signature, role, validity window, expiry, and the
     revocation *snapshot* (signed, monotonic, freshness-and-future
     bounded — unsigned revocation evidence is refused);
  3. physically measures the model, tokenizer, and adapter bytes —
     caller-supplied hashes are never trusted;
  4. the `admission` authority issues a short-lived `AdmissionGrantV1`
     binding the measured artifact set, the backend (implementation +
     policy epoch when a backend manifest is configured), and the
     operative revocation epoch;
  5. the `ServingSupervisor` drives REQUESTED → AUTHORIZED → STAGED →
     PREPARED → READY → COMMITTED → ACTIVE over the transactional
     authority store — grant reservation is durable and atomic, and the
     commit intent and pointer swap share one transaction;
  6. activation ids, snapshot destinations, and receipt paths are
     generated inside the trusted launcher — campaign/seed are request
     metadata, never path components; clients receive an artifact id,
     not a writable location (SEC-204);
  7. the production activation receipt binds the decision,
     qualification, manifest, candidate, the digests measured at load,
     the backend, and a replay nonce — ledgered before any traffic.

Any failure after PREPARED aborts the activation and unloads the
backend — a failed launch cannot leave a loaded, unevidenced model
behind. The only supported serving entry point is a `ServingBackend`
whose `load()` accepts a `MeasuredSnapshot`, and a `MeasuredSnapshot`
carries no deployment authority by itself: the supervisor re-verifies
the grant before anything is staged.
"""
from __future__ import annotations

import json
import secrets
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from egai.common.canonical import validate_digest
from egai.common.crypto import Ed25519Signer

from minagi.runtime.access_policy import contained_child
from minagi.runtime.authority_store import AuthorityStore
from minagi.runtime.supervisor import (ActivationError,
                                       ActivationRefused,
                                       ServingSupervisor)
from minagi.security.admission_grants import issue_grant
from minagi.security.signed_revocations import (RevocationRefused,
                                              RevocationSnapshotV2,
                                              RevocationStore)
from minagi.security.signed_revocations import (
    verify_snapshot as verify_revocation_snapshot)

from .artifact_closure import (ArtifactClosureError, expected_from_manifest,
                               tokenizer_artifact_digest, verify_entries)
from .authority import AuthorityLedger, as_utc
from .immutable_snapshot import (MeasuredSnapshot, SnapshotError,
                                 stage_snapshot, verify_snapshot)
from .runtime_admission import (AdmissionRefused, ActivationReceipt,
                                RevocationList, RuntimeAdmissionController,
                                write_activation_receipt)


class LaunchRefused(PermissionError):
    """The trusted launcher refused to start the serving backend."""


class LaunchReplay(RuntimeError):
    """An idempotent request id that already executed — carries the
    recorded outcome instead of repeating the activation."""

    def __init__(self, prior: dict):
        super().__init__(
            f"request already recorded: {prior.get('outcome')}")
        self.prior = dict(prior)


class ServingBackend(Protocol):
    backend_id: str

    def load(self, snapshot: MeasuredSnapshot):  # pragma: no cover
        ...

    def health_probe(self, handle) -> None:  # pragma: no cover
        ...

    def unload(self, handle) -> None:  # pragma: no cover
        ...


@dataclass(frozen=True)
class LaunchRequest:
    campaign_id: str
    seed: str
    decision_doc: dict
    qualification_doc: dict
    plan_doc: dict
    runtime_manifest: dict
    adapter_dir: str
    model_path: str
    tokenizer_path: str = ""         # optional cross-check root; the
                                     # model snapshot carries the
                                     # tokenizer the plan binds
    expected_backend: str = "hf-peft"
    candidate_digest: str = ""
    rollback_of: str = ""            # decision digest being rolled back from

    def __post_init__(self):
        if not self.campaign_id or not self.seed:
            raise ValueError("launch request requires campaign_id and seed")
        if self.candidate_digest:
            validate_digest(self.candidate_digest)
        if self.rollback_of:
            validate_digest(self.rollback_of)


@dataclass(frozen=True)
class LaunchResult:
    snapshot: MeasuredSnapshot
    receipt: ActivationReceipt
    receipt_doc: dict
    backend_id: str
    loaded: object
    activation_id: str = ""
    grant_digest: str = ""
    receipt_path: str = ""


class TrustedRuntimeLauncher:
    """The supervised model-loading path: admission is mandatory and
    issuance of a short-lived grant precedes any staging; the
    supervisor owns the transactional lifecycle, and the activation
    receipt is emitted only after the commit is durable."""

    def __init__(self, registry, *, runtime_signer: Ed25519Signer,
                 admission_signer: Ed25519Signer,
                 snapshot_root, nonce_journal=None,
                 receipts_dir=None,
                 revocation_store: RevocationStore | None = None,
                 revocation_snapshot=None,
                 supervisor: ServingSupervisor | None = None,
                 authority_store: AuthorityStore | None = None,
                 runtime_identity: str = "local-supervisor",
                 policy_epoch: int = 0,
                 backend_manifest_doc=None,
                 max_revocation_age_seconds: int = 7 * 86400,
                 max_clock_skew_seconds: int = 300,
                 ledger_path=None, now: datetime | None = None):
        if revocation_store is None and revocation_snapshot is None:
            raise LaunchRefused(
                "authenticated revocation evidence is required — the "
                "launcher refuses to admit without a signed revocation "
                "snapshot (v16.4.2)")
        if runtime_signer is None:
            raise LaunchRefused(
                "a protected runtime signing identity is required — only "
                "a signed receipt may authorize serving")
        if admission_signer is None:
            raise LaunchRefused(
                "a protected admission signing identity is required — "
                "activation requires an AdmissionGrantV1")
        self.registry = registry
        self.revocation_store = revocation_store
        self.revocation_snapshot = revocation_snapshot
        self.max_revocation_age_seconds = int(max_revocation_age_seconds)
        self.max_clock_skew_seconds = int(max_clock_skew_seconds)
        self.runtime_signer = runtime_signer
        self.admission_signer = admission_signer
        self.snapshot_root = Path(snapshot_root)
        self.receipts_dir = Path(receipts_dir) if receipts_dir else \
            self.snapshot_root.parent / "receipts"
        self.nonce_journal = Path(nonce_journal) if nonce_journal else \
            self.receipts_dir / "activation_nonces.jsonl"
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.policy_epoch = int(policy_epoch)
        self.backend_manifest_doc = backend_manifest_doc
        self._now = now
        self.supervisor = supervisor if supervisor is not None else \
            ServingSupervisor(
                authority_store or
                AuthorityStore(self.snapshot_root.parent / "state"
                               / "authority.sqlite"),
                runtime_signer=runtime_signer, registry=registry,
                runtime_identity=runtime_identity, now=now)

    def _at(self) -> datetime:
        return as_utc(self._now)

    def _operative_revocations(self) -> RevocationSnapshotV2:
        """The newest valid authorized revocation snapshot — verified
        at admission time, never cached across launches."""
        try:
            if self.revocation_snapshot is not None:
                return verify_revocation_snapshot(
                    self.revocation_snapshot, self.registry,
                    now=self._now,
                    max_age_seconds=self.max_revocation_age_seconds,
                    max_clock_skew_seconds=self.max_clock_skew_seconds)
            return self.revocation_store.latest_valid(
                self.registry, now=self._now,
                max_age_seconds=self.max_revocation_age_seconds,
                max_clock_skew_seconds=self.max_clock_skew_seconds)
        except RevocationRefused as exc:
            raise LaunchRefused(
                f"revocation evidence refused: {exc}") from exc

    def launch(self, request: LaunchRequest, backend: ServingBackend,
               *, request_id: str | None = None) -> LaunchResult:
        at = self._at()
        if getattr(backend, "backend_id", None) != request.expected_backend:
            raise LaunchRefused(
                f"backend {getattr(backend, 'backend_id', None)!r} does "
                f"not match the requested backend "
                f"{request.expected_backend!r}")
        if request_id:
            prior = self.supervisor.store.lookup_request(request_id)
            if prior is not None:
                raise LaunchReplay(prior)

        from . import strict_schema
        try:
            strict_schema.validate("runtime_manifest",
                                   request.runtime_manifest,
                                   require_production=True)
        except strict_schema.SchemaRefused as exc:
            raise LaunchRefused(
                f"runtime manifest refused for serving: {exc}") from exc

        snapshot = self._operative_revocations()
        controller = RuntimeAdmissionController(
            self.registry,
            revocation_snapshot=snapshot,
            max_revocation_age_seconds=self.max_revocation_age_seconds,
            now=self._now)
        try:
            admission = controller.admit(
                decision_doc=request.decision_doc,
                qualification_doc=request.qualification_doc,
                plan_doc=request.plan_doc,
                runtime_manifest=request.runtime_manifest,
                adapter_dir=request.adapter_dir, seed=request.seed,
                runtime_model_path=request.model_path,
                runtime_tokenizer_path=request.tokenizer_path or None,
                expected_backend=request.expected_backend,
                rollback_of=request.rollback_of)
        except AdmissionRefused as exc:
            raise LaunchRefused(f"admission refused: {exc}") from exc

        manifest = request.runtime_manifest
        # The trusted side generates the activation id — campaign/seed
        # are metadata, never path components (SEC-204).
        act = self.supervisor.request()
        activation_id = act.activation_id
        dest = contained_child(self.snapshot_root, activation_id,
                               what="snapshot destination")
        staged: MeasuredSnapshot | None = None
        grant_doc: dict = {}
        try:
            # AUTHORIZED — the grant binds exactly what will be staged:
            # the measured model+adapter set on exactly this backend.
            artifact_root = self._artifact_set_digest(manifest)
            grant_doc = issue_grant(
                self.admission_signer,
                decision_digest=admission.decision_digest,
                qualification_digest=admission.qualification_record_digest,
                runtime_manifest_digest=str(manifest["digest"]),
                artifact_root_digest=artifact_root,
                backend_id=request.expected_backend,
                backend_binary_digest=(
                    str(self.backend_manifest_doc["digest"])
                    if self.backend_manifest_doc is not None else ""),
                policy_epoch=self.policy_epoch,
                audience_runtime_identity=self.supervisor.runtime_identity,
                now=self._now,
                revocation_epoch=snapshot.epoch)
            self.supervisor.authorize(activation_id, grant_doc)

            # STAGED — physical copy-and-verify under the destination
            try:
                staged = stage_snapshot(
                    dest,
                    {"model": request.model_path,
                     "adapter": request.adapter_dir},
                    expected_digests={
                        "model": str(manifest["model_digest"]),
                        "adapter": str(manifest["adapter_digest"])},
                    resolve_symlinks={"model": True, "adapter": False},
                    manifest_digest=str(manifest["digest"]))
                verify_snapshot(staged)
                # The plan binds the tokenizer as the tokenizer-named
                # files inside the model snapshot; re-measure from the
                # staged model bytes.
                tokenizer_digest = tokenizer_artifact_digest(
                    staged.path("model"), resolve_symlinks=False)
                if tokenizer_digest != str(manifest["tokenizer_digest"]):
                    raise ArtifactClosureError(
                        "staged model's tokenizer artifact digest "
                        f"{tokenizer_digest} != authorized "
                        f"{manifest['tokenizer_digest']}")
                listed = manifest.get("adapter_files")
                if listed is not None:
                    verify_entries(staged.closure("adapter").entries,
                                   expected_from_manifest(listed),
                                   what="staged adapter")
            except SnapshotError as exc:
                _discard(dest)
                raise LaunchRefused(f"snapshot refused: {exc}") from exc
            except Exception as exc:  # noqa: BLE001 - fail closed
                _discard(dest)
                raise LaunchRefused(f"snapshot refused: {exc}") from exc
            self.supervisor.stage(activation_id, staged)

            # PREPARED -> READY -> COMMITTED -> ACTIVE
            self.supervisor.prepare(activation_id, backend)
            self.supervisor.health_check(activation_id)
            self.supervisor.commit_activation(activation_id)
        except (LaunchRefused, ActivationRefused, ActivationError) as exc:
            # the supervisor has already aborted + unloaded anything
            # post-request; surface the refusal
            if isinstance(exc, LaunchRefused):
                raise
            raise LaunchRefused(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - fail closed
            self.supervisor.abort(activation_id,
                                  reason=f"launch error: {exc}")
            _discard(dest)
            raise LaunchRefused(f"launch failed: {exc}") from exc

        nonce = self._reserve_nonce()
        receipt = ActivationReceipt(
            decision_digest=admission.decision_digest,
            qualification_record_digest=admission.qualification_record_digest,
            runtime_manifest_digest=admission.runtime_manifest_digest,
            adapter_digest=admission.adapter_digest,
            backend=request.expected_backend,
            admitted_at=at.isoformat(),
            candidate_digest=(request.candidate_digest
                              or admission.candidate_digest),
            # model + adapter bytes from the staged snapshot, plus the
            # tokenizer digest re-measured from the staged model at load
            loaded_artifact_digests=tuple(staged.artifact_digests)
            + (("tokenizer", tokenizer_digest),),
            activation_nonce=nonce,
            rollback_of=admission.rollback_of)
        # The completion evidence is fail-closed: ledger first, then the
        # service-owned receipt file. If persistence fails after the
        # commit, the candidate is quarantined — nothing unevidenced
        # stays active. Receipt destinations are chosen by the trusted
        # launcher — clients receive the artifact id, not a path (SEC-204).
        receipt_path = contained_child(
            self.receipts_dir, activation_id,
            what="receipt destination").with_suffix(".json")
        try:
            if self.ledger_path is not None:
                AuthorityLedger(self.ledger_path).append(
                    self.runtime_signer, "activation_receipt",
                    receipt.to_doc()["value"])
            doc = receipt.to_doc(signer=self.runtime_signer)
            self.receipts_dir.mkdir(parents=True, exist_ok=True)
            write_activation_receipt(receipt, receipt_path,
                                     signer=self.runtime_signer)
            if request_id:
                self.supervisor.store.record_request(
                    request_id=request_id, activation_id=activation_id,
                    outcome="committed",
                    outcome_digest=str(doc["digest"]), at=int(
                        self._at().timestamp()))
        except Exception as exc:  # noqa: BLE001 - no evidence, no serving
            self.supervisor.quarantine_active(
                reason=f"activation evidence could not be recorded: {exc}")
            raise LaunchRefused(
                f"activation evidence could not be recorded — the "
                f"candidate was quarantined rather than left serving: "
                f"{exc}") from exc
        return LaunchResult(snapshot=staged, receipt=receipt,
                            receipt_doc=doc,
                            backend_id=request.expected_backend,
                            loaded=self.supervisor._live_handles.get(
                                activation_id, (None, None))[1],
                            activation_id=activation_id,
                            grant_digest=grant_doc["digest"],
                            receipt_path=str(receipt_path))

    # --- helpers ----------------------------------------------------
    @staticmethod
    def _artifact_set_digest(manifest: dict) -> str:
        """The digest a grant's artifact_root_digest binds: the
        authorized digests of exactly the artifact set the supervisor
        will stage (model + adapter; the tokenizer is measured inside
        the staged model)."""
        from egai.common.canonical import digest
        return digest({"adapter": str(manifest["adapter_digest"]),
                       "model": str(manifest["model_digest"])})

    # --- replay protection ------------------------------------------
    def _reserve_nonce(self) -> str:
        used = self._used_nonces()
        for _ in range(8):
            nonce = secrets.token_hex(16)
            if nonce not in used:
                self.nonce_journal.parent.mkdir(parents=True,
                                                exist_ok=True)
                entry = {"nonce": nonce, "at": self._at().isoformat(),
                         "receipt_schema": "mini-agi-v16.4.1-activation-"
                                           "receipt-v2"}
                with self.nonce_journal.open("a") as f:
                    f.write(json.dumps(entry, sort_keys=True) + "\n")
                return nonce
        raise LaunchRefused("could not allocate a fresh activation nonce")

    def _used_nonces(self) -> set[str]:
        if not self.nonce_journal.is_file():
            return set()
        used = set()
        for line in self.nonce_journal.read_text().splitlines():
            if line.strip():
                try:
                    used.add(str(json.loads(line)["nonce"]))
                except (ValueError, KeyError, TypeError):
                    continue
        return used


def load_revocation_list(path, *, max_age_seconds: int | None = None,
                         now: datetime | None = None) -> RevocationList:
    """Load a legacy unsigned revocation list (research-side checks
    only — production activation requires a signed
    RevocationSnapshotV2)."""
    rl = RevocationList.load(path)
    if max_age_seconds is not None:
        at = as_utc(now)
        age = int(at.timestamp()) - int(rl.generated_at)
        if age > int(max_age_seconds):
            raise LaunchRefused(
                f"revocation list is stale ({age}s > {max_age_seconds}s)")
    return rl


def _discard(path: Path) -> None:
    if not path.exists():
        return
    try:
        for p in sorted(path.rglob("*"), reverse=True):
            try:
                p.chmod(0o700)
            except OSError:
                pass
        path.chmod(0o700)
        shutil.rmtree(path)
    except OSError:
        pass
