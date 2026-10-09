"""v16.4.1 TrustedRuntimeLauncher — mandatory runtime admission.

v16.4.0 shipped an admission *controller* but no enforcement point: a
serving process could load a model, tokenizer, and adapter directly
from mutable paths, and the audit could not establish that any
model-loading path passed through admission at all. The launcher is
that enforcement point. It:

  1. resolves the exact signed campaign plan, qualification record, and
     promotion decision (plus the runtime manifest the decision
     authorizes for this seed);
  2. verifies every signature, role, validity window, expiry, and the
     revocation list (with freshness);
  3. physically measures the model, tokenizer, and adapter bytes —
     caller-supplied hashes are never trusted, and a supplied hash that
     disagrees with the measured bytes is refused. The tokenizer
     identity is the plan's convention (the tokenizer-named files
     inside the model snapshot root), re-measured from the staged
     model — the tokenizer the backend opens is part of the verified
     model artifact, not a separately substitutable path;
  4. stages the approved artifacts into an immutable snapshot
     (single-pass copy-and-verify, then read-only), re-verifies it, and
     hands the backend ONLY that snapshot;
  5. starts the selected serving backend and, only after it loads
     successfully, emits a signed production activation receipt binding
     the decision, qualification, manifest, candidate, the artifact
     digests actually measured at load, the executing backend, and a
     replay nonce; the receipt is recorded in the authority ledger.

Any failure refuses the launch and no receipt exists. The only
supported serving entry point is a `ServingBackend` whose `load()`
accepts an `ApprovedSnapshot` — see `PeftServingBackend`, which
refuses raw paths.
"""
from __future__ import annotations

import json
import secrets
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from egai.common.canonical import validate_digest
from egai.common.crypto import Ed25519Signer

from .artifact_closure import (ArtifactClosureError, expected_from_manifest,
                               tokenizer_artifact_digest, verify_entries)
from .authority import AuthorityLedger, as_utc
from .immutable_snapshot import (ApprovedSnapshot, SnapshotError,
                                 stage_snapshot, verify_snapshot)
from .runtime_admission import (AdmissionRefused, ActivationReceipt,
                                RevocationList, RuntimeAdmissionController,
                                write_activation_receipt)


class LaunchRefused(PermissionError):
    """The trusted launcher refused to start the serving backend."""


class ServingBackend(Protocol):
    backend_id: str

    def load(self, snapshot: ApprovedSnapshot):  # pragma: no cover - protocol
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
    snapshot: ApprovedSnapshot
    receipt: ActivationReceipt
    receipt_doc: dict
    backend_id: str
    loaded: object


class TrustedRuntimeLauncher:
    """The only supported model-loading path: admission is mandatory,
    the backend sees only an approved immutable snapshot, and the
    activation receipt is emitted only after a successful load."""

    def __init__(self, registry, *, revocation_list: RevocationList,
                 runtime_signer: Ed25519Signer,
                 snapshot_root, nonce_journal,
                 max_revocation_age_seconds: int = 7 * 86400,
                 ledger_path=None, now: datetime | None = None):
        if not isinstance(revocation_list, RevocationList):
            raise LaunchRefused(
                "a revocation list is required — the launcher refuses to "
                "admit without revocation evidence")
        if runtime_signer is None:
            raise LaunchRefused(
                "a protected runtime signing identity is required — only a "
                "signed receipt may authorize serving")
        self.registry = registry
        self.revocation_list = revocation_list
        self.max_revocation_age_seconds = int(max_revocation_age_seconds)
        self.runtime_signer = runtime_signer
        self.snapshot_root = Path(snapshot_root)
        self.nonce_journal = Path(nonce_journal)
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self._now = now

    def _at(self) -> datetime:
        return as_utc(self._now)

    def launch(self, request: LaunchRequest, backend: ServingBackend,
               *, receipt_path=None) -> LaunchResult:
        at = self._at()
        if getattr(backend, "backend_id", None) != request.expected_backend:
            raise LaunchRefused(
                f"backend {getattr(backend, 'backend_id', None)!r} does not "
                f"match the requested backend {request.expected_backend!r}")

        from . import strict_schema
        try:
            strict_schema.validate("runtime_manifest",
                                   request.runtime_manifest,
                                   require_production=True)
        except strict_schema.SchemaRefused as exc:
            raise LaunchRefused(
                f"runtime manifest refused for serving: {exc}") from exc

        controller = RuntimeAdmissionController(
            self.registry, revocation_list=self.revocation_list,
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
        dest = self.snapshot_root / (
            f"{request.campaign_id}-{request.seed}-{uuid.uuid4().hex[:12]}")
        try:
            snapshot = stage_snapshot(
                dest,
                {"model": request.model_path,
                 "adapter": request.adapter_dir},
                expected_digests={
                    "model": str(manifest["model_digest"]),
                    "adapter": str(manifest["adapter_digest"])},
                resolve_symlinks={"model": True, "adapter": False},
                manifest_digest=str(manifest["digest"]))
            verify_snapshot(snapshot)
            # The plan binds the tokenizer as the tokenizer-named files
            # inside the model snapshot; re-measure that convention from
            # the staged model bytes. The tokenizer the backend serves
            # is part of the verified model artifact — there is no
            # separate tokenizer path in the served set.
            tokenizer_digest = tokenizer_artifact_digest(
                snapshot.path("model"), resolve_symlinks=False)
            if tokenizer_digest != str(manifest["tokenizer_digest"]):
                raise ArtifactClosureError(
                    "staged model's tokenizer artifact digest "
                    f"{tokenizer_digest} != authorized "
                    f"{manifest['tokenizer_digest']}")
            listed = manifest.get("adapter_files")
            if listed is not None:
                verify_entries(snapshot.closure("adapter").entries,
                               expected_from_manifest(listed),
                               what="staged adapter")
        except SnapshotError as exc:
            _discard(dest)
            raise LaunchRefused(f"snapshot refused: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - fail closed, never serve
            _discard(dest)
            raise LaunchRefused(f"snapshot refused: {exc}") from exc

        try:
            loaded = backend.load(snapshot)
        except Exception as exc:  # noqa: BLE001 - no receipt on failure
            _discard(dest)
            raise LaunchRefused(
                f"serving backend {request.expected_backend!r} failed to "
                f"load the approved snapshot: {exc}") from exc

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
            loaded_artifact_digests=tuple(snapshot.artifact_digests)
            + (("tokenizer", tokenizer_digest),),
            activation_nonce=nonce,
            rollback_of=admission.rollback_of)
        # Evidence is recorded fail-closed: the ledger entry lands first,
        # then the receipt file — if either cannot be written, no receipt
        # authorizes the load.
        try:
            if self.ledger_path is not None:
                AuthorityLedger(self.ledger_path).append(
                    self.runtime_signer, "activation_receipt",
                    receipt.to_doc()["value"])
            doc = receipt.to_doc(signer=self.runtime_signer)
            if receipt_path is not None:
                write_activation_receipt(receipt, receipt_path,
                                         signer=self.runtime_signer)
        except Exception as exc:  # noqa: BLE001 - no evidence, no activation
            raise LaunchRefused(
                f"activation evidence could not be recorded: {exc}") from exc
        return LaunchResult(snapshot=snapshot, receipt=receipt,
                            receipt_doc=doc,
                            backend_id=request.expected_backend,
                            loaded=loaded)

    # --- replay protection ------------------------------------------
    def _reserve_nonce(self) -> str:
        used = self._used_nonces()
        for _ in range(8):
            nonce = secrets.token_hex(16)
            if nonce not in used:
                self.nonce_journal.parent.mkdir(parents=True, exist_ok=True)
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
    """Load a revocation list, optionally enforcing freshness here."""
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
