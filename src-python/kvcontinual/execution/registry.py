from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from kvcontinual.execution.authority import Ed25519ReceiptSigner, Ed25519ReceiptVerifier
from kvcontinual.execution.digests import sha256_file, sha256_json
from kvcontinual.execution.qualification_bundle import verify_qualification_bundle
from kvcontinual.execution.types import QualificationRecord, PromotionDecision


@dataclass
class CandidateManifest:
    candidate_id: str
    base_model_digest: str
    parent_adapter_digest: str | None
    dataset_digest: str
    training_config_digest: str
    adapter_digest: str
    created_at: str
    artifact_relpath: str = "adapter.safetensors"
    tokenizer_digest: str | None = None
    execution_identity_digest: str | None = None


def _safe_component(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ValueError(f"invalid {field}")
    if value in (".", "..") or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe {field}")
    return value


def _safe_member(root: Path, relpath: str) -> Path:
    if not isinstance(relpath, str) or not relpath or "\x00" in relpath:
        raise ValueError("invalid artifact_relpath")
    rel = Path(relpath)
    if rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
        raise ValueError("unsafe artifact_relpath")
    root_resolved = root.resolve()
    raw = root / rel
    cursor = root
    for part in rel.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("artifact_relpath traverses a symlink")
    out = raw.resolve()
    try:
        out.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("artifact_relpath escapes candidate directory") from exc
    return out


def _fsync_file(path: Path) -> None:
    """Durably flush a regular file to stable storage where the OS supports it."""
    with path.open("rb") as f:
        os.fsync(f.fileno())


def _fsync_dir(path: Path) -> None:
    """Durably flush directory entry updates on the Unix/macOS target platforms."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    """Flush a small registry subtree before publishing it by directory rename."""
    for p in sorted(root.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if p.is_symlink():
            raise ValueError(f"registry subtree contains symlink: {p}")
        if p.is_file():
            _fsync_file(p)
        elif p.is_dir():
            _fsync_dir(p)
    _fsync_dir(root)


class AdapterRegistry:
    """Digest-bound candidate registry with fail-closed promotion authority.

    RC11.1 adds path confinement and verifies the complete signed chain during
    rollback/current-state checks. In signed mode promotion creates a second
    receipt binding the staged cache namespace to the promoted candidate; a
    qualification signature alone no longer authorizes an unbound namespace.
    """

    def __init__(self, root: str, *, require_signed_promotions: bool = False,
                 require_qualification_bundle: bool = False):
        self.root = Path(root)
        self.candidates = self.root / "candidates"
        self.promoted = self.root / "promoted"
        self.meta = self.root / "meta"
        self.namespaces = self.root / "cache_namespaces"
        self.transitions = self.meta / "production_transitions.jsonl"
        self.lock_path = self.meta / ".registry.lock"
        self.registry_identity_path = self.meta / "registry_identity.json"
        self.require_signed_promotions = bool(require_signed_promotions)
        self.require_qualification_bundle = bool(require_qualification_bundle)
        for p in (self.candidates, self.promoted, self.meta, self.namespaces):
            p.mkdir(parents=True, exist_ok=True)
        self.registry_id = self._load_or_create_registry_identity()

    def _load_or_create_registry_identity(self) -> str:
        if self.registry_identity_path.exists():
            payload = json.loads(self.registry_identity_path.read_text())
            rid = payload.get("registry_id")
            return _safe_component(rid, field="registry_id")
        payload = {
            "schema_version": 1,
            "registry_id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._atomic_json(self.registry_identity_path, payload)
        return payload["registry_id"]

    @contextmanager
    def _mutation_lock(self):
        """Serialize promotion/rollback state transitions on Unix/macOS.

        The release targets macOS and Linux reference validation. A lock file is
        deliberately runtime state and is not part of release identity.
        """
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - fail closed off target platforms
            raise RuntimeError("registry mutation locking requires fcntl") from exc
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _transition_digest(record: dict) -> str:
        return sha256_json(record)

    def _transition_records(self) -> list[dict]:
        if not self.transitions.exists():
            return []
        records = []
        with self.transitions.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"malformed production transition ledger at line {lineno}") from exc
                if not isinstance(record, dict):
                    raise RuntimeError(f"malformed production transition ledger at line {lineno}")
                records.append(record)
        return records

    def verify_transition_chain(self, *, verifier: Ed25519ReceiptVerifier | None = None,
                                minimum_generation: int = 0,
                                expected_tail_digest: str | None = None) -> dict | None:
        if not isinstance(minimum_generation, int) or isinstance(minimum_generation, bool) or minimum_generation < 0:
            raise ValueError("minimum_generation must be a non-negative integer")
        previous_digest = None
        previous_generation = 0
        previous_current = None
        seen_events: set[str] = set()
        tail = None
        for record in self._transition_records():
            body = record.get("body")
            if not isinstance(body, dict) or body.get("receipt_type") not in {
                "rc11.production-transition.v1", "rc11.production-transition.v2"
            }:
                raise RuntimeError("production transition ledger contains an invalid body")
            generation = body.get("generation")
            if not isinstance(generation, int) or isinstance(generation, bool) or generation != previous_generation + 1:
                raise RuntimeError("production transition generation is not monotonic")
            if body.get("previous_transition_digest") != previous_digest:
                raise RuntimeError("production transition hash chain is broken")
            if body.get("action") not in {"promote", "rollback"}:
                raise RuntimeError("production transition action is invalid")
            event_id = body.get("event_id")
            if not isinstance(event_id, str) or not event_id or event_id in seen_events:
                raise RuntimeError("production transition event ID is invalid or replayed")
            seen_events.add(event_id)
            if generation == 1:
                if body.get("previous_candidate_id") is not None:
                    raise RuntimeError("first production transition must not name a previous candidate")
            elif body.get("previous_candidate_id") != previous_current:
                raise RuntimeError("production transition candidate continuity is broken")
            if body.get("receipt_type") == "rc11.production-transition.v2":
                if body.get("registry_id") != self.registry_id:
                    raise RuntimeError("production transition registry identity mismatch")
                embedded = body.get("production_receipt")
                expected_embedded_digest = sha256_json(embedded) if embedded is not None else None
                if body.get("production_receipt_digest") != expected_embedded_digest:
                    raise RuntimeError("production transition embedded binding digest mismatch")
                if embedded is not None and verifier is not None:
                    if not verifier.verify(embedded):
                        raise RuntimeError("production transition embedded binding receipt is invalid")
            receipt = record.get("authority_receipt")
            if self.require_signed_promotions:
                if verifier is None or not isinstance(receipt, dict):
                    raise RuntimeError("signed production transition receipt is required")
                if not verifier.verify(receipt, expected_body=body):
                    raise RuntimeError("production transition receipt is invalid")
            elif receipt is not None and verifier is not None:
                if not verifier.verify(receipt, expected_body=body):
                    raise RuntimeError("production transition receipt is invalid")
            previous_digest = self._transition_digest(record)
            previous_generation = generation
            previous_current = body.get("current_candidate_id")
            tail = record
        if previous_generation < minimum_generation:
            raise RuntimeError("production transition generation is below the external monotonic floor")
        if expected_tail_digest is not None:
            actual = self._transition_digest(tail) if tail is not None else None
            if actual != expected_tail_digest:
                raise RuntimeError("production transition tail does not match external anchor")
        return tail

    def _append_transition(self, body: dict, *, signer: Ed25519ReceiptSigner | None) -> dict:
        if self.require_signed_promotions and signer is None:
            raise RuntimeError("signed production transition authority is required")
        receipt = signer.issue(body) if signer is not None else None
        record = {"body": body, "authority_receipt": receipt}
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        self.transitions.parent.mkdir(parents=True, exist_ok=True)
        with self.transitions.open("a", encoding="utf-8") as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        _fsync_dir(self.transitions.parent)
        return record

    def register_candidate(self, adapter_file: str, base_model_digest: str, dataset_digest: str,
                           training_config: dict, parent_adapter_digest: str | None = None,
                           tokenizer_digest: str | None = None,
                           execution_identity_digest: str | None = None) -> CandidateManifest:
        src = Path(adapter_file)
        if not src.is_file() or src.is_symlink():
            raise ValueError("candidate adapter must be a regular non-symlink file")
        cid = str(uuid.uuid4())
        dst_dir = self.candidates / cid
        pending = self.candidates / (".pending-" + cid)
        shutil.rmtree(pending, ignore_errors=True)
        pending.mkdir()
        dst = pending / src.name
        shutil.copy2(src, dst)
        _fsync_file(dst)
        manifest = CandidateManifest(
            cid, base_model_digest, parent_adapter_digest, dataset_digest,
            sha256_json(training_config), sha256_file(dst),
            datetime.now(timezone.utc).isoformat(), dst.name,
            tokenizer_digest, execution_identity_digest,
        )
        self._atomic_json(pending / "manifest.json", asdict(manifest))
        _fsync_tree(pending)
        pending.replace(dst_dir)
        _fsync_dir(self.candidates)
        return manifest

    def _candidate_dir(self, candidate_id: str) -> Path:
        return self.candidates / _safe_component(candidate_id, field="candidate_id")

    def _promoted_dir(self, candidate_id: str) -> Path:
        return self.promoted / _safe_component(candidate_id, field="candidate_id")

    def _namespace_dir(self, candidate_id: str) -> Path:
        return self.namespaces / _safe_component(candidate_id, field="candidate_id")

    def _manifest(self, candidate_id: str) -> tuple[dict, Path]:
        cdir = self._candidate_dir(candidate_id)
        mpath = cdir / "manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(candidate_id)
        manifest = json.loads(mpath.read_text())
        if manifest.get("candidate_id") != candidate_id:
            raise RuntimeError("candidate manifest ID mismatch")
        artifact = _safe_member(cdir, manifest.get("artifact_relpath", ""))
        if not artifact.is_file() or artifact.is_symlink():
            raise RuntimeError("candidate artifact is missing or unsafe")
        return manifest, artifact

    def _verified_candidate_binding(self, candidate_id: str) -> tuple[dict, Path, str, str]:
        manifest, artifact = self._manifest(candidate_id)
        actual = sha256_file(artifact)
        if actual != manifest.get("adapter_digest"):
            raise RuntimeError("candidate artifact changed after registration")
        manifest_digest = sha256_json(manifest)
        return manifest, artifact, actual, manifest_digest

    def write_qualification(self, q: QualificationRecord, *, signer: Ed25519ReceiptSigner | None = None,
                            qualification_bundle: dict | None = None) -> None:
        candidate_id = _safe_component(q.candidate_id, field="candidate_id")
        cdir = self._candidate_dir(candidate_id)
        path = cdir / "qualification.json"
        if not cdir.exists():
            raise FileNotFoundError(candidate_id)
        if path.exists():
            raise RuntimeError("qualification record is immutable once written")
        if self._promoted_dir(candidate_id).exists():
            raise RuntimeError("promoted candidates cannot be re-qualified")
        manifest, _, adapter_digest, manifest_digest = self._verified_candidate_binding(candidate_id)
        payload = asdict(q)
        payload["decision"] = q.decision.value
        payload["adapter_digest"] = adapter_digest
        payload["candidate_manifest_digest"] = manifest_digest
        payload["registry_id"] = self.registry_id
        payload["qualified_at"] = datetime.now(timezone.utc).isoformat()
        if qualification_bundle is not None:
            if not isinstance(qualification_bundle, dict) or not verify_qualification_bundle(qualification_bundle):
                raise RuntimeError("qualification bundle is invalid or did not pass")
            if qualification_bundle.get("model_weights_digest") != manifest.get("base_model_digest"):
                raise RuntimeError("qualification bundle model digest does not match candidate base model")
            if manifest.get("tokenizer_digest") is not None and qualification_bundle.get("tokenizer_digest") != manifest.get("tokenizer_digest"):
                raise RuntimeError("qualification bundle tokenizer digest does not match candidate manifest")
            if manifest.get("execution_identity_digest") is not None and qualification_bundle.get("execution_identity_digest") != manifest.get("execution_identity_digest"):
                raise RuntimeError("qualification bundle execution identity does not match candidate manifest")
            bundle_digest = qualification_bundle.get("bundle_digest")
            payload["qualification_bundle_digest"] = bundle_digest
            self._atomic_json(cdir / "qualification_bundle.json", qualification_bundle)
        elif self.require_qualification_bundle:
            raise RuntimeError("a verified qualification bundle is required")
        if signer is not None:
            payload["authority_receipt"] = signer.issue(dict(payload))
        self._atomic_json(path, payload)

    def stage_cache_namespace(self, candidate_id: str, namespace_digest: str, warm_fraction: float = 0.0) -> Path:
        candidate_id = _safe_component(candidate_id, field="candidate_id")
        if not self._candidate_dir(candidate_id).exists():
            raise FileNotFoundError(candidate_id)
        if not isinstance(namespace_digest, str) or not namespace_digest:
            raise ValueError("namespace_digest is required")
        if not (0.0 <= warm_fraction <= 1.0):
            raise ValueError("warm_fraction must be in [0,1]")
        p = self._namespace_dir(candidate_id)
        p.mkdir(exist_ok=True)
        payload = {
            "candidate_id": candidate_id,
            "namespace_digest": namespace_digest,
            "warm_fraction": float(warm_fraction),
        }
        existing = p / "namespace.json"
        if self._promoted_dir(candidate_id).exists() and existing.exists():
            if json.loads(existing.read_text()) != payload:
                raise RuntimeError("promoted cache namespace is immutable")
            return p
        self._atomic_json(existing, payload)
        return p

    @staticmethod
    def _expected_receipt_payload(record: dict) -> dict:
        expected = dict(record)
        expected.pop("authority_receipt", None)
        return expected

    def _verify_qualification_record(self, q: dict, *, candidate_id: str,
                                     adapter_digest: str, manifest_digest: str,
                                     verifier: Ed25519ReceiptVerifier | None,
                                     record_dir: Path | None = None) -> None:
        if q.get("candidate_id") != candidate_id:
            raise RuntimeError("qualification candidate ID mismatch")
        if q.get("decision") != PromotionDecision.PROMOTE.value:
            raise RuntimeError("Candidate is not qualified for promotion")
        if q.get("adapter_digest") != adapter_digest:
            raise RuntimeError("qualified adapter digest does not match candidate bytes")
        if q.get("candidate_manifest_digest") != manifest_digest:
            raise RuntimeError("qualified manifest digest does not match candidate manifest")
        q_registry_id = q.get("registry_id")
        if q_registry_id is not None and q_registry_id != self.registry_id:
            raise RuntimeError("qualification registry identity mismatch")
        bundle_digest = q.get("qualification_bundle_digest")
        if self.require_qualification_bundle and not isinstance(bundle_digest, str):
            raise RuntimeError("verified qualification bundle is required")
        if bundle_digest is not None:
            root = record_dir or self._candidate_dir(candidate_id)
            bpath = root / "qualification_bundle.json"
            if not bpath.exists():
                raise RuntimeError("qualification bundle is missing")
            bundle = json.loads(bpath.read_text())
            if not verify_qualification_bundle(bundle):
                raise RuntimeError("qualification bundle is invalid")
            if bundle.get("bundle_digest") != bundle_digest:
                raise RuntimeError("qualification bundle digest mismatch")
        receipt = q.get("authority_receipt")
        if self.require_signed_promotions:
            if verifier is None or not isinstance(receipt, dict):
                raise RuntimeError("signed promotion authority receipt is required")
            if not verifier.verify(receipt, expected_body=self._expected_receipt_payload(q)):
                raise RuntimeError("promotion authority receipt is invalid")
        elif receipt is not None and verifier is not None:
            if not verifier.verify(receipt, expected_body=self._expected_receipt_payload(q)):
                raise RuntimeError("promotion authority receipt is invalid")

    def _production_binding_body(self, *, candidate_id: str, adapter_digest: str,
                                 manifest_digest: str, namespace: dict,
                                 qualification_record: dict, promoted_at: str,
                                 generation: int, previous_transition_digest: str | None,
                                 event_id: str) -> dict:
        q_receipt = qualification_record.get("authority_receipt")
        return {
            "receipt_type": "rc11.production-binding.v3",
            "registry_id": self.registry_id,
            "event_id": event_id,
            "generation": generation,
            "previous_transition_digest": previous_transition_digest,
            "candidate_id": candidate_id,
            "adapter_digest": adapter_digest,
            "candidate_manifest_digest": manifest_digest,
            "cache_namespace_digest": sha256_json(namespace),
            "qualification_payload_digest": sha256_json(AdapterRegistry._expected_receipt_payload(qualification_record)),
            "qualification_receipt_digest": q_receipt.get("body_sha256") if isinstance(q_receipt, dict) else None,
            "qualification_bundle_digest": qualification_record.get("qualification_bundle_digest"),
            "promoted_at": promoted_at,
        }

    @staticmethod
    def _legacy_production_binding_body(*, candidate_id: str, adapter_digest: str,
                                        manifest_digest: str, namespace: dict,
                                        qualification_record: dict, promoted_at: str,
                                        generation: int, previous_transition_digest: str | None,
                                        event_id: str) -> dict:
        """Reconstruct the RC11.1-RC11.3 v2 binding body for migration verification."""
        q_receipt = qualification_record.get("authority_receipt")
        return {
            "receipt_type": "rc11.production-binding.v2",
            "event_id": event_id,
            "generation": generation,
            "previous_transition_digest": previous_transition_digest,
            "candidate_id": candidate_id,
            "adapter_digest": adapter_digest,
            "candidate_manifest_digest": manifest_digest,
            "cache_namespace_digest": sha256_json(namespace),
            "qualification_payload_digest": sha256_json(AdapterRegistry._expected_receipt_payload(qualification_record)),
            "qualification_receipt_digest": q_receipt.get("body_sha256") if isinstance(q_receipt, dict) else None,
            "promoted_at": promoted_at,
        }

    def _transition_body(self, *, action: str, generation: int,
                         previous_transition_digest: str | None,
                         current_candidate_id: str, previous_candidate_id: str | None,
                         adapter_digest: str, manifest_digest: str,
                         namespace: dict, production_receipt: dict | None,
                         qualification_bundle_digest: str | None,
                         occurred_at: str, event_id: str) -> dict:
        if action not in {"promote", "rollback"}:
            raise ValueError("invalid production transition action")
        return {
            "receipt_type": "rc11.production-transition.v2",
            "registry_id": self.registry_id,
            "event_id": event_id,
            "generation": generation,
            "previous_transition_digest": previous_transition_digest,
            "action": action,
            "current_candidate_id": current_candidate_id,
            "previous_candidate_id": previous_candidate_id,
            "adapter_digest": adapter_digest,
            "candidate_manifest_digest": manifest_digest,
            "cache_namespace_digest": sha256_json(namespace),
            "production_receipt_digest": sha256_json(production_receipt) if production_receipt is not None else None,
            "production_receipt": production_receipt,
            "qualification_bundle_digest": qualification_bundle_digest,
            "occurred_at": occurred_at,
        }

    def _persist_promotion_receipt(self, candidate_id: str, event_id: str,
                                   receipt: dict | None) -> None:
        if receipt is None:
            return
        pdir = self._promoted_dir(candidate_id)
        receipts = pdir / "promotion_receipts"
        receipts.mkdir(parents=True, exist_ok=True)
        event_name = _safe_component(event_id, field="event_id") + ".json"
        event_path = receipts / event_name
        if event_path.exists():
            existing = json.loads(event_path.read_text())
            if existing != receipt:
                raise RuntimeError("promotion receipt event ID collision")
        else:
            self._atomic_json(event_path, receipt)
        self._atomic_json(pdir / "promotion_receipt.json", receipt)

    def promote(self, candidate_id: str, *, verifier: Ed25519ReceiptVerifier | None = None,
                signer: Ed25519ReceiptSigner | None = None) -> Path:
        candidate_id = _safe_component(candidate_id, field="candidate_id")
        qpath = self._candidate_dir(candidate_id) / "qualification.json"
        if not qpath.exists():
            raise RuntimeError("Candidate has no qualification record")
        q = json.loads(qpath.read_text())
        _, _, adapter_digest, manifest_digest = self._verified_candidate_binding(candidate_id)
        self._verify_qualification_record(
            q, candidate_id=candidate_id, adapter_digest=adapter_digest,
            manifest_digest=manifest_digest, verifier=verifier,
            record_dir=self._candidate_dir(candidate_id),
        )

        nspath = self._namespace_dir(candidate_id) / "namespace.json"
        if not nspath.exists():
            raise RuntimeError("Candidate cache namespace is not staged")
        namespace = json.loads(nspath.read_text())
        if namespace.get("candidate_id") != candidate_id:
            raise RuntimeError("cache namespace candidate ID mismatch")

        if self.require_signed_promotions and signer is None:
            raise RuntimeError("signed production binding is required")
        if signer is not None and verifier is not None and signer.key_id != verifier.key_id:
            raise RuntimeError("signer/verifier key IDs differ")

        with self._mutation_lock():
            # Re-read all mutable inputs after acquiring the writer lock.
            q = json.loads(qpath.read_text())
            _, _, adapter_digest, manifest_digest = self._verified_candidate_binding(candidate_id)
            self._verify_qualification_record(
                q, candidate_id=candidate_id, adapter_digest=adapter_digest,
                manifest_digest=manifest_digest, verifier=verifier,
                record_dir=self._candidate_dir(candidate_id),
            )
            namespace = json.loads(nspath.read_text())
            if namespace.get("candidate_id") != candidate_id:
                raise RuntimeError("cache namespace candidate ID mismatch")

            tail = self.verify_transition_chain(verifier=verifier)
            previous_transition_digest = self._transition_digest(tail) if tail is not None else None
            generation = int(tail["body"]["generation"]) + 1 if tail is not None else 1
            # The ledger, not the derived production snapshot, is authoritative.
            # This keeps promotion safe after the crash window where the ledger
            # advanced but production.json was not yet replaced.
            previous = tail["body"].get("current_candidate_id") if tail is not None else None
            event_id = str(uuid.uuid4())
            promoted_at = datetime.now(timezone.utc).isoformat()

            src = self._candidate_dir(candidate_id)
            dst = self._promoted_dir(candidate_id)
            pending = self.promoted / (".pending-" + candidate_id)

            production_body = self._production_binding_body(
                candidate_id=candidate_id,
                adapter_digest=adapter_digest,
                manifest_digest=manifest_digest,
                namespace=namespace,
                qualification_record=q,
                promoted_at=promoted_at,
                generation=generation,
                previous_transition_digest=previous_transition_digest,
                event_id=event_id,
            )
            production_receipt = signer.issue(production_body) if signer is not None else None
            if dst.exists():
                # Re-promotions reuse immutable promoted bytes instead of deleting
                # and replacing the live artifact directory. Only the generation-
                # specific binding receipt is atomically refreshed.
                p_manifest, p_artifact, p_digest, p_manifest_digest, p_q = self._verified_promoted_binding(candidate_id)
                if p_digest != adapter_digest or p_manifest_digest != manifest_digest or p_q != q:
                    raise RuntimeError("existing promoted candidate does not match immutable candidate state")
            else:
                shutil.rmtree(pending, ignore_errors=True)
                shutil.copytree(src, pending, symlinks=False)
                copied_manifest = json.loads((pending / "manifest.json").read_text())
                copied_artifact = _safe_member(pending, copied_manifest.get("artifact_relpath", ""))
                if sha256_file(copied_artifact) != adapter_digest:
                    shutil.rmtree(pending, ignore_errors=True)
                    raise RuntimeError("promotion copy failed digest verification")
                _fsync_tree(pending)
                pending.replace(dst)
                _fsync_dir(self.promoted)

            transition_body = self._transition_body(
                action="promote", generation=generation,
                previous_transition_digest=previous_transition_digest,
                current_candidate_id=candidate_id, previous_candidate_id=previous,
                adapter_digest=adapter_digest, manifest_digest=manifest_digest,
                namespace=namespace, production_receipt=production_receipt,
                qualification_bundle_digest=q.get("qualification_bundle_digest"),
                occurred_at=promoted_at, event_id=event_id,
            )
            transition = self._append_transition(transition_body, signer=signer)
            self._persist_promotion_receipt(candidate_id, event_id, production_receipt)
            state = {
                "registry_id": self.registry_id,
                "current": candidate_id,
                "previous": previous,
                "generation": generation,
                "transition_record_digest": self._transition_digest(transition),
                "transition_receipt": transition.get("authority_receipt"),
                "adapter_digest": adapter_digest,
                "candidate_manifest_digest": manifest_digest,
                "qualification_bundle_digest": q.get("qualification_bundle_digest"),
                "cache_namespace": namespace,
                "authority_receipt": q.get("authority_receipt"),
                "production_receipt": production_receipt,
                "updated_at": promoted_at,
            }
            self._atomic_json(self.meta / "production.json", state)
        return dst

    def current_id(self) -> str | None:
        p = self.meta / "production.json"
        if not p.exists():
            return None
        current = json.loads(p.read_text()).get("current")
        return _safe_component(current, field="current candidate_id") if current is not None else None

    def _verified_promoted_binding(self, candidate_id: str) -> tuple[dict, Path, str, str, dict]:
        candidate_id = _safe_component(candidate_id, field="candidate_id")
        pdir = self._promoted_dir(candidate_id)
        mpath = pdir / "manifest.json"
        if not mpath.exists():
            raise RuntimeError("promoted candidate manifest is missing")
        manifest = json.loads(mpath.read_text())
        if manifest.get("candidate_id") != candidate_id:
            raise RuntimeError("promoted manifest ID mismatch")
        artifact = _safe_member(pdir, manifest.get("artifact_relpath", ""))
        if not artifact.is_file() or artifact.is_symlink():
            raise RuntimeError("promoted artifact is missing or unsafe")
        actual = sha256_file(artifact)
        if actual != manifest.get("adapter_digest"):
            raise RuntimeError("promoted artifact digest mismatch")
        qpath = pdir / "qualification.json"
        if not qpath.exists():
            raise RuntimeError("promoted qualification record is missing")
        q = json.loads(qpath.read_text())
        return manifest, artifact, actual, sha256_json(manifest), q

    def verify_promoted(self, candidate_id: str, *, verifier: Ed25519ReceiptVerifier | None = None,
                        production_receipt: dict | None = None) -> dict:
        candidate_id = _safe_component(candidate_id, field="candidate_id")
        _, _, adapter_digest, manifest_digest, q = self._verified_promoted_binding(candidate_id)
        self._verify_qualification_record(
            q, candidate_id=candidate_id, adapter_digest=adapter_digest,
            manifest_digest=manifest_digest, verifier=verifier,
            record_dir=self._promoted_dir(candidate_id),
        )
        namespace_path = self._namespace_dir(candidate_id) / "namespace.json"
        if not namespace_path.exists():
            raise RuntimeError("promoted cache namespace is unavailable")
        namespace = json.loads(namespace_path.read_text())
        if namespace.get("candidate_id") != candidate_id:
            raise RuntimeError("cache namespace candidate ID mismatch")

        receipt_path = self._promoted_dir(candidate_id) / "promotion_receipt.json"
        if production_receipt is None and receipt_path.exists():
            production_receipt = json.loads(receipt_path.read_text())
        if production_receipt is None:
            receipts_dir = self._promoted_dir(candidate_id) / "promotion_receipts"
            if receipts_dir.exists():
                candidates = []
                for rp in receipts_dir.glob("*.json"):
                    try:
                        rec = json.loads(rp.read_text())
                        body = rec.get("body") if isinstance(rec, dict) else None
                        gen = body.get("generation") if isinstance(body, dict) else None
                        if isinstance(gen, int) and not isinstance(gen, bool):
                            candidates.append((gen, rp.name, rec))
                    except Exception:
                        continue
                if candidates:
                    production_receipt = sorted(candidates)[-1][2]
        if self.require_signed_promotions:
            if verifier is None or not isinstance(production_receipt, dict):
                raise RuntimeError("signed production binding is required")
            body = production_receipt.get("body")
            if not isinstance(body, dict):
                raise RuntimeError("production binding receipt is malformed")
            promoted_at = body.get("promoted_at")
            if body.get("receipt_type") == "rc11.production-binding.v2":
                expected = self._legacy_production_binding_body(
                    candidate_id=candidate_id, adapter_digest=adapter_digest,
                    manifest_digest=manifest_digest, namespace=namespace,
                    qualification_record=q, promoted_at=promoted_at,
                    generation=body.get("generation"),
                    previous_transition_digest=body.get("previous_transition_digest"),
                    event_id=body.get("event_id"),
                )
            else:
                expected = self._production_binding_body(
                    candidate_id=candidate_id, adapter_digest=adapter_digest,
                    manifest_digest=manifest_digest, namespace=namespace,
                    qualification_record=q, promoted_at=promoted_at,
                    generation=body.get("generation"),
                    previous_transition_digest=body.get("previous_transition_digest"),
                    event_id=body.get("event_id"),
                )
            if not verifier.verify(production_receipt, expected_body=expected):
                raise RuntimeError("production binding receipt is invalid")
        return {
            "candidate_id": candidate_id,
            "adapter_digest": adapter_digest,
            "candidate_manifest_digest": manifest_digest,
            "qualification_bundle_digest": q.get("qualification_bundle_digest"),
            "cache_namespace": namespace,
            "authority_receipt": q.get("authority_receipt"),
            "production_receipt": production_receipt,
        }

    def verify_current(self, *, verifier: Ed25519ReceiptVerifier | None = None,
                       minimum_generation: int = 0,
                       expected_tail_digest: str | None = None) -> dict:
        p = self.meta / "production.json"
        if not p.exists():
            raise RuntimeError("No production state")
        state = json.loads(p.read_text())
        if state.get("registry_id") is not None and state.get("registry_id") != self.registry_id:
            raise RuntimeError("production state registry identity mismatch")
        candidate_id = _safe_component(state.get("current"), field="current candidate_id")
        tail = self.verify_transition_chain(
            verifier=verifier,
            minimum_generation=minimum_generation,
            expected_tail_digest=expected_tail_digest,
        )
        if tail is None:
            raise RuntimeError("production transition ledger is missing")
        body = tail["body"]
        if state.get("generation") != body.get("generation"):
            raise RuntimeError("production state generation does not match transition ledger")
        if state.get("transition_record_digest") != self._transition_digest(tail):
            raise RuntimeError("production state transition digest mismatch")
        if state.get("transition_receipt") != tail.get("authority_receipt"):
            raise RuntimeError("production state transition receipt mismatch")
        if body.get("current_candidate_id") != candidate_id or body.get("previous_candidate_id") != state.get("previous"):
            raise RuntimeError("production state candidate transition mismatch")
        embedded_receipt = body.get("production_receipt") if body.get("receipt_type") == "rc11.production-transition.v2" else None
        verified = self.verify_promoted(candidate_id, verifier=verifier, production_receipt=embedded_receipt)
        if state.get("adapter_digest") != verified["adapter_digest"]:
            raise RuntimeError("production state adapter digest mismatch")
        if state.get("candidate_manifest_digest") != verified["candidate_manifest_digest"]:
            raise RuntimeError("production state manifest digest mismatch")
        if state.get("cache_namespace") != verified["cache_namespace"]:
            raise RuntimeError("production state cache namespace mismatch")
        if state.get("production_receipt") != verified["production_receipt"]:
            raise RuntimeError("production state receipt mismatch")
        if body.get("adapter_digest") != verified["adapter_digest"]:
            raise RuntimeError("production transition adapter digest mismatch")
        if body.get("candidate_manifest_digest") != verified["candidate_manifest_digest"]:
            raise RuntimeError("production transition manifest digest mismatch")
        if body.get("qualification_bundle_digest") != verified.get("qualification_bundle_digest"):
            raise RuntimeError("production transition qualification bundle mismatch")
        if body.get("cache_namespace_digest") != sha256_json(verified["cache_namespace"]):
            raise RuntimeError("production transition namespace digest mismatch")
        expected_production_receipt_digest = (
            sha256_json(verified["production_receipt"]) if verified["production_receipt"] is not None else None
        )
        if body.get("production_receipt_digest") != expected_production_receipt_digest:
            raise RuntimeError("production transition binding receipt mismatch")
        return state

    def rollback(self, *, verifier: Ed25519ReceiptVerifier | None = None,
                 signer: Ed25519ReceiptSigner | None = None) -> str:
        if self.require_signed_promotions and signer is None:
            raise RuntimeError("signed rollback authority is required")
        if signer is not None and verifier is not None and signer.key_id != verifier.key_id:
            raise RuntimeError("signer/verifier key IDs differ")
        p = self.meta / "production.json"
        with self._mutation_lock():
            if not p.exists():
                raise RuntimeError("No production state")
            state = json.loads(p.read_text())
            prev = state.get("previous")
            if not prev:
                raise RuntimeError("No previous promoted adapter to roll back to")
            prev = _safe_component(prev, field="previous candidate_id")
            current = _safe_component(state.get("current"), field="current candidate_id")

            # Verify the complete current chain before authorizing a new state
            # transition, then independently verify the rollback target.
            self.verify_current(verifier=verifier)
            verified = self.verify_promoted(prev, verifier=verifier)
            tail = self.verify_transition_chain(verifier=verifier)
            if tail is None:
                raise RuntimeError("production transition ledger is missing")
            previous_transition_digest = self._transition_digest(tail)
            generation = int(tail["body"]["generation"]) + 1
            event_id = str(uuid.uuid4())
            occurred_at = datetime.now(timezone.utc).isoformat()
            transition_body = self._transition_body(
                action="rollback", generation=generation,
                previous_transition_digest=previous_transition_digest,
                current_candidate_id=prev, previous_candidate_id=current,
                adapter_digest=verified["adapter_digest"],
                manifest_digest=verified["candidate_manifest_digest"],
                namespace=verified["cache_namespace"],
                production_receipt=verified["production_receipt"],
                qualification_bundle_digest=verified.get("qualification_bundle_digest"),
                occurred_at=occurred_at, event_id=event_id,
            )
            transition = self._append_transition(transition_body, signer=signer)
            state["current"], state["previous"] = prev, current
            state["registry_id"] = self.registry_id
            state["generation"] = generation
            state["transition_record_digest"] = self._transition_digest(transition)
            state["transition_receipt"] = transition.get("authority_receipt")
            state["adapter_digest"] = verified["adapter_digest"]
            state["candidate_manifest_digest"] = verified["candidate_manifest_digest"]
            state["qualification_bundle_digest"] = verified.get("qualification_bundle_digest")
            state["cache_namespace"] = verified["cache_namespace"]
            state["authority_receipt"] = verified["authority_receipt"]
            state["production_receipt"] = verified["production_receipt"]
            state["updated_at"] = occurred_at
            self._atomic_json(p, state)
            return prev

    def transition_anchor(self, *, verifier: Ed25519ReceiptVerifier | None = None) -> dict:
        """Return a compact external anti-rollback anchor for independent storage."""
        tail = self.verify_transition_chain(verifier=verifier)
        if tail is None:
            return {
                "schema_version": 1,
                "registry_id": self.registry_id,
                "generation": 0,
                "transition_record_digest": None,
                "event_id": None,
            }
        body = tail["body"]
        return {
            "schema_version": 1,
            "registry_id": self.registry_id,
            "generation": body["generation"],
            "transition_record_digest": self._transition_digest(tail),
            "event_id": body["event_id"],
        }

    def recover_current(self, *, verifier: Ed25519ReceiptVerifier | None = None,
                        minimum_generation: int = 0,
                        expected_tail_digest: str | None = None) -> dict:
        """Rebuild the production snapshot from the authoritative ledger tail.

        This is safe for the crash window after a durable transition append but
        before the derived ``production.json`` snapshot was atomically replaced.
        """
        with self._mutation_lock():
            tail = self.verify_transition_chain(
                verifier=verifier,
                minimum_generation=minimum_generation,
                expected_tail_digest=expected_tail_digest,
            )
            if tail is None:
                raise RuntimeError("production transition ledger is missing")
            body = tail["body"]
            candidate_id = _safe_component(body.get("current_candidate_id"), field="current candidate_id")
            embedded_receipt = body.get("production_receipt") if body.get("receipt_type") == "rc11.production-transition.v2" else None
            verified = self.verify_promoted(candidate_id, verifier=verifier, production_receipt=embedded_receipt)
            if body.get("adapter_digest") != verified["adapter_digest"]:
                raise RuntimeError("production transition adapter digest mismatch")
            if body.get("candidate_manifest_digest") != verified["candidate_manifest_digest"]:
                raise RuntimeError("production transition manifest digest mismatch")
            if body.get("cache_namespace_digest") != sha256_json(verified["cache_namespace"]):
                raise RuntimeError("production transition namespace digest mismatch")
            if body.get("qualification_bundle_digest") != verified.get("qualification_bundle_digest"):
                raise RuntimeError("production transition qualification bundle mismatch")
            expected_receipt_digest = (
                sha256_json(verified["production_receipt"]) if verified["production_receipt"] is not None else None
            )
            if body.get("production_receipt_digest") != expected_receipt_digest:
                raise RuntimeError("production transition binding receipt mismatch")
            state = {
                "registry_id": self.registry_id,
                "current": candidate_id,
                "previous": body.get("previous_candidate_id"),
                "generation": body["generation"],
                "transition_record_digest": self._transition_digest(tail),
                "transition_receipt": tail.get("authority_receipt"),
                "adapter_digest": verified["adapter_digest"],
                "candidate_manifest_digest": verified["candidate_manifest_digest"],
                "qualification_bundle_digest": verified.get("qualification_bundle_digest"),
                "cache_namespace": verified["cache_namespace"],
                "authority_receipt": verified["authority_receipt"],
                "production_receipt": verified["production_receipt"],
                "updated_at": body.get("occurred_at"),
            }
            self._atomic_json(self.meta / "production.json", state)
            if embedded_receipt is not None:
                self._persist_promotion_receipt(candidate_id, body["event_id"], embedded_receipt)
        return self.verify_current(
            verifier=verifier,
            minimum_generation=minimum_generation,
            expected_tail_digest=expected_tail_digest,
        )

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
            tmp = Path(f.name)
        os.replace(tmp, path)
        _fsync_dir(path.parent)
