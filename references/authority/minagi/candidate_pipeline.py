"""Candidate qualification and cryptographic promotion authority.

A conversation, growth experiment, consolidation run, or replay policy can
produce a *candidate*.  It does not become production state simply because it
exists.  This module provides an append-only SHA-256 hash-chained state machine:

    proposed -> trained -> evaluated -> qualified|rejected -> promoted

v4.2 separates *signing* from *verification*.  Ed25519 promotion uses a private
key held outside the candidate workspace, while a public verifier can be
embedded anywhere without acquiring promotion authority.  The legacy HMAC
``PromotionAuthority`` remains for v4.1 compatibility, but HMAC verification
necessarily possesses the same secret that can sign and is therefore a weaker
trust boundary.

Promotion receipts can bind either one file or an entire artifact directory.
Directory digests are deterministic tree manifests over relative paths, file
sizes and SHA-256 hashes, so a candidate cannot swap an unlisted side artifact
without changing the receipt.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import base64
import hashlib
import hmac
import json
import os
import time
import uuid

from .stats import paired_bootstrap


_GENESIS = "0" * 64
_ALLOWED = {
    None: {"proposed"},
    "proposed": {"trained", "rejected"},
    "trained": {"evaluated", "rejected"},
    "evaluated": {"qualified", "rejected"},
    "qualified": {"promoted", "rejected"},
    "rejected": set(),
    "promoted": set(),
}


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _digest_record(record: dict) -> str:
    d = dict(record)
    d.pop("hash", None)
    return hashlib.sha256(_canon(d)).hexdigest()


def sha256_path(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def artifact_manifest(path: str | os.PathLike) -> dict:
    """Return a canonical content manifest for a file or directory tree."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_symlink():
        raise ValueError("promotion artifacts may not be symlinks")
    if p.is_file():
        return {
            "kind": "file",
            "bytes": int(p.stat().st_size),
            "sha256": sha256_path(p),
        }
    if not p.is_dir():
        raise ValueError(f"unsupported artifact type: {p}")
    files = []
    for q in sorted(p.rglob("*"), key=lambda x: x.as_posix()):
        if q.is_symlink():
            raise ValueError(f"artifact tree contains symlink: {q.relative_to(p)}")
        if not q.is_file():
            continue
        rel = q.relative_to(p).as_posix()
        files.append({
            "path": rel,
            "bytes": int(q.stat().st_size),
            "sha256": sha256_path(q),
        })
    body = {"kind": "directory", "files": files}
    body["tree_sha256"] = hashlib.sha256(_canon(body)).hexdigest()
    return body


def artifact_digest(path: str | os.PathLike) -> tuple[str, dict]:
    m = artifact_manifest(path)
    if m["kind"] == "file":
        return str(m["sha256"]), m
    return str(m["tree_sha256"]), m


@dataclass(frozen=True)
class QualificationPolicy:
    min_functional_accuracy: float = 0.0
    max_heldout_regression: float = 0.0
    max_old_domain_regression: float = 0.02
    min_causal_gain: float | None = None
    require_checks: tuple[str, ...] = ()
    # Optional paired-evidence gate. ``metrics['paired_candidate']`` and
    # ``metrics['paired_control']`` must be matched higher-is-better values.
    min_paired_lcb: float | None = None
    paired_confidence: float = 0.95
    paired_min_n: int = 0
    paired_seed: int = 0

    def evaluate(self, metrics: dict) -> tuple[bool, list[str]]:
        failures: list[str] = []
        fa = float(metrics.get("functional_accuracy", 0.0))
        if fa < self.min_functional_accuracy:
            failures.append(
                f"functional_accuracy {fa:.6g} < {self.min_functional_accuracy:.6g}")
        hr = float(metrics.get("heldout_regression", 0.0))
        if hr > self.max_heldout_regression:
            failures.append(
                f"heldout_regression {hr:.6g} > {self.max_heldout_regression:.6g}")
        od = float(metrics.get("old_domain_regression", 0.0))
        if od > self.max_old_domain_regression:
            failures.append(
                f"old_domain_regression {od:.6g} > {self.max_old_domain_regression:.6g}")
        if self.min_causal_gain is not None:
            if "causal_gain" not in metrics:
                failures.append("causal_gain is required")
            elif float(metrics["causal_gain"]) < self.min_causal_gain:
                failures.append(
                    f"causal_gain {float(metrics['causal_gain']):.6g} < {self.min_causal_gain:.6g}")
        if self.min_paired_lcb is not None:
            a = metrics.get("paired_candidate")
            b = metrics.get("paired_control")
            if a is None or b is None:
                failures.append("paired_candidate and paired_control are required")
            else:
                try:
                    ev = paired_bootstrap(
                        a, b, confidence=self.paired_confidence,
                        seed=self.paired_seed,
                    )
                    if ev.n < int(self.paired_min_n):
                        failures.append(
                            f"paired evidence n={ev.n} < required {int(self.paired_min_n)}")
                    if ev.lcb < float(self.min_paired_lcb):
                        failures.append(
                            f"paired {self.paired_confidence:.1%} LCB {ev.lcb:.6g} < "
                            f"{float(self.min_paired_lcb):.6g}")
                except Exception as e:
                    failures.append(f"paired evidence invalid: {e}")
        checks = metrics.get("checks") or {}
        for name in self.require_checks:
            if checks.get(name) is not True:
                failures.append(f"required check {name!r} did not pass")
        return not failures, failures


class CandidateLedger:
    """Append-only JSONL state machine with a SHA-256 hash chain."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "candidate_ledger.jsonl"
        self._records = self.verify()

    def verify(self) -> list[dict]:
        records: list[dict] = []
        prev = _GENESIS
        states: dict[str, str] = {}
        if not self.path.exists():
            return records
        with self.path.open(encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"candidate ledger malformed at line {ln}") from e
                if rec.get("prev") != prev:
                    raise RuntimeError(f"candidate ledger chain break at line {ln}")
                got = _digest_record(rec)
                if not hmac.compare_digest(str(rec.get("hash", "")), got):
                    raise RuntimeError(f"candidate ledger hash mismatch at line {ln}")
                cid = str(rec.get("candidate_id", ""))
                state = str(rec.get("state", ""))
                before = states.get(cid)
                if state not in _ALLOWED.get(before, set()):
                    raise RuntimeError(
                        f"candidate ledger invalid transition at line {ln}: {before!r}->{state!r}")
                states[cid] = state
                records.append(rec)
                prev = got
        return records

    @property
    def head(self) -> str:
        return self._records[-1]["hash"] if self._records else _GENESIS

    def state(self, candidate_id: str) -> str | None:
        s = None
        for r in self._records:
            if r["candidate_id"] == candidate_id:
                s = r["state"]
        return s

    def history(self, candidate_id: str) -> list[dict]:
        return [dict(r) for r in self._records if r["candidate_id"] == candidate_id]

    def append(self, candidate_id: str, state: str, *, evidence: dict | None = None,
               actor: str = "operator") -> dict:
        candidate_id = str(candidate_id)
        before = self.state(candidate_id)
        if state not in _ALLOWED.get(before, set()):
            raise ValueError(f"invalid candidate transition: {before!r} -> {state!r}")
        rec = {
            "version": 2,
            "seq": len(self._records) + 1,
            "candidate_id": candidate_id,
            "state": state,
            "ts": time.time(),
            "actor": str(actor),
            "evidence": evidence or {},
            "prev": self.head,
        }
        rec["hash"] = _digest_record(rec)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            fd = os.open(str(self.root), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
        self._records.append(rec)
        return dict(rec)

    def propose(self, *, base_generation: str, source: str,
                dataset_digest: str | None = None, candidate_id: str | None = None,
                actor: str = "operator") -> str:
        cid = candidate_id or ("cand-" + uuid.uuid4().hex)
        self.append(cid, "proposed", actor=actor, evidence={
            "base_generation": str(base_generation),
            "source": str(source),
            "dataset_digest": dataset_digest,
        })
        return cid

    def qualify(self, candidate_id: str, metrics: dict, policy: QualificationPolicy,
                actor: str = "qualifier") -> dict:
        if self.state(candidate_id) != "evaluated":
            raise ValueError("candidate must be in evaluated state before qualification")
        ok, failures = policy.evaluate(metrics)
        evidence = {
            "metrics": metrics,
            "policy": asdict(policy),
            "failures": failures,
        }
        if policy.min_paired_lcb is not None and metrics.get("paired_candidate") is not None:
            try:
                evidence["paired_evidence"] = paired_bootstrap(
                    metrics["paired_candidate"], metrics["paired_control"],
                    confidence=policy.paired_confidence,
                    seed=policy.paired_seed,
                ).as_dict()
            except Exception:
                pass
        return self.append(candidate_id, "qualified" if ok else "rejected",
                           actor=actor, evidence=evidence)

    def qualification_record_hash(self, candidate_id: str) -> str:
        hist = self.history(candidate_id)
        for rec in reversed(hist):
            if rec.get("state") == "qualified":
                return str(rec.get("hash"))
        raise ValueError("candidate has no qualified record")


class PromotionAuthority:
    """Legacy HMAC promotion authority retained for v4.1 compatibility.

    HMAC verification and signing share one secret. New deployments should use
    ``Ed25519PromotionSigner`` + ``Ed25519PromotionVerifier`` instead.
    """

    def __init__(self, key: bytes, key_id: str = "operator"):
        if not isinstance(key, (bytes, bytearray)) or len(key) < 32:
            raise ValueError("promotion key must contain at least 32 bytes")
        self.key = bytes(key)
        self.key_id = str(key_id)

    def issue(self, ledger: CandidateLedger, candidate_id: str, *, artifact_path: str,
              expected_base_generation: str, actor: str = "promotion-authority") -> dict:
        if ledger.state(candidate_id) != "qualified":
            raise ValueError("only a qualified candidate may be promoted")
        hist = ledger.history(candidate_id)
        proposal = hist[0]
        base = (proposal.get("evidence") or {}).get("base_generation")
        if str(base) != str(expected_base_generation):
            raise ValueError("candidate base generation does not match current promotion base")
        digest, manifest = artifact_digest(artifact_path)
        payload = {
            "version": 2,
            "scheme": "hmac-sha256-legacy",
            "candidate_id": candidate_id,
            "base_generation": str(base),
            "artifact_sha256": digest,
            "artifact_kind": manifest["kind"],
            "qualification_record_hash": ledger.qualification_record_hash(candidate_id),
            "ledger_head": ledger.head,
            "key_id": self.key_id,
            "issued_at": time.time(),
            "nonce": uuid.uuid4().hex,
        }
        sig = hmac.new(self.key, _canon(payload), hashlib.sha256).hexdigest()
        receipt = {**payload, "hmac_sha256": sig}
        ledger.append(candidate_id, "promoted", actor=actor,
                      evidence={"receipt": receipt})
        return receipt

    def verify_receipt(self, receipt: dict, artifact_path: str | None = None) -> bool:
        payload = dict(receipt)
        sig = str(payload.pop("hmac_sha256", ""))
        want = hmac.new(self.key, _canon(payload), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, want):
            return False
        if artifact_path is not None:
            digest, _ = artifact_digest(artifact_path)
            return hmac.compare_digest(str(receipt.get("artifact_sha256", "")), digest)
        return True


class Ed25519PromotionSigner:
    """Offline/private promotion signer. Keep this outside model-writable state."""

    def __init__(self, private_key, key_id: str = "operator-ed25519"):
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        except ImportError as e:
            raise RuntimeError("Ed25519 promotion requires the 'cryptography' package") from e
        if isinstance(private_key, Ed25519PrivateKey):
            self.private_key = private_key
        elif isinstance(private_key, (bytes, bytearray)):
            raw = bytes(private_key)
            if len(raw) != 32:
                raise ValueError("Ed25519 private key must be 32 raw bytes")
            self.private_key = Ed25519PrivateKey.from_private_bytes(raw)
        else:
            raise TypeError("private_key must be Ed25519PrivateKey or 32 raw bytes")
        self.key_id = str(key_id)

    @classmethod
    def generate(cls, key_id: str = "operator-ed25519"):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        return cls(Ed25519PrivateKey.generate(), key_id=key_id)

    def private_bytes_raw(self) -> bytes:
        from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
        return self.private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())

    def public_bytes_raw(self) -> bytes:
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        return self.private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    def verifier(self):
        return Ed25519PromotionVerifier(self.public_bytes_raw(), self.key_id)

    def issue(self, ledger: CandidateLedger, candidate_id: str, *, artifact_path: str,
              expected_base_generation: str, actor: str = "promotion-authority") -> dict:
        if ledger.state(candidate_id) != "qualified":
            raise ValueError("only a qualified candidate may be promoted")
        hist = ledger.history(candidate_id)
        base = (hist[0].get("evidence") or {}).get("base_generation")
        if str(base) != str(expected_base_generation):
            raise ValueError("candidate base generation does not match current promotion base")
        digest, manifest = artifact_digest(artifact_path)
        payload = {
            "version": 2,
            "scheme": "ed25519",
            "candidate_id": candidate_id,
            "base_generation": str(base),
            "artifact_sha256": digest,
            "artifact_kind": manifest["kind"],
            "qualification_record_hash": ledger.qualification_record_hash(candidate_id),
            "ledger_head": ledger.head,
            "key_id": self.key_id,
            "issued_at": time.time(),
            "nonce": uuid.uuid4().hex,
        }
        sig = self.private_key.sign(_canon(payload))
        receipt = {**payload, "ed25519_signature": base64.b64encode(sig).decode("ascii")}
        ledger.append(candidate_id, "promoted", actor=actor,
                      evidence={"receipt": receipt})
        return receipt


class Ed25519PromotionVerifier:
    """Public-only promotion verifier. It cannot issue promotion receipts."""

    def __init__(self, public_key, key_id: str | None = None):
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError as e:
            raise RuntimeError("Ed25519 promotion requires the 'cryptography' package") from e
        if isinstance(public_key, Ed25519PublicKey):
            self.public_key = public_key
        elif isinstance(public_key, (bytes, bytearray)):
            raw = bytes(public_key)
            if len(raw) != 32:
                raise ValueError("Ed25519 public key must be 32 raw bytes")
            self.public_key = Ed25519PublicKey.from_public_bytes(raw)
        else:
            raise TypeError("public_key must be Ed25519PublicKey or 32 raw bytes")
        self.key_id = None if key_id is None else str(key_id)

    def verify_receipt(self, receipt: dict, artifact_path: str | None = None) -> bool:
        if receipt.get("scheme") != "ed25519":
            return False
        if self.key_id is not None and str(receipt.get("key_id")) != self.key_id:
            return False
        payload = dict(receipt)
        sig_b64 = str(payload.pop("ed25519_signature", ""))
        try:
            sig = base64.b64decode(sig_b64, validate=True)
            self.public_key.verify(sig, _canon(payload))
        except Exception:
            return False
        if artifact_path is not None:
            try:
                digest, manifest = artifact_digest(artifact_path)
            except Exception:
                return False
            if not hmac.compare_digest(str(receipt.get("artifact_sha256", "")), digest):
                return False
            if str(receipt.get("artifact_kind")) != str(manifest.get("kind")):
                return False
        return True


class LeasedCandidateLedger(CandidateLedger):
    """Candidate ledger whose mutations acquire an OS-level writer lease.

    ``CandidateLedger`` is kept for v4/v5 compatibility. New v6 control-plane
    code should prefer this class when multiple processes may share the same
    authority directory. Each append reloads and verifies the chain while the
    lease is held, preventing two writers from extending the same head.
    """

    def append(self, candidate_id: str, state: str, *, evidence: dict | None = None,
               actor: str = "operator") -> dict:
        from .authority import WriterLease
        # CandidateLedger.__init__ itself verifies but does not mutate. Reload
        # chain inside the lease immediately before computing the next head.
        with WriterLease(self.root, purpose="candidate-ledger"):
            self._records = self.verify()
            return super().append(candidate_id, state, evidence=evidence, actor=actor)
