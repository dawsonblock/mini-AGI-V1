"""RC11 cryptographic authority primitives.

Private signing authority is deliberately separated from verification. Ordinary
runtimes should receive public verification keys only. RC11.1 additionally
rejects non-finite JSON numbers and malformed signatures so signed receipts have
one deterministic byte representation.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _validate_json_value(obj: Any, *, path: str = "$") -> None:
    if obj is None or isinstance(obj, (str, bool, int)):
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float is forbidden in signed JSON at {path}")
        return
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            _validate_json_value(item, path=f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"signed JSON object keys must be strings at {path}")
            _validate_json_value(value, path=f"{path}.{key}")
        return
    raise TypeError(f"unsupported signed JSON value {type(obj).__name__} at {path}")


def canonical_json(obj: Any) -> bytes:
    _validate_json_value(obj)
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def artifact_manifest(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_symlink():
        raise ValueError("authority artifacts may not be symlinks")
    if p.is_file():
        return {"kind": "file", "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    files = []
    for q in sorted(p.rglob("*"), key=lambda x: x.as_posix()):
        if q.is_symlink():
            raise ValueError(f"artifact tree contains symlink: {q.relative_to(p)}")
        if q.is_file():
            h = hashlib.sha256()
            with q.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            files.append({"path": q.relative_to(p).as_posix(), "bytes": q.stat().st_size, "sha256": h.hexdigest()})
    body = {"kind": "directory", "files": files}
    body["tree_sha256"] = hashlib.sha256(canonical_json(body)).hexdigest()
    return body


def artifact_digest(path: str | Path) -> str:
    m = artifact_manifest(path)
    raw = m["sha256"] if m["kind"] == "file" else m["tree_sha256"]
    return "sha256:" + str(raw)


class Ed25519ReceiptSigner:
    def __init__(self, private_key, *, key_id: str = "rc11-promotion"):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        if isinstance(private_key, Ed25519PrivateKey):
            self.private_key = private_key
        else:
            raw = bytes(private_key)
            if len(raw) != 32:
                raise ValueError("Ed25519 private key must be 32 raw bytes")
            self.private_key = Ed25519PrivateKey.from_private_bytes(raw)
        if not key_id or len(key_id) > 128:
            raise ValueError("invalid key_id")
        self.key_id = key_id

    @classmethod
    def generate(cls, *, key_id: str = "rc11-promotion") -> "Ed25519ReceiptSigner":
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        return cls(Ed25519PrivateKey.generate(), key_id=key_id)

    def verifier(self) -> "Ed25519ReceiptVerifier":
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        raw = self.private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return Ed25519ReceiptVerifier(raw, key_id=self.key_id)

    def issue(self, body: dict) -> dict:
        if not isinstance(body, dict):
            raise TypeError("receipt body must be a JSON object")
        payload = canonical_json(body)
        sig = self.private_key.sign(payload)
        return {
            "schema_version": 2,
            "algorithm": "Ed25519",
            "key_id": self.key_id,
            "body": body,
            "body_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "signature_b64": base64.b64encode(sig).decode("ascii"),
        }


class Ed25519ReceiptVerifier:
    def __init__(self, public_key, *, key_id: str = "rc11-promotion", revocation_check=None):
        self.revocation_check = revocation_check
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        if isinstance(public_key, Ed25519PublicKey):
            self.public_key = public_key
        else:
            raw = bytes(public_key)
            if len(raw) != 32:
                raise ValueError("Ed25519 public key must be 32 raw bytes")
            self.public_key = Ed25519PublicKey.from_public_bytes(raw)
        if not key_id or len(key_id) > 128:
            raise ValueError("invalid key_id")
        self.key_id = key_id

    def verify(self, receipt: dict, expected_body: dict | None = None) -> bool:
        try:
            if self.revocation_check is not None and self.revocation_check(self.key_id):
                return False
            if not isinstance(receipt, dict):
                return False
            if receipt.get("schema_version") not in (None, 1, 2):
                return False
            if receipt.get("algorithm") != "Ed25519" or receipt.get("key_id") != self.key_id:
                return False
            body = receipt.get("body")
            if not isinstance(body, dict):
                return False
            if expected_body is not None and body != expected_body:
                return False
            payload = canonical_json(body)
            if receipt.get("body_sha256") != "sha256:" + hashlib.sha256(payload).hexdigest():
                return False
            sig_text = receipt.get("signature_b64")
            if not isinstance(sig_text, str):
                return False
            sig = base64.b64decode(sig_text, validate=True)
            if len(sig) != 64:
                return False
            self.public_key.verify(sig, payload)
        except (ValueError, TypeError, KeyError, binascii.Error, Exception):
            return False
        return True
