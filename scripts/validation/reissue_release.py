#!/usr/bin/env python3
"""Regenerate SOURCE_MANIFEST.json over the current tree and re-sign it.

Used when a release needs its manifest/metadata re-issued after post-assembly
source updates. The signing key lives OUTSIDE the repository
(~/.config/miniagi/release-ed25519.pem by default) and is generated on first
use. Never commit the private key.

Enumeration matches scripts/verify_release.py exactly: the four release
metadata files, .git, __pycache__, .pytest_cache, *.egg-info and *.pyc are
excluded; everything else is hashed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXCLUDED = {"SOURCE_MANIFEST.json", "RELEASE_SIGNATURE.bin",
            "RELEASE_PUBLIC_KEY.pem", "RELEASE_ATTESTATION.json"}


def skip(rel: str, p: Path) -> bool:
    parts = Path(rel).parts
    return (rel in EXCLUDED or "__pycache__" in parts or ".git" in parts
            or ".pytest_cache" in parts or p.name.endswith(".pyc")
            or any(part.endswith(".egg-info") for part in parts))


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="mini-AGI-v16.1-Colab-Converged-Full-Upgraded")
    ap.add_argument("--key", default=str(Path.home() / ".config/miniagi/release-ed25519.pem"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    files = {
        p.relative_to(ROOT).as_posix(): sha256_file(p)
        for p in sorted(ROOT.rglob("*"))
        if p.is_file() and not p.is_symlink() and not skip(p.relative_to(ROOT).as_posix(), p)
    }
    doc = {"schema_version": 1, "hash_algorithm": "sha256",
           "release": args.release, "files": files}
    body = canonical(doc)

    key_path = Path(args.key)
    if key_path.is_file():
        priv = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    else:
        priv = Ed25519PrivateKey.generate()
        if not args.dry_run:
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.write_bytes(priv.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()))
            key_path.chmod(0o600)
    assert isinstance(priv, Ed25519PrivateKey)

    import hashlib
    sig = priv.sign(body)
    pub_pem = priv.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)

    print(json.dumps({
        "files": len(files),
        "manifest_sha256": hashlib.sha256(body).hexdigest(),
        "public_key_sha256": hashlib.sha256(pub_pem).hexdigest(),
        "key_path": str(key_path),
        "dry_run": args.dry_run,
    }, indent=2, sort_keys=True))

    if not args.dry_run:
        (ROOT / "SOURCE_MANIFEST.json").write_text(
            json.dumps(doc, indent=2, sort_keys=True) + "\n")
        (ROOT / "RELEASE_SIGNATURE.bin").write_bytes(sig)
        (ROOT / "RELEASE_PUBLIC_KEY.pem").write_bytes(pub_pem)
        print("wrote SOURCE_MANIFEST.json, RELEASE_SIGNATURE.bin, RELEASE_PUBLIC_KEY.pem")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
