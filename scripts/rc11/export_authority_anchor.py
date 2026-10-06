#!/usr/bin/env python3
"""Export the current RC11 transition head for storage outside the registry.

An external copy supplies the monotonic floor that local files alone cannot
provide against whole-registry rollback/replay.
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
from kvcontinual.execution.authority import Ed25519ReceiptVerifier
from kvcontinual.execution.registry import AdapterRegistry


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno()); tmp = Path(f.name)
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--public-key", required=True)
    ap.add_argument("--key-id", default="rc11-promotion")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = Path(a.public_key).read_bytes()
    if len(key) != 32:
        raise SystemExit("public key must contain exactly 32 raw Ed25519 bytes")
    verifier = Ed25519ReceiptVerifier(key, key_id=a.key_id)
    reg = AdapterRegistry(a.registry, require_signed_promotions=True)
    anchor = reg.transition_anchor(verifier=verifier)
    atomic_json(Path(a.out), anchor)
    print(json.dumps({"ok": True, "out": a.out, **anchor}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
