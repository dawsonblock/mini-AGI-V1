#!/usr/bin/env python3
"""Generate an RC11 Ed25519 promotion keypair.

Private keys are never shipped. Existing keys are not overwritten unless
--force is explicitly supplied.
"""
from pathlib import Path
import argparse
import json
import os
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption


def atomic_create(path: Path, data: bytes, mode: int, *, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"refusing to overwrite existing key file: {path}; use --force only for intentional rotation")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/authority', help='output directory')
    ap.add_argument('--key-id', default='rc11-promotion', help='receipt key identifier')
    ap.add_argument('--force', action='store_true', help='replace an existing keypair intentionally')
    a = ap.parse_args()
    if not a.key_id or len(a.key_id) > 128:
        raise SystemExit('invalid --key-id')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    priv = Ed25519PrivateKey.generate()
    private_raw = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    public_raw = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    priv_path = out / 'promotion_ed25519.private'
    pub_path = out / 'promotion_ed25519.public'
    atomic_create(priv_path, private_raw, 0o600, force=a.force)
    atomic_create(pub_path, public_raw, 0o644, force=a.force)
    meta = out / 'promotion_key.json'
    meta.write_text(json.dumps({"algorithm": "Ed25519", "key_id": a.key_id, "public_key_file": pub_path.name}, indent=2) + "\n")
    os.chmod(meta, 0o644)
    print(f'private: {priv_path} (mode 0600; keep offline/restricted)')
    print(f'public:  {pub_path}')
    print(f'metadata:{meta}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
