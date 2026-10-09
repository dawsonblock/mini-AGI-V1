#!/usr/bin/env python3
"""Deterministic release ZIP builder (v16.4.1).

Builds a byte-reproducible archive of the release tree: entries sorted
by path, fixed timestamps (1980-01-01), fixed permissions, generated
material (.git, __pycache__, .pytest_cache, build*, *.egg-info,
compiled objects) and symlinks excluded. Run it after
`scripts/validation/reissue_release.py` so the archive carries the
signed manifest, signature, and public key:

    python scripts/release/build_zip.py --out ~/Downloads/Runtime-Security-Closure.zip

The printed sha256 is the archive hash to record in release notes; it
is reproducible for the same tree and the same Python/zlib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

BANNED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git",
                "dist"}
BANNED_SUFFIXES = {".pyc", ".pyo", ".whl", ".o", ".a", ".so", ".dylib"}


def included(rel: Path) -> bool:
    """Match scripts/verify_release.py: gitignored top-level build*
    directories and caches are excluded; files merely *named* build_*.py
    are release-controlled source."""
    parts = rel.parts
    if any(p in BANNED_PARTS or p.endswith(".egg-info") for p in parts):
        return False
    if parts and parts[0].startswith("build"):
        return False
    return rel.suffix not in BANNED_SUFFIXES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None,
                    help="top-level folder inside the archive")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    name = args.name or root.name
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root)
        if included(rel):
            rows.append((rel.as_posix(), p))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        for rel, p in rows:
            info = zipfile.ZipInfo(f"{name}/{rel}", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            mode = 0o755 if p.suffix == ".sh" else 0o644
            info.external_attr = (0o100000 | mode) << 16
            z.writestr(info, p.read_bytes())
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(json.dumps({"zip": str(out), "entries": len(rows),
                      "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
