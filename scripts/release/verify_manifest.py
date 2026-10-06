#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "release" / "source-manifest.sha256"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not MANIFEST.is_file():
        print("manifest missing", file=sys.stderr)
        return 1
    failures = []
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        path = ROOT / rel
        if not path.is_file():
            failures.append(f"missing: {rel}")
        elif digest(path) != expected:
            failures.append(f"digest mismatch: {rel}")
    if failures:
        print("source manifest verification: FAIL", file=sys.stderr)
        for failure in failures:
            print(f" - {failure}", file=sys.stderr)
        return 1
    print("source manifest verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
