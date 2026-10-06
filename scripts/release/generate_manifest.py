#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "release" / "source-manifest.sha256"
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache"}
SKIP_PREFIXES = ("build",)


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in SKIP_PARTS for part in rel.parts):
        return False
    if any(part.startswith(SKIP_PREFIXES) for part in rel.parts):
        return False
    if path == OUT:
        return False
    return path.is_file()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_PARTS and not d.startswith(SKIP_PREFIXES)
        ]
        base = Path(dirpath)
        files.extend(base / name for name in filenames)
    rows = [
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in sorted(files)
        if include(path)
    ]
    OUT.write_text("\n".join(rows) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} with {len(rows)} files")


if __name__ == "__main__":
    main()
