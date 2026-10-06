from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def sha256_tree(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(root)
    h = hashlib.sha256()
    for child in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        rel = child.relative_to(root).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(8, "big"))
        h.update(rel)
        fh = hashlib.sha256()
        with child.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                fh.update(chunk)
        digest = fh.digest()
        h.update(len(digest).to_bytes(8, "big"))
        h.update(digest)
    return "sha256:" + h.hexdigest()


def sha256_path(path: str | Path) -> str:
    p = Path(path)
    if p.is_file():
        return sha256_file(p)
    if p.is_dir():
        return sha256_tree(p)
    raise FileNotFoundError(p)
