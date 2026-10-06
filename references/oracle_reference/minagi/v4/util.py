from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def tree_digest(path: str | Path) -> str:
    p=Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.is_file():
        return sha256_file(p)
    rows=[]
    for f in sorted(p.rglob("*")):
        if not f.is_file():
            continue
        if any(part in {"__pycache__", ".git", ".pytest_cache"} for part in f.parts):
            continue
        rel=f.relative_to(p).as_posix()
        rows.append((rel, f.stat().st_size, sha256_file(f)))
    return sha256_bytes(canonical_json(rows).encode())
