"""Dataset and experiment provenance helpers.

Validation leakage is easy to create accidentally when copies of one document
have different filenames.  v4.1 assigns train/validation membership from the
*content digest*, not the pathname. Exact duplicates therefore cannot cross the
split boundary. The packer can also deduplicate them and emits a source manifest
that binds each packed lane to the files, tokenizer and split rule that created
it.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def content_split(digest: str, val_permille: int = 5) -> str:
    v = int(val_permille)
    if not 0 <= v <= 1000:
        raise ValueError("val_permille must be in [0,1000]")
    # Use 64 digest bits so the assignment is stable across platforms and path
    # layouts while retaining more than enough entropy for corpus splitting.
    bucket = int(str(digest)[:16], 16) % 1000
    return "val" if bucket < v else "train"


def source_records(files, root, val_permille=5, deduplicate=True):
    root = Path(root).resolve()
    seen = {}
    out = []
    for fp in sorted(map(Path, files), key=lambda p: str(p)):
        digest = sha256_file(fp)
        duplicate_of = seen.get(digest)
        rel = os.path.relpath(fp.resolve(), root)
        rec = {
            "path": rel,
            "sha256": digest,
            "bytes": int(fp.stat().st_size),
            "split": content_split(digest, val_permille),
            "duplicate_of": duplicate_of,
            "included": not (deduplicate and duplicate_of is not None),
        }
        out.append(rec)
        if duplicate_of is None:
            seen[digest] = rel
    return out


def canonical_digest(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def source_tree_digest(root: str | os.PathLike, suffixes=(".py", ".yaml", ".toml")):
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix not in suffixes:
            continue
        if any(x in p.parts for x in (".git", "__pycache__", ".checkpoints")):
            continue
        rows.append({"path": str(p.relative_to(root)), "sha256": sha256_file(p)})
    return canonical_digest(rows), rows


def build_run_manifest(root: str | os.PathLike, *, config_path=None,
                       data_dirs=(), tokenizer_path=None, extra=None):
    root = Path(root)
    code_digest, code_files = source_tree_digest(root)
    data = []
    for d in data_dirs:
        p = Path(d)
        for name in ("meta.json", "source_manifest.json", "tokenizer.json"):
            q = p / name
            if q.is_file():
                data.append({"path": str(q), "sha256": sha256_file(q)})
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch
        env.update({
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda": getattr(torch.version, "cuda", None),
        })
    except Exception:
        env["torch"] = None
    doc = {
        "schema": 1,
        "created_at": time.time(),
        "source_tree_sha256": code_digest,
        "source_files": code_files,
        "config": ({"path": str(config_path), "sha256": sha256_file(config_path)}
                   if config_path else None),
        "tokenizer": ({"path": str(tokenizer_path), "sha256": sha256_file(tokenizer_path)}
                      if tokenizer_path else None),
        "data_artifacts": data,
        "environment": env,
        "extra": extra or {},
    }
    doc["manifest_sha256"] = canonical_digest({k: v for k, v in doc.items()
                                                if k != "manifest_sha256"})
    return doc
