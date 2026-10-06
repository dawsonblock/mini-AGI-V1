"""Checkpoint and expert-file integrity helpers.

The active weights directory remains the mutable training workspace.  Each
successful checkpoint gets a compact commit record, and expert files carry a
sidecar digest that is refreshed whenever the expert is atomically replaced.
The commit makes mixed bundle generations detectable; the sidecars make silent
expert corruption detectable at page-in time.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

SCHEMA = "miniagi-checkpoint-v2"

DIGEST_SUFFIX = ".sha256"
COMMIT = "COMMIT.json"


def sha256_file(path: str | os.PathLike, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fsync_dir(path: str | os.PathLike) -> None:
    """Best-effort directory fsync after an atomic rename.

    ``os.replace`` makes a name change atomic, but on Unix durability across a
    sudden power loss also requires syncing the directory entry.  Platforms
    that do not permit opening/fsyncing directories simply fall back to the
    file-level guarantees already in place.
    """
    try:
        fd = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_text(path: str | os.PathLike, text: str) -> None:
    path = os.fspath(path)
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(parent)


def write_sidecar(path: str | os.PathLike) -> str:
    digest = sha256_file(path)
    _atomic_text(os.fspath(path) + DIGEST_SUFFIX, digest + "\n")
    return digest


def verify_sidecar(path: str | os.PathLike, required: bool = False) -> bool:
    side = os.fspath(path) + DIGEST_SUFFIX
    if not os.path.exists(side):
        if required:
            raise RuntimeError(f"missing integrity sidecar: {side}")
        return True
    with open(side, encoding="utf-8") as f:
        expected = f.read().strip().split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"integrity failure for {path}: expected sha256 {expected}, got {actual}")
    return True


def _commit_content_digest(doc: dict) -> str:
    base = {k: v for k, v in doc.items() if k != "content_sha256"}
    raw = json.dumps(base, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_commit(root: str | os.PathLike, files: Iterable[str], meta=None) -> dict:
    root = os.fspath(root)
    artifacts = {}
    for rel in files:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            artifacts[rel] = {
                "sha256": sha256_file(p),
                "bytes": int(os.path.getsize(p)),
            }
    doc = {"version": 2, "schema": SCHEMA, "artifacts": artifacts}
    if meta:
        doc["meta"] = dict(meta)
    # This is a digest of the canonical commit body rather than the commit file
    # itself, avoiding self-hash recursion while still detecting metadata edits.
    doc["content_sha256"] = _commit_content_digest(doc)
    _atomic_text(os.path.join(root, COMMIT), json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


def verify_commit(root: str | os.PathLike, required: bool = False) -> bool:
    root = os.fspath(root)
    p = os.path.join(root, COMMIT)
    if not os.path.exists(p):
        if required:
            raise RuntimeError(f"missing checkpoint commit: {p}")
        return True  # legacy checkpoints remain loadable
    with open(p, encoding="utf-8") as f:
        doc = json.load(f)
    if int(doc.get("version", 1)) >= 2:
        if doc.get("schema") != SCHEMA:
            raise RuntimeError(f"unsupported checkpoint commit schema: {doc.get('schema')!r}")
        expected_body = doc.get("content_sha256")
        actual_body = _commit_content_digest(doc)
        if expected_body != actual_body:
            raise RuntimeError(
                f"checkpoint commit metadata integrity failure: expected {expected_body}, got {actual_body}")
    for rel, rec in doc.get("artifacts", {}).items():
        fp = os.path.join(root, rel)
        if not os.path.exists(fp):
            raise RuntimeError(f"checkpoint is incomplete: missing {rel}")
        if int(os.path.getsize(fp)) != int(rec.get("bytes", -1)):
            raise RuntimeError(f"checkpoint size mismatch: {rel}")
        got = sha256_file(fp)
        if got != rec.get("sha256"):
            raise RuntimeError(
                f"checkpoint integrity failure for {rel}: expected {rec.get('sha256')}, got {got}")
    return True


def verify_checkpoint(root: str | os.PathLike, deep: bool = False) -> dict:
    """Verify bundle commit and, optionally, every expert file."""
    root = os.fspath(root)
    verify_commit(root, required=False)
    checked = 0
    if deep:
        ed = Path(root) / "experts"
        if ed.exists():
            for p in sorted(ed.glob("e*.npz")):
                verify_sidecar(p, required=False)
                checked += 1
    return {"ok": True, "root": root, "experts_checked": checked, "deep": bool(deep)}
