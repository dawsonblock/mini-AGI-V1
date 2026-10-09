"""v16.4.1 canonical artifact closure (REPAIR-101..104).

The v16.4.0 directory hasher (`runtime_closure3.sha256_path`) walked a
tree with `rglob` and *silently skipped* every symlink and every
non-regular file: adding a symlink to an approved adapter directory did
not change its digest, so a substituted artifact could pass closure
checks. It also could not say which files it had authorized — only a
single opaque digest — so "unexpected file" was indistinguishable from
"approved file" to any downstream check.

This module is the single closure implementation used by the runtime:

  * `close_tree()` walks a file or directory once and returns a
    `TreeClosure`: the ordered (path, size, sha256) entries plus the
    canonical closure digest. Symbolic links and unsupported special
    files (fifo/socket/device) are REFUSED, not skipped. Directory
    structure changes (adding/removing any regular file) change the
    digest.
  * `verify_entries()` compares a measured closure against an expected
    file listing (from a signed manifest): missing, unexpected,
    length-mismatched, or content-mismatched entries all fail.
  * `copy_closure()` copies a tree while hashing the bytes it writes
    (single pass), so the staged copy is the bytes that were verified —
    the time-of-check/time-of-use window on the source is closed.

Digest compatibility: for a tree with no symlinks or special files the
digest is byte-identical to the v16.4.0 `sha256_path` digest (same
sorted (relpath, size, sha256) rows), so evidence recorded before this
release still verifies. Trees that previously hashed *successfully but
incompletely* (i.e. contained skipped symlinks) are now refused —
exactly the case the audit flagged.

`resolve_symlinks=True` exists for one documented case: Hugging Face
cache snapshot directories are made of symlinks into the blob store
(see `scripts/run_campaign1.py::_snapshot_files`). In that mode a
symlink must resolve to a regular file, the target's bytes are what is
measured, and the policy is recorded in the closure so a manifest can
bind which policy produced the digest.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from egai.common.canonical import digest, validate_digest

CHUNK = 1 << 20


class ArtifactClosureError(ValueError):
    """Artifact tree fails closure checks — never serve it."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _safe_rel(rel: str) -> str:
    if not rel or rel.startswith(("/", "\\")) or "\\" in rel:
        raise ArtifactClosureError(f"unsafe artifact entry path: {rel!r}")
    parts = rel.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ArtifactClosureError(f"unsafe artifact entry path: {rel!r}")
    return rel


@dataclass(frozen=True)
class TreeEntry:
    """One authorized regular file: relative path, byte length, content
    digest. Constructing one validates the path cannot escape the tree."""

    path: str
    size: int
    sha256: str

    def __post_init__(self):
        _safe_rel(self.path)
        if self.size < 0:
            raise ValueError("artifact entry size must be >= 0")
        validate_digest(self.sha256)

    def to_doc(self) -> dict:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_doc(cls, doc) -> "TreeEntry":
        if not isinstance(doc, dict):
            raise ArtifactClosureError("artifact entry must be an object")
        try:
            return cls(path=str(doc["path"]), size=int(doc["size"]),
                       sha256=str(doc["sha256"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactClosureError(
                f"malformed artifact entry {doc!r}: {exc}") from exc


@dataclass(frozen=True)
class TreeClosure:
    """Canonical closure of a measured artifact path."""

    root: str
    kind: str                       # "file" | "dir"
    digest: str                     # canonical closure digest
    entries: tuple[TreeEntry, ...]
    symlink_policy: str = "reject"  # "reject" | "resolve"
    schema: str = "mini-agi-v16.4.1-tree-closure-v1"

    def __post_init__(self):
        validate_digest(self.digest)
        if self.kind not in ("file", "dir"):
            raise ValueError("closure kind must be file|dir")
        if self.symlink_policy not in ("reject", "resolve"):
            raise ValueError("symlink_policy must be reject|resolve")
        seen = set()
        for e in self.entries:
            if e.path in seen:
                raise ValueError(f"duplicate artifact entry: {e.path}")
            seen.add(e.path)

    @property
    def total_bytes(self) -> int:
        return sum(e.size for e in self.entries)

    def entry_map(self) -> dict[str, TreeEntry]:
        return {e.path: e for e in self.entries}

    def manifest(self) -> list[dict]:
        """JSON-ready listing for binding inside a signed manifest."""
        return [e.to_doc() for e in self.entries]


def _digest_rows(entries: Iterable[TreeEntry]) -> str:
    rows = [(e.path, e.size, e.sha256) for e in entries]
    return digest(rows)


def _walk(root: Path, *, resolve_symlinks: bool, rel: str = "",
          depth: int = 0) -> list[TreeEntry]:
    if depth > 64:
        raise ArtifactClosureError(f"artifact tree too deep at {root}")
    out: list[TreeEntry] = []
    try:
        with os.scandir(root) as it:
            items = sorted(it, key=lambda e: e.name)
    except OSError as exc:
        raise ArtifactClosureError(
            f"cannot read artifact directory {root}: {exc}") from exc
    for e in items:
        relpath = f"{rel}/{e.name}" if rel else e.name
        if e.is_symlink():
            if not resolve_symlinks:
                raise ArtifactClosureError(
                    "symbolic link inside authorized artifact tree "
                    f"(refused, not skipped): {relpath}")
            target = Path(e.path).resolve()
            if not target.is_file():
                raise ArtifactClosureError(
                    f"symbolic link does not resolve to a regular file: "
                    f"{relpath}")
            out.append(TreeEntry(relpath, target.stat().st_size,
                                 sha256_file(target)))
        elif e.is_dir(follow_symlinks=False):
            out.extend(_walk(Path(e.path), resolve_symlinks=resolve_symlinks,
                             rel=relpath, depth=depth + 1))
        elif e.is_file(follow_symlinks=False):
            p = Path(e.path)
            out.append(TreeEntry(relpath, p.stat().st_size, sha256_file(p)))
        else:
            raise ArtifactClosureError(
                f"unsupported special file in authorized artifact tree: "
                f"{relpath}")
    return out


def close_tree(path, *, resolve_symlinks: bool = False) -> TreeClosure:
    """Measure a file or directory exactly: reject symlinks/special
    files, bind every regular file's path, length, and content."""
    p = Path(path)
    if p.is_symlink():
        raise ArtifactClosureError(
            f"artifact path is a symbolic link: {p}")
    if p.is_file():
        entry = TreeEntry(p.name, p.stat().st_size, sha256_file(p))
        return TreeClosure(root=str(p), kind="file",
                           digest=entry.sha256, entries=(entry,),
                           symlink_policy=("resolve" if resolve_symlinks
                                           else "reject"))
    if not p.is_dir():
        raise ArtifactClosureError(f"artifact path missing: {p}")
    entries = tuple(sorted(_walk(p, resolve_symlinks=resolve_symlinks),
                           key=lambda e: e.path))
    return TreeClosure(root=str(p), kind="dir", digest=_digest_rows(entries),
                       entries=entries,
                       symlink_policy=("resolve" if resolve_symlinks
                                       else "reject"))


def verify_entries(actual: Sequence[TreeEntry],
                   expected: Sequence[TreeEntry], *,
                   what: str = "artifact") -> None:
    """Fail closed unless the measured entries exactly equal the
    authorized listing (no missing, unexpected, resized, or modified
    files)."""
    a = {e.path: e for e in actual}
    x = {e.path: e for e in expected}
    problems: list[str] = []
    missing = sorted(set(x) - set(a))
    unexpected = sorted(set(a) - set(x))
    if missing:
        problems.append(f"missing required files: {missing}")
    if unexpected:
        problems.append(f"unexpected files not covered by the manifest: "
                        f"{unexpected}")
    for rel in sorted(set(a) & set(x)):
        if a[rel].size != x[rel].size:
            problems.append(
                f"{rel}: length {a[rel].size} != authorized {x[rel].size}")
        elif a[rel].sha256 != x[rel].sha256:
            problems.append(f"{rel}: content digest mismatch")
    if problems:
        raise ArtifactClosureError(
            f"{what} closure violated: " + "; ".join(problems))


def expected_from_manifest(entries: Iterable[dict]) -> tuple[TreeEntry, ...]:
    expected = tuple(TreeEntry.from_doc(d) for d in entries)
    seen = set()
    for e in expected:
        if e.path in seen:
            raise ArtifactClosureError(f"duplicate manifest entry: {e.path}")
        seen.add(e.path)
    return expected


# ---------------------------------------------------------------------------
# Tokenizer artifact identity (the signed plan's convention).
#
# The campaign plan binds the tokenizer as the tokenizer-named files
# inside the model snapshot root (`scripts/run_campaign1.py::
# physical_identity_digests`), as a map of relative path -> content
# digest. The runtime re-measures exactly this convention from the
# staged model artifact, so the tokenizer bytes the serving backend
# opens are the bytes the plan bound — and there is no separate
# tokenizer artifact to substitute.
#
# WARNING: changing the name set or the digest shape changes every plan
# digest. Recorded plans must keep verifying — do not edit casually.
# ---------------------------------------------------------------------------

TOKENIZER_ARTIFACT_NAMES = frozenset({
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "vocab.txt",
    "merges.txt", "special_tokens_map.json", "added_tokens.json",
    "chat_template.jinja", "tokenizer.model", "spiece.model"})


def tokenizer_artifact_files(root, *,
                             resolve_symlinks: bool = True
                             ) -> list[tuple[str, Path]]:
    """(relative path, target) for every tokenizer artifact file under
    `root`, enumerated exactly like the plan's identity digest."""
    r = Path(root)
    if not r.is_dir():
        raise ArtifactClosureError(f"tokenizer artifact root missing: {r}")
    out: list[tuple[str, Path]] = []
    for f in sorted(r.rglob("*")):
        rel = f.relative_to(r).as_posix()
        if Path(rel).name not in TOKENIZER_ARTIFACT_NAMES:
            continue
        if f.is_symlink():
            if not resolve_symlinks:
                raise ArtifactClosureError(
                    "symbolic link inside tokenizer artifact tree "
                    f"(refused, not skipped): {rel}")
            target = f.resolve()
            if target.is_file():
                out.append((rel, target))
        elif f.is_file():
            out.append((rel, f))
    return out


def tokenizer_artifact_digest(root, *,
                              resolve_symlinks: bool = True) -> str:
    """Digest of the tokenizer artifact files under `root` — the
    convention the signed campaign plan binds (a map of relative path ->
    content digest). Refuses a root with no tokenizer artifacts."""
    rows = {rel: sha256_file(target)
            for rel, target in tokenizer_artifact_files(
                root, resolve_symlinks=resolve_symlinks)}
    if not rows:
        raise ArtifactClosureError(
            f"no tokenizer artifact files under {root}")
    return digest(rows)


def copy_closure(src, dst, *, resolve_symlinks: bool = False) -> TreeClosure:
    """Copy `src` to `dst` while hashing the bytes actually written.

    Returns the closure of the copy (single pass: what was measured is
    what was staged). The caller compares this digest against the
    authorized digest — a source that changes mid-copy cannot make the
    staged bytes look authorized, because the staged bytes are what the
    digest covers."""
    s = Path(src)
    d = Path(dst)
    if s.is_symlink():
        raise ArtifactClosureError(f"artifact path is a symbolic link: {s}")
    if d.exists():
        raise ArtifactClosureError(f"snapshot destination already exists: {d}")
    if s.is_file():
        d.parent.mkdir(parents=True, exist_ok=True)
        size, sha = _copy_file_hashed(s, d)
        entry = TreeEntry(d.name, size, sha)
        return TreeClosure(root=str(d), kind="file", digest=sha,
                           entries=(entry,),
                           symlink_policy=("resolve" if resolve_symlinks
                                           else "reject"))
    if not s.is_dir():
        raise ArtifactClosureError(f"artifact path missing: {s}")
    entries: list[TreeEntry] = []
    d.mkdir(parents=True)
    _copy_walk(s, d, resolve_symlinks=resolve_symlinks, rel="",
               out=entries)
    entries = sorted(entries, key=lambda e: e.path)
    return TreeClosure(root=str(d), kind="dir", digest=_digest_rows(entries),
                       entries=tuple(entries),
                       symlink_policy=("resolve" if resolve_symlinks
                                       else "reject"))


def _copy_walk(src: Path, dst: Path, *, resolve_symlinks: bool, rel: str,
               out: list[TreeEntry], depth: int = 0) -> None:
    if depth > 64:
        raise ArtifactClosureError(f"artifact tree too deep at {src}")
    with os.scandir(src) as it:
        items = sorted(it, key=lambda e: e.name)
    for e in items:
        relpath = f"{rel}/{e.name}" if rel else e.name
        if e.is_symlink():
            if not resolve_symlinks:
                raise ArtifactClosureError(
                    "symbolic link inside authorized artifact tree "
                    f"(refused, not skipped): {relpath}")
            target = Path(e.path).resolve()
            if not target.is_file():
                raise ArtifactClosureError(
                    f"symbolic link does not resolve to a regular file: "
                    f"{relpath}")
            size, sha = _copy_file_hashed(target, dst / e.name)
        elif e.is_dir(follow_symlinks=False):
            (dst / e.name).mkdir()
            _copy_walk(Path(e.path), dst / e.name,
                       resolve_symlinks=resolve_symlinks, rel=relpath,
                       out=out, depth=depth + 1)
            continue
        elif e.is_file(follow_symlinks=False):
            size, sha = _copy_file_hashed(Path(e.path), dst / e.name)
        else:
            raise ArtifactClosureError(
                f"unsupported special file in authorized artifact tree: "
                f"{relpath}")
        out.append(TreeEntry(relpath, size, sha))


def _copy_file_hashed(src: Path, dst: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(dst, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as out, src.open("rb") as f:
            for block in iter(lambda: f.read(CHUNK), b""):
                h.update(block)
                size += len(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
    except BaseException:
        try:
            dst.unlink()
        except OSError:
            pass
        raise
    return size, "sha256:" + h.hexdigest()


def freeze_tree(root) -> None:
    """Best-effort immutability for a staged snapshot: files read-only,
    directories non-writable. The snapshot is additionally protected by
    the single-pass copy + post-copy verification, so this is defense in
    depth, not the assurance boundary."""
    r = Path(root)
    for dirpath, dirnames, filenames in os.walk(r):
        for name in filenames:
            try:
                os.chmod(Path(dirpath) / name, 0o444)
            except OSError:
                pass
    for dirpath, _dirnames, _filenames in os.walk(r, topdown=False):
        try:
            os.chmod(dirpath, 0o555)
        except OSError:
            pass
