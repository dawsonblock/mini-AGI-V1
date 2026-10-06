"""Transactional, immutable checkpoint generations with content-addressed storage.

``weights/`` is the mutable paging workspace. A successful formal save is
committed there, then frozen under ``weights/.checkpoints/generations/``.
``LAST`` always names the most recent committed generation; ``BEST`` names the
lowest-validation generation. Startup of a writable process restores the
workspace from LAST, so a crash cannot mix newer expert writebacks with an
older core/router bundle.

Large artifacts are stored once in a SHA-256 object store. Generations hardlink
only to immutable objects, never to the mutable workspace. This avoids the
aliasing bug of hardlinking workspace files directly while still deduplicating
unchanged experts across checkpoints. Object creation uses reflink/CoW when
available, otherwise a normal copy.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path

from .integrity import COMMIT, fsync_dir, sha256_file, verify_commit, write_commit

META_DIR = ".checkpoints"
GENERATIONS = "generations"
OBJECTS = "objects"
COMMIT_LOG = "commits"
LAST = "LAST"
BEST = "BEST"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def _clone_or_copy(src: Path, dst: Path) -> None:
    """CoW clone where supported, otherwise copy; never hardlink mutable data."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp-{uuid.uuid4().hex}")
    cloned = False
    try:
        import fcntl
        FICLONE = 0x40049409
        with open(src, "rb") as s, open(tmp, "wb") as d:
            fcntl.ioctl(d.fileno(), FICLONE, s.fileno())
        shutil.copystat(src, tmp, follow_symlinks=True)
        cloned = True
    except (OSError, ImportError):
        tmp.unlink(missing_ok=True)
    if not cloned:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    fsync_dir(dst.parent)


def _hardlink_immutable(src: Path, dst: Path) -> None:
    """Link an immutable CAS object into an immutable generation."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp-{uuid.uuid4().hex}")
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    fsync_dir(dst.parent)


def _pointer(root: Path, name: str) -> str | None:
    p = root / META_DIR / name
    if not p.exists():
        return None
    v = p.read_text(encoding="utf-8").strip()
    return v or None


def _object_path(root: Path, digest: str) -> Path:
    return root / META_DIR / OBJECTS / digest[:2] / digest[2:]


def _commit_path(root: Path, generation: str) -> Path | None:
    live = root / META_DIR / GENERATIONS / generation / COMMIT
    if live.exists():
        return live
    tomb = root / META_DIR / COMMIT_LOG / (generation + ".json")
    return tomb if tomb.exists() else None


def _verify_lineage(root: Path, generation: str) -> None:
    """Verify the retained commit-hash chain back to its recorded root.

    Large old generation payloads may be pruned; their tiny COMMIT records are
    retained as tombstones so parent hashes remain verifiable indefinitely.
    """
    cur = generation
    seen = set()
    while cur:
        if cur in seen:
            raise RuntimeError(f"checkpoint lineage cycle at {cur}")
        seen.add(cur)
        cp = _commit_path(root, cur)
        if cp is None:
            raise RuntimeError(f"checkpoint lineage is missing commit for {cur}")
        with open(cp, encoding="utf-8") as f:
            doc = json.load(f)
        meta = doc.get("meta") or {}
        parent = meta.get("parent")
        parent_hash = meta.get("parent_commit_sha256")
        if not parent:
            return
        if not parent_hash:
            # Legacy generation: no hash edge existed yet; stop without
            # inventing an authority relationship that was never recorded.
            return
        pp = _commit_path(root, parent)
        if pp is None or sha256_file(pp) != parent_hash:
            raise RuntimeError(f"checkpoint parent-chain integrity failure at {cur}")
        cur = parent


def _store_object(root: Path, src: Path, digest: str | None = None) -> Path:
    digest = digest or sha256_file(src)
    obj = _object_path(root, digest)
    if obj.exists():
        # Size is a cheap first guard; hash if something looks suspicious.
        if obj.stat().st_size != src.stat().st_size or sha256_file(obj) != digest:
            raise RuntimeError(f"checkpoint object corruption: {obj}")
        return obj
    obj.parent.mkdir(parents=True, exist_ok=True)
    stage = obj.with_name(obj.name + f".tmp-{uuid.uuid4().hex}")
    _clone_or_copy(src, stage)
    if sha256_file(stage) != digest:
        stage.unlink(missing_ok=True)
        raise RuntimeError(f"checkpoint object copy hash mismatch for {src}")
    # Objects are immutable by policy. Read-only mode is an additional tripwire,
    # not the integrity authority; commit hashes remain authoritative.
    try:
        os.chmod(stage, 0o444)
    except OSError:
        pass
    try:
        os.replace(stage, obj)
        fsync_dir(obj.parent)
    except OSError:
        stage.unlink(missing_ok=True)
        if not obj.exists() or sha256_file(obj) != digest:
            raise
    return obj


def resolve_checkpoint(root: str | os.PathLike, which: str = LAST) -> str:
    """Resolve ``weights/`` to an immutable generation when one exists."""
    root = Path(root)
    rel = _pointer(root, which)
    if not rel:
        return str(root)
    p = root / META_DIR / GENERATIONS / rel
    if not p.is_dir():
        raise RuntimeError(f"checkpoint pointer {which} targets missing generation: {p}")
    verify_commit(p, required=True)
    # Bind each generation to the exact parent commits it claims. Historical
    # payloads may be pruned, but their small commit records are retained.
    try:
        _verify_lineage(root, rel)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"cannot validate checkpoint lineage for {p}") from e
    return str(p)


def _manifest_val(path: Path) -> float:
    with open(path / "manifest.json", encoding="utf-8") as f:
        v = json.load(f).get("val")
    return float(v) if v is not None else float("inf")


def _retention(root: Path) -> int:
    try:
        from .config import load, get
        return max(1, int(get(load(), "checkpoint.keep_last", 3) or 3))
    except Exception:
        return 3


def prune_generations(root: str | os.PathLike, keep_last: int | None = None) -> dict:
    """Bound checkpoint growth while always preserving LAST and BEST."""
    root = Path(root)
    gen_root = root / META_DIR / GENERATIONS
    if not gen_root.exists():
        return {"removed": 0, "objects_removed": 0}
    keep_last = _retention(root) if keep_last is None else max(1, int(keep_last))
    dirs = [p for p in gen_root.iterdir()
            if p.is_dir() and not p.name.startswith(".tmp-")]
    dirs.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    protected = {x for x in (_pointer(root, LAST), _pointer(root, BEST)) if x}
    protected.update(p.name for p in dirs[:keep_last])
    removed = 0
    commit_log = root / META_DIR / COMMIT_LOG
    for p in dirs:
        if p.name not in protected:
            cp = p / COMMIT
            if cp.exists():
                commit_log.mkdir(parents=True, exist_ok=True)
                _atomic_text(commit_log / (p.name + ".json"),
                             cp.read_text(encoding="utf-8"))
            shutil.rmtree(p)
            fsync_dir(gen_root)
            removed += 1

    # Generation artifacts normally hardlink to CAS objects. Any object with
    # one remaining link is referenced only by its own object-store pathname.
    obj_root = root / META_DIR / OBJECTS
    objects_removed = 0
    if obj_root.exists():
        for p in list(obj_root.rglob("*")):
            if p.is_file():
                try:
                    if p.stat().st_nlink <= 1:
                        p.unlink()
                        objects_removed += 1
                except FileNotFoundError:
                    pass
        for d in sorted((p for p in obj_root.rglob("*") if p.is_dir()),
                        key=lambda p: len(p.parts), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass
    return {"removed": removed, "objects_removed": objects_removed}


def snapshot_workspace(root: str | os.PathLike, step=None, val=None) -> str:
    """Freeze the coherent root commit and atomically advance LAST/BEST."""
    root = Path(root)
    verify_commit(root, required=True)
    with open(root / COMMIT, encoding="utf-8") as f:
        base = json.load(f)
    base_artifacts = dict(base.get("artifacts", {}))
    if not base_artifacts:
        raise RuntimeError("refusing to snapshot an empty checkpoint commit")

    gen_root = root / META_DIR / GENERATIONS
    gen_root.mkdir(parents=True, exist_ok=True)
    step_part = "none" if step is None else f"{int(step):012d}"
    name = f"g{step_part}-{time.time_ns()}"
    stage = gen_root / (".tmp-" + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    try:
        copied = []
        for rel, rec in base_artifacts.items():
            src = root / rel
            if not src.exists():
                raise RuntimeError(f"workspace changed during checkpoint: missing {rel}")
            # Re-hash before importing: if a writeback raced the commit, fail
            # rather than freezing a mixed generation.
            digest = sha256_file(src)
            if digest != rec.get("sha256"):
                raise RuntimeError(f"workspace changed during checkpoint: digest changed for {rel}")
            obj = _store_object(root, src, digest)
            _hardlink_immutable(obj, stage / rel)
            copied.append(rel)

            side = Path(str(src) + ".sha256")
            if side.exists():
                side_rel = rel + ".sha256"
                side_digest = sha256_file(side)
                side_obj = _store_object(root, side, side_digest)
                _hardlink_immutable(side_obj, stage / side_rel)
                copied.append(side_rel)

        parent = _pointer(root, LAST)
        parent_commit_sha256 = None
        if parent:
            parent_commit = gen_root / parent / COMMIT
            if not parent_commit.exists():
                raise RuntimeError(f"LAST points to generation without commit: {parent}")
            parent_commit_sha256 = sha256_file(parent_commit)
        write_commit(stage, copied, meta={
            "step": step, "val": val, "parent": parent,
            "parent_commit_sha256": parent_commit_sha256,
            "immutable_generation": True,
            "object_store": "sha256",
        })
        verify_commit(stage, required=True)
        final = gen_root / name
        os.replace(stage, final)
        fsync_dir(gen_root)
        _atomic_text(root / META_DIR / LAST, name + "\n")

        best_rel = _pointer(root, BEST)
        best_val = (_manifest_val(gen_root / best_rel)
                    if best_rel else float("inf"))
        current_val = _manifest_val(final)
        if current_val <= best_val:
            _atomic_text(root / META_DIR / BEST, name + "\n")
        prune_generations(root)
        return str(final)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def recover_workspace(root: str | os.PathLike) -> bool:
    """Restore the mutable workspace exactly to LAST.

    Unsaved expert writes left by a crash are deliberately discarded.
    Non-checkpoint files (episodic memory, logs, cold archive) are untouched.
    """
    root = Path(root)
    rel = _pointer(root, LAST)
    if not rel:
        return False
    src = Path(resolve_checkpoint(root, LAST))
    with open(src / COMMIT, encoding="utf-8") as f:
        commit = json.load(f)
    artifacts = list(commit.get("artifacts", {}))

    expected_experts = {
        Path(relp).name for relp in artifacts
        if relp.startswith("experts/") and relp.endswith(".npz")
    }
    ed = root / "experts"
    if ed.exists():
        for p in ed.glob("e*.npz"):
            if p.name not in expected_experts:
                p.unlink(missing_ok=True)
                Path(str(p) + ".sha256").unlink(missing_ok=True)

    primary = []
    for relp in artifacts:
        _clone_or_copy(src / relp, root / relp)
        if not relp.endswith(".sha256"):
            primary.append(relp)
    write_commit(root, primary, meta={
        "recovered_from": rel,
        "workspace": True,
    })
    verify_commit(root, required=True)
    return True


def generation_info(root: str | os.PathLike) -> dict:
    root = Path(root)
    out = {"last": _pointer(root, LAST), "best": _pointer(root, BEST)}
    for key, ptr in list(out.items()):
        if ptr:
            p = root / META_DIR / GENERATIONS / ptr
            out[key + "_path"] = str(p)
            out[key + "_val"] = _manifest_val(p)
    return out
