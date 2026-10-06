import json
from pathlib import Path

import pytest

from minagi.checkpointing import (BEST, LAST, generation_info,
                                  recover_workspace, resolve_checkpoint,
                                  snapshot_workspace)
from minagi.integrity import verify_commit, write_commit, write_sidecar


def _workspace(root: Path, val: float, payload: bytes):
    (root / "experts").mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({"val": val, "step": 1}))
    (root / "core.npz").write_bytes(b"core-" + payload)
    (root / "routers.npz").write_bytes(b"routers-" + payload)
    (root / "experts" / "e00000.npz").write_bytes(b"expert-" + payload)
    for p in (root / "manifest.json", root / "core.npz", root / "routers.npz",
              root / "experts" / "e00000.npz"):
        write_sidecar(p)
    write_commit(root, ["manifest.json", "core.npz", "routers.npz",
                        "experts/e00000.npz"], meta={"val": val})


def test_immutable_generation_survives_dirty_workspace(tmp_path):
    _workspace(tmp_path, 1.0, b"a")
    snap = Path(snapshot_workspace(tmp_path, step=1, val=1.0))
    assert verify_commit(snap, required=True)

    # A later expert writeback invalidates the mutable root commit, exactly the
    # crash/mixed-generation case the generation store is meant to survive.
    (tmp_path / "experts" / "e00000.npz").write_bytes(b"dirty")
    with pytest.raises(RuntimeError):
        verify_commit(tmp_path, required=True)
    assert Path(resolve_checkpoint(tmp_path, LAST)) == snap

    assert recover_workspace(tmp_path)
    assert verify_commit(tmp_path, required=True)
    assert (tmp_path / "experts" / "e00000.npz").read_bytes() == b"expert-a"


def test_best_and_last_are_distinct(tmp_path):
    _workspace(tmp_path, 1.0, b"a")
    first = snapshot_workspace(tmp_path, step=1, val=1.0)

    _workspace(tmp_path, 1.2, b"b")
    second = snapshot_workspace(tmp_path, step=2, val=1.2)
    info = generation_info(tmp_path)
    assert Path(resolve_checkpoint(tmp_path, LAST)) == Path(second)
    assert Path(resolve_checkpoint(tmp_path, BEST)) == Path(first)
    assert info["best_val"] == 1.0

    _workspace(tmp_path, 0.8, b"c")
    third = snapshot_workspace(tmp_path, step=3, val=0.8)
    assert Path(resolve_checkpoint(tmp_path, BEST)) == Path(third)


def test_generation_retention_preserves_best_plus_recent(tmp_path):
    for step, val in enumerate([1.0, 1.1, 1.2, 1.3, 1.4], start=1):
        _workspace(tmp_path, val, str(step).encode())
        snapshot_workspace(tmp_path, step=step, val=val)
    gen_root = tmp_path / ".checkpoints" / "generations"
    gens = [p for p in gen_root.iterdir() if p.is_dir() and not p.name.startswith(".tmp")]
    # keep_last defaults to 3, plus the older BEST generation.
    assert len(gens) <= 4
    assert Path(resolve_checkpoint(tmp_path, BEST)).exists()
    assert Path(resolve_checkpoint(tmp_path, LAST)).exists()


def test_generation_commit_binds_parent_hash(tmp_path):
    from minagi.integrity import COMMIT, sha256_file
    _workspace(tmp_path, 1.0, b"a")
    first = Path(snapshot_workspace(tmp_path, step=1, val=1.0))
    first_hash = sha256_file(first / COMMIT)
    _workspace(tmp_path, 0.9, b"b")
    second = Path(snapshot_workspace(tmp_path, step=2, val=0.9))
    doc = json.loads((second / COMMIT).read_text())
    assert doc["version"] == 2
    assert doc["meta"]["parent_commit_sha256"] == first_hash
