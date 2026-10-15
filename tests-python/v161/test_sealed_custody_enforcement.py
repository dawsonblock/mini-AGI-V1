"""Campaign 3B.2 Step 19 & 21 — OS-enforced sealed evaluator custody.

Verifies that:
  1. A same-UID research/evaluator arrangement is NEVER classified as
     independently isolated merely because the sealed file is mode 0600.
  2. Independent isolation requires all three physical controls:
     owner-only mode (0600), path outside research storage, AND a
     distinct OS identity (evaluator uid != research storage uid).
  3. Without relying on MINAGI_RESEARCH_WORKER, a path inside research
     storage or a group/world-readable file is refused by load_sealed.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))


def _fixture():
    spec = importlib.util.spec_from_file_location(
        "sealed_confirmation_fixture",
        Path(__file__).parent / "test_sealed_confirmation_execution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_F = _fixture()
Custody, TOOL = _F.Custody, _F.TOOL


def _write_sealed(c, path: Path, *, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, sort_keys=True)
                              for r in c.rows) + "\n")
    path.chmod(mode)
    return path


def test_same_uid_mode_0600_is_not_classified_as_independent_isolation(
        tmp_path):
    """Step 19: a same-UID research/evaluator arrangement must not be
    classified as independently isolated merely because a file is 0600."""
    c = Custody(tmp_path)
    sealed = _write_sealed(c, tmp_path / "eval" / "sealed.jsonl", mode=0o600)
    report = TOOL.sealed_custody_report(sealed, storage=c.storage, environ={})
    assert report["permissions_enforced"] is True
    assert report["distinct_os_identity_enforced"] is False
    assert report["independent_isolation_established"] is False
    assert "Same-UID" in report["statement"]


def test_distinct_uid_plus_0600_plus_outside_establishes_isolation(tmp_path):
    c = Custody(tmp_path)
    sealed = _write_sealed(c, tmp_path / "eval" / "sealed.jsonl", mode=0o600)
    report = TOOL.sealed_custody_report(
        sealed, storage=c.storage, environ={},
        research_uid=os.geteuid() + 1000)
    assert report["permissions_enforced"] is True
    assert report["distinct_os_identity_enforced"] is True
    assert report["independent_isolation_established"] is True


def test_access_boundary_without_relying_on_env_marker(tmp_path):
    """Step 21: even when MINAGI_RESEARCH_WORKER is unset, load_sealed
    enforces the filesystem path and mode boundaries."""
    c = Custody(tmp_path)
    in_storage = _write_sealed(c, c.storage / "leaked.jsonl", mode=0o600)
    with pytest.raises(TOOL.ConfirmationRefused, match="evaluator custody"):
        TOOL.load_sealed(in_storage, storage=c.storage, environ={})

    world_readable = _write_sealed(
        c, tmp_path / "eval" / "world.jsonl", mode=0o644)
    with pytest.raises(TOOL.ConfirmationRefused,
                       match="readable beyond its owner"):
        TOOL.load_sealed(world_readable, storage=c.storage, environ={})
