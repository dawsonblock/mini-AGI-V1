"""v16.4.3 — path containment and caller-controlled writes (SEC-204).

Adversarial battery: no client-provided string may redirect privileged
staging, receipt persistence, or cleanup outside the authorized storage
root; activation ids and receipt destinations are server-generated.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v161"))

from minagi.runtime.access_policy import (  # noqa: E402
    PolicyRefused, contained_child, opaque_id, secure_dir)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)


# ---------- contained_child ------------------------------------------------

def test_traversal_refused(tmp_path):
    root = tmp_path / "snapshots"
    root.mkdir()
    for name in ("../../outside", "..%2f..%2fescape",
                 "a/b", ".", "..", "a\\b"):
        with pytest.raises(PolicyRefused):
            contained_child(root, name)


def test_absolute_path_refused(tmp_path):
    root = tmp_path / "snapshots"
    root.mkdir()
    with pytest.raises(PolicyRefused):
        contained_child(root, str(tmp_path / "elsewhere" / ("aa" * 8)))


def test_unicode_and_nonhex_names_refused(tmp_path):
    root = tmp_path / "snapshots"
    root.mkdir()
    for name in ("café" * 8, "ABCDEF12" * 4, "g" * 32,
                 "token with space"):
        with pytest.raises(PolicyRefused):
            contained_child(root, name)


def test_symlinked_parent_refused(tmp_path):
    real = tmp_path / "snapshots"
    real.mkdir()
    evil = tmp_path / "evil"
    evil.mkdir()
    link = tmp_path / "snapshots-link"
    try:
        link.symlink_to(evil)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(PolicyRefused):
        contained_child(link, "aa" * 16)


def test_existing_symlink_destination_refused(tmp_path):
    root = tmp_path / "snapshots"
    root.mkdir()
    name = "ab" * 16
    outside = tmp_path / "loot"
    outside.mkdir()
    try:
        (root / name).symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(PolicyRefused):
        contained_child(root, name)


def test_legitimate_opaque_id_contained(tmp_path):
    root = tmp_path / "snapshots"
    root.mkdir()
    name = opaque_id()
    dest = contained_child(root, name)
    assert dest == root / name
    dest.mkdir()
    assert dest.is_dir()


# ---------- secure_dir ------------------------------------------------------

def test_secure_dir_enforces_private_ownership(tmp_path):
    d = secure_dir(tmp_path / "state")
    import stat
    assert stat.S_IMODE(d.lstat().st_mode) & 0o077 == 0
    # a symlinked dir refuses
    link = tmp_path / "link"
    try:
        link.symlink_to(d)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(PolicyRefused):
        secure_dir(link)


# ---------- launcher-level containment --------------------------------------

def test_launch_activation_id_is_opaque_and_contained(tmp_path):
    """The activation id is server-generated; campaign/seed are
    metadata and never appear in the snapshot path (SEC-204)."""
    from test_v1641_trusted_launcher import (  # noqa: E402
        FakeBackend, _build_chain, _launcher, _request)
    chain = _build_chain(tmp_path)
    launcher = _launcher(chain, tmp_path)
    result = launcher.launch(
        _request(chain, campaign_id="../../evil/../x",
                 seed="seed-0"),
        FakeBackend())
    assert len(result.activation_id) == 32
    assert result.snapshot.root.parent.resolve() == (
        launcher.snapshot_root.resolve())
    assert result.activation_id in result.snapshot.root.name
    assert Path(result.receipt_path).parent.resolve() == \
        launcher.receipts_dir.resolve()


def test_client_cannot_pick_storage_names(tmp_path):
    """A caller may not select the activation id, snapshot dir, or
    receipt file — all are derived inside the trusted side."""
    from test_v1641_trusted_launcher import (  # noqa: E402
        FakeBackend, _build_chain, _launcher, _request)
    chain = _build_chain(tmp_path)
    launcher = _launcher(chain, tmp_path)
    r1 = launcher.launch(_request(chain), FakeBackend())
    r2 = launcher.launch(_request(chain), FakeBackend())
    assert r1.activation_id != r2.activation_id
    assert Path(r1.receipt_path).parent == Path(r2.receipt_path).parent
