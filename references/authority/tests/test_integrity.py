from pathlib import Path
import pytest

from minagi.integrity import write_commit, write_sidecar, verify_commit, verify_sidecar


def test_sidecar_detects_corruption(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    write_sidecar(p)
    assert verify_sidecar(p)
    p.write_bytes(b"abd")
    with pytest.raises(RuntimeError):
        verify_sidecar(p)


def test_commit_detects_mixed_bundle(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "core.npz").write_bytes(b"core")
    write_commit(tmp_path, ["manifest.json", "core.npz"])
    assert verify_commit(tmp_path)
    (tmp_path / "core.npz").write_bytes(b"different")
    with pytest.raises(RuntimeError):
        verify_commit(tmp_path)
