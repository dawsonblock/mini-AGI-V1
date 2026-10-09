"""v16.4.1 — canonical artifact closure + immutable snapshots.

Reproduces the audited defects and their repairs:

  * a symlink added to an approved adapter directory did not change the
    v16.4.0 directory digest (it was silently skipped) — now the closure
    refuses the tree outright;
  * an unexpected/unlisted file was indistinguishable from an approved
    one — manifests now bind the complete file listing and every
    missing/unexpected/resized/modified entry fails individually;
  * an artifact could change between verification and load — staging
    copies the bytes it verifies in a single pass, freezes the result,
    and re-verifies immediately before load.
"""
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from minagi.v161.artifact_closure import (  # noqa: E402
    ArtifactClosureError, TreeEntry, close_tree, copy_closure,
    expected_from_manifest, freeze_tree, tokenizer_artifact_digest,
    verify_entries)
from minagi.v161.immutable_snapshot import (  # noqa: E402
    MeasuredSnapshot, SnapshotError, stage_snapshot, verify_snapshot)
from minagi.v161.runtime_closure3 import sha256_path  # noqa: E402


def _adapter(tmp_path, name="adapter"):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 4,
        "lora_alpha": 8, "lora_dropout": 0.0,
        "target_modules": ["c_attn"]}))
    (d / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    return d


def _legacy_digest(root: Path) -> str:
    """The v16.4.0 directory digest, computed independently: sorted
    (relpath, size, sha256) rows, symlinks skipped."""
    rows = []
    for f in sorted(x for x in root.rglob("*")
                    if x.is_file() and not x.is_symlink()):
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        rows.append((f.relative_to(root).as_posix(), f.stat().st_size,
                     "sha256:" + h))
    return digest(rows)


# ---------- digest compatibility ---------------------------------------

def test_clean_tree_digest_matches_v1640_semantics(tmp_path):
    d = _adapter(tmp_path)
    assert sha256_path(d) == _legacy_digest(d)
    closure = close_tree(d)
    assert closure.digest == _legacy_digest(d)
    assert {e.path for e in closure.entries} == {
        "adapter_config.json", "adapter_model.safetensors"}


def test_file_digest_unchanged(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"payload")
    assert sha256_path(p) == "sha256:" + hashlib.sha256(b"payload").hexdigest()


# ---------- tokenizer identity convention (frozen) ----------------------

_LEGACY_TOKENIZER_NAMES = {
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "vocab.txt",
    "merges.txt", "special_tokens_map.json", "added_tokens.json",
    "chat_template.jinja", "tokenizer.model", "spiece.model"}


def _legacy_tokenizer_digest(root: Path) -> str:
    """The plan's tokenizer identity, computed independently (literal
    name set, map of relative path -> content digest, symlinks
    resolved). Recorded plans bind this shape — it must not drift."""
    rows = {}
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root).as_posix()
        if Path(rel).name not in _LEGACY_TOKENIZER_NAMES:
            continue
        real = f.resolve() if f.is_symlink() else f
        if real.is_file():
            rows[rel] = "sha256:" + hashlib.sha256(
                real.read_bytes()).hexdigest()
    return digest(rows)


def test_tokenizer_digest_matches_recorded_plan_convention(tmp_path):
    snap = tmp_path / "snapshot"
    snap.mkdir()
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    for name, data in (("tokenizer.json", b'{"vocab": []}'),
                       ("tokenizer_config.json", b'{"ct": "x"}'),
                       ("vocab.json", b"{}"),
                       ("merges.txt", b"a b\n"),
                       ("config.json", b'{"m": 1}'),
                       ("sub/added_tokens.json", b"[]")):
        blob = blobs / name.replace("/", "_")
        blob.write_bytes(data)
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        (snap / name).symlink_to(blob)
    assert tokenizer_artifact_digest(snap) == _legacy_tokenizer_digest(snap)

    flat = tmp_path / "flat"
    flat.mkdir()
    for name, data in (("tokenizer.json", b'{"vocab": []}'),
                       ("vocab.json", b"{}"), ("README.md", b"docs")):
        (flat / name).write_bytes(data)
    assert tokenizer_artifact_digest(flat) == _legacy_tokenizer_digest(flat)

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.json").write_bytes(b"{}")
    with pytest.raises(ArtifactClosureError, match="no tokenizer artifact"):
        tokenizer_artifact_digest(plain)
    with pytest.raises(ArtifactClosureError, match="symbolic link"):
        tokenizer_artifact_digest(snap, resolve_symlinks=False)


# ---------- symlinks are refused, not skipped --------------------------

def test_symlink_added_to_adapter_tree_is_refused(tmp_path):
    d = _adapter(tmp_path)
    (d / "extra_link.safetensors").symlink_to(d / "adapter_model.safetensors")
    # v16.4.0 silently skipped it: the digest was unchanged
    assert _legacy_digest(d) == _legacy_digest(_adapter(tmp_path, "clean"))
    with pytest.raises(ArtifactClosureError, match="symbolic link"):
        sha256_path(d)
    with pytest.raises(ArtifactClosureError, match="symbolic link"):
        close_tree(d)


def test_symlinked_file_masquerading_as_adapter_refused(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    real = tmp_path / "elsewhere.safetensors"
    real.write_bytes(b"weights")
    (d / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "target_modules": ["c_attn"], "r": 4,
        "lora_alpha": 8, "lora_dropout": 0.0}))
    (d / "adapter_model.safetensors").symlink_to(real)
    with pytest.raises(ArtifactClosureError, match="symbolic link"):
        close_tree(d)


def test_symlinked_directory_refused(tmp_path):
    d = _adapter(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "smuggled.bin").write_bytes(b"x")
    (d / "sub").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactClosureError, match="symbolic link"):
        close_tree(d)


def test_top_level_symlink_refused(tmp_path):
    d = _adapter(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(d, target_is_directory=True)
    with pytest.raises(ArtifactClosureError, match="symbolic link"):
        sha256_path(link)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX fifo required")
def test_special_file_refused(tmp_path):
    d = _adapter(tmp_path)
    os.mkfifo(d / "fifo")
    with pytest.raises(ArtifactClosureError, match="special file"):
        close_tree(d)


# ---------- expected listing enforcement --------------------------------

def test_missing_required_file_refused(tmp_path):
    d = _adapter(tmp_path)
    expected = expected_from_manifest(close_tree(d).manifest())
    (d / "adapter_model.safetensors").unlink()
    with pytest.raises(ArtifactClosureError, match="missing required files"):
        verify_entries(close_tree(d).entries, expected)


def test_unexpected_file_refused(tmp_path):
    d = _adapter(tmp_path)
    expected = expected_from_manifest(close_tree(d).manifest())
    (d / "smuggled.bin").write_bytes(b"payload")
    with pytest.raises(ArtifactClosureError, match="unexpected files"):
        verify_entries(close_tree(d).entries, expected)


def test_modified_and_resized_entries_refused(tmp_path):
    d = _adapter(tmp_path)
    expected = expected_from_manifest(close_tree(d).manifest())
    (d / "adapter_model.safetensors").write_bytes(b"adapter-weightS")
    with pytest.raises(ArtifactClosureError, match="content digest mismatch"):
        verify_entries(close_tree(d).entries, expected)
    (d / "adapter_model.safetensors").write_bytes(b"short")
    with pytest.raises(ArtifactClosureError, match="length"):
        verify_entries(close_tree(d).entries, expected)


def test_manifest_path_traversal_entry_refused():
    for bad in ("../escape", "/abs/path", "a/../../b", "", "a//b", "."):
        with pytest.raises((ArtifactClosureError, ValueError)):
            TreeEntry(path=bad, size=0, sha256="sha256:" + "0" * 64)


def test_duplicate_manifest_entry_refused(tmp_path):
    d = _adapter(tmp_path)
    doc = close_tree(d).manifest()[0]
    with pytest.raises(ArtifactClosureError, match="duplicate"):
        expected_from_manifest([doc, dict(doc)])


# ---------- immutable snapshot / TOCTOU ---------------------------------

def test_stage_verifies_single_pass_and_freezes(tmp_path):
    d = _adapter(tmp_path)
    authorized = close_tree(d).digest
    snap = stage_snapshot(tmp_path / "snap", {"adapter": d},
                          expected_digests={"adapter": authorized},
                          manifest_digest="sha256:" + "7" * 64)
    assert snap.artifact_digests == (("adapter", authorized),)
    staged = snap.path("adapter") / "adapter_config.json"
    assert staged.is_file()
    assert not (staged.stat().st_mode & stat.S_IWUSR), \
        "staged files must be frozen read-only"
    verify_snapshot(snap)


def test_source_mutation_after_verification_is_refused(tmp_path):
    """TOCTOU: the digest is verified, then the source is swapped
    before staging — the staged bytes cannot match the authorized
    digest, so the snapshot is refused."""
    d = _adapter(tmp_path)
    authorized = close_tree(d).digest
    (d / "adapter_model.safetensors").write_bytes(b"substituted-bytes")
    with pytest.raises(SnapshotError, match="changed between verification"):
        stage_snapshot(tmp_path / "snap", {"adapter": d},
                       expected_digests={"adapter": authorized},
                       manifest_digest="sha256:" + "7" * 64)
    assert not (tmp_path / "snap").exists(), \
        "a refused snapshot must not be left behind"


def test_tampering_with_staged_snapshot_is_detected(tmp_path):
    d = _adapter(tmp_path)
    snap = stage_snapshot(tmp_path / "snap", {"adapter": d},
                          expected_digests={"adapter": close_tree(d).digest},
                          manifest_digest="sha256:" + "7" * 64)
    target = snap.path("adapter") / "adapter_model.safetensors"
    os.chmod(snap.path("adapter"), 0o755)
    os.chmod(target, 0o644)
    target.write_bytes(b"tampered")
    with pytest.raises(SnapshotError, match="no longer matches"):
        verify_snapshot(snap)


def test_approved_snapshot_cannot_be_forged(tmp_path):
    with pytest.raises(SnapshotError, match="stage_snapshot"):
        MeasuredSnapshot(root=tmp_path, digests={"adapter": "sha256:" + "1" * 64},
                         manifest_digest="sha256:" + "7" * 64, closures={})


def test_snapshot_refuses_existing_destination(tmp_path):
    d = _adapter(tmp_path)
    dest = tmp_path / "snap"
    dest.mkdir()
    with pytest.raises(SnapshotError, match="already exists"):
        stage_snapshot(dest, {"adapter": d},
                       expected_digests={"adapter": close_tree(d).digest},
                       manifest_digest="sha256:" + "7" * 64)


def test_snapshot_requires_every_authorized_artifact(tmp_path):
    d = _adapter(tmp_path)
    with pytest.raises(SnapshotError, match="not supplied for staging"):
        stage_snapshot(tmp_path / "snap", {"adapter": d},
                       expected_digests={
                           "adapter": close_tree(d).digest,
                           "model": "sha256:" + "3" * 64},
                       manifest_digest="sha256:" + "7" * 64)


def test_snapshot_refuses_unauthorized_artifact_names(tmp_path):
    """An artifact name the manifest did not authorize cannot be
    staged, and a refused staging leaves nothing behind."""
    d = _adapter(tmp_path)
    with pytest.raises(SnapshotError, match="not authorized"):
        stage_snapshot(tmp_path / "snap", {"adapter": d, "extra": d},
                       expected_digests={"adapter": close_tree(d).digest},
                       manifest_digest="sha256:" + "7" * 64)
    assert not (tmp_path / "snap").exists()


def test_copy_closure_refuses_symlinked_source(tmp_path):
    d = _adapter(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(d, target_is_directory=True)
    with pytest.raises(ArtifactClosureError, match="symbolic link"):
        copy_closure(link, tmp_path / "out")


def test_hf_cache_style_symlinks_allowed_only_when_policy_says_so(tmp_path):
    """HF snapshot dirs are all symlinks into the blob store; the
    documented resolve policy measures the targets, while the strict
    default refuses them."""
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "weights").write_bytes(b"real-weights")
    snapdir = tmp_path / "snapshot"
    snapdir.mkdir()
    (snapdir / "model.safetensors").symlink_to(blobs / "weights")
    with pytest.raises(ArtifactClosureError, match="symbolic link"):
        close_tree(snapdir)
    resolved = close_tree(snapdir, resolve_symlinks=True)
    assert resolved.entries[0].sha256 == "sha256:" + hashlib.sha256(
        b"real-weights").hexdigest()
    assert resolved.symlink_policy == "resolve"


def test_freeze_tree_is_idempotent(tmp_path):
    d = _adapter(tmp_path)
    freeze_tree(d)
    freeze_tree(d)
    assert not ((d / "adapter_config.json").stat().st_mode & stat.S_IWUSR)
