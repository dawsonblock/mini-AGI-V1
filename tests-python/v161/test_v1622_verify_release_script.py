"""FIX-006 — adversarial release-verifier suite.

Exercises the failure paths of `scripts/verify_release.py` against a
signed fixture tree: tampered files, extra/missing files, corrupted
manifests, stale or missing attestation metadata, version-identity
disagreement, and forged signatures must all be refused with the
documented exit codes. The plan's Phase-1 gate lists "corrupted
manifests" among the malicious-input cases; this is that test.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: E402
    Ed25519PrivateKey)

VERIFIER = ROOT / "scripts" / "verify_release.py"


def _canonical(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(tree: Path, manifest: dict):
    (tree / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _build_tree(tmp_path: Path):
    """A minimal signed fixture tree that verifies cleanly. Returns
    (tree, trusted_pub_path, signing_key)."""
    tree = tmp_path / "release"
    tree.mkdir()
    (tree / "a.txt").write_text("alpha\n")
    (tree / "sub").mkdir()
    (tree / "sub" / "b.txt").write_text("beta\n")
    (tree / "VERSION").write_text("9.9.9-test\n")

    key = Ed25519PrivateKey.generate()
    pub_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo)

    files = {p.relative_to(tree).as_posix(): _sha_file(p)
             for p in sorted(tree.rglob("*")) if p.is_file()}
    manifest = {"schema_version": 1, "hash_algorithm": "sha256",
                "release": "mini-AGI-v9.9.9-Test", "files": files}
    _write_manifest(tree, manifest)
    (tree / "RELEASE_SIGNATURE.bin").write_bytes(
        key.sign(_canonical(manifest)))
    (tree / "RELEASE_PUBLIC_KEY.pem").write_bytes(pub_pem)

    digest = hashlib.sha256(_canonical(manifest)).hexdigest()
    attestation = {
        "release": "mini-AGI-v9.9.9-Test",
        "manifest_sha256": digest,
        "reissued_utc": "2026-10-08T00:00:00+00:00",
        "reissue_history": [
            {"manifest_sha256": digest,
             "utc": "2026-10-08T00:00:00+00:00",
             "reason": "fixture"}],
        "status": "PASS_TEST_FIXTURE",
    }
    (tree / "RELEASE_ATTESTATION.json").write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n")
    pub_path = tmp_path / "trusted.pem"
    pub_path.write_bytes(pub_pem)
    return tree, pub_path, key


def _verify(tree: Path, pub_path: Path):
    proc = subprocess.run(
        [sys.executable, str(VERIFIER), "--root", str(tree),
         "--trusted-key", str(pub_path)],
        capture_output=True, text=True)
    try:
        doc = json.loads(proc.stdout)
    except ValueError:
        doc = {"raw": proc.stdout, "stderr": proc.stderr}
    return proc.returncode, doc


def _mutate_manifest(tree: Path, mutate):
    manifest = json.loads((tree / "SOURCE_MANIFEST.json").read_text())
    mutate(manifest)
    _write_manifest(tree, manifest)


def test_clean_fixture_verifies(tmp_path):
    tree, pub, _ = _build_tree(tmp_path)
    rc, doc = _verify(tree, pub)
    assert rc == 0, doc
    assert doc["status"] == "PASS"
    assert doc["files_verified"] == 3


def test_tampered_file_fails(tmp_path):
    tree, pub, _ = _build_tree(tmp_path)
    (tree / "a.txt").write_text("tampered\n")
    rc, doc = _verify(tree, pub)
    assert rc == 2
    assert doc["changed"] == ["a.txt"]


def test_extra_file_fails(tmp_path):
    tree, pub, _ = _build_tree(tmp_path)
    (tree / "smuggled.txt").write_text("extra\n")
    rc, doc = _verify(tree, pub)
    assert rc == 2
    assert doc["extra"] == ["smuggled.txt"]


def test_missing_file_fails(tmp_path):
    tree, pub, _ = _build_tree(tmp_path)
    (tree / "sub" / "b.txt").unlink()
    rc, doc = _verify(tree, pub)
    assert rc == 2
    assert doc["missing"] == ["sub/b.txt"]


def test_symlink_in_tree_fails(tmp_path):
    """v16.4.1 closure policy: symbolic links are refused, not
    silently skipped — an extra link is not an extra file."""
    tree, pub, _ = _build_tree(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n")
    (tree / "link.txt").symlink_to(outside)
    rc, doc = _verify(tree, pub)
    assert rc == 2
    assert doc["reason"] == "symlinks_in_tree"
    assert doc["symlinks"] == ["link.txt"]


def test_symlink_at_governed_path_fails(tmp_path):
    """A governed file replaced by a symlink is refused (the symlink
    policy fires before the missing/changed comparison)."""
    tree, pub, _ = _build_tree(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n")
    (tree / "a.txt").unlink()
    (tree / "a.txt").symlink_to(outside)
    rc, doc = _verify(tree, pub)
    assert rc == 2
    assert doc["reason"] == "symlinks_in_tree"


def test_tool_caches_are_skipped(tmp_path):
    """Dev-tool caches (pytest, ruff) are skipped like the other
    gitignored artifacts — a local ruff run must not break release
    verification."""
    tree, pub, _ = _build_tree(tmp_path)
    for name in (".pytest_cache", ".ruff_cache"):
        cache = tree / name
        cache.mkdir()
        (cache / "data").write_text("cache\n")
    rc, doc = _verify(tree, pub)
    assert rc == 0, doc


def test_forged_signature_fails(tmp_path):
    tree, pub, _ = _build_tree(tmp_path)
    rogue = Ed25519PrivateKey.generate()
    manifest = json.loads((tree / "SOURCE_MANIFEST.json").read_text())
    (tree / "RELEASE_SIGNATURE.bin").write_bytes(
        rogue.sign(_canonical(manifest)))
    rc, doc = _verify(tree, pub)
    assert rc == 4
    assert doc["reason"] == "bad_signature"


def test_corrupted_manifest_digest_entry_fails(tmp_path):
    """A manifest whose recorded hash for a file does not match the
    file (the classic corrupted-manifest substitution) is refused."""
    tree, pub, _ = _build_tree(tmp_path)
    _mutate_manifest(tree,
                     lambda m: m["files"].__setitem__("a.txt", "0" * 64))
    rc, doc = _verify(tree, pub)
    assert rc == 2
    assert doc["changed"] == ["a.txt"]


def test_unsupported_manifest_schema_fails(tmp_path):
    tree, pub, _ = _build_tree(tmp_path)
    _mutate_manifest(tree, lambda m: m.__setitem__("schema_version", 2))
    proc = subprocess.run(
        [sys.executable, str(VERIFIER), "--root", str(tree),
         "--trusted-key", str(pub)],
        capture_output=True, text=True)
    assert proc.returncode != 0
    assert "unsupported SOURCE_MANIFEST schema" in proc.stdout + proc.stderr


def test_stale_attestation_digest_fails(tmp_path):
    tree, pub, _ = _build_tree(tmp_path)
    att = json.loads((tree / "RELEASE_ATTESTATION.json").read_text())
    att["manifest_sha256"] = "0" * 64
    (tree / "RELEASE_ATTESTATION.json").write_text(json.dumps(att))
    rc, doc = _verify(tree, pub)
    assert rc == 5
    assert doc["reason"] == "release_metadata_inconsistent"


def test_stale_reissue_history_entry_fails(tmp_path):
    """Top-level digest correct but the last history entry stale — the
    exact v16.2.1 drift shape — must also fail."""
    tree, pub, _ = _build_tree(tmp_path)
    att = json.loads((tree / "RELEASE_ATTESTATION.json").read_text())
    att["reissue_history"][-1]["manifest_sha256"] = "0" * 64
    (tree / "RELEASE_ATTESTATION.json").write_text(json.dumps(att))
    rc, doc = _verify(tree, pub)
    assert rc == 5
    assert any("reissue_history" in p for p in doc["problems"])


def test_missing_attestation_fails(tmp_path):
    tree, pub, _ = _build_tree(tmp_path)
    (tree / "RELEASE_ATTESTATION.json").unlink()
    rc, doc = _verify(tree, pub)
    assert rc == 5
    assert any("missing" in p for p in doc["problems"])


def test_version_identity_disagreement_fails(tmp_path):
    """A properly signed tree whose VERSION disagrees with the
    attestation release name is refused at the metadata stage."""
    tree, pub, key = _build_tree(tmp_path)
    (tree / "VERSION").write_text("1.0.0-other\n")
    _mutate_manifest(tree,
                     lambda m: m["files"].__setitem__(
                         "VERSION", _sha_file(tree / "VERSION")))
    manifest = json.loads((tree / "SOURCE_MANIFEST.json").read_text())
    (tree / "RELEASE_SIGNATURE.bin").write_bytes(
        key.sign(_canonical(manifest)))
    rc, doc = _verify(tree, pub)
    assert rc == 5
    assert any("name VERSION" in p for p in doc["problems"])
