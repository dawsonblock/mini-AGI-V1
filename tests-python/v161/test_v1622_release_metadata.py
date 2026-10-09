"""FIX-006 — release metadata reconciliation.

On v16.2.1 the attestation's top-level manifest_sha256/reissued_utc
were stale relative to the last reissue_history entry and the actual
signed manifest, and version identities disagreed across VERSION,
pyproject, package, SBOM and validation docs. These tests fail on the
original metadata and pass on the reconciled release.

Scope note: this checks metadata *internal* consistency (attestation ↔
manifest ↔ version identities). Tree-vs-manifest drift between releases
is `scripts/verify_release.py`'s job and is checked at release points.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from cryptography.hazmat.primitives import serialization  # noqa: E402


def _canonical_manifest_bytes(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()


def _manifest_digest(doc):
    return hashlib.sha256(_canonical_manifest_bytes(doc)).hexdigest()


def _attestation():
    return json.loads((ROOT / "RELEASE_ATTESTATION.json").read_text())


def _manifest():
    return json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())


def test_attestation_digests_match_signed_manifest():
    att = _attestation()
    actual = _manifest_digest(_manifest())
    assert att["manifest_sha256"] == actual
    history = att["reissue_history"]
    assert history and history[-1]["manifest_sha256"] == actual
    assert att["reissued_utc"] == history[-1]["utc"]


def test_manifest_signature_verifies_with_bundled_key():
    key = serialization.load_pem_public_key(
        (ROOT / "RELEASE_PUBLIC_KEY.pem").read_bytes())
    key.verify((ROOT / "RELEASE_SIGNATURE.bin").read_bytes(),
               _canonical_manifest_bytes(_manifest()))


def test_version_identities_agree():
    version = (ROOT / "VERSION").read_text().strip()
    number = version.split("-", 1)[0]
    assert re.fullmatch(r"\d+\.\d+\.\d+", number)
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert re.search(rf'^version = "{re.escape(number)}"$', pyproject,
                     re.M), "pyproject version disagrees with VERSION"
    init = (ROOT / "src-python" / "minagi" / "__init__.py").read_text()
    assert f'__version__ = "{number}"' in init, \
        "package version disagrees with VERSION"
    sbom = json.loads((ROOT / "SBOM.cdx.json").read_text())
    assert sbom["metadata"]["component"]["version"] == number, \
        "SBOM version disagrees with VERSION"
    validation = json.loads((ROOT / "RELEASE_VALIDATION.json").read_text())
    assert validation["version"] == number, \
        "RELEASE_VALIDATION version disagrees with VERSION"
    att = _attestation()
    assert f"v{number}" in att["release"], \
        "attestation release name does not name the VERSION"


def test_attestation_and_validation_status_agree():
    assert _attestation()["status"] == \
        json.loads((ROOT / "RELEASE_VALIDATION.json").read_text())["status"]
