"""v16.4.2 — AdmissionGrantV1: the capability that separates artifact
measurement from deployment authorization (UPGRADE_PLAN §3.1).

A grant is short-lived, audience-bound, digest-bound to exactly one
measured artifact set on exactly one backend, and epoch-bound to the
revocation evidence it was issued against. Every check fails closed.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.security.admission_grants import (  # noqa: E402
    AdmissionGrantV1, GrantRefused, issue_grant, verify_grant)
from minagi.v161.authority import (AUTHORITY_ROLES, AuthorityRegistry,  # noqa: E402
                                   write_trust_root)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
TS = int(NOW.timestamp())
AUDIENCE = "local-supervisor"
MANIFEST = digest({"manifest": 1})
ARTIFACT = digest({"artifact": 1})
DECISION = digest({"decision": 1})
QUAL = digest({"qual": 1})


def _chain(tmp_path):
    storage = tmp_path / "storage"
    write_trust_root(storage / ".keys", storage / "trust_root.json")
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    signers = {r: Ed25519Signer.from_private_bytes(
        (storage / ".keys" / f"{r}.pem").read_bytes())
        for r in AUTHORITY_ROLES}
    return registry, signers


def _issue(signer, **kw):
    base = dict(decision_digest=DECISION, qualification_digest=QUAL,
                runtime_manifest_digest=MANIFEST,
                artifact_root_digest=ARTIFACT, backend_id="hf-peft",
                audience_runtime_identity=AUDIENCE, now=NOW,
                revocation_epoch=2)
    base.update(kw)
    return issue_grant(signer, **base)


# ---------- happy path --------------------------------------------------

def test_issued_grant_verifies_end_to_end(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = _issue(signers["admission"])
    grant = verify_grant(
        doc, registry, now=NOW, audience_runtime_identity=AUDIENCE,
        require_backend_id="hf-peft", require_manifest_digest=MANIFEST,
        require_artifact_root_digest=ARTIFACT, min_revocation_epoch=2)
    assert grant.backend_id == "hf-peft"
    assert grant.revocation_epoch == 2
    assert grant.audience_runtime_identity == AUDIENCE


# ---------- refusal rows ------------------------------------------------

def test_unsigned_grant_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = _issue(signers["admission"])
    unsigned = {"value": doc["value"], "digest": doc["digest"]}
    with pytest.raises(GrantRefused, match="unsigned"):
        verify_grant(unsigned, registry, now=NOW)


def test_grant_signed_by_wrong_role_refused(tmp_path):
    """A research or promotion key cannot mint production credentials —
    authority separation is cryptographic, not conventional."""
    registry, signers = _chain(tmp_path)
    for role in ("plan", "promotion", "runtime", "qualification"):
        doc = _issue(signers[role])
        with pytest.raises(GrantRefused, match="admission authority"):
            verify_grant(doc, registry, now=NOW)


def test_forged_grant_signature_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = dict(_issue(signers["admission"]))
    doc["value"] = dict(doc["value"])
    doc["value"]["backend_id"] = "qwen-native-cuda"
    doc["digest"] = digest(doc["value"])
    with pytest.raises(GrantRefused, match="signature invalid"):
        verify_grant(doc, registry, now=NOW)


def test_wrong_audience_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = _issue(signers["admission"])
    with pytest.raises(GrantRefused, match="not this runtime"):
        verify_grant(doc, registry, now=NOW,
                     audience_runtime_identity="other-runtime")


def test_wrong_backend_refused(tmp_path):
    """An hf-peft grant does not authorize a native-CUDA load."""
    registry, signers = _chain(tmp_path)
    doc = _issue(signers["admission"])
    with pytest.raises(GrantRefused, match="authorizes backend"):
        verify_grant(doc, registry, now=NOW,
                     require_backend_id="qwen-native-cuda")


def test_wrong_manifest_digest_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = _issue(signers["admission"])
    with pytest.raises(GrantRefused, match="runtime manifest"):
        verify_grant(doc, registry, now=NOW,
                     require_manifest_digest=digest({"other": 1}))


def test_wrong_artifact_root_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = _issue(signers["admission"])
    with pytest.raises(GrantRefused,
                       match="measured artifact root"):
        verify_grant(doc, registry, now=NOW,
                     require_artifact_root_digest=digest({"swapped": 1}))


def test_stale_revocation_epoch_refused(tmp_path):
    """A grant issued against epoch 2 is refused once the operative
    revocation epoch has advanced past it."""
    registry, signers = _chain(tmp_path)
    doc = _issue(signers["admission"], revocation_epoch=2)
    with pytest.raises(GrantRefused, match="stale authorization"):
        verify_grant(doc, registry, now=NOW, min_revocation_epoch=3)


def test_expired_grant_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = _issue(signers["admission"], ttl_seconds=60)
    later = datetime(2026, 10, 8, 12, 5, tzinfo=timezone.utc)
    with pytest.raises(GrantRefused, match="expired"):
        verify_grant(doc, registry, now=later)


def test_future_dated_grant_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    past = datetime(2026, 10, 8, 11, 0, tzinfo=timezone.utc)
    doc = _issue(signers["admission"], now=past, ttl_seconds=7200)
    # rewind the verifier's clock far behind issued_at
    earlier = datetime(2026, 10, 8, 10, 0, tzinfo=timezone.utc)
    with pytest.raises(GrantRefused, match="future-dated"):
        verify_grant(doc, registry, now=earlier)


def test_envelope_digest_mismatch_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = dict(_issue(signers["admission"]))
    doc["digest"] = "sha256:" + "0" * 64
    with pytest.raises(GrantRefused, match="digest mismatch"):
        verify_grant(doc, registry, now=NOW)


def test_grant_body_roundtrips_through_schema(tmp_path):
    """from_value must reconstruct exactly the object to_body wrote."""
    registry, signers = _chain(tmp_path)
    doc = _issue(signers["admission"], backend_binary_digest=digest("b"))
    grant = verify_grant(doc, registry, now=NOW)
    again = AdmissionGrantV1.from_value(grant.to_body())
    assert again.digest == grant.digest
