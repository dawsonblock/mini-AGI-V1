"""v16.4.5 restoration-authority unit tests (SEC-403 / G1).

The verifier must establish the COMPLETE original chain before any
restoration grant is issued. Every defect in the chain — absent,
forged, revoked, stale, or mismatched — is a permanent refusal, and
no fallback digest is ever synthesized.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "tests-python" / "helpers"))

import authority_chain as ac  # noqa: E402
from egai.common.canonical import digest  # noqa: E402
from minagi.security.restoration_authority import (  # noqa: E402
    RestorationAuthorityRefused,
    RestorationAuthorizationVerifier)
from minagi.security.signed_revocations import (  # noqa: E402
    RevocationSnapshotV2)


def _setup(tmp_path):
    registry, signers = ac.trust_chain(tmp_path)
    rstore = ac.revocation_store(signers, tmp_path, epoch=0)
    chain = ac.docs(signers, tmp_path, "a")
    verifier = RestorationAuthorizationVerifier(
        registry, revocation_store=rstore, now=ac.NOW)
    ctx = {"decision_digest": chain["decision"]["digest"],
           "qualification_digest": chain["qualification"]["digest"],
           "manifest": chain["runtime_manifest"]["digest"],
           "artifact_root_digest": chain["artifact_root"],
           "backend": "hf-peft",
           "backend_binary_digest": "",
           "policy_epoch": 0}
    kw = dict(candidate_id="cand-1", authorized_context=ctx,
              authority_docs=ac.authority_docs(chain),
              measured_digests=dict(chain["digests"]),
              artifact_root_digest=chain["artifact_root"],
              backend_id="hf-peft")
    return registry, signers, rstore, chain, verifier, ctx, kw


def test_happy_path_returns_verified_bindings(tmp_path):
    _, _, _, chain, verifier, _, kw = _setup(tmp_path)
    auth = verifier.verify(**kw)
    assert auth.decision_digest == chain["decision"]["digest"]
    assert auth.runtime_manifest_digest == \
        chain["runtime_manifest"]["digest"]
    assert auth.revocation_epoch == 0
    assert auth.seed == "seed-0"


def test_missing_documents_refuse(tmp_path):
    _, _, _, _, verifier, _, kw = _setup(tmp_path)
    kw["authority_docs"] = {"decision": kw["authority_docs"]["decision"]}
    with pytest.raises(RestorationAuthorityRefused, match="retained"):
        verifier.verify(**kw)


def test_absent_journaled_digests_refuse_no_fallback(tmp_path):
    """Missing historical fields mean refusal — never a synthesized
    substitute digest (the v16.4.4 defect)."""
    _, _, _, _, verifier, ctx, kw = _setup(tmp_path)
    ctx["decision_digest"] = ""
    with pytest.raises(RestorationAuthorityRefused,
                       match="lacks decision_digest"):
        verifier.verify(**kw)


def test_wrong_measured_adapter_refused(tmp_path):
    """An approved-but-altered adapter fails the re-measure."""
    _, _, _, _, verifier, _, kw = _setup(tmp_path)
    kw["measured_digests"] = dict(kw["measured_digests"],
                                adapter=digest({"tampered": 1}))
    with pytest.raises(RestorationAuthorityRefused,
                       match="manifest-authorized"):
        verifier.verify(**kw)


def test_extra_artifact_refused(tmp_path):
    """Artifacts beyond the manifest-authorized set refuse — the
    snapshot must contain exactly {model, adapter}."""
    _, _, _, _, verifier, _, kw = _setup(tmp_path)
    kw["measured_digests"] = dict(kw["measured_digests"],
                                rogue=digest({"rogue": 1}))
    with pytest.raises(RestorationAuthorityRefused,
                       match="artifact set"):
        verifier.verify(**kw)


def test_qualification_not_qualified_refused(tmp_path):
    registry, signers, _, chain, verifier, _, kw = _setup(tmp_path)
    qual = dict(chain["qualification"]["value"], decision="REFUSED")
    kw["authority_docs"] = dict(kw["authority_docs"],
                              qualification=ac.signed(
                                  signers["qualification"], qual))
    with pytest.raises(RestorationAuthorityRefused):
        verifier.verify(**kw)


def test_unsigned_decision_refused(tmp_path):
    registry, signers, _, chain, verifier, _, kw = _setup(tmp_path)
    forged = dict(chain["decision"], signature_b64="AAAA")
    kw["authority_docs"] = dict(kw["authority_docs"], decision=forged)
    with pytest.raises(RestorationAuthorityRefused):
        verifier.verify(**kw)


def test_decision_digest_mismatch_refused(tmp_path):
    """The retained decision isn't the one the activation journaled —
    substituted authority is refused."""
    registry, signers, _, chain, verifier, ctx, kw = _setup(tmp_path)
    other = dict(chain["decision"]["value"], campaign_id="other")
    forged = ac.signed(signers["promotion"], other)
    kw["authority_docs"] = dict(kw["authority_docs"], decision=forged)
    with pytest.raises(RestorationAuthorityRefused):
        verifier.verify(**kw)


def test_expired_decision_refused(tmp_path):
    registry, signers, rstore, chain, verifier, _, kw = _setup(tmp_path)
    short = ac.docs(signers, tmp_path, "b", expires_at=ac.TS - 10)
    ctx2 = dict(kw["authorized_context"],
                decision_digest=short["decision"]["digest"],
                qualification_digest=short["qualification"]["digest"],
                manifest=short["runtime_manifest"]["digest"],
                artifact_root_digest=short["artifact_root"])
    kw["authorized_context"] = ctx2
    kw["authority_docs"] = ac.authority_docs(short)
    kw["measured_digests"] = dict(short["digests"])
    kw["artifact_root_digest"] = short["artifact_root"]
    with pytest.raises(RestorationAuthorityRefused, match="EXPIRED"):
        verifier.verify(**kw)


def test_wrong_backend_binding_refused(tmp_path):
    _, _, _, _, verifier, ctx, kw = _setup(tmp_path)
    kw["backend_id"] = "hf-peft"
    ctx["backend"] = "some-other-backend"
    with pytest.raises(RestorationAuthorityRefused):
        verifier.verify(**kw)


def test_backend_binary_drift_refused(tmp_path):
    """The operative backend implementation must equal the one the
    original authorization bound — backend drift refuses."""
    _, _, _, _, verifier, ctx, kw = _setup(tmp_path)
    ctx["backend_binary_digest"] = digest({"binary": "old"})
    kw["backend_manifest_digest"] = digest({"binary": "new"})
    with pytest.raises(RestorationAuthorityRefused,
                       match="backend"):
        verifier.verify(**kw)


def test_revoked_decision_refused(tmp_path):
    registry, signers, rstore, chain, verifier, _, kw = _setup(tmp_path)
    rstore.publish(RevocationSnapshotV2(
        epoch=1, issued_at=ac.TS, valid_until=ac.TS + 86400,
        revoked_decision_digests=(chain["decision"]["digest"],)).to_doc(
            signer=signers["revocation"]))
    with pytest.raises(RestorationAuthorityRefused, match="REVOKED"):
        verifier.verify(**kw)


def test_no_store_fails_closed(tmp_path):
    registry, _, _, chain, _, _, kw = _setup(tmp_path)
    verifier = RestorationAuthorizationVerifier(
        registry, revocation_store=None, now=ac.NOW)
    with pytest.raises(RestorationAuthorityRefused,
                       match="revocation"):
        verifier.verify(**kw)
