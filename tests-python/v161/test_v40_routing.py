"""V4-14/V4-17 — verified adapter routing: family gating, OOD,
revocation/rollback, health suspension, registry validation.

V4-21/V4-22 — registry persistence of safety state (revocation,
suspension, health) across save/reload, signed manifest authorization,
and artifact-content verification."""
import base64
import json
from dataclasses import asdict

import pytest

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from minagi.v161.authority import AuthorityRecord, AuthorityRegistry
from minagi.v161.runtime_closure3 import sha256_path
from minagi.v161.verified_routing import (QualifiedAdapterManifest,
                                          VerifiedAdapterRouter)

Z = "sha256:" + "0" * 64


def _manifest(aid="a1", families=("tr-reverse",), terms=None, **kw):
    return QualifiedAdapterManifest(
        adapter_id=aid,
        adapter_digest=kw.pop("adapter_digest", digest({"a": aid})),
        model_id=kw.pop("model_id", "m"),
        model_revision=kw.pop("model_revision", "rev1"),
        qualified_families=families,
        qualification_evidence_digest=kw.pop("evidence", digest({"e": aid})),
        scorer_artifact_digest=kw.pop("scorer", Z),
        family_terms=terms or tuple(t.replace("tr-", "")
                                    for t in families),
        **kw)


def test_manifest_requires_digests_and_families():
    with pytest.raises(Exception):
        QualifiedAdapterManifest(
            adapter_id="x", adapter_digest="not-a-digest",
            model_id="m", model_revision="r", qualified_families=("f",),
            qualification_evidence_digest=Z, scorer_artifact_digest=Z)
    with pytest.raises(ValueError):
        _manifest("x", families=())


def test_qualified_family_routes():
    r = VerifiedAdapterRouter([_manifest()])
    d = r.route("reverse this word", family="tr-reverse",
                model_revision="rev1")
    assert d.selected == "a1" and not d.ood


def test_unqualified_family_fails_closed():
    r = VerifiedAdapterRouter([_manifest()])
    d = r.route("do something else", family="hid-sort")
    assert d.selected is None and d.ood
    assert any("family not qualified" in x for x in d.refused)


def test_model_revision_mismatch_refused():
    r = VerifiedAdapterRouter([_manifest()])
    d = r.route("reverse this word", family="tr-reverse",
                model_revision="rev2")
    assert d.selected is None
    assert any("model revision mismatch" in x for x in d.refused)


def test_unknown_family_uses_term_overlap():
    r = VerifiedAdapterRouter([_manifest(terms=("reverse", "string"))])
    # query shares terms -> eligible even without a family hint
    d = r.route("please reverse the string token")
    assert d.selected == "a1"


def test_ood_query_abstains():
    r = VerifiedAdapterRouter([_manifest(terms=("reverse",))],
                              ood_threshold=0.9)
    d = r.route("reverse this")
    assert d.selected is None and d.ood


def test_revocation_blocks_routing():
    r = VerifiedAdapterRouter([_manifest()])
    r.revoke("a1")
    d = r.route("reverse this", family="tr-reverse")
    assert d.selected is None
    assert any("revoked" in x for x in d.refused)


def test_reinstate_requires_new_evidence():
    m = _manifest()
    r = VerifiedAdapterRouter([m])
    r.revoke("a1")
    with pytest.raises(ValueError):
        r.reinstate(m)  # same evidence digest — forbidden
    r.reinstate(_manifest(evidence=digest({"e": "new"})))
    d = r.route("reverse this", family="tr-reverse")
    assert d.selected == "a1"


def test_health_suspension():
    r = VerifiedAdapterRouter([_manifest()], min_health=0.5,
                              health_window=4)
    for _ in range(4):
        r.record_outcome("a1", False)
    d = r.route("reverse", family="tr-reverse")
    assert d.selected is None
    assert any("suspended" in x for x in d.refused)


def test_rollback_restores_snapshot():
    m1, m2 = _manifest("a1"), _manifest("a2", families=("tr-upper",))
    r = VerifiedAdapterRouter([m1])
    r.snapshot()
    r.register(m2)
    assert len(r._adapters) == 2
    r.rollback()
    assert "a2" not in r._adapters
    assert r.route("reverse", family="tr-reverse").selected == "a1"


def test_rollback_never_undoes_revocation():
    """A revocation is a safety action — rolling the registry back must
    not silently re-enable a revoked adapter."""
    m1, m2 = _manifest("a1"), _manifest("a2", families=("tr-upper",))
    r = VerifiedAdapterRouter([m1])
    r.snapshot()
    r.register(m2)
    r.revoke("a1")
    r.rollback()
    assert "a2" not in r._adapters            # snapshot restored
    assert "a1" in r._adapters
    assert r.route("reverse", family="tr-reverse").selected is None
    assert r.route("reverse", family="tr-reverse").refused == \
        ("a1: revoked",)


def test_registry_roundtrip(tmp_path):
    r = VerifiedAdapterRouter([_manifest("a1"),
                               _manifest("a2", families=("f2",))])
    p = tmp_path / "registry.json"
    p.write_text(r.to_json())
    r2 = VerifiedAdapterRouter.from_json(p)
    assert sorted(r2._adapters) == ["a1", "a2"]
    with pytest.raises(ValueError):
        (tmp_path / "bad.json").write_text(json.dumps({"schema": "x"}))
        VerifiedAdapterRouter.from_json(tmp_path / "bad.json")


def test_every_route_emits_an_event():
    r = VerifiedAdapterRouter([_manifest()])
    r.route("q1", family="tr-reverse")
    r.route("q2")
    assert len(r.events) == 2
    assert r.events[0]["decision"]["selected"] == "a1"
    assert r.events[1]["decision"]["selected"] is None


# ---- V4-21: safety state must survive save/reload ---------------------

def test_revocation_survives_registry_roundtrip(tmp_path):
    """A revoked adapter must NOT become selectable after save/reload —
    the pre-v4.0.1 bug this pins (revocations were silently dropped)."""
    r = VerifiedAdapterRouter([_manifest("a1"),
                               _manifest("a2", families=("tr-upper",))])
    r.revoke("a1")
    p = tmp_path / "registry.json"
    p.write_text(r.to_json())
    r2 = VerifiedAdapterRouter.from_json(p)
    assert r2.state_complete
    d = r2.route("reverse", family="tr-reverse")
    assert d.selected is None
    assert any("revoked" in x for x in d.refused)
    # the non-revoked adapter still routes
    assert r2.route("upper", family="tr-upper").selected == "a2"


def test_suspension_and_health_survive_roundtrip(tmp_path):
    r = VerifiedAdapterRouter([_manifest()], min_health=0.5,
                              health_window=4)
    for _ in range(4):
        r.record_outcome("a1", False)
    assert "a1" in r._suspended
    p = tmp_path / "registry.json"
    p.write_text(r.to_json())
    r2 = VerifiedAdapterRouter.from_json(p, min_health=0.5,
                                         health_window=4)
    assert r2._health["a1"] == [0, 0, 0, 0]
    d = r2.route("reverse", family="tr-reverse")
    assert d.selected is None
    assert any("suspended" in x for x in d.refused)


def test_v1_registry_load_marks_incomplete_state(tmp_path):
    """A v1 file cannot prove its adapters were not revoked before save —
    it loads, but the router must say the safety state is incomplete."""
    doc = {"schema": "mini-agi-v4.0-adapter-registry-v1",
           "adapters": [json.loads(json.dumps(asdict(_manifest())))]}
    p = tmp_path / "registry-v1.json"
    p.write_text(json.dumps(doc))
    r = VerifiedAdapterRouter.from_json(p)
    assert not r.state_complete
    assert r.registry_warnings


def test_v1_registry_strict_state_refuses(tmp_path):
    doc = {"schema": "mini-agi-v4.0-adapter-registry-v1",
           "adapters": []}
    p = tmp_path / "registry-v1.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="strict_state|v1"):
        VerifiedAdapterRouter.from_json(p, strict_state=True)


# ---- V4-22: signed authorization + artifact verification --------------

def _signer_and_verifier():
    s = Ed25519Signer.generate()
    v = Ed25519Verifier()
    v.register(s.key_id, s.public_bytes())
    return s, v


def test_unsigned_manifest_refused_when_verifier_configured():
    s, v = _signer_and_verifier()
    r = VerifiedAdapterRouter([_manifest()], manifest_verifier=v)
    d = r.route("reverse", family="tr-reverse")
    assert d.selected is None
    assert any("unsigned" in x for x in d.refused)
    assert r.ineligible()["a1"] == "manifest unsigned"
    # and the signed variant routes
    r2 = VerifiedAdapterRouter([_manifest().sign(s)],
                               manifest_verifier=v)
    assert r2.route("reverse", family="tr-reverse").selected == "a1"


def test_wrong_key_signature_refused():
    _, v = _signer_and_verifier()
    other = Ed25519Signer.generate()  # signed, but not by a trusted key
    m = _manifest().sign(other)
    r = VerifiedAdapterRouter([m], manifest_verifier=v)
    d = r.route("reverse", family="tr-reverse")
    assert d.selected is None
    assert any("signature invalid" in x for x in d.refused)


def test_authority_role_enforced():
    """A signature from a key not authorized for the qualification role
    is inadmissible even when the signature itself verifies."""
    s_eval, s_other = Ed25519Signer.generate(), Ed25519Signer.generate()
    reg = AuthorityRegistry([
        AuthorityRecord(role="qualification", key_id=s_eval.key_id,
                        public_key_b64=base64.b64encode(
                            s_eval.public_bytes()).decode()),
        AuthorityRecord(role="evaluation", key_id=s_other.key_id,
                        public_key_b64=base64.b64encode(
                            s_other.public_bytes()).decode())])
    ok = VerifiedAdapterRouter([_manifest().sign(s_eval)],
                               authority=reg)
    assert ok.route("reverse", family="tr-reverse").selected == "a1"
    # signed by the *evaluation* key — verifies cryptographically, but
    # the signer is not authorized for the qualification role
    bad = VerifiedAdapterRouter([_manifest().sign(s_other)],
                                authority=reg)
    d = bad.route("reverse", family="tr-reverse")
    assert d.selected is None
    assert any("not authorized" in x for x in d.refused)


def test_artifact_digest_verified_against_disk(tmp_path):
    adir = tmp_path / "adapter-a1"
    adir.mkdir()
    (adir / "w.bin").write_bytes(b"weights")
    good = _manifest(adapter_digest=sha256_path(adir))
    r = VerifiedAdapterRouter(
        [good], artifact_resolver=lambda aid: tmp_path / f"adapter-{aid}")
    assert r.route("reverse", family="tr-reverse").selected == "a1"
    # tamper with the artifact after registration
    (adir / "w.bin").write_bytes(b"weights-TAMPERED")
    r2 = VerifiedAdapterRouter(
        [good], artifact_resolver=lambda aid: tmp_path / f"adapter-{aid}")
    d = r2.route("reverse", family="tr-reverse")
    assert d.selected is None
    assert any("artifact digest mismatch" in x for x in d.refused)


def test_artifact_missing_is_ineligible_not_crash(tmp_path):
    r = VerifiedAdapterRouter(
        [_manifest()],
        artifact_resolver=lambda aid: tmp_path / "does-not-exist")
    d = r.route("reverse", family="tr-reverse")
    assert d.selected is None
    assert any("unverifiable" in x for x in d.refused)


def test_reinstate_rechecked_under_policy(tmp_path):
    m = _manifest()
    r = VerifiedAdapterRouter(
        [m], artifact_resolver=lambda aid: tmp_path / "missing")
    # manifest was ineligible at construction — nothing to reinstate
    with pytest.raises(ValueError):
        r.reinstate(_manifest(evidence=digest({"e": "new"})))
