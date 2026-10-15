"""V4-14/V4-17 — verified adapter routing: family gating, OOD,
revocation/rollback, health suspension, registry validation."""
import json
import pytest

from egai.common.canonical import digest
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
    r.revoke("a1")
    r.rollback()
    assert "a2" not in r._adapters
    assert r.route("reverse", family="tr-reverse").selected == "a1"


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
