from dataclasses import replace

import pytest
from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from minagi.v14 import (
    BiTemporalBeliefStore,
    EvidenceStrength,
    GovernanceCoordinates,
    GovernanceDB,
    GovernancePolicyV14,
    GovernedContinualRuntimeV141,
    ImmutableCAS,
    LearningMechanism,
    LearningProposalV14,
    PermanenceLevel,
    SkillLifecycleStore,
)


def d(ch="a"):
    return "sha256:" + ch * 64


def proposal():
    return LearningProposalV14.create(
        target="skill:reverse",
        mechanism=LearningMechanism.SKILL,
        coordinates=GovernanceCoordinates(PermanenceLevel.L4_REUSABLE_SKILL, EvidenceStrength.E3_REPLICATED),
        evidence_digests=(d("1"), d("2")),
        rationale_digest=d("3"), proposer_id="learner", production_identity_digest=d("4"),
        expected_gain=.2, expected_interference=.01,
    )


def runtime(tmp_path):
    signer=Ed25519Signer.generate("promotion")
    verifier=Ed25519Verifier(); verifier.register(signer.key_id, signer.public_bytes())
    rt=GovernedContinualRuntimeV141(
        tmp_path/"runtime", policy=GovernancePolicyV14(), promotion_signer=signer,
        promotion_verifier=verifier, trusted_promotion_key_ids={signer.key_id})
    return rt, signer, verifier


def test_cas_rejects_traversal_and_detects_corruption(tmp_path):
    cas=ImmutableCAS(tmp_path/"cas")
    obj=cas.put_json({"x":1})
    assert cas.exists(obj)
    for bad in ("../x", "sha256:../"+"a"*61, "md5:"+"a"*32, "sha256:"+"A"*64, "sha256:"+"a"*63):
        with pytest.raises(ValueError): cas.get_bytes(bad)
    path=cas._path(obj); path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="digest mismatch"): cas.get_bytes(obj)


def test_authoritative_chain_rejects_skipped_build(tmp_path):
    db=GovernanceDB(tmp_path/"g.sqlite3")
    c=d("a"); db.register_candidate(candidate_digest=c, proposal_digest=d("b"))
    with pytest.raises(Exception):
        db.record_evaluation(candidate_digest=c, build_digest=d("c"), evaluation_digest=d("d"))
    assert db.verify_audit_chain()


def test_full_chain_requires_promote_qualification_and_signed_authorization(tmp_path):
    rt,_,_=runtime(tmp_path)
    c=rt.register_candidate(proposal(),candidate_payload={"kind":"skill"})
    b=rt.record_build(candidate_digest=c,build_payload={"artifact":"x"})
    e=rt.record_evaluation(candidate_digest=c,build_digest=b,evaluation_payload={"retention":1.0,"security":1.0})
    q=rt.record_qualification(candidate_digest=c,evaluation_digest=e,decision="PROMOTE",metrics_payload={"rings":["R0","R1","R2"]})
    auth=rt.authorize(candidate_digest=c,qualification_digest=q)
    manifest=rt.activate(
        auth=auth,production_identity_digest=d("4"),belief_snapshot_digest=d("5"),skill_snapshot_digest=d("6"),
        adapter_set_digest=d("7"),routing_policy_digest=d("8"),execution_runtime_digest=d("9"))
    assert rt.db.current_runtime_manifest()==manifest.digest
    assert rt.db.verify_audit_chain()
    rt.close()


def test_rejected_qualification_cannot_be_authorized(tmp_path):
    rt,_,_=runtime(tmp_path)
    c=rt.register_candidate(proposal(),candidate_payload={"kind":"skill"})
    b=rt.record_build(candidate_digest=c,build_payload={})
    e=rt.record_evaluation(candidate_digest=c,build_digest=b,evaluation_payload={})
    q=rt.record_qualification(candidate_digest=c,evaluation_digest=e,decision="REJECT",metrics_payload={"reason":"regression"})
    with pytest.raises(PermissionError,match="does not permit promotion"):
        rt.authorize(candidate_digest=c,qualification_digest=q)


def test_tampered_promotion_signature_cannot_activate(tmp_path):
    rt,_,_=runtime(tmp_path)
    c=rt.register_candidate(proposal(),candidate_payload={})
    b=rt.record_build(candidate_digest=c,build_payload={})
    e=rt.record_evaluation(candidate_digest=c,build_digest=b,evaluation_payload={})
    q=rt.record_qualification(candidate_digest=c,evaluation_digest=e,decision="PROMOTE",metrics_payload={})
    auth=rt.authorize(candidate_digest=c,qualification_digest=q)
    bad=replace(auth, signature_b64="AAAA")
    with pytest.raises(PermissionError,match="invalid promotion"):
        rt.activate(auth=bad,production_identity_digest=d("4"),belief_snapshot_digest=d("5"),skill_snapshot_digest=d("6"),
                    adapter_set_digest=d("7"),routing_policy_digest=d("8"),execution_runtime_digest=d("9"))


def test_bitemporal_beliefs_append_revisions_without_overwrite(tmp_path):
    s=BiTemporalBeliefStore(tmp_path/"beliefs.sqlite3")
    first=s.promote(subject="Acme",predicate="ceo",object_value="Alice",valid_from=1.0,
                    evidence_digests=(d("1"),),confidence=.9,authorization_digest=d("a"),belief_id="b1")
    second=s.promote(subject="Acme",predicate="ceo",object_value="Bob",valid_from=2.0,
                     evidence_digests=(d("2"),),confidence=.95,authorization_digest=d("b"),belief_id="b1")
    hist=s.history("b1")
    assert [x.revision for x in hist]==[1,2]
    assert hist[1].supersedes_digest==first.digest
    assert s.current("b1").digest==second.digest


def test_skill_lifecycle_keeps_negative_evidence_and_requires_auth_to_activate(tmp_path):
    s=SkillLifecycleStore(tmp_path/"skills.sqlite3")
    common=dict(procedure_digest=d("1"),positive_evidence=(d("2"),d("3")),negative_evidence=(d("4"),),failure_envelope_digest=d("5"))
    a=s.transition(status="DISCOVERED",**common)
    sid=a.skill_id
    s.transition(status="PROPOSED",skill_id=sid,**common)
    s.transition(status="CANDIDATE",skill_id=sid,**common)
    s.transition(status="QUALIFIED",skill_id=sid,**common)
    with pytest.raises(ValueError): s.transition(status="ACTIVE",skill_id=sid,**common)
    active=s.transition(status="ACTIVE",skill_id=sid,authorization_digest=d("a"),**common)
    assert active.negative_evidence==(d("4"),)
    assert len(s.history(sid))==5


def test_audit_tampering_detected(tmp_path):
    db=GovernanceDB(tmp_path/"g.sqlite3")
    db.register_candidate(candidate_digest=d("a"),proposal_digest=d("b"))
    db.conn.execute("UPDATE audit_events SET payload_json='{}' WHERE seq=1")
    with pytest.raises(RuntimeError,match="hash chain"): db.verify_audit_chain()


def test_activation_transaction_rolls_back_if_authorization_binding_is_wrong(tmp_path):
    rt,_,_=runtime(tmp_path)
    c=rt.register_candidate(proposal(),candidate_payload={})
    b=rt.record_build(candidate_digest=c,build_payload={})
    e=rt.record_evaluation(candidate_digest=c,build_digest=b,evaluation_payload={})
    q=rt.record_qualification(candidate_digest=c,evaluation_digest=e,decision="PROMOTE",metrics_payload={})
    auth=rt.authorize(candidate_digest=c,qualification_digest=q)
    with pytest.raises(PermissionError):
        rt.db.activate(candidate_digest=d("f"),authorization_digest=auth.digest,runtime_manifest_digest=d("7"),activation_digest=d("8"))
    assert rt.db.current_runtime_manifest() is None
