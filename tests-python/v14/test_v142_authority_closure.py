from dataclasses import replace
import sqlite3
import time

import pytest
from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from minagi.v14 import (
    EvidenceStrength, GovernanceCoordinates, GovernancePolicyV14, GovernedContinualRuntimeV142,
    LearningMechanism, LearningProposalV14, PermanenceLevel, QualificationPolicyV142,
    DurableFreshTaskAuthority, EpisodeVerificationAuthority, EpisodeVerificationValidator,
)


def d(ch="a"): return "sha256:" + ch * 64


def proposal():
    return LearningProposalV14.create(
        target="skill:reverse", mechanism=LearningMechanism.SKILL,
        coordinates=GovernanceCoordinates(PermanenceLevel.L4_REUSABLE_SKILL,EvidenceStrength.E3_REPLICATED),
        evidence_digests=(d("1"),d("2")), rationale_digest=d("3"), proposer_id="learner",
        production_identity_digest=d("4"), expected_gain=.2, expected_interference=.01)


def signer_pair(name):
    s=Ed25519Signer.generate(name); v=Ed25519Verifier(); v.register(s.key_id,s.public_bytes()); return s,v


def runtime(tmp_path, qpolicy=None):
    es,ev=signer_pair("eval"); qs,qv=signer_pair("qual"); ps,pv=signer_pair("prom"); aus,auv=signer_pair("audit")
    rt=GovernedContinualRuntimeV142(tmp_path/"runtime",policy=GovernancePolicyV14(),
        evaluation_signer=es,evaluation_verifier=ev,trusted_evaluation_key_ids={es.key_id},
        qualification_signer=qs,qualification_verifier=qv,trusted_qualification_key_ids={qs.key_id},
        promotion_signer=ps,promotion_verifier=pv,trusted_promotion_key_ids={ps.key_id},
        audit_signer=aus,audit_verifier=auv,trusted_audit_key_ids={aus.key_id},qualification_policy=qpolicy)
    return rt


def passing_metrics():
    return {"retention":.99,"security":1.0,"forgetting":.01,"ood_delta":.02,
            "transfer_rings":["R0","R1","R2"],"negative_controls_passed":True,"fresh_hidden_validated":True,"falsification_passed":True}


def components(rt):
    return {name:rt.put_runtime_component(name,{"x":1}) for name in (
        "production_identity_digest","belief_snapshot_digest","skill_snapshot_digest",
        "adapter_set_digest","routing_policy_digest","execution_runtime_digest")}


def build_chain(rt, kind="skill", metrics=None):
    c=rt.register_candidate(proposal(),candidate_payload={"kind":kind})
    b=rt.record_build(candidate_digest=c,build_payload={"artifact":"x"})
    e=rt.evaluate(candidate_digest=c,build_digest=b,metrics_payload=metrics or passing_metrics())
    q=rt.qualify(e)
    return c,b,e,q


def test_qualification_decision_is_mechanically_derived(tmp_path):
    rt=runtime(tmp_path)
    _,_,_,q=build_chain(rt,metrics={"retention":.2,"security":1.0,"forgetting":.0,"ood_delta":.1,
        "transfer_rings":["R0","R1","R2"],"negative_controls_passed":True,"fresh_hidden_validated":True,"falsification_passed":True})
    assert q.decision=="REJECT" and any("retention" in x for x in q.reasons)
    with pytest.raises(PermissionError): rt.authorize(q)


def test_signed_evaluation_tamper_is_rejected(tmp_path):
    rt=runtime(tmp_path); c=rt.register_candidate(proposal(),candidate_payload={"kind":"skill"}); b=rt.record_build(candidate_digest=c,build_payload={})
    e=rt.evaluator.issue(candidate_digest=c,build_digest=b,metrics_payload=passing_metrics())
    bad=replace(e,metrics_payload={**passing_metrics(),"retention":.1})
    with pytest.raises(PermissionError,match="evaluation signature"): rt.qualifier.qualify(bad,candidate_digest=c,build_digest=b)


def test_signed_qualification_tamper_is_rejected_by_promotion(tmp_path):
    rt=runtime(tmp_path); _,_,_,q=build_chain(rt)
    bad=replace(q,decision="REJECT")
    with pytest.raises(PermissionError): rt.promotion.authorize(qualification=bad,mutation_scopes=("runtime.activate",))


def test_runtime_manifest_requires_real_typed_cas_dependencies(tmp_path):
    rt=runtime(tmp_path); _,_,_,q=build_chain(rt); auth=rt.authorize(q); good=components(rt)
    bad=dict(good); bad["belief_snapshot_digest"]=d("e")
    with pytest.raises(PermissionError,match="missing/corrupt"): rt.activate(auth=auth,**bad)
    wrong=rt.cas.put_json({"schema":"wrong-schema","payload":{}}); bad=dict(good); bad["belief_snapshot_digest"]=wrong
    with pytest.raises(PermissionError,match="schema mismatch"): rt.activate(auth=auth,**bad)
    m=rt.activate(auth=auth,**good); assert rt.db.current_runtime_manifest()==m.digest


def test_authorization_scope_is_resolved_from_signed_body_for_belief(tmp_path):
    rt=runtime(tmp_path); c=rt.register_candidate(proposal(),candidate_payload={"kind":"belief"}); b=rt.record_build(candidate_digest=c,build_payload={})
    e=rt.evaluate(candidate_digest=c,build_digest=b,metrics_payload=passing_metrics()); q=rt.qualify(e); auth=rt.authorize(q)
    bel=rt.beliefs.promote(subject="A",predicate="p",object_value="B",valid_from=1.0,evidence_digests=(d("1"),),
                           confidence=.9,authorization_digest=auth.digest)
    assert bel.revision==1
    with pytest.raises(PermissionError,match="does not permit skill.activate"):
        common=dict(procedure_digest=d("1"),positive_evidence=(d("2"),),negative_evidence=(d("3"),),failure_envelope_digest=d("4"))
        s=rt.skills.transition(status="DISCOVERED",**common); rt.skills.transition(status="PROPOSED",skill_id=s.skill_id,**common); rt.skills.transition(status="CANDIDATE",skill_id=s.skill_id,**common); rt.skills.transition(status="QUALIFIED",skill_id=s.skill_id,**common); rt.skills.transition(status="ACTIVE",skill_id=s.skill_id,authorization_digest=auth.digest,**common)


def test_signed_audit_checkpoint_detects_rewritten_chain(tmp_path):
    rt=runtime(tmp_path); build_chain(rt); cp=rt.checkpoint_audit(); assert cp.signer_key_id; assert rt.verify_audit()
    # An attacker with DB write access can recompute an ordinary hash chain, but not the signed checkpoint.
    row=rt.db.conn.execute("SELECT * FROM audit_events WHERE seq=?",(cp.event_seq,)).fetchone()
    rt.db.conn.execute("UPDATE audit_events SET event_digest=? WHERE seq=?",(d("f"),cp.event_seq))
    with pytest.raises((RuntimeError,PermissionError)): rt.verify_audit()


def test_fresh_task_vault_separates_secret_and_reclaims_expired_lease(tmp_path):
    meta=tmp_path/"fresh.sqlite3"; a=DurableFreshTaskAuthority(meta); c=a.seal({"secret":"x"},generation=2)
    cols={r[1] for r in sqlite3.connect(meta).execute("PRAGMA table_info(tasks)")}; assert "task_json" not in cols and "salt" not in cols
    l=a.lease(task_id=c.task_id,consumer_id="e",ttl_seconds=.001); time.sleep(.01)
    assert a.reclaim_expired() == 1
    l2=a.lease(task_id=c.task_id,consumer_id="e2"); assert a.consume(l2)=={"secret":"x"}


def test_verification_receipt_binds_expected_result_and_scoring_policy(tmp_path):
    s,v=signer_pair("verify"); authority=EpisodeVerificationAuthority(verifier_id="v",signer=s)
    r=authority.issue(episode_id="e",prompt="p",attempted_output="bad",repaired_output="good",expected_output="gold",
                      scoring_policy_id="judge-v2",evidence_root_digest=d("1"),score=.9,pass_threshold=.8)
    val=EpisodeVerificationValidator(verifier=v,trusted_key_ids={s.key_id})
    assert val.validate(r,episode_id="e",prompt="p",attempted_output="bad",repaired_output="good",expected_output="gold",scoring_policy_id="judge-v2")
    with pytest.raises(PermissionError): val.validate(r,episode_id="e",prompt="p",attempted_output="bad",expected_output="wrong")


def test_canonical_learning_cycle_runs_end_to_end(tmp_path):
    rt=runtime(tmp_path)
    payloads={name:{"component":name} for name in (
        "production_identity_digest","belief_snapshot_digest","skill_snapshot_digest",
        "adapter_set_digest","routing_policy_digest","execution_runtime_digest")}
    out=rt.learning_cycle(proposal=proposal(),candidate_payload={"kind":"skill"},build_payload={"artifact":"skill-v1"},
                          metrics_payload=passing_metrics(),runtime_component_payloads=payloads)
    assert out["status"]=="ACTIVE"
    assert rt.db.current_runtime_manifest()==out["runtime_manifest"].digest
    assert rt.verify_audit()
