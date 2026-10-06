from dataclasses import replace
import sqlite3
import pytest

from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from minagi.v14 import (
    EvidenceStrength,GovernanceCoordinates,GovernancePolicyV14,LearningMechanism,LearningProposalV14,PermanenceLevel,
    ImmutableCAS,GovernanceDBV143,EvidenceStrengthAuthorityV143,EvidenceStrengthValidatorV143,EvidenceAuthorityClientV143,
    EvaluationAuthorityV143,EvaluationValidatorV143,EvaluationAuthorityClientV143,EvaluationCaseResultV143,
    QualificationAuthorityV143,QualificationValidatorV143,QualificationAuthorityClientV143,QualificationPolicyV142,
    PromotionAuthorityV143,PromotionAuthorizationValidatorV143,PromotionAuthorityClientV143,
    GovernedContinualRuntimeV143,RUNTIME_COMPONENT_SCHEMAS_V143,FileAuditAnchorV143,AuditAuthorityClientV143,
    FreshTaskMetadataStoreV143,FreshTaskVaultAuthorityV143,
)


def pair(name):
    s=Ed25519Signer.generate(name);v=Ed25519Verifier();v.register(s.key_id,s.public_bytes());return s,v

def stack(tmp_path):
    root=tmp_path/"rt";cas=ImmutableCAS(root/"cas");db=GovernanceDBV143(root/"governance.sqlite3");policy=GovernancePolicyV14()
    es,ev=pair("evidence");vs,vv=pair("eval");qs,qv=pair("qual");ps,pv=pair("prom");aus,auv=pair("audit")
    ea=EvidenceStrengthAuthorityV143(authority_id="evidence",authority_generation=1,signer=es,cas=cas)
    evalv=EvaluationValidatorV143(verifier=vv,trusted_key_ids={vs.key_id},evaluator_generation=1,cas=cas)
    eva=EvaluationAuthorityV143(evaluator_id="eval",evaluator_generation=1,signer=vs,cas=cas)
    qualv=QualificationValidatorV143(verifier=qv,trusted_key_ids={qs.key_id},qualifier_generation=1,authority_generation=1,policy_generation=1)
    qala=QualificationAuthorityV143(qualifier_id="qual",qualifier_generation=1,authority_generation=1,policy_generation=1,
                                    signer=qs,evaluation_validator=evalv,cas=cas,policy=QualificationPolicyV142())
    proma=PromotionAuthorityV143(governance_db=db,signer=ps,qualification_validator=qualv,authority_generation=1,policy_generation=1)
    promv=PromotionAuthorizationValidatorV143(verifier=pv,trusted_key_ids={ps.key_id},authority_generation=1,policy_generation=1)
    evproofv=EvidenceStrengthValidatorV143(verifier=ev,trusted_key_ids={es.key_id},authority_generation=1,cas=cas)
    anchor=FileAuditAnchorV143(root/"audit-anchor.json",signer=aus,verifier=auv,trusted_key_ids={aus.key_id})
    auditc=AuditAuthorityClientV143(db=db,signer=aus,anchor=anchor,authority_generation=1)
    rt=GovernedContinualRuntimeV143(root,policy=policy,evidence_client=EvidenceAuthorityClientV143(ea),evidence_validator=evproofv,
        evaluation_client=EvaluationAuthorityClientV143(eva),evaluation_validator=evalv,
        qualification_client=QualificationAuthorityClientV143(qala),qualification_validator=qualv,
        promotion_client=PromotionAuthorityClientV143(proma),promotion_validator=promv,
        audit_client=auditc,audit_verifier=auv,trusted_audit_key_ids={aus.key_id},audit_anchor=anchor,cas=cas,governance_db=db)
    return rt,ea

def make_evidence(rt,ea):
    e1=rt.put_evidence({"fact":"a"});e2=rt.put_evidence({"fact":"b"})
    vr=rt.put_supporting_receipt("verification",{"ok":True});rr=rt.put_supporting_receipt("replication",{"ok":True})
    proof=ea.issue(evidence_digests=(e1,e2),verification_receipt_digests=(vr,),replication_receipt_digests=(rr,))
    return (e1,e2),proof

def proposal(evidence):
    return LearningProposalV14.create(target="skill:reverse",mechanism=LearningMechanism.SKILL,
        coordinates=GovernanceCoordinates(PermanenceLevel.L4_REUSABLE_SKILL,EvidenceStrength.E3_REPLICATED),
        evidence_digests=tuple(evidence),rationale_digest="sha256:"+"3"*64,proposer_id="learner",
        production_identity_digest="sha256:"+"4"*64,expected_gain=.2,expected_interference=.01)

def cases(rt):
    def c(cid,cat,score,passed=True,ring=""):
        ed=rt.put_supporting_receipt("raw-eval",{"case":cid,"cat":cat,"score":score,"passed":passed,"ring":ring})
        return EvaluationCaseResultV143(cid,cat,score,passed,ed,ring)
    return (
        c("ret","retention",.99),c("sec","security",1.0),c("fg","forgetting",.01),c("ood","ood_delta",.02),
        c("r0","transfer",1,True,"R0"),c("r1","transfer",1,True,"R1"),c("r2","transfer",1,True,"R2"),
        c("neg","negative_control",1),c("fresh","fresh_hidden",1),c("fals","falsification",1),
    )

def components(): return {k:{"component":k} for k in RUNTIME_COMPONENT_SCHEMAS_V143}

def chain(rt,ea,mutation_targets=None):
    ev,proof=make_evidence(rt,ea);p=proposal(ev);c=rt.register_candidate(p,evidence_proof=proof,candidate_payload={"note":"no scopes here"})
    b=rt.record_build(candidate_digest=c,build_payload={"artifact":"skill"},mutation_targets=mutation_targets)
    e=rt.evaluate(candidate_digest=c,build_digest=b,case_results=cases(rt));q=rt.qualify(e);m=rt.prepare_runtime_manifest(qualification=q,runtime_component_payloads=components())
    return p,c,b,e,q,m


def test_evidence_strength_is_authority_derived_and_missing_evidence_cannot_self_claim(tmp_path):
    rt,ea=stack(tmp_path); e,proof=make_evidence(rt,ea); p=proposal(e)
    bad=replace(p,coordinates=GovernanceCoordinates(PermanenceLevel.L4_REUSABLE_SKILL,EvidenceStrength.E5_INDEPENDENTLY_REPRODUCED))
    with pytest.raises(PermissionError,match="not authority-derived"): rt.register_candidate(bad,evidence_proof=proof,candidate_payload={})


def test_skill_candidate_cannot_escalate_to_adapter_scope(tmp_path):
    rt,ea=stack(tmp_path);target=rt.put_supporting_receipt("skill-target",{"v":1})
    ev,proof=make_evidence(rt,ea);p=proposal(ev);c=rt.register_candidate(p,evidence_proof=proof,candidate_payload={"mutation_scopes":["adapter.activate"]})
    with pytest.raises(PermissionError,match="outside policy-derived scopes"):
        rt.record_build(candidate_digest=c,build_payload={},mutation_targets={"adapter.activate":target})


def test_qualification_recomputes_from_signed_raw_cases(tmp_path):
    rt,ea=stack(tmp_path);ev,proof=make_evidence(rt,ea);p=proposal(ev);c=rt.register_candidate(p,evidence_proof=proof,candidate_payload={});b=rt.record_build(candidate_digest=c,build_payload={})
    bad=list(cases(rt));bad[0]=replace(bad[0],score=.1)
    bundle=rt.evaluate(candidate_digest=c,build_digest=b,case_results=tuple(bad));q=rt.qualify(bundle)
    assert q.decision=="REJECT" and any("retention" in r for r in q.reasons)
    tampered=replace(bundle,case_results=cases(rt))
    with pytest.raises(PermissionError): rt.qualification_client.qualify(tampered,candidate_digest=c,build_digest=b)


def test_promotion_is_bound_to_exact_manifest_and_single_use(tmp_path):
    rt,ea=stack(tmp_path);_,_,_,_,q,m=chain(rt,ea);a=rt.authorize(q,m);first=rt.activate(auth=a,manifest=m);assert first
    with pytest.raises(PermissionError,match="already consumed"): rt.activate(auth=a,manifest=m)
    m2=replace(m,created_at=m.created_at+1)
    md2=rt.cas.put_json(m2.__dict__);assert md2==m2.digest
    with pytest.raises(PermissionError,match="exact-content binding"): rt.promotion_validator.validate(a,candidate_digest=m2.candidate_digest,build_digest=m2.build_digest,
        evaluation_digest=m2.evaluation_digest,qualification_digest=m2.qualification_digest,runtime_manifest_digest=m2.digest,artifact_root_digest=m2.artifact_root_digest,
        required_scope="runtime.activate",target_digest=m2.digest)


def test_nonruntime_mutation_is_bound_to_exact_target(tmp_path):
    rt,ea=stack(tmp_path);target=rt.put_supporting_receipt("skill-target",{"v":1});other=rt.put_supporting_receipt("skill-target",{"v":2})
    _,_,_,_,q,m=chain(rt,ea,{"skill.activate":target});a=rt.authorize(q,m)
    with pytest.raises(PermissionError,match="exact mutation"): rt.authorization_gate.require(a.digest,mutation_scope="skill.activate",target_digest=other)
    rt.authorization_gate.require(a.digest,mutation_scope="skill.activate",target_digest=target)
    with pytest.raises(PermissionError,match="already consumed"): rt.authorization_gate.require(a.digest,mutation_scope="skill.activate",target_digest=target)


def test_external_audit_anchor_makes_checkpoint_deletion_visible(tmp_path):
    rt,ea=stack(tmp_path);chain(rt,ea);rt.checkpoint_audit();assert rt.verify_audit()
    rt.db.conn.execute("DELETE FROM audit_checkpoints")
    with pytest.raises(RuntimeError,match="anchor not present"): rt.verify_audit()


def test_runtime_does_not_hold_private_signers(tmp_path):
    rt,_=stack(tmp_path)
    assert not any("signer" in k for k in rt.__dict__)
    assert hasattr(rt,"evaluation_client") and hasattr(rt,"promotion_client")


def test_fresh_task_secret_is_not_available_through_metadata_client(tmp_path):
    s,v=pair("fresh");meta=FreshTaskMetadataStoreV143(tmp_path/"meta.sqlite3");vault=FreshTaskVaultAuthorityV143(tmp_path/"vault.sqlite3",authority_id="fresh",authority_generation=1,signer=s)
    c=vault.seal({"secret":"x"},generation=4,metadata=meta);lease=meta.lease(task_id=c.task_id,consumer_id="evaluator")
    cols={r[1] for r in sqlite3.connect(tmp_path/"meta.sqlite3").execute("PRAGMA table_info(tasks)")};assert "task_json" not in cols and "salt" not in cols
    task,receipt=vault.consume(lease,metadata=meta);assert task=={"secret":"x"};assert receipt.signer_key_id==s.key_id


def test_alpha3_learning_cycle_runs_exact_content_path(tmp_path):
    rt,ea=stack(tmp_path);ev,proof=make_evidence(rt,ea);p=proposal(ev)
    out=rt.learning_cycle(proposal=p,evidence_proof=proof,candidate_payload={},build_payload={"artifact":"skill"},mutation_targets=None,
                          case_results=cases(rt),runtime_component_payloads=components())
    assert out["status"]=="ACTIVE" and rt.db.current_runtime_manifest()==out["runtime_manifest"].digest

def test_production_mode_rejects_colocated_local_authorities(tmp_path):
    rt,_=stack(tmp_path)
    with pytest.raises(RuntimeError,match="out-of-process evidence"):
        GovernedContinualRuntimeV143(tmp_path/"prod",policy=rt.policy,evidence_client=rt.evidence_client,evidence_validator=rt.evidence_validator,
            evaluation_client=rt.evaluation_client,evaluation_validator=rt.evaluation_validator,
            qualification_client=rt.qualification_client,qualification_validator=rt.qualification_validator,
            promotion_client=rt.promotion_client,promotion_validator=rt.promotion_validator,production_mode=True)
