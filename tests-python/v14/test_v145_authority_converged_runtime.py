from dataclasses import asdict, replace
import time
import pytest

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from minagi.v14.models import EvidenceStrength, GovernanceCoordinates, LearningMechanism, LearningProposalV14, PermanenceLevel
from minagi.v14.policy import GovernancePolicyV14
from minagi.v14.storage import ImmutableCAS
from minagi.v14.storage_v145 import GovernanceDBV145
from minagi.v14.verification import EpisodeVerificationAuthority, EpisodeVerificationValidator
from minagi.v14.evidence_v144 import EvidenceStageAuthorityV144, EvidenceStageValidatorV144, EvidenceStrengthAuthorityV144, EvidenceStrengthValidatorV144
from minagi.v14.services_v144 import VerificationAuthorityClientV144, EvidenceStageAuthorityClientV144, EvidenceStrengthAuthorityClientV144
from minagi.v14.services_v143 import EvaluationAuthorityClientV143, QualificationAuthorityClientV143, PromotionAuthorityClientV143
from minagi.v14.qualification_v142 import QualificationPolicyV142
from minagi.v14.qualification_v143 import QualificationAuthorityV143, QualificationValidatorV143
from minagi.v14.authority_v143 import PromotionAuthorityV143, PromotionAuthorizationValidatorV143
from minagi.v14.fresh_tasks_v143 import FreshTaskConsumptionReceiptV143
from minagi.v14.provenance_v145 import (
    FalsificationCaseCommitmentV145, EvaluationCaseResultV145,
    EvaluationProvenanceValidatorV145, BoundEvaluationAuthorityV145, BoundEvaluationValidatorV145,
)
from minagi.v14.state_epoch_v145 import EpochTransitionAuthorityV145, EpochTransitionValidatorV145
from minagi.v14.runtime_v145 import GovernedContinualRuntimeV145
from minagi.v14.runtime_v144 import RUNTIME_COMPONENT_SCHEMAS_V144
from minagi.v14.trajectory_v145 import VerifiedTrajectoryV145
from minagi.v14.experiment_v145 import FrozenArmRecordV145, FrozenBaselineGateV145


def signer(name, verifier):
    s=Ed25519Signer.generate(name); verifier.register(s.key_id,s.public_bytes()); return s


def stack(tmp_path):
    root=tmp_path/"rt"; cas=ImmutableCAS(root/"cas"); db=GovernanceDBV145(root/"governance.sqlite3"); policy=GovernancePolicyV14()
    shared=Ed25519Verifier()
    verify_s=signer("episode",shared); stage_s=signer("stage",shared); strength_s=signer("strength",shared)
    eval_s=signer("eval",shared); qual_s=signer("qual",shared); prom_s=signer("prom",shared)
    fresh_s=signer("fresh",shared); epoch_s=signer("epoch",shared)

    ev_auth=EpisodeVerificationAuthority(verifier_id="episode",signer=verify_s)
    ev_val=EpisodeVerificationValidator(verifier=shared,trusted_key_ids={verify_s.key_id})
    stage_val=EvidenceStageValidatorV144(verifier=shared,episode_trusted_key_ids={verify_s.key_id},stage_key_policy={2:{stage_s.key_id}},authority_generation=1,cas=cas)
    stage_auth=EvidenceStageAuthorityV144(authority_id="stage",authority_generation=1,signer=stage_s,validator=stage_val,cas=cas)
    strength_auth=EvidenceStrengthAuthorityV144(authority_id="strength",authority_generation=1,signer=strength_s,stage_validator=stage_val,cas=cas)
    strength_val=EvidenceStrengthValidatorV144(verifier=shared,trusted_key_ids={strength_s.key_id},authority_generation=1,stage_validator=stage_val,cas=cas)

    prov=EvaluationProvenanceValidatorV145(cas=cas,verifier=shared,trusted_fresh_task_key_ids={fresh_s.key_id},fresh_task_authority_generation=1)
    eval_auth=BoundEvaluationAuthorityV145(evaluator_id="eval",evaluator_generation=1,signer=eval_s,cas=cas,provenance_validator=prov)
    eval_val=BoundEvaluationValidatorV145(verifier=shared,trusted_key_ids={eval_s.key_id},evaluator_generation=1,cas=cas,provenance_validator=prov)
    qual_val=QualificationValidatorV143(verifier=shared,trusted_key_ids={qual_s.key_id},qualifier_generation=1,authority_generation=1,policy_generation=1)
    qual_auth=QualificationAuthorityV143(qualifier_id="qual",qualifier_generation=1,authority_generation=1,policy_generation=1,signer=qual_s,evaluation_validator=eval_val,cas=cas,policy=QualificationPolicyV142())
    prom_val=PromotionAuthorizationValidatorV143(verifier=shared,trusted_key_ids={prom_s.key_id},authority_generation=1,policy_generation=1)
    prom_auth=PromotionAuthorityV143(governance_db=db,signer=prom_s,qualification_validator=qual_val,authority_generation=1,policy_generation=1)
    epoch_val=EpochTransitionValidatorV145(verifier=shared,trusted_key_ids={epoch_s.key_id},authority_generation=1)

    rt=GovernedContinualRuntimeV145(root,policy=policy,
        verification_client=VerificationAuthorityClientV144(ev_auth),verification_validator=ev_val,
        evidence_stage_client=EvidenceStageAuthorityClientV144(stage_auth),
        evidence_client=EvidenceStrengthAuthorityClientV144(strength_auth),evidence_validator=strength_val,
        evaluation_client=EvaluationAuthorityClientV143(eval_auth),evaluation_validator=eval_val,
        qualification_client=QualificationAuthorityClientV143(qual_auth),qualification_validator=qual_val,
        promotion_client=PromotionAuthorityClientV143(prom_auth),promotion_validator=prom_val,
        epoch_transition_validator=epoch_val,cas=cas,governance_db=db)
    return rt,fresh_s,epoch_s


def proposal_chain(rt):
    exp,receipt,verified=rt.ingest_verified_experience(prompt="q",attempted_output="bad",repaired_output="good",expected_output="good",
        evidence_payload={"fact":"x"},production_identity_digest="sha256:"+"4"*64)
    _,proof=rt.issue_e2_evidence_proof(verified_record=verified,verification_receipt=receipt)
    p=LearningProposalV14.create(target="belief:x",mechanism=LearningMechanism.BELIEF_UPDATE,
        coordinates=GovernanceCoordinates(PermanenceLevel.L3_SEMANTIC_BELIEF,EvidenceStrength.E2_INDEPENDENTLY_VERIFIED),
        evidence_digests=(verified.evidence_root_digest,),rationale_digest="sha256:"+"3"*64,proposer_id="learner",
        production_identity_digest="sha256:"+"4"*64,expected_gain=.1,expected_interference=0)
    c=rt.register_candidate(p,evidence_proof=proof,candidate_payload={"experience":exp.digest})
    belief=rt.persistence.prepare_belief(subject="s",predicate="p",object_value="o",valid_from=1,
        evidence_digests=(verified.evidence_root_digest,),confidence=.9)
    b=rt.record_build(candidate_digest=c,build_payload={"kind":"belief"},mutation_targets={"belief.promote":belief.digest})
    return c,b,belief,verified.evidence_root_digest


def bound_cases(rt,candidate,fresh_signer):
    def raw(cid,cat,score=1.0,passed=True,ring="",**kw):
        ev=rt.put_supporting_receipt("eval",{"case":cid,"category":cat,"score":score})
        return EvaluationCaseResultV145(cid,cat,score,passed,ev,ring,**kw)
    fresh_task_digest=digest({"hidden-task":"fresh"})
    fresh=FreshTaskConsumptionReceiptV143("task1","lease1","eval",1,digest({"commitment":"x"}),fresh_task_digest,time.time(),"fresh",1)
    env=fresh_signer.sign(asdict(fresh.unsigned())); fresh=replace(fresh,signer_key_id=env.key_id,signature_b64=env.signature_b64)
    fresh_digest=rt.cas.put_json(asdict(fresh)); assert fresh_digest==fresh.digest
    fals=[]
    for cid,cat in (("neg","negative_control"),("fals","falsification")):
        task=digest({"task":cid}); fc=FalsificationCaseCommitmentV145(candidate,cid,cat,task,digest({"expected":cid}),digest({"plan":"pre-registered"}),time.time())
        fd=rt.cas.put_json(asdict(fc)); assert fd==fc.digest; fals.append((cid,cat,task,fd))
    out=[raw("ret","retention",.99),raw("sec","security",1.0),raw("fg","forgetting",.01),raw("ood","ood_delta",.02),
         raw("r0","transfer",1,True,"R0"),raw("r1","transfer",1,True,"R1"),raw("r2","transfer",1,True,"R2"),
         raw("fresh","fresh_hidden",1,True,"",provenance_kind="fresh_task",provenance_digest=fresh_digest,task_digest=fresh_task_digest)]
    for cid,cat,task,fd in fals: out.append(raw(cid,cat,1,True,"",provenance_kind="falsification",provenance_digest=fd,task_digest=task))
    return tuple(out)


def components(belief_digest):
    out={k:{"component":k} for k in RUNTIME_COMPONENT_SCHEMAS_V144}
    out["belief_snapshot_digest"]["included_mutation_digests"]=[belief_digest]
    return out


def test_privileged_evaluation_labels_require_authority_provenance(tmp_path):
    rt,fresh,_=stack(tmp_path); c,b,_,_=proposal_chain(rt)
    ev=rt.put_supporting_receipt("eval",{"case":"fresh"})
    forged=(EvaluationCaseResultV145("fresh","fresh_hidden",1,True,ev),)
    with pytest.raises(PermissionError,match="lacks fresh-task provenance"):
        rt.evaluate(candidate_digest=c,build_digest=b,case_results=forged)


def test_bound_evaluation_qualifies_and_state_epoch_becomes_servable(tmp_path):
    rt,fresh,epoch_s=stack(tmp_path); c,b,belief,_=proposal_chain(rt)
    e=rt.evaluate(candidate_digest=c,build_digest=b,case_results=bound_cases(rt,c,fresh)); q=rt.qualify(e)
    assert q.decision=="PROMOTE"
    m=rt.prepare_runtime_manifest(qualification=q,runtime_component_payloads=components(belief.digest)); a=rt.authorize(q,m)
    epoch=rt.prepare_state_epoch(manifest=m,authorization=a)
    rt.activate_into_epoch(auth=a,manifest=m,epoch=epoch)
    assert rt.db.state_epoch_row_v145(epoch.digest)["state"]=="LOCALLY_COMMITTED"
    authority=EpochTransitionAuthorityV145(authority_id="epoch",authority_generation=1,signer=epoch_s)
    witness=rt.cas.put_json({"external_witness":"ok","epoch":epoch.digest})
    rt.state_epochs.transition(epoch_digest=epoch.digest,receipt=authority.issue(epoch_digest=epoch.digest,from_state="LOCALLY_COMMITTED",to_state="EXTERNALLY_WITNESSED",artifact_digest=witness))
    attest=rt.cas.put_json({"runtime_attestation":"ok","epoch":epoch.digest})
    rt.state_epochs.transition(epoch_digest=epoch.digest,receipt=authority.issue(epoch_digest=epoch.digest,from_state="EXTERNALLY_WITNESSED",to_state="ATTESTED",artifact_digest=attest))
    serve=rt.cas.put_json({"servable":"approved","epoch":epoch.digest})
    rt.state_epochs.transition(epoch_digest=epoch.digest,receipt=authority.issue(epoch_digest=epoch.digest,from_state="ATTESTED",to_state="SERVABLE",artifact_digest=serve))
    lease=rt.state_epochs.acquire_request_lease(request_id="request-1",ttl_seconds=60)
    assert lease.epoch_digest==epoch.digest and rt.state_epochs.validate_request_lease(lease)
    retire=rt.cas.put_json({"retire":"rotation"})
    rt.state_epochs.transition(epoch_digest=epoch.digest,receipt=authority.issue(epoch_digest=epoch.digest,from_state="SERVABLE",to_state="RETIRED",artifact_digest=retire))
    assert rt.state_epochs.validate_request_lease(lease)  # already-started request remains epoch pinned
    rt.state_epochs.release_request_lease(lease.lease_id)


def test_dependency_revocation_cascades_to_derived_objects(tmp_path):
    rt,_,_=stack(tmp_path)
    evidence=rt.put_evidence({"fact":"root"}); belief=rt.cas.put_json({"belief":"derived"}); runtime=rt.cas.put_json({"runtime":"derived"})
    rt.register_learned_dependency(child_digest=belief,parent_digests=(evidence,),relation="evidence_support")
    rt.register_learned_dependency(child_digest=runtime,parent_digests=(belief,),relation="runtime_support")
    descendants=rt.revoke_evidence(evidence_digest=evidence,reason="source retracted")
    assert set(descendants)=={belief,runtime}
    assert rt.db.control_state_v145(evidence)=="REVOKED"
    assert rt.db.control_state_v145(belief)=="QUARANTINED"
    assert rt.db.control_state_v145(runtime)=="QUARANTINED"
    with pytest.raises(PermissionError): rt.db.require_usable_v145(runtime)
    with pytest.raises(RuntimeError): rt.db._set_control_v145(object_digest=evidence,state="ACTIVE",reason="undo")


def vt(i,text,out,ev,rec):
    return VerifiedTrajectoryV145(f"t{i}","normalize",text,"bad",out,"sha256:"+"9"*64,(ev,),(rec,))


def test_trajectory_induction_requires_repeated_noncontradictory_support(tmp_path):
    rt,_,_=stack(tmp_path); ev1=rt.put_evidence({"e":1}); ev2=rt.put_evidence({"e":2}); rec1=rt.put_supporting_receipt("verification",{"v":1}); rec2=rt.put_supporting_receipt("verification",{"v":2})
    report=rt.induce_skill_candidates((vt(1,"abc","ABC",ev1,rec1),vt(2,"def","DEF",ev2,rec2)))
    assert len(report.candidates)==1 and not report.contradictions
    c=report.candidates[0]; assert c.operation_signature=="uppercase"
    # Same input with two independently verified repaired outputs is blocked.
    bad=rt.trajectory_inducer.induce((vt(3,"xyz","XYZ",ev1,rec1),vt(4,"xyz","zyx",ev2,rec2)))
    assert bad.contradictions and not bad.candidates


def test_frozen_baseline_gate_requires_same_model_and_positive_transfer():
    gate=FrozenBaselineGateV145(required_rings=("R0","R1","R2"),bootstrap_iterations=600)
    model="sha256:"+"a"*64; ident="sha256:"+"b"*64
    a0=[]; a1=[]
    for i,ring in enumerate(("R0","R1","R2")):
        td=digest({"task":i}); a0.append(FrozenArmRecordV145(td,"fam",ring,model,ident,"A0",0.0)); a1.append(FrozenArmRecordV145(td,"fam",ring,model,ident,"A1",1.0))
    report=gate.evaluate(a0,a1); assert report.decision=="PASS" and report.ci95[0] > 0
    changed=list(a1); changed[0]=replace(changed[0],foundation_model_digest="sha256:"+"c"*64)
    with pytest.raises(PermissionError,match="changed foundation model"): gate.evaluate(a0,changed)


def test_v15_namespace_exposes_one_canonical_runtime():
    import minagi.v15 as api
    assert api.GovernedContinualRuntime is GovernedContinualRuntimeV145
    assert api.GovernanceDB is GovernanceDBV145
    assert not hasattr(api,"GovernedContinualRuntimeV144")
