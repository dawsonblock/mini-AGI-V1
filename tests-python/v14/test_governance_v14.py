import pytest
from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from minagi.v14 import (
    AdaptiveReplayScheduler,
    CandidateStateStore,
    DreamCandidate,
    DreamPolicyResearcher,
    DurableFreshTaskAuthority,
    EpisodeVerificationAuthority,
    EpisodeVerificationValidator,
    EvidenceStrength,
    FalsificationCase,
    FalsificationKind,
    FalsificationPlan,
    GovernanceCoordinates,
    GovernancePolicyV14,
    GovernedContinualOrchestrator,
    LearningMechanism,
    LearningProposalV14,
    ModelTimeClock,
    PermanenceLevel,
)


def d(ch="a"):
    return "sha256:" + ch * 64


def proposal(level=PermanenceLevel.L4_REUSABLE_SKILL, strength=EvidenceStrength.E3_REPLICATED):
    return LearningProposalV14.create(
        target="skill:reverse",
        mechanism=LearningMechanism.SKILL,
        coordinates=GovernanceCoordinates(level, strength),
        evidence_digests=(d("1"), d("2")),
        rationale_digest=d("3"),
        proposer_id="learner",
        production_identity_digest=d("4"),
        expected_gain=.1,
        expected_interference=.01,
    )


def test_permanence_and_evidence_are_independent_policy_axes():
    policy=GovernancePolicyV14()
    ok,reasons=policy.evaluate(proposal(PermanenceLevel.L4_REUSABLE_SKILL,EvidenceStrength.E2_INDEPENDENTLY_VERIFIED))
    assert not ok and any("below required" in x for x in reasons)
    ok,_=policy.evaluate(proposal())
    assert ok


def test_shared_adapter_remains_proposal_only_by_default():
    p=LearningProposalV14.create(
        target="adapter:x",mechanism=LearningMechanism.LORA,
        coordinates=GovernanceCoordinates(PermanenceLevel.L7_SHARED_ADAPTER,EvidenceStrength.E5_INDEPENDENTLY_REPRODUCED),
        evidence_digests=(d("1"),),rationale_digest=d("2"),proposer_id="learner",production_identity_digest=d("3"),
        expected_gain=.2,expected_interference=.01)
    ok,reasons=GovernancePolicyV14().evaluate(p)
    assert not ok and any("proposal-only" in x or "disabled" in x for x in reasons)


def test_candidate_state_machine_blocks_skip_to_active(tmp_path):
    state=CandidateStateStore(tmp_path/"state.sqlite3")
    with pytest.raises(PermissionError):
        state.transition(candidate_digest=d("a"),new_state="ACTIVE",actor="attacker",authority_generation=1,policy_generation=1,authorization_digest=d("b"))


def test_end_to_end_state_machine_requires_authorization(tmp_path):
    state=CandidateStateStore(tmp_path/"state.sqlite3")
    o=GovernedContinualOrchestrator(state,GovernancePolicyV14())
    c=d("a"); p=proposal()
    o.register(p,candidate_digest=c);o.built(c);o.evaluated(c);o.qualified(c)
    with pytest.raises(PermissionError):
        state.transition(candidate_digest=c,new_state="AUTHORIZED",actor="x",authority_generation=1,policy_generation=1)
    auth=d("f");o.authorized(c,authorization_digest=auth);o.active(c,authorization_digest=auth)
    assert state.current(c)=="ACTIVE"


def test_fresh_task_is_restart_safe_and_one_shot(tmp_path):
    path=tmp_path/"fresh.sqlite3"
    a=DurableFreshTaskAuthority(path); commitment=a.seal({"q":"secret"},generation=7)
    lease=a.lease(task_id=commitment.task_id,consumer_id="eval")
    b=DurableFreshTaskAuthority(path)
    assert b.consume(lease)=={"q":"secret"}
    with pytest.raises(PermissionError): b.consume(lease)
    revealed=b.reveal(commitment.task_id)
    assert revealed["commitment_valid"] is True


def test_falsification_plan_requires_negative_retention_security():
    def base(i, k):
        return FalsificationCase(i,k,d(str(i%10)),d(str((i+1)%10)))
    with pytest.raises(ValueError):
        FalsificationPlan(d("a"),(base(1,FalsificationKind.OOD),),"falsifier").validate()
    plan=FalsificationPlan(d("a"),(
        base(1,FalsificationKind.NEGATIVE_CONTROL),base(2,FalsificationKind.RETENTION),base(3,FalsificationKind.SECURITY)),"falsifier")
    plan.validate()


def test_dream_researcher_is_proposal_only():
    r=DreamPolicyResearcher(); assert r.can_promote is False and r.can_activate is False
    best=r.propose((DreamCandidate("a",d("a"),.1,.05,(d("1"),)),DreamCandidate("b",d("b"),.2,.01,(d("2"),))))
    assert best.candidate_id=="b"


def test_model_time_replay_uses_parameter_displacement():
    clock=ModelTimeClock(); sched=AdaptiveReplayScheduler(model_time_interval=1.0,min_priority=.9)
    clock.advance(.2); assert not sched.decide(model_time=clock.value,estimated_forgetting_risk=.0,memory_importance=.0).should_replay
    clock.advance(.9); assert sched.decide(model_time=clock.value,estimated_forgetting_risk=.0,memory_importance=.0).should_replay


def test_episode_receipt_is_bound_and_signed():
    signer=Ed25519Signer.generate("v"); verifier=Ed25519Verifier(); verifier.register(signer.key_id,signer.public_bytes())
    authority=EpisodeVerificationAuthority(verifier_id="independent",signer=signer)
    receipt=authority.issue(episode_id="e1",prompt="p",attempted_output="bad",repaired_output="good",evidence_root_digest=d("e"),score=1.0)
    validator=EpisodeVerificationValidator(verifier=verifier,trusted_key_ids={signer.key_id})
    assert validator.validate(receipt,episode_id="e1",prompt="p",attempted_output="bad")
    with pytest.raises(PermissionError): validator.validate(receipt,episode_id="e1",prompt="p2",attempted_output="bad")
