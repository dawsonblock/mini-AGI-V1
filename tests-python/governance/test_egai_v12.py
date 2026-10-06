from dataclasses import replace
from chain_helpers import prepare_chain
import json

import pytest

from kvcontinual.execution.authority import Ed25519ReceiptSigner
from minagi.egai import (
    Belief,
    BeliefRepository,
    BeliefStatus,
    CognitivePlane,
    EvidenceClass,
    EvidenceLedger,
    EvidenceOrigin,
    ExperienceNode,
    ExperimentCheckpoint,
    FrozenFoundationExperimentSpec,
    FrozenFoundationProtocol,
    FutureConsequence,
    GroundedReplayEngine,
    Hypothesis,
    HypothesisStatus,
    ImprovementLedger,
    IndependentQualificationGate,
    LearningAction,
    LearningLevel,
    LearningPlane,
    LeastPermanentPlasticityProposer,
    PlasticityDecisionContext,
    ProposalOption,
    PromotionAuthority,
    PromotionConstraints,
    PromotionVerdict,
    QualificationBundle,
    QualificationMetrics,
    ReplayWorldCompiler,
    SimulationRecord,
    SkillManifest,
    SkillRepository,
    TrustedEvidenceIngestor,
    compute_fte,
)
from minagi.egai.canonical import sha256_json


IDENTITY = "sha256:" + "1" * 64


def make_proposal(level=LearningLevel.L3_PROCEDURAL_SKILL, action=LearningAction.CREATE_PROCEDURE):
    proposer = LeastPermanentPlasticityProposer(proposer_id="plasticity-v12")
    return proposer.propose(
        options=[ProposalOption(action, level, 0.2, 0.01, reason="verified repeated pattern")],
        target="dependency-resolution",
        evidence_digests=["sha256:" + "2" * 64],
        production_identity_digest=IDENTITY,
        context=PlasticityDecisionContext(0.5, 0.01, 1.0, 0.2, 0.01, {}),
    )


def make_bundle(proposal, *, candidate="sha256:" + "3" * 64, worlds=None, rings=None, **metric_overrides):
    worlds = worlds if worlds is not None else 16
    rings = rings if rings is not None else ("episode", "procedure", "reusable_skill")
    metrics = dict(
        forward_transfer_delta=0.10,
        forgetting=0.0,
        ood_delta=0.01,
        calibration_regression=0.0,
        security_regressions=0,
        unauthorized_writes=0,
        provenance_closure=1.0,
        ablation_attribution=0.08,
        compute_delta=1.0,
        capacity_growth=0.0,
    )
    metrics.update(metric_overrides)
    return QualificationBundle(
        proposal_digest=proposal.digest,
        candidate_digest=candidate,
        production_identity_digest=proposal.production_identity_digest,
        evaluator_id="independent-q",
        qualification_worlds=worlds,
        transfer_rings_passed=tuple(rings),
        fresh_one_shot_worlds=True,
        hidden_until_evaluation=True,
        metrics=QualificationMetrics(**metrics),
        suite_digest="sha256:" + "4" * 64,
    )


def test_trusted_evidence_ingest_distinguishes_observation_inference_and_simulation(tmp_path):
    ledger = EvidenceLedger(tmp_path / "evidence")
    ingestor = TrustedEvidenceIngestor(ledger, ingestor_id="runtime")
    env = ingestor.ingest(
        {"exit_code": 0}, origin_class=EvidenceOrigin.DETERMINISTIC_TOOL,
        production_identity_digest=IDENTITY, provenance={"tool": "pytest"},
    )
    inf = ingestor.ingest(
        {"claim": "likely fixed"}, origin_class=EvidenceOrigin.MODEL_INFERENCE,
        production_identity_digest=IDENTITY, provenance={"model": "frozen"},
    )
    sim = ingestor.ingest(
        {"predicted": "pass"}, origin_class=EvidenceOrigin.SIMULATION,
        production_identity_digest=IDENTITY, provenance={"world_model": "wm1"},
    )
    assert env.evidence_class is EvidenceClass.OBSERVED and env.promotion_eligible
    assert inf.evidence_class is EvidenceClass.DERIVED and inf.promotion_eligible
    assert sim.evidence_class is EvidenceClass.SIMULATED and not sim.promotion_eligible
    assert ledger.verify()["records"] == 3
    assert not hasattr(ledger, "append")


def test_evidence_payload_tamper_is_detected(tmp_path):
    ledger = EvidenceLedger(tmp_path / "evidence")
    ingestor = TrustedEvidenceIngestor(ledger, ingestor_id="runtime")
    r = ingestor.ingest(
        {"x": 1}, origin_class=EvidenceOrigin.ENVIRONMENT,
        production_identity_digest=IDENTITY, provenance={"source": "env"},
    )
    path = ledger.payloads / f"{r.payload_digest.removeprefix('sha256:')}.json"
    path.write_text('{"x": 2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError):
        ledger.verify()


def test_plasticity_proposer_prefers_least_permanent_feasible_mechanism():
    p = LeastPermanentPlasticityProposer(proposer_id="p")
    low = ProposalOption(LearningAction.CREATE_PROCEDURE, LearningLevel.L3_PROCEDURAL_SKILL, 0.05, 0.01)
    high = ProposalOption(LearningAction.ADAPT, LearningLevel.L6_MODULE_ADAPTATION, 0.9, 0.01)
    chosen = p.select_option([high, low])
    assert chosen.level is LearningLevel.L3_PROCEDURAL_SKILL


def test_plasticity_proposer_has_no_promotion_authority(tmp_path):
    repo = SkillRepository(tmp_path / "skills")
    learning = LearningPlane(skills=repo)
    assert learning.can_promote is False
    assert not hasattr(learning, "promote")
    assert not hasattr(LeastPermanentPlasticityProposer(proposer_id="p"), "promote")


def test_grounded_replay_policy_only_receives_prefix_view():
    nodes = [
        ExperienceNode("r", None, {"state": "root"}, "sha256:" + "a" * 64, creation_index=0),
        ExperienceNode("a", "r", {"state": "A"}, "sha256:" + "b" * 64, score=1, creation_index=1),
        ExperienceNode("secret", "a", {"future_secret": 99}, "sha256:" + "c" * 64, score=5, creation_index=2),
    ]
    world = ReplayWorldCompiler.compile_grounded(
        world_id="w1", nodes=nodes, source_evidence_digests=["sha256:" + "d" * 64]
    )
    seen = []

    def policy(view, workers):
        seen.append([x.node_id for x in view.nodes])
        if view.round_index == 0:
            assert "secret" not in seen[-1]
            return ["r"]
        if view.round_index == 1:
            assert "secret" not in seen[-1]
            return ["a"]
        return []

    result = GroundedReplayEngine(max_rounds=5).evaluate(world, policy)
    assert result.best_score == 5
    assert seen[0] == ["r"]
    assert seen[1] == ["r", "a"]


def test_grounded_replay_rejects_hidden_node_selection():
    nodes = [
        ExperienceNode("r", None, {}, "sha256:" + "a" * 64, creation_index=0),
        ExperienceNode("hidden", "r", {}, "sha256:" + "b" * 64, creation_index=1),
    ]
    world = ReplayWorldCompiler.compile_grounded(
        world_id="w", nodes=nodes, source_evidence_digests=["sha256:" + "c" * 64]
    )
    with pytest.raises(ValueError):
        GroundedReplayEngine().evaluate(world, lambda view, workers: ["hidden"])


def test_simulation_is_structurally_not_promotion_evidence():
    sim = SimulationRecord("wm", "sha256:" + "a" * 64, {"outcome": 1}, 0.2)
    assert not sim.promotion_evidence_eligible


def test_promotion_gate_approves_clean_procedural_change():
    proposal = make_proposal()
    bundle = make_bundle(proposal)
    decision = IndependentQualificationGate().evaluate_policy(proposal, bundle)
    assert decision.verdict is PromotionVerdict.APPROVE
    assert not decision.reasons


def test_security_failure_cannot_be_traded_for_forward_transfer():
    proposal = make_proposal()
    bundle = make_bundle(proposal, forward_transfer_delta=100.0, security_regressions=1)
    decision = IndependentQualificationGate().evaluate_policy(proposal, bundle)
    assert decision.verdict is PromotionVerdict.REJECT
    assert "security regression" in decision.reasons


def test_unauthorized_write_is_hard_failure():
    proposal = make_proposal()
    bundle = make_bundle(proposal, unauthorized_writes=1)
    assert IndependentQualificationGate().evaluate_policy(proposal, bundle).verdict is PromotionVerdict.REJECT


def test_permanence_increases_required_evidence():
    gate = IndependentQualificationGate()
    p3 = make_proposal(LearningLevel.L3_PROCEDURAL_SKILL)
    p8 = make_proposal(LearningLevel.L8_SHARED_CONSOLIDATION, LearningAction.CONSOLIDATE)
    d3 = gate.evaluate_policy(p3, make_bundle(p3, worlds=8, rings=("episode", "procedure")))
    d8 = gate.evaluate_policy(p8, make_bundle(
        p8, worlds=8,
        rings=("episode", "procedure", "reusable_skill", "composition", "task_family", "new_domain", "extrapolation")
    ))
    assert d3.verdict is PromotionVerdict.APPROVE
    assert d8.verdict is PromotionVerdict.REJECT
    assert d8.required_worlds > d3.required_worlds


def test_fresh_hidden_qualification_is_required():
    proposal = make_proposal()
    bundle = make_bundle(proposal)
    bundle = replace(bundle, fresh_one_shot_worlds=False)
    d = IndependentQualificationGate().evaluate_policy(proposal, bundle)
    assert d.verdict is PromotionVerdict.REJECT
    assert any("fresh one-shot" in x for x in d.reasons)


def test_skill_promotion_requires_signed_authorization(tmp_path):
    repo = SkillRepository(tmp_path / "skills")
    skill = SkillManifest(
        skill_id="dep-resolve", version=1, name="Dependency Resolution",
        implementation_digest="sha256:" + "5" * 64,
        activation_conditions=("dependency conflict",), preconditions=("repository available",),
        contraindications=("offline lockfile policy",), permissions=("read_repo", "run_tests"),
        resource_budget={"seconds": 30, "tool_calls": 10}, termination_conditions=("tests pass",),
        verifier_digest="sha256:" + "6" * 64, supporting_evidence=("sha256:" + "2" * 64,),
    )
    repo.register_candidate(skill)
    proposal = make_proposal()
    bundle = make_bundle(proposal, candidate=skill.digest)
    gate, bundle, chain, _, _ = prepare_chain(tmp_path, proposal, bundle, skill)
    signer = Ed25519ReceiptSigner.generate(key_id="promotion")
    authority = PromotionAuthority(
        gate=gate,
        improvement_ledger=ImprovementLedger(tmp_path / "improvements.jsonl"),
        signer=signer,
    )
    decision, authorization, _ = authority.decide(
        proposal=proposal, bundle=bundle, research_chain=chain, origin_evidence=proposal.evidence_digests,
        affected_components=(skill.skill_id,), production_identity_before=IDENTITY,
    )
    assert decision.verdict is PromotionVerdict.APPROVE and authorization is not None
    promoted = repo.promote(skill.digest, authorization, signer.verifier())
    assert promoted.qualification_digest == bundle.digest


def test_skill_promotion_rejects_unsigned_or_mismatched_authorization(tmp_path):
    repo = SkillRepository(tmp_path / "skills")
    skill = SkillManifest(
        skill_id="s", version=1, name="S", implementation_digest="sha256:" + "5" * 64,
        activation_conditions=(), preconditions=(), contraindications=(), permissions=(),
        resource_budget={}, termination_conditions=(), verifier_digest="sha256:" + "6" * 64,
        supporting_evidence=("sha256:" + "2" * 64,),
    )
    repo.register_candidate(skill)
    signer = Ed25519ReceiptSigner.generate(key_id="p")
    fake_body = {
        "schema": "mini-agi-egai-promotion-authorization-v2", "candidate_digest": "sha256:" + "9" * 64,
        "verdict": "approve", "qualification_digest": "sha256:q",
    }
    fake = signer.issue(fake_body)
    with pytest.raises(PermissionError):
        repo.promote(skill.digest, fake, signer.verifier())


def test_rejected_promotion_emits_no_authorization(tmp_path):
    proposal = make_proposal()
    bundle = make_bundle(proposal, security_regressions=1)
    signer = Ed25519ReceiptSigner.generate(key_id="promotion")
    authority = PromotionAuthority(
        gate=IndependentQualificationGate(),
        improvement_ledger=ImprovementLedger(tmp_path / "improvements.jsonl"), signer=signer,
    )
    decision, auth, _ = authority.decide(
        proposal=proposal, bundle=bundle, origin_evidence=proposal.evidence_digests,
        affected_components=("skill",), production_identity_before=IDENTITY,
    )
    assert decision.verdict is PromotionVerdict.REJECT
    assert auth is None


def test_improvement_ledger_records_future_consequence_for_meta_learning(tmp_path):
    proposal = make_proposal()
    bundle = make_bundle(proposal)
    signer = Ed25519ReceiptSigner.generate(key_id="promotion")
    ledger = ImprovementLedger(tmp_path / "improvements.jsonl")
    authority = PromotionAuthority(gate=IndependentQualificationGate(), improvement_ledger=ledger, signer=signer)
    _, _, improvement = authority.decide(
        proposal=proposal, bundle=bundle, origin_evidence=proposal.evidence_digests,
        affected_components=("skill",), production_identity_before=IDENTITY,
    )
    ledger.append_consequence(FutureConsequence(
        improvement_digest=improvement.digest, observation_horizon="100 future tasks",
        forward_transfer=0.12, retention_delta=-0.001, calibration_delta=-0.01,
        compute_delta=0.02, capacity_delta=0.0,
    ))
    rows = ledger.meta_learning_rows()
    assert len(rows) == 1
    assert rows[0]["future_consequence"]["forward_transfer"] == 0.12


def test_improvement_ledger_detects_tamper(tmp_path):
    path = tmp_path / "improvements.jsonl"
    ledger = ImprovementLedger(path)
    proposal = make_proposal()
    bundle = make_bundle(proposal)
    signer = Ed25519ReceiptSigner.generate(key_id="promotion")
    PromotionAuthority(gate=IndependentQualificationGate(), improvement_ledger=ledger, signer=signer).decide(
        proposal=proposal, bundle=bundle, origin_evidence=proposal.evidence_digests,
        affected_components=("x",), production_identity_before=IDENTITY,
    )
    line = json.loads(path.read_text().splitlines()[0])
    line["payload"]["metrics"]["forward_transfer_delta"] = 999
    path.write_text(json.dumps(line) + "\n")
    with pytest.raises(RuntimeError):
        ledger.verify()


def test_frozen_foundation_protocol_rejects_model_change():
    spec = FrozenFoundationExperimentSpec(
        base_model_digest="sha256:" + "a" * 64,
        sealed_future_task_digest="sha256:" + "b" * 64,
        checkpoints=(0, 50),
    )
    protocol = FrozenFoundationProtocol(spec)
    protocol.record(ExperimentCheckpoint(0, spec.base_model_digest, 0.5, 0.8, 0.1, 100, 4, 10, 0.0, 0.3))
    with pytest.raises(RuntimeError):
        protocol.record(ExperimentCheckpoint(50, "sha256:" + "c" * 64, 0.6, 0.8, 0.09, 90, 3, 9, 0.2, 0.2))


def test_frozen_foundation_protocol_tests_primary_hypothesis():
    spec = FrozenFoundationExperimentSpec(
        base_model_digest="sha256:" + "a" * 64,
        sealed_future_task_digest="sha256:" + "b" * 64,
        checkpoints=(0, 50), max_retention_drop=0.01,
    )
    protocol = FrozenFoundationProtocol(spec)
    protocol.record(ExperimentCheckpoint(0, spec.base_model_digest, 0.48, 0.80, 0.10, 100, 4, 10, 0.0, 0.3))
    protocol.record(ExperimentCheckpoint(50, spec.base_model_digest, 0.55, 0.795, 0.09, 90, 3, 9, 0.2, 0.2))
    summary = protocol.summary()
    assert summary["passes_primary_hypothesis"]
    assert summary["future_task_delta"] == pytest.approx(0.07)


def test_compute_fte_penalizes_compute_and_capacity():
    cheap = compute_fte(0.5, 0.6, experience=100)
    expensive = compute_fte(0.5, 0.6, experience=100, compute_cost=100, capacity_growth=100)
    assert cheap > expensive > 0


def test_cognitive_plane_has_read_only_evidence_and_promoted_skills(tmp_path):
    ledger = EvidenceLedger(tmp_path / "e")
    repo = SkillRepository(tmp_path / "s")
    cognitive = CognitivePlane(evidence=ledger, skills=repo)
    assert cognitive.plane.value == "cognitive"
    assert not hasattr(cognitive, "promote")


def test_promoted_belief_requires_qualification_digest():
    with pytest.raises(ValueError):
        Belief(
            belief_id="B-1", statement_digest="sha256:" + "a" * 64,
            supporting_evidence=("sha256:" + "b" * 64,), contradicting_evidence=(),
            confidence=0.9, production_identity_digest=IDENTITY, status=BeliefStatus.PROMOTED,
        )
    b = Belief(
        belief_id="B-1", statement_digest="sha256:" + "a" * 64,
        supporting_evidence=("sha256:" + "b" * 64,), contradicting_evidence=(),
        confidence=0.9, production_identity_digest=IDENTITY, status=BeliefStatus.PROMOTED,
        qualification_digest="sha256:" + "c" * 64,
    )
    assert b.status is BeliefStatus.PROMOTED


def test_hypothesis_requires_falsification_tests_and_known_transfer_rings():
    with pytest.raises(ValueError):
        Hypothesis(
            hypothesis_id="H-1", claim_digest="sha256:" + "d" * 64,
            evidence_digests=("sha256:" + "e" * 64,), falsification_tests=(),
            predicted_transfer_rings=("procedure",), production_identity_digest=IDENTITY,
        )
    h = Hypothesis(
        hypothesis_id="H-1", claim_digest="sha256:" + "d" * 64,
        evidence_digests=("sha256:" + "e" * 64,), falsification_tests=("counterexample-search",),
        predicted_transfer_rings=("procedure", "reusable_skill"), production_identity_digest=IDENTITY,
        status=HypothesisStatus.OPEN,
    )
    assert h.status is HypothesisStatus.OPEN


def test_belief_persistence_requires_signed_promotion(tmp_path):
    repo = BeliefRepository(tmp_path / "beliefs")
    belief = Belief(
        belief_id="B-deps", statement_digest="sha256:" + "1" * 64,
        supporting_evidence=("sha256:" + "2" * 64,), contradicting_evidence=(),
        confidence=0.9, production_identity_digest=IDENTITY,
    )
    repo.register_candidate(belief)
    proposal = make_proposal(LearningLevel.L2_BELIEF_UPDATE, LearningAction.REVISE_BELIEF)
    bundle = make_bundle(proposal, candidate=belief.digest, worlds=4, rings=("episode",))
    gate, bundle, chain, _, _ = prepare_chain(tmp_path, proposal, bundle, belief)
    signer = Ed25519ReceiptSigner.generate(key_id="belief-promotion")
    authority = PromotionAuthority(
        gate=gate,
        improvement_ledger=ImprovementLedger(tmp_path / "improvements.jsonl"), signer=signer,
    )
    decision, auth, _ = authority.decide(
        proposal=proposal, bundle=bundle, research_chain=chain, origin_evidence=proposal.evidence_digests,
        affected_components=(belief.belief_id,), production_identity_before=IDENTITY,
    )
    assert decision.verdict is PromotionVerdict.APPROVE and auth is not None
    promoted = repo.promote(belief.digest, auth, signer.verifier())
    assert promoted.status is BeliefStatus.PROMOTED
    assert promoted.qualification_digest == bundle.digest


def test_learning_plane_can_stage_belief_candidate_but_not_promote(tmp_path):
    beliefs = BeliefRepository(tmp_path / "beliefs")
    skills = SkillRepository(tmp_path / "skills")
    plane = LearningPlane(skills=skills, beliefs=beliefs)
    belief = Belief(
        belief_id="B1", statement_digest="sha256:" + "a" * 64,
        supporting_evidence=("sha256:" + "b" * 64,), contradicting_evidence=(),
        confidence=0.6, production_identity_digest=IDENTITY,
    )
    digest = beliefs.register_candidate(belief)
    assert digest == belief.digest
    assert not hasattr(plane, "promote_belief")
