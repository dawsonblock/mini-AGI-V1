from dataclasses import replace
import pytest

from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from minagi.v14 import (
    EvidenceStrength, GovernancePolicyV14, ImmutableCAS, GovernanceDBV144,
    EpisodeVerificationAuthority, EpisodeVerificationValidator,
    EvidenceStageAuthorityV144, EvidenceStageValidatorV144,
    EvidenceStrengthAuthorityV144, EvidenceStrengthValidatorV144,
    VerificationAuthorityClientV144, EvidenceStageAuthorityClientV144, EvidenceStrengthAuthorityClientV144,
    EvaluationAuthorityV143, EvaluationValidatorV143, EvaluationAuthorityClientV143, EvaluationCaseResultV143,
    QualificationAuthorityV143, QualificationValidatorV143, QualificationAuthorityClientV143, QualificationPolicyV142,
    PromotionAuthorityV143, PromotionAuthorizationValidatorV143, PromotionAuthorityClientV143,
    GovernedContinualRuntimeV144, RUNTIME_COMPONENT_SCHEMAS_V144,
)


def pair(name, verifier=None):
    s = Ed25519Signer.generate(name)
    v = verifier or Ed25519Verifier()
    v.register(s.key_id, s.public_bytes())
    return s, v


def stack(tmp_path):
    root = tmp_path / "rt"
    cas = ImmutableCAS(root / "cas")
    db = GovernanceDBV144(root / "governance.sqlite3")
    policy = GovernancePolicyV14()

    shared = Ed25519Verifier()
    verify_s, _ = pair("episode-verifier", shared)
    stage_s, _ = pair("e2-stage", shared)
    strength_s, _ = pair("evidence-strength", shared)
    eval_s, _ = pair("evaluator", shared)
    qual_s, _ = pair("qualifier", shared)
    prom_s, _ = pair("promoter", shared)

    episode_authority = EpisodeVerificationAuthority(verifier_id="episode-verifier", signer=verify_s)
    episode_validator = EpisodeVerificationValidator(verifier=shared, trusted_key_ids={verify_s.key_id})
    stage_validator = EvidenceStageValidatorV144(
        verifier=shared, episode_trusted_key_ids={verify_s.key_id},
        stage_key_policy={2: {stage_s.key_id}}, authority_generation=1, cas=cas,
    )
    stage_authority = EvidenceStageAuthorityV144(
        authority_id="verification-stage", authority_generation=1, signer=stage_s,
        validator=stage_validator, cas=cas,
    )
    strength_authority = EvidenceStrengthAuthorityV144(
        authority_id="evidence-strength", authority_generation=1, signer=strength_s,
        stage_validator=stage_validator, cas=cas,
    )
    strength_validator = EvidenceStrengthValidatorV144(
        verifier=shared, trusted_key_ids={strength_s.key_id}, authority_generation=1,
        stage_validator=stage_validator, cas=cas,
    )

    eval_validator = EvaluationValidatorV143(verifier=shared, trusted_key_ids={eval_s.key_id}, evaluator_generation=1, cas=cas)
    eval_authority = EvaluationAuthorityV143(evaluator_id="eval", evaluator_generation=1, signer=eval_s, cas=cas)
    qual_validator = QualificationValidatorV143(verifier=shared, trusted_key_ids={qual_s.key_id}, qualifier_generation=1,
                                                authority_generation=1, policy_generation=1)
    qual_authority = QualificationAuthorityV143(
        qualifier_id="qual", qualifier_generation=1, authority_generation=1, policy_generation=1,
        signer=qual_s, evaluation_validator=eval_validator, cas=cas, policy=QualificationPolicyV142(),
    )
    prom_validator = PromotionAuthorizationValidatorV143(verifier=shared, trusted_key_ids={prom_s.key_id},
                                                          authority_generation=1, policy_generation=1)
    prom_authority = PromotionAuthorityV143(governance_db=db, signer=prom_s, qualification_validator=qual_validator,
                                             authority_generation=1, policy_generation=1)

    rt = GovernedContinualRuntimeV144(
        root, policy=policy,
        verification_client=VerificationAuthorityClientV144(episode_authority), verification_validator=episode_validator,
        evidence_stage_client=EvidenceStageAuthorityClientV144(stage_authority),
        evidence_client=EvidenceStrengthAuthorityClientV144(strength_authority), evidence_validator=strength_validator,
        evaluation_client=EvaluationAuthorityClientV143(eval_authority), evaluation_validator=eval_validator,
        qualification_client=QualificationAuthorityClientV143(qual_authority), qualification_validator=qual_validator,
        promotion_client=PromotionAuthorityClientV143(prom_authority), promotion_validator=prom_validator,
        cas=cas, governance_db=db,
    )
    return rt


def cases(rt):
    def c(cid, cat, score, passed=True, ring=""):
        raw = rt.put_supporting_receipt("raw-eval", {"case": cid, "category": cat, "score": score, "passed": passed, "ring": ring})
        return EvaluationCaseResultV143(cid, cat, score, passed, raw, ring)
    return (
        c("ret", "retention", .99), c("sec", "security", 1.0), c("fg", "forgetting", .01),
        c("ood", "ood_delta", .02), c("r0", "transfer", 1, True, "R0"), c("r1", "transfer", 1, True, "R1"),
        c("r2", "transfer", 1, True, "R2"), c("neg", "negative_control", 1), c("fresh", "fresh_hidden", 1),
        c("fals", "falsification", 1),
    )


def components():
    return {k: {"component": k} for k in RUNTIME_COMPONENT_SCHEMAS_V144}


def test_arbitrary_cas_receipt_cannot_create_e2_strength(tmp_path):
    rt = stack(tmp_path)
    exp, receipt, verified = rt.ingest_verified_experience(
        prompt="q", attempted_output="bad", repaired_output="good", expected_output="good",
        evidence_payload={"fact": "x"}, production_identity_digest="sha256:" + "4" * 64,
    )
    fake = rt.put_supporting_receipt("verification", {"ok": True})
    with pytest.raises((PermissionError, TypeError, ValueError, KeyError)):
        rt.evidence_stage_client.issue(stage_strength=2, evidence_digests=(verified.evidence_root_digest,),
                                       parent_receipt_digests=(fake,))


def test_episode_verification_must_bind_same_evidence_root(tmp_path):
    rt = stack(tmp_path)
    exp, receipt, verified = rt.ingest_verified_experience(
        prompt="q", attempted_output="bad", repaired_output="good", expected_output="good",
        evidence_payload={"fact": "x"}, production_identity_digest="sha256:" + "4" * 64,
    )
    other = rt.put_evidence({"different": True})
    rd = rt.cas.put_json(receipt.__dict__)
    with pytest.raises(PermissionError, match="does not bind evidence root"):
        rt.evidence_stage_client.issue(stage_strength=2, evidence_digests=(other,), parent_receipt_digests=(rd,))


def test_e2_strength_is_cryptographically_chained(tmp_path):
    rt = stack(tmp_path)
    exp, receipt, verified = rt.ingest_verified_experience(
        prompt="q", attempted_output="bad", repaired_output="good", expected_output="good",
        evidence_payload={"fact": "x"}, production_identity_digest="sha256:" + "4" * 64,
    )
    stage, proof = rt.issue_e2_evidence_proof(verified_record=verified, verification_receipt=receipt)
    assert stage.stage_strength == 2
    assert rt.evidence_validator.validate(proof, expected_evidence_digests=(verified.evidence_root_digest,)) == EvidenceStrength.E2_INDEPENDENTLY_VERIFIED
    tampered = replace(proof, derived_strength=5)
    with pytest.raises(PermissionError):
        rt.evidence_validator.validate(tampered, expected_evidence_digests=(verified.evidence_root_digest,))


def test_full_experience_to_belief_to_runtime_cycle_is_atomic(tmp_path):
    rt = stack(tmp_path)
    out = rt.full_belief_learning_cycle(
        prompt="capital?", attempted_output="old", repaired_output="new", expected_output="new",
        evidence_payload={"source": "verified-source", "fact": "new"},
        production_identity_digest="sha256:" + "4" * 64,
        rationale_digest="sha256:" + "3" * 64, proposer_id="learner", target="belief:capital",
        belief_subject="country:x", belief_predicate="capital", belief_object="new",
        belief_valid_from=1.0, belief_confidence=.99,
        case_results=cases(rt), runtime_component_payloads=components(),
    )
    assert out["status"] == "ACTIVE"
    b = out["belief_revision"]
    assert rt.db.belief_head(b.belief_id)["belief_digest"] == b.digest
    assert rt.db.current_runtime_manifest() == out["runtime_manifest"].digest
    row = rt.db.conn.execute("SELECT consumed_at FROM mutation_commitments WHERE authorization_digest=? AND mutation_scope='belief.promote' AND target_digest=?",
                             (out["authorization"].digest, b.digest)).fetchone()
    assert row is not None and row["consumed_at"] is not None


def test_runtime_snapshot_must_include_exact_belief_mutation(tmp_path):
    rt = stack(tmp_path)
    exp, receipt, verified = rt.ingest_verified_experience(
        prompt="q", attempted_output="bad", repaired_output="good", expected_output="good",
        evidence_payload={"fact": "x"}, production_identity_digest="sha256:" + "4" * 64,
    )
    stage, proof = rt.issue_e2_evidence_proof(verified_record=verified, verification_receipt=receipt)
    from minagi.v14 import LearningProposalV14, GovernanceCoordinates, PermanenceLevel, LearningMechanism
    p = LearningProposalV14.create(target="belief:x", mechanism=LearningMechanism.BELIEF_UPDATE,
        coordinates=GovernanceCoordinates(PermanenceLevel.L3_SEMANTIC_BELIEF, EvidenceStrength.E2_INDEPENDENTLY_VERIFIED),
        evidence_digests=(verified.evidence_root_digest,), rationale_digest="sha256:" + "3"*64, proposer_id="l",
        production_identity_digest="sha256:" + "4"*64, expected_gain=.1, expected_interference=0)
    c = rt.register_candidate(p, evidence_proof=proof, candidate_payload={})
    belief = rt.persistence.prepare_belief(subject="s", predicate="p", object_value="o", valid_from=1,
                                            evidence_digests=(verified.evidence_root_digest,), confidence=.9)
    b = rt.record_build(candidate_digest=c, build_payload={}, mutation_targets={"belief.promote": belief.digest})
    e = rt.evaluate(candidate_digest=c, build_digest=b, case_results=cases(rt)); q = rt.qualify(e)
    with pytest.raises(PermissionError, match="snapshot does not include"):
        rt.prepare_runtime_manifest(qualification=q, runtime_component_payloads=components())


def test_atomic_activation_rolls_back_learned_state_if_mutation_body_invalid(tmp_path):
    rt = stack(tmp_path)
    # Run normally until authorization, then remove the CAS target body to force pre-transaction failure.
    exp, receipt, verified = rt.ingest_verified_experience(
        prompt="q", attempted_output="bad", repaired_output="good", expected_output="good",
        evidence_payload={"fact": "x"}, production_identity_digest="sha256:" + "4" * 64,
    )
    stage, proof = rt.issue_e2_evidence_proof(verified_record=verified, verification_receipt=receipt)
    from minagi.v14 import LearningProposalV14, GovernanceCoordinates, PermanenceLevel, LearningMechanism
    p = LearningProposalV14.create(target="belief:x", mechanism=LearningMechanism.BELIEF_UPDATE,
        coordinates=GovernanceCoordinates(PermanenceLevel.L3_SEMANTIC_BELIEF, EvidenceStrength.E2_INDEPENDENTLY_VERIFIED),
        evidence_digests=(verified.evidence_root_digest,), rationale_digest="sha256:" + "3"*64, proposer_id="l",
        production_identity_digest="sha256:" + "4"*64, expected_gain=.1, expected_interference=0)
    c = rt.register_candidate(p, evidence_proof=proof, candidate_payload={})
    belief = rt.persistence.prepare_belief(subject="s", predicate="p", object_value="o", valid_from=1,
                                            evidence_digests=(verified.evidence_root_digest,), confidence=.9)
    b = rt.record_build(candidate_digest=c, build_payload={}, mutation_targets={"belief.promote": belief.digest})
    e = rt.evaluate(candidate_digest=c, build_digest=b, case_results=cases(rt)); q = rt.qualify(e)
    comp = components(); comp["belief_snapshot_digest"]["included_mutation_digests"]=[belief.digest]
    m = rt.prepare_runtime_manifest(qualification=q, runtime_component_payloads=comp); a = rt.authorize(q,m)
    target_path = rt.cas._path(belief.digest); saved = target_path.read_bytes(); target_path.unlink()
    with pytest.raises(FileNotFoundError): rt.activate(auth=a, manifest=m)
    assert rt.db.belief_head(belief.belief_id) is None
    assert rt.db.current_runtime_manifest() is None
    target_path.parent.mkdir(parents=True, exist_ok=True); target_path.write_bytes(saved)


def test_production_mode_requires_remote_new_authorities(tmp_path):
    rt = stack(tmp_path)
    with pytest.raises(RuntimeError, match="out-of-process verification"):
        GovernedContinualRuntimeV144(tmp_path/"prod", policy=rt.policy,
            verification_client=rt.verification_client, verification_validator=rt.verification_validator,
            evidence_stage_client=rt.evidence_stage_client, evidence_client=rt.evidence_client, evidence_validator=rt.evidence_validator,
            evaluation_client=rt.evaluation_client, evaluation_validator=rt.evaluation_validator,
            qualification_client=rt.qualification_client, qualification_validator=rt.qualification_validator,
            promotion_client=rt.promotion_client, promotion_validator=rt.promotion_validator,
            production_mode=True)
