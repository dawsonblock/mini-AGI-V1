
import pytest

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer
from minagi.integration.qw3_state import GovernedServingContract, RuntimeStateMismatch, ServedArtifactManifest
from minagi.v14.storage import ImmutableCAS
from minagi.v14.trajectory_v145 import (
    HeldOutTransferQualifierV145,
    TrajectorySkillInducerV145,
    VerifiedTrajectoryV145,
)
from minagi.v14.experiment_v145 import FrozenBaselineGateV145
from minagi.v14.fresh_tasks_v143 import FreshTaskMetadataStoreV143, FreshTaskVaultAuthorityV143
from minagi.v15.skill_ir import SkillIRCompiler, SkillIRExecutor, SkillPolicyRuntime
from minagi.v15.sealed_learning import FreshTaskPairedEvaluator
from minagi.v15.learning_loop import GovernedSkillLearningLoop


def d(label: str) -> str:
    return digest({"label": label})


def traj(i: int, inp: str, out: str, family: str = "normalize.upper") -> VerifiedTrajectoryV145:
    return VerifiedTrajectoryV145(
        trajectory_id=f"t-{i}",
        task_family=family,
        input_text=inp,
        attempted_output=inp,
        repaired_output=out,
        production_identity_digest=d("prod"),
        evidence_digests=(d(f"ev-{i}"),),
        verification_receipt_digests=(d(f"vr-{i}"),),
    )


def qualified_candidate(cas):
    report = TrajectorySkillInducerV145().induce((traj(1, "abc", "ABC"), traj(2, "def", "DEF")))
    assert len(report.candidates) == 1
    q = HeldOutTransferQualifierV145(evaluator_id="heldout", minimum_score=1.0, cas=cas)
    return q.qualify(
        report.candidates[0],
        (traj(3, "ghi", "GHI"),),
        lambda candidate, t: 1.0 if t.repaired_output == t.input_text.upper() else 0.0,
    )


def base_manifest():
    return ServedArtifactManifest(
        epoch_digest=d("epoch-0"),
        runtime_manifest_digest=d("runtime-0"),
        foundation_digest="1" * 64,
        tokenizer_digest="2" * 64,
        kv_archive_root="3" * 64,
        adapter_set_root="4" * 64,
        retrieval_policy_root="5" * 64,
        skill_policy_root="6" * 64,
        runtime_binary_digest="7" * 64,
    )


def test_compiled_skill_ir_executes_transfer_qualified_candidate(tmp_path):
    cas = ImmutableCAS(tmp_path / "cas")
    candidate = qualified_candidate(cas)
    skill = SkillIRCompiler().compile(candidate)
    assert SkillIRExecutor().execute(skill, "hello") == "HELLO"
    assert skill.source_candidate_digest == candidate.digest


def test_compiler_rejects_nonqualified_candidate(tmp_path):
    ImmutableCAS(tmp_path / "cas")
    report = TrajectorySkillInducerV145().induce((traj(1, "abc", "ABC"), traj(2, "def", "DEF")))
    with pytest.raises(PermissionError):
        SkillIRCompiler().compile(report.candidates[0])


def test_fresh_task_a0_a1_learning_changes_served_policy_root(tmp_path):
    cas = ImmutableCAS(tmp_path / "cas")
    candidate = qualified_candidate(cas)
    skill = SkillIRCompiler().compile(candidate)

    signer = Ed25519Signer.generate("fresh-task")
    metadata = FreshTaskMetadataStoreV143(tmp_path / "fresh-meta.sqlite3")
    vault = FreshTaskVaultAuthorityV143(
        tmp_path / "fresh-secret.sqlite3",
        authority_id="fresh-task-vault",
        authority_generation=1,
        signer=signer,
    )
    commitments = tuple(
        vault.seal(
            {"family_id": "normalize.upper", "ring": "R0", "input": x, "expected": x.upper()},
            generation=1,
            metadata=metadata,
        )
        for x in ("one", "two", "three", "four")
    )

    # Candidate policy is frozen before fresh tasks are revealed.
    from minagi.v15.skill_ir import SkillPolicyBundle
    bundle = SkillPolicyBundle.from_skills((skill,))
    a1_runtime = SkillPolicyRuntime(bundle)
    gate = FrozenBaselineGateV145(required_rings=("R0",), minimum_effect=0.0,
                                  minimum_retention=0.95, bootstrap_iterations=600, seed=7)
    evaluator = FreshTaskPairedEvaluator(vault=vault, metadata=metadata, cas=cas, gate=gate)
    experiment = evaluator.evaluate(
        commitments=commitments,
        foundation_model_digest=d("foundation"),
        production_identity_digest=d("prod"),
        a0=lambda family, text: text,
        a1=lambda family, text: a1_runtime.run(task_family=family, input_text=text),
    )
    assert experiment.baseline_report.decision == "PASS"
    assert experiment.baseline_report.mean_delta == 1.0
    assert experiment.baseline_report.ci95[0] > 0.0

    q, learned_bundle, learned_skill = GovernedSkillLearningLoop(cas=cas).prepare(
        candidate=candidate, experiment=experiment
    )
    assert q.decision == "PASS"
    assert q.skill_policy_root == learned_bundle.root_hex
    assert learned_skill.digest == skill.digest

    old = base_manifest()
    new = GovernedSkillLearningLoop.prepare_served_manifest(base=old, qualified=q)
    assert new.skill_policy_root != old.skill_policy_root
    assert new.artifact_root != old.artifact_root
    assert new.manifest_digest != old.manifest_digest


def test_old_qw3_state_is_rejected_after_learning_policy_changes(tmp_path):
    cas = ImmutableCAS(tmp_path / "cas")
    candidate = qualified_candidate(cas)
    skill = SkillIRCompiler().compile(candidate)
    from minagi.v15.skill_ir import SkillPolicyBundle
    bundle = SkillPolicyBundle.from_skills((skill,))

    signer = Ed25519Signer.generate("fresh-task")
    metadata = FreshTaskMetadataStoreV143(tmp_path / "fresh-meta.sqlite3")
    vault = FreshTaskVaultAuthorityV143(tmp_path / "fresh-secret.sqlite3", authority_id="vault",
                                        authority_generation=1, signer=signer)
    commitments = tuple(vault.seal(
        {"family_id": "normalize.upper", "ring": "R0", "input": x, "expected": x.upper()},
        generation=1, metadata=metadata) for x in ("a", "b", "c"))
    exp = FreshTaskPairedEvaluator(
        vault=vault, metadata=metadata, cas=cas,
        gate=FrozenBaselineGateV145(required_rings=("R0",), bootstrap_iterations=500),
    ).evaluate(
        commitments=commitments,
        foundation_model_digest=d("foundation"), production_identity_digest=d("prod"),
        a0=lambda family, text: text,
        a1=lambda family, text: SkillPolicyRuntime(bundle).run(task_family=family, input_text=text),
    )
    q, _, _ = GovernedSkillLearningLoop(cas=cas).prepare(candidate=candidate, experiment=exp)
    old = base_manifest()
    new = GovernedSkillLearningLoop.prepare_served_manifest(base=old, qualified=q)

    class Lease:
        epoch_digest = new.epoch_digest

    old_state = {"governed": True, "epoch_id": new.epoch_digest,
                 "manifest_digest": old.manifest_digest, "artifact_root": old.artifact_root,
                 "adapter_set_root": old.adapter_set_root}
    contract = GovernedServingContract(base_url="http://unused", fetch_json=lambda path: old_state)
    with pytest.raises(RuntimeStateMismatch):
        contract.request_headers(lease=Lease(), manifest=new)

    new_state = {"governed": True, "epoch_id": new.epoch_digest,
                 "manifest_digest": new.manifest_digest, "artifact_root": new.artifact_root,
                 "adapter_set_root": new.adapter_set_root}
    contract = GovernedServingContract(base_url="http://unused", fetch_json=lambda path: new_state)
    headers = contract.request_headers(lease=Lease(), manifest=new)
    assert headers["X-MiniAGI-Artifact-Root"] == new.artifact_root


def test_blocked_experiment_cannot_change_served_state(tmp_path):
    cas = ImmutableCAS(tmp_path / "cas")
    candidate = qualified_candidate(cas)
    signer = Ed25519Signer.generate("fresh-task")
    metadata = FreshTaskMetadataStoreV143(tmp_path / "meta.sqlite3")
    vault = FreshTaskVaultAuthorityV143(tmp_path / "vault.sqlite3", authority_id="vault", authority_generation=1, signer=signer)
    commitments = tuple(vault.seal(
        {"family_id": "normalize.upper", "ring": "R0", "input": x, "expected": x.upper()}, generation=1, metadata=metadata)
        for x in ("x", "y", "z"))
    exp = FreshTaskPairedEvaluator(
        vault=vault, metadata=metadata, cas=cas,
        gate=FrozenBaselineGateV145(required_rings=("R0",), bootstrap_iterations=500),
    ).evaluate(
        commitments=commitments,
        foundation_model_digest=d("foundation"), production_identity_digest=d("prod"),
        a0=lambda family, text: text.upper(),
        a1=lambda family, text: text.upper(),
    )
    assert exp.baseline_report.decision == "BLOCK"
    q, _, _ = GovernedSkillLearningLoop(cas=cas).prepare(candidate=candidate, experiment=exp)
    with pytest.raises(PermissionError):
        GovernedSkillLearningLoop.prepare_served_manifest(base=base_manifest(), qualified=q)
