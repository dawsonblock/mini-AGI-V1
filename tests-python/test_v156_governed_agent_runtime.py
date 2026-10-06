from dataclasses import asdict
from pathlib import Path

import pytest

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer
from minagi.integration.qw3_state import GovernedServingContract, RuntimeStateMismatch, ServedArtifactManifest
from minagi.v14.experiment_v145 import FrozenBaselineGateV145
from minagi.v14.fresh_tasks_v143 import FreshTaskMetadataStoreV143, FreshTaskVaultAuthorityV143
from minagi.v14.storage import ImmutableCAS
from minagi.v15.agent_runtime import GovernedAgentRuntime
from minagi.v15.artifact_binding import materialize_canonical_artifact
from minagi.v15.retrieval_policy import EpisodicMemoryRecord, RetrievalPolicy
from minagi.v15.sealed_learning import FreshTaskPairedEvaluator
from minagi.v15.skill_ir import SkillIR, SkillInstruction, SkillPolicyBundle

ZERO = "0" * 64


def d(label: str) -> str:
    return digest({"label": label})


def make_skill() -> SkillIR:
    return SkillIR(
        skill_id="reverse-v1", task_family="text.reverse",
        instructions=(SkillInstruction("reverse"),),
        source_candidate_digest=d("candidate"),
        supporting_trajectory_digests=(d("traj1"), d("traj2")),
        transfer_receipt_digest=d("transfer"),
    )


def materialized_policies(tmp_path: Path):
    bundle = SkillPolicyBundle.from_skills((make_skill(),))
    retrieval = RetrievalPolicy(top_k=2, minimum_score=0.0, family_bonus=2.0, lexical_weight=1.0)
    skill_path = tmp_path / "skill-policy.json"
    retrieval_path = tmp_path / "retrieval-policy.json"
    materialize_canonical_artifact(asdict(bundle), skill_path)
    materialize_canonical_artifact(asdict(retrieval), retrieval_path)
    return bundle, retrieval, skill_path, retrieval_path


def manifest_for(bundle, retrieval):
    return ServedArtifactManifest(
        epoch_digest=d("epoch-156"), runtime_manifest_digest=d("runtime-156"),
        foundation_digest="1" * 64, tokenizer_digest="2" * 64,
        kv_archive_root=ZERO, adapter_set_root=ZERO,
        retrieval_policy_root=retrieval.root_hex,
        skill_policy_root=bundle.root_hex,
        runtime_binary_digest="7" * 64,
    )


def contract_for(manifest):
    state = {
        "governed": True,
        "epoch_id": manifest.epoch_digest,
        "manifest_digest": manifest.manifest_digest,
        "artifact_root": manifest.artifact_root,
        "adapter_set_root": manifest.adapter_set_root,
    }
    return GovernedServingContract(base_url="http://unused", fetch_json=lambda _: state)


class Lease:
    def __init__(self, epoch): self.epoch_digest = epoch


def test_bound_skill_policy_is_actually_executed(tmp_path):
    bundle, retrieval, skill_path, retrieval_path = materialized_policies(tmp_path)
    manifest = manifest_for(bundle, retrieval)
    runtime = GovernedAgentRuntime(
        lease=Lease(manifest.epoch_digest), manifest=manifest,
        serving_contract=contract_for(manifest),
        skill_policy_artifact=skill_path, retrieval_policy_artifact=retrieval_path,
    )
    result = runtime.run(task_family="text.reverse", input_text="abcdef")
    assert result.output == "fedcba"
    assert result.receipt.route == "skill_ir"
    assert result.receipt.skill_id == "reverse-v1"
    assert result.receipt.skill_policy_root == manifest.skill_policy_root


def test_runtime_rejects_policy_tamper_after_start(tmp_path):
    bundle, retrieval, skill_path, retrieval_path = materialized_policies(tmp_path)
    manifest = manifest_for(bundle, retrieval)
    runtime = GovernedAgentRuntime(
        lease=Lease(manifest.epoch_digest), manifest=manifest,
        serving_contract=contract_for(manifest),
        skill_policy_artifact=skill_path, retrieval_policy_artifact=retrieval_path,
    )
    skill_path.write_bytes(skill_path.read_bytes() + b"\n")
    with pytest.raises(RuntimeStateMismatch, match="skill_policy physical root mismatch"):
        runtime.run(task_family="text.reverse", input_text="abc")


def test_bound_retrieval_policy_controls_qw3_context(tmp_path):
    bundle, retrieval, skill_path, retrieval_path = materialized_policies(tmp_path)
    manifest = manifest_for(bundle, retrieval)
    records = (
        EpisodicMemoryRecord("r1", "qa", "capital france", "Paris", d("e1")),
        EpisodicMemoryRecord("r2", "other", "capital france", "irrelevant", d("e2")),
        EpisodicMemoryRecord("r3", "qa", "capital germany", "Berlin", d("e3")),
    )
    captured = {}
    def model_call(payload, headers):
        captured["payload"] = payload; captured["headers"] = headers
        return "Paris"
    runtime = GovernedAgentRuntime(
        lease=Lease(manifest.epoch_digest), manifest=manifest,
        serving_contract=contract_for(manifest),
        skill_policy_artifact=skill_path, retrieval_policy_artifact=retrieval_path,
        memory_records=records, model_call=model_call,
    )
    result = runtime.run(task_family="qa", input_text="capital france")
    assert result.output == "Paris"
    assert result.receipt.route == "qw3"
    # family bias keeps qa memories ahead of the otherwise lexical-identical other-family record
    assert result.receipt.retrieved_record_digests[0] == records[0].digest
    content = captured["payload"]["messages"][1]["content"]
    assert "[r1]" in content
    assert captured["headers"]["X-MiniAGI-Skill-Policy-Root"] == manifest.skill_policy_root


def test_policy_file_root_must_match_manifest_at_construction(tmp_path):
    bundle, retrieval, skill_path, retrieval_path = materialized_policies(tmp_path)
    manifest = manifest_for(bundle, retrieval)
    retrieval_path.write_text('{"tampered":true}')
    with pytest.raises(RuntimeStateMismatch, match="retrieval_policy physical root mismatch"):
        GovernedAgentRuntime(
            lease=Lease(manifest.epoch_digest), manifest=manifest,
            serving_contract=contract_for(manifest),
            skill_policy_artifact=skill_path, retrieval_policy_artifact=retrieval_path,
        )


def test_sealed_unseen_tasks_show_agent_epoch_improvement(tmp_path):
    bundle, retrieval, skill_path, retrieval_path = materialized_policies(tmp_path)
    manifest = manifest_for(bundle, retrieval)
    runtime = GovernedAgentRuntime(
        lease=Lease(manifest.epoch_digest), manifest=manifest,
        serving_contract=contract_for(manifest),
        skill_policy_artifact=skill_path, retrieval_policy_artifact=retrieval_path,
    )

    cas = ImmutableCAS(tmp_path / "cas")
    signer = Ed25519Signer.generate("fresh-task")
    metadata = FreshTaskMetadataStoreV143(tmp_path / "fresh-meta.sqlite3")
    vault = FreshTaskVaultAuthorityV143(
        tmp_path / "fresh-vault.sqlite3", authority_id="fresh-task-vault",
        authority_generation=1, signer=signer,
    )
    commitments = tuple(vault.seal(
        {"family_id": "text.reverse", "ring": "R0", "input": x, "expected": x[::-1]},
        generation=1, metadata=metadata,
    ) for x in ("planet", "governed", "runtime", "unseen"))
    experiment = FreshTaskPairedEvaluator(
        vault=vault, metadata=metadata, cas=cas,
        gate=FrozenBaselineGateV145(required_rings=("R0",), minimum_effect=0.0,
                                    minimum_retention=0.95, bootstrap_iterations=600, seed=11),
    ).evaluate(
        commitments=commitments,
        foundation_model_digest=d("foundation"), production_identity_digest=d("prod"),
        a0=lambda family, text: text,
        a1=lambda family, text: runtime.run(task_family=family, input_text=text).output,
    )
    assert experiment.baseline_report.decision == "PASS"
    assert experiment.baseline_report.mean_delta == 1.0
    assert experiment.baseline_report.ci95[0] > 0.0


def test_native_epoch_mismatch_blocks_even_local_skill(tmp_path):
    bundle, retrieval, skill_path, retrieval_path = materialized_policies(tmp_path)
    manifest = manifest_for(bundle, retrieval)
    bad = {
        "governed": True,
        "epoch_id": d("wrong-epoch"),
        "manifest_digest": manifest.manifest_digest,
        "artifact_root": manifest.artifact_root,
        "adapter_set_root": manifest.adapter_set_root,
    }
    runtime = GovernedAgentRuntime(
        lease=Lease(manifest.epoch_digest), manifest=manifest,
        serving_contract=GovernedServingContract(base_url="http://unused", fetch_json=lambda _: bad),
        skill_policy_artifact=skill_path, retrieval_policy_artifact=retrieval_path,
    )
    with pytest.raises(RuntimeStateMismatch):
        runtime.run(task_family="text.reverse", input_text="abc")
