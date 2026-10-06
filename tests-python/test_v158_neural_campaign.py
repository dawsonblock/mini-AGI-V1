from dataclasses import replace

import pytest

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer
from minagi.integration.qw3_state import GovernedServingContract, ServedArtifactManifest
from minagi.v14.experiment_v145 import FrozenBaselineGateV145
from minagi.v14.fresh_tasks_v143 import FreshTaskMetadataStoreV143, FreshTaskVaultAuthorityV143
from minagi.v14.storage import ImmutableCAS
from minagi.v15.neural_eval import QW3NeuralArm, SealedNeuralAdapterEvaluator
from minagi.v15.neural_campaign import (
    NeuralCampaignPlan, NeuralGateSpec, PreregisteredNeuralCampaignRunner,
    QW3LaunchSpec,
)

ZERO = "0" * 64

def d(label): return digest({"label": label})

class Lease:
    def __init__(self, epoch): self.epoch_digest = epoch


def manifest(*, arm, adapter_root, native_root=ZERO):
    return ServedArtifactManifest(
        epoch_digest=d("epoch-" + arm), runtime_manifest_digest=d("runtime-" + arm),
        foundation_digest="1" * 64, tokenizer_digest="2" * 64,
        kv_archive_root=ZERO, adapter_set_root=adapter_root,
        retrieval_policy_root=ZERO, skill_policy_root=ZERO,
        runtime_binary_digest="7" * 64, native_adapter_bundle_root=native_root,
    )


def state(m):
    return {
        "governed": True, "epoch_id": m.epoch_digest,
        "manifest_digest": m.manifest_digest, "artifact_root": m.artifact_root,
        "adapter_set_root": m.adapter_set_root,
        "native_adapter_bundle_root": m.native_adapter_bundle_root,
        "native_adapter_loaded": m.native_adapter_bundle_root != ZERO,
        "runtime_closure_verified": True,
        "measured_foundation_digest": m.foundation_digest,
        "measured_tokenizer_digest": m.tokenizer_digest,
        "measured_runtime_binary_digest": m.runtime_binary_digest,
        "measured_kvmem_archive_root": m.kv_archive_root,
        "measured_adapter_set_root": m.adapter_set_root,
        "measured_retrieval_policy_root": m.retrieval_policy_root,
        "measured_skill_policy_root": m.skill_policy_root,
    }


def arm(name, m, fn, endpoint):
    return QW3NeuralArm(
        arm=name, lease=Lease(m.epoch_digest), manifest=m,
        contract=GovernedServingContract(base_url=endpoint, fetch_json=lambda _: state(m)),
        model_call=lambda payload, headers: fn(payload["messages"][0]["content"]),
    )


def fresh(tmp_path):
    signer=Ed25519Signer.generate("v158-fresh")
    metadata=FreshTaskMetadataStoreV143(tmp_path/"meta.sqlite3")
    vault=FreshTaskVaultAuthorityV143(tmp_path/"vault.sqlite3", authority_id="v158-vault", authority_generation=1, signer=signer)
    commitments=[]
    for ring in ("R0","R1","R2"):
        for text in (ring.lower()+"-alpha", ring.lower()+"-beta"):
            commitments.append(vault.seal({"family_id":"normalize.upper","ring":ring,"input":text,"expected":text.upper()}, generation=1, metadata=metadata))
    return vault,metadata,tuple(commitments)


def setup(tmp_path):
    m0=manifest(arm="a0",adapter_root=ZERO)
    m1=manifest(arm="a1",adapter_root="a"*64,native_root="b"*64)
    a0=arm("A0",m0,lambda x:x,"http://127.0.0.1:18080")
    a1=arm("A1",m1,lambda x:x.upper(),"http://127.0.0.1:18081")
    vault,metadata,commitments=fresh(tmp_path)
    cas=ImmutableCAS(tmp_path/"cas")
    gate_spec=NeuralGateSpec(bootstrap_iterations=600,seed=19,minimum_retention=.98)
    evaluator=SealedNeuralAdapterEvaluator(vault=vault,metadata=metadata,cas=cas,gate=gate_spec.build(),consumer_id="v158-campaign")
    plan=NeuralCampaignPlan.create(run_id="campaign-001",a0=a0,a1=a1,commitments=commitments,gate=gate_spec,
        scorer_id="exact-match-v1",retention_policy_id="legacy-retention-v1",security_policy_id="security-zero-v1")
    return a0,a1,commitments,cas,evaluator,plan


def test_campaign_must_be_preregistered_before_consumption(tmp_path):
    a0,a1,cs,cas,evaluator,plan=setup(tmp_path)
    runner=PreregisteredNeuralCampaignRunner(evaluator=evaluator,cas=cas)
    with pytest.raises(PermissionError,match="preregistered"):
        runner.run(plan=plan,commitments=cs,a0=a0,a1=a1)


def test_preregistered_campaign_passes_and_persists_promotion_ready_bundle(tmp_path):
    a0,a1,cs,cas,evaluator,plan=setup(tmp_path)
    runner=PreregisteredNeuralCampaignRunner(evaluator=evaluator,cas=cas)
    assert runner.preregister(plan)==plan.digest
    result=runner.run(plan=plan,commitments=cs,a0=a0,a1=a1,
        retention_fn=lambda *_:.995,security_fn=lambda *_:0)
    assert result.decision=="PASS" and result.promotion_ready
    assert len(result.pair_digests)==6 and len(result.runtime_evidence_digests)==12
    assert cas.exists(result.digest) and cas.exists(result.experiment_digest)


def test_campaign_rejects_arm_substitution_after_preregistration(tmp_path):
    a0,a1,cs,cas,evaluator,plan=setup(tmp_path)
    runner=PreregisteredNeuralCampaignRunner(evaluator=evaluator,cas=cas); runner.preregister(plan)
    changed=replace(a1.manifest,runtime_manifest_digest=d("different"))
    bad=arm("A1",changed,lambda x:x.upper(),"http://127.0.0.1:18081")
    with pytest.raises(PermissionError,match="run spec"):
        runner.run(plan=plan,commitments=cs,a0=a0,a1=bad)


def test_campaign_rejects_task_battery_substitution(tmp_path):
    a0,a1,cs,cas,evaluator,plan=setup(tmp_path)
    runner=PreregisteredNeuralCampaignRunner(evaluator=evaluator,cas=cas); runner.preregister(plan)
    with pytest.raises(PermissionError,match="task commitments"):
        runner.run(plan=plan,commitments=tuple(reversed(cs)),a0=a0,a1=a1)


def test_campaign_rejects_gate_changes_after_preregistration(tmp_path):
    a0,a1,cs,cas,evaluator,plan=setup(tmp_path)
    runner=PreregisteredNeuralCampaignRunner(evaluator=evaluator,cas=cas); runner.preregister(plan)
    evaluator.gate=FrozenBaselineGateV145(required_rings=("R0","R1","R2"),minimum_retention=.50,bootstrap_iterations=600,seed=19)
    with pytest.raises(PermissionError,match="gate differs"):
        runner.run(plan=plan,commitments=cs,a0=a0,a1=a1)


def test_qw3_launch_spec_emits_exact_governed_identity_flags():
    m=manifest(arm="a1",adapter_root="a"*64,native_root="b"*64)
    spec=QW3LaunchSpec(arm="A1",executable="/opt/qw3",model_path="/models/qwen",host="127.0.0.1",port=8081,
        manifest=m,adapter_set_artifact="/state/adapters.json",native_adapter_bundle="/state/native")
    argv=spec.argv()
    assert argv[:4]==("/opt/qw3","serve","--model","/models/qwen")
    assert m.epoch_digest in argv and m.manifest_digest in argv and m.artifact_root in argv
    assert "--state-adapter-set-artifact" in argv and "--native-adapter-bundle" in argv


def test_launch_spec_refuses_missing_physical_candidate_adapter():
    m=manifest(arm="a1",adapter_root="a"*64,native_root="b"*64)
    with pytest.raises(ValueError,match="adapter-set path required"):
        QW3LaunchSpec(arm="A1",executable="qw3",model_path="model.gguf",host="127.0.0.1",port=8081,
            manifest=m,native_adapter_bundle="native")
