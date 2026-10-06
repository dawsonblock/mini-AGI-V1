from dataclasses import replace

import pytest

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer
from minagi.integration.qw3_state import GovernedServingContract, ServedArtifactManifest
from minagi.v14.experiment_v145 import FrozenBaselineGateV145
from minagi.v14.fresh_tasks_v143 import FreshTaskMetadataStoreV143, FreshTaskVaultAuthorityV143
from minagi.v14.storage import ImmutableCAS
from minagi.v15.neural_eval import QW3NeuralArm, SealedNeuralAdapterEvaluator

ZERO = "0" * 64

def d(label): return digest({"label": label})

class Lease:
    def __init__(self, epoch): self.epoch_digest = epoch


def manifest(*, arm, adapter_root, native_root=ZERO):
    return ServedArtifactManifest(
        epoch_digest=d("epoch-" + arm), runtime_manifest_digest=d("runtime-" + arm),
        foundation_digest="1" * 64, tokenizer_digest="2" * 64,
        kv_archive_root=ZERO, adapter_set_root=adapter_root,
        retrieval_policy_root="5" * 64, skill_policy_root="6" * 64,
        runtime_binary_digest="7" * 64, native_adapter_bundle_root=native_root,
    )


def state(m):
    return {
        "governed": True,
        "epoch_id": m.epoch_digest,
        "manifest_digest": m.manifest_digest,
        "artifact_root": m.artifact_root,
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


def arm(name, m, fn):
    return QW3NeuralArm(
        arm=name, lease=Lease(m.epoch_digest), manifest=m,
        contract=GovernedServingContract(base_url="http://unused", fetch_json=lambda _: state(m)),
        model_call=lambda payload, headers: fn(payload["messages"][0]["content"]),
    )


def commitments(tmp_path):
    signer = Ed25519Signer.generate("v157-fresh")
    metadata = FreshTaskMetadataStoreV143(tmp_path / "meta.sqlite3")
    vault = FreshTaskVaultAuthorityV143(
        tmp_path / "vault.sqlite3", authority_id="v157-vault", authority_generation=1, signer=signer,
    )
    vals = []
    for ring in ("R0", "R1", "R2"):
        for text in (ring.lower()+"-alpha", ring.lower()+"-beta"):
            vals.append(vault.seal(
                {"family_id":"normalize.upper", "ring":ring, "input":text, "expected":text.upper()},
                generation=1, metadata=metadata,
            ))
    return vault, metadata, tuple(vals)


def test_qw3_neural_arm_records_measured_runtime_evidence():
    m = manifest(arm="a0", adapter_root=ZERO)
    a = arm("A0", m, lambda x: x)
    out = a.run(task_family="x", input_text="hello")
    assert out.output == "hello"
    assert out.evidence.artifact_root == m.artifact_root
    assert out.evidence.adapter_set_root == ZERO
    assert out.evidence.runtime_state_digest.startswith("sha256:")


def test_neural_arm_rejects_unmeasured_runtime():
    m = manifest(arm="a0", adapter_root=ZERO)
    s = state(m); s["runtime_closure_verified"] = False
    a = QW3NeuralArm(
        arm="A0", lease=Lease(m.epoch_digest), manifest=m,
        contract=GovernedServingContract(base_url="http://unused", fetch_json=lambda _: s),
        model_call=lambda payload, headers: "x",
    )
    with pytest.raises(PermissionError, match="measured QW3 runtime closure"):
        a.run(task_family="x", input_text="x")


def test_a0_a1_may_change_only_adapter_state(tmp_path):
    m0 = manifest(arm="a0", adapter_root=ZERO)
    m1 = manifest(arm="a1", adapter_root="a"*64, native_root="b"*64)
    m1 = replace(m1, tokenizer_digest="c"*64)
    vault, metadata, cs = commitments(tmp_path)
    ev = SealedNeuralAdapterEvaluator(vault=vault, metadata=metadata, cas=ImmutableCAS(tmp_path/"cas"), gate=FrozenBaselineGateV145())
    with pytest.raises(PermissionError, match="non-adapter serving state"):
        ev.evaluate(commitments=cs, a0=arm("A0",m0,lambda x:x), a1=arm("A1",m1,lambda x:x.upper()))


def test_nonzero_candidate_adapter_requires_native_bundle(tmp_path):
    m0 = manifest(arm="a0", adapter_root=ZERO)
    m1 = manifest(arm="a1", adapter_root="a"*64, native_root=ZERO)
    vault, metadata, cs = commitments(tmp_path)
    ev = SealedNeuralAdapterEvaluator(vault=vault, metadata=metadata, cas=ImmutableCAS(tmp_path/"cas"), gate=FrozenBaselineGateV145())
    with pytest.raises(PermissionError, match="native adapter bundle"):
        ev.evaluate(commitments=cs, a0=arm("A0",m0,lambda x:x), a1=arm("A1",m1,lambda x:x.upper()))


def test_sealed_neural_qw3_experiment_can_pass_transfer_retention_and_security(tmp_path):
    m0 = manifest(arm="a0", adapter_root=ZERO)
    m1 = manifest(arm="a1", adapter_root="a"*64, native_root="b"*64)
    vault, metadata, cs = commitments(tmp_path)
    ev = SealedNeuralAdapterEvaluator(
        vault=vault, metadata=metadata, cas=ImmutableCAS(tmp_path/"cas"),
        gate=FrozenBaselineGateV145(required_rings=("R0","R1","R2"), minimum_retention=.98,
                                    bootstrap_iterations=600, seed=17),
    )
    exp = ev.evaluate(
        commitments=cs,
        a0=arm("A0",m0,lambda x:x),
        a1=arm("A1",m1,lambda x:x.upper()),
        retention_fn=lambda family, inp, expected: .995,
        security_fn=lambda family, inp, expected: 0,
    )
    assert exp.baseline_report.decision == "PASS"
    assert exp.baseline_report.mean_delta == 1.0
    assert exp.baseline_report.ci95[0] > 0
    assert len(exp.pairs) == 6
    assert all(p.a0_runtime_evidence_digest != p.a1_runtime_evidence_digest for p in exp.pairs)


def test_security_regression_blocks_neural_candidate(tmp_path):
    m0 = manifest(arm="a0", adapter_root=ZERO)
    m1 = manifest(arm="a1", adapter_root="a"*64, native_root="b"*64)
    vault, metadata, cs = commitments(tmp_path)
    exp = SealedNeuralAdapterEvaluator(
        vault=vault, metadata=metadata, cas=ImmutableCAS(tmp_path/"cas"),
        gate=FrozenBaselineGateV145(required_rings=("R0","R1","R2"), bootstrap_iterations=500),
    ).evaluate(
        commitments=cs, a0=arm("A0",m0,lambda x:x), a1=arm("A1",m1,lambda x:x.upper()),
        security_fn=lambda family, inp, expected: 1,
    )
    assert exp.baseline_report.decision == "BLOCK"
    assert exp.baseline_report.security_regressions > 0
