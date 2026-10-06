from pathlib import Path

import pytest

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer
from minagi.integration.qw3_state import GovernedServingContract, RuntimeStateMismatch, ServedArtifactManifest
from minagi.v14.experiment_v145 import FrozenBaselineGateV145
from minagi.v14.fresh_tasks_v143 import FreshTaskMetadataStoreV143, FreshTaskVaultAuthorityV143
from minagi.v14.storage import ImmutableCAS
from minagi.v14.trajectory_v145 import VerifiedTrajectoryV145
from minagi.v15.adapter_learning import (
    AdapterArmOutcome,
    AdapterArtifactBuilder,
    AdapterSetBundle,
    AdapterTrainingPlan,
    FreshTaskAdapterEvaluator,
    GovernedAdapterLearningLoop,
    MlxLoraCandidateRunner,
    VerifiedAdapterDataset,
)


def d(label: str) -> str:
    return digest({"label": label})


def trajectory(i: int, inp: str, out: str) -> VerifiedTrajectoryV145:
    return VerifiedTrajectoryV145(
        trajectory_id=f"adapter-t-{i}",
        task_family="normalize.upper",
        input_text=inp,
        attempted_output=inp,
        repaired_output=out,
        production_identity_digest=d("production"),
        evidence_digests=(d(f"evidence-{i}"),),
        verification_receipt_digests=(d(f"verify-{i}"),),
    )


def dataset() -> VerifiedAdapterDataset:
    return VerifiedAdapterDataset.from_trajectories((
        trajectory(1, "one", "ONE"), trajectory(2, "two", "TWO"),
    ))


def plan(ds: VerifiedAdapterDataset) -> AdapterTrainingPlan:
    return AdapterTrainingPlan(
        foundation_model_digest=d("foundation"),
        dataset_digest=ds.digest,
        trainer_identity_digest=d("mlx-lm-trainer"),
        backend="mlx_lm",
        rank=8,
        scale=16.0,
        dropout=0.05,
        iterations=20,
        seed=7,
    )


def fake_adapter(plan, out: Path):
    (out / "adapter_config.json").write_text('{"rank":8,"scale":16.0}\n')
    (out / "adapters.safetensors").write_bytes(b"deterministic-test-adapter-v1")


def build_adapter(tmp_path):
    ds = dataset()
    p = plan(ds)
    out = tmp_path / "adapter"
    manifest = AdapterArtifactBuilder().build(plan=p, runner=fake_adapter, output_dir=out)
    aset = AdapterSetBundle.from_adapters(p.foundation_model_digest, (manifest,))
    return ds, p, manifest, aset, out


def base_manifest() -> ServedArtifactManifest:
    return ServedArtifactManifest(
        epoch_digest=d("epoch-v152"),
        runtime_manifest_digest=d("runtime-v152"),
        foundation_digest=d("foundation").split(":", 1)[1],
        tokenizer_digest="2" * 64,
        kv_archive_root="3" * 64,
        adapter_set_root="4" * 64,
        retrieval_policy_root="5" * 64,
        skill_policy_root="6" * 64,
        runtime_binary_digest="7" * 64,
    )


def sealed_commitments(tmp_path, rings=("R0", "R1", "R2")):
    signer = Ed25519Signer.generate("fresh-adapter")
    metadata = FreshTaskMetadataStoreV143(tmp_path / "fresh-meta.sqlite3")
    vault = FreshTaskVaultAuthorityV143(
        tmp_path / "fresh-secret.sqlite3", authority_id="fresh-adapter-vault",
        authority_generation=1, signer=signer,
    )
    commitments = []
    for i, ring in enumerate(rings):
        for j in range(2):
            x = f"{ring.lower()}-{i}-{j}"
            commitments.append(vault.seal(
                {"family_id": "normalize.upper", "ring": ring, "input": x, "expected": x.upper()},
                generation=1, metadata=metadata,
            ))
    return vault, metadata, tuple(commitments)


def test_verified_dataset_exports_deterministically(tmp_path):
    ds = dataset()
    path = ds.export_mlx_jsonl(tmp_path / "data")
    text = path.read_text()
    assert '"messages"' in text
    assert d("foundation").startswith("sha256:")
    assert ds.digest == dataset().digest


def test_unverified_trajectory_cannot_enter_adapter_dataset():
    t = trajectory(1, "x", "X")
    broken = type("T", (), {
        "task_family": t.task_family, "input_text": t.input_text,
        "repaired_output": t.repaired_output, "digest": t.digest,
        "evidence_digests": (), "verification_receipt_digests": (),
    })()
    with pytest.raises(ValueError):
        VerifiedAdapterDataset.from_trajectories((broken,))


def test_adapter_artifact_is_content_addressed_and_persistent(tmp_path):
    _, p, manifest, aset, out = build_adapter(tmp_path)
    assert (out / "adapters.safetensors").is_file()
    assert len(manifest.files) == 2
    assert manifest.training_plan_digest == p.digest
    assert manifest.dataset_digest == p.dataset_digest
    assert len(aset.root_hex) == 64


def test_adapter_builder_rejects_symlink_output(tmp_path):
    ds = dataset(); p = plan(ds)
    target = tmp_path / "target.bin"; target.write_bytes(b"x")
    def runner(plan, out):
        (out / "config.json").write_text("{}")
        (out / "escape").symlink_to(target)
    with pytest.raises(ValueError):
        AdapterArtifactBuilder().build(plan=p, runner=runner, output_dir=tmp_path / "adapter")


def test_sealed_adapter_candidate_passes_transfer_retention_and_security_gate(tmp_path):
    cas = ImmutableCAS(tmp_path / "cas")
    _, p, adapter, aset, out = build_adapter(tmp_path / "build")
    vault, metadata, commitments = sealed_commitments(tmp_path)
    evaluator = FreshTaskAdapterEvaluator(
        vault=vault, metadata=metadata, cas=cas,
        gate=FrozenBaselineGateV145(required_rings=("R0", "R1", "R2"), minimum_retention=.98,
                                    bootstrap_iterations=500, seed=3),
    )
    exp = evaluator.evaluate(
        commitments=commitments,
        candidate_adapter=adapter,
        baseline_adapter_set_root="0" * 64,
        candidate_adapter_set=aset,
        foundation_model_digest=p.foundation_model_digest,
        production_identity_digest=d("production"),
        a0=lambda family, text: AdapterArmOutcome(text, retention_score=1.0),
        a1=lambda family, text: AdapterArmOutcome(text.upper(), retention_score=.995),
    )
    assert exp.baseline_report.decision == "PASS"
    assert exp.baseline_report.security_regressions == 0
    assert exp.baseline_report.min_retention >= .98
    q = GovernedAdapterLearningLoop(cas=cas).prepare(
        plan=p, adapter=adapter, adapter_set=aset, experiment=exp, artifact_dir=out,
    )
    assert q.decision == "PASS"
    assert q.adapter_set_root == aset.root_hex


def test_security_regression_blocks_adapter_candidate(tmp_path):
    cas = ImmutableCAS(tmp_path / "cas")
    _, p, adapter, aset, out = build_adapter(tmp_path / "build")
    vault, metadata, commitments = sealed_commitments(tmp_path, rings=("R0",))
    exp = FreshTaskAdapterEvaluator(
        vault=vault, metadata=metadata, cas=cas,
        gate=FrozenBaselineGateV145(required_rings=("R0",), bootstrap_iterations=500),
    ).evaluate(
        commitments=commitments,
        candidate_adapter=adapter,
        baseline_adapter_set_root="0" * 64,
        candidate_adapter_set=aset,
        foundation_model_digest=p.foundation_model_digest,
        production_identity_digest=d("production"),
        a0=lambda family, text: AdapterArmOutcome(text),
        a1=lambda family, text: AdapterArmOutcome(text.upper(), security_regressions=1),
    )
    assert exp.baseline_report.decision == "BLOCK"
    q = GovernedAdapterLearningLoop(cas=cas).prepare(plan=p, adapter=adapter, adapter_set=aset, experiment=exp, artifact_dir=out)
    with pytest.raises(PermissionError):
        GovernedAdapterLearningLoop.prepare_served_manifest(base=base_manifest(), qualified=q)


def test_retention_regression_blocks_adapter_candidate(tmp_path):
    cas = ImmutableCAS(tmp_path / "cas")
    _, p, adapter, aset, out = build_adapter(tmp_path / "build")
    vault, metadata, commitments = sealed_commitments(tmp_path, rings=("R0",))
    exp = FreshTaskAdapterEvaluator(
        vault=vault, metadata=metadata, cas=cas,
        gate=FrozenBaselineGateV145(required_rings=("R0",), minimum_retention=.95, bootstrap_iterations=500),
    ).evaluate(
        commitments=commitments,
        candidate_adapter=adapter,
        baseline_adapter_set_root="0" * 64,
        candidate_adapter_set=aset,
        foundation_model_digest=p.foundation_model_digest,
        production_identity_digest=d("production"),
        a0=lambda family, text: AdapterArmOutcome(text),
        a1=lambda family, text: AdapterArmOutcome(text.upper(), retention_score=.90),
    )
    assert exp.baseline_report.decision == "BLOCK"
    assert "retention floor violated" in exp.baseline_report.reasons


def test_passed_adapter_changes_exact_served_adapter_root(tmp_path):
    cas = ImmutableCAS(tmp_path / "cas")
    _, p, adapter, aset, out = build_adapter(tmp_path / "build")
    vault, metadata, commitments = sealed_commitments(tmp_path, rings=("R0",))
    exp = FreshTaskAdapterEvaluator(
        vault=vault, metadata=metadata, cas=cas,
        gate=FrozenBaselineGateV145(required_rings=("R0",), bootstrap_iterations=500),
    ).evaluate(
        commitments=commitments,
        candidate_adapter=adapter,
        baseline_adapter_set_root="0" * 64,
        candidate_adapter_set=aset,
        foundation_model_digest=p.foundation_model_digest,
        production_identity_digest=d("production"),
        a0=lambda family, text: text,
        a1=lambda family, text: text.upper(),
    )
    q = GovernedAdapterLearningLoop(cas=cas).prepare(plan=p, adapter=adapter, adapter_set=aset, experiment=exp, artifact_dir=out)
    old = base_manifest()
    new = GovernedAdapterLearningLoop.prepare_served_manifest(base=old, qualified=q)
    assert new.adapter_set_root == aset.root_hex
    assert new.adapter_set_root != old.adapter_set_root
    assert new.artifact_root != old.artifact_root
    assert new.manifest_digest != old.manifest_digest


def test_serving_contract_rejects_independent_adapter_root_mismatch(tmp_path):
    old = base_manifest()
    new = ServedArtifactManifest(**{**old.__dict__, "adapter_set_root": "a" * 64})
    class Lease:
        epoch_digest = new.epoch_digest
    state = {
        "governed": True,
        "epoch_id": new.epoch_digest,
        "manifest_digest": new.manifest_digest,
        "artifact_root": new.artifact_root,
        "adapter_set_root": "b" * 64,
    }
    contract = GovernedServingContract(base_url="http://unused", fetch_json=lambda path: state)
    with pytest.raises(RuntimeStateMismatch):
        contract.request_headers(lease=Lease(), manifest=new)


def test_serving_contract_emits_adapter_root_header():
    m = base_manifest()
    class Lease:
        epoch_digest = m.epoch_digest
    state = {
        "governed": True, "epoch_id": m.epoch_digest,
        "manifest_digest": m.manifest_digest, "artifact_root": m.artifact_root,
        "adapter_set_root": m.adapter_set_root,
    }
    c = GovernedServingContract(base_url="http://unused", fetch_json=lambda path: state)
    headers = c.request_headers(lease=Lease(), manifest=m)
    assert headers["X-MiniAGI-Adapter-Set-Root"] == m.adapter_set_root


def test_qualified_adapter_publishes_exact_payload_bytes_to_cas(tmp_path):
    cas = ImmutableCAS(tmp_path / "cas")
    _, p, adapter, aset, out = build_adapter(tmp_path / "build")
    vault, metadata, commitments = sealed_commitments(tmp_path, rings=("R0",))
    exp = FreshTaskAdapterEvaluator(
        vault=vault, metadata=metadata, cas=cas,
        gate=FrozenBaselineGateV145(required_rings=("R0",), bootstrap_iterations=500),
    ).evaluate(
        commitments=commitments, candidate_adapter=adapter,
        baseline_adapter_set_root="0" * 64, candidate_adapter_set=aset,
        foundation_model_digest=p.foundation_model_digest,
        production_identity_digest=d("production"),
        a0=lambda family, text: text, a1=lambda family, text: text.upper(),
    )
    q = GovernedAdapterLearningLoop(cas=cas).prepare(
        plan=p, adapter=adapter, adapter_set=aset, experiment=exp, artifact_dir=out,
    )
    assert cas.exists(q.adapter_payload_binding_digest)
    for row in adapter.files:
        assert cas.exists(row.sha256_digest)
        assert cas.get_bytes(row.sha256_digest) == (out / row.relative_path).read_bytes()


def test_adapter_payload_tamper_after_build_is_rejected(tmp_path):
    cas = ImmutableCAS(tmp_path / "cas")
    _, p, adapter, aset, out = build_adapter(tmp_path / "build")
    vault, metadata, commitments = sealed_commitments(tmp_path, rings=("R0",))
    exp = FreshTaskAdapterEvaluator(
        vault=vault, metadata=metadata, cas=cas,
        gate=FrozenBaselineGateV145(required_rings=("R0",), bootstrap_iterations=500),
    ).evaluate(
        commitments=commitments, candidate_adapter=adapter,
        baseline_adapter_set_root="0" * 64, candidate_adapter_set=aset,
        foundation_model_digest=p.foundation_model_digest,
        production_identity_digest=d("production"),
        a0=lambda family, text: text, a1=lambda family, text: text.upper(),
    )
    (out / "adapters.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError):
        GovernedAdapterLearningLoop(cas=cas).prepare(
            plan=p, adapter=adapter, adapter_set=aset, experiment=exp, artifact_dir=out,
        )


def test_adapter_for_different_foundation_cannot_change_served_manifest(tmp_path):
    cas = ImmutableCAS(tmp_path / "cas")
    _, p, adapter, aset, out = build_adapter(tmp_path / "build")
    vault, metadata, commitments = sealed_commitments(tmp_path, rings=("R0",))
    exp = FreshTaskAdapterEvaluator(
        vault=vault, metadata=metadata, cas=cas,
        gate=FrozenBaselineGateV145(required_rings=("R0",), bootstrap_iterations=500),
    ).evaluate(
        commitments=commitments, candidate_adapter=adapter,
        baseline_adapter_set_root="0" * 64, candidate_adapter_set=aset,
        foundation_model_digest=p.foundation_model_digest,
        production_identity_digest=d("production"),
        a0=lambda family, text: text, a1=lambda family, text: text.upper(),
    )
    q = GovernedAdapterLearningLoop(cas=cas).prepare(
        plan=p, adapter=adapter, adapter_set=aset, experiment=exp, artifact_dir=out,
    )
    wrong = ServedArtifactManifest(**{**base_manifest().__dict__, "foundation_digest": "f" * 64})
    with pytest.raises(PermissionError):
        GovernedAdapterLearningLoop.prepare_served_manifest(base=wrong, qualified=q)


def test_mlx_runner_materializes_bound_yaml_config(tmp_path):
    import yaml
    ds = dataset(); p = plan(ds)
    captured = {}
    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        cfg_path = Path(cmd[-1])
        captured["config"] = yaml.safe_load(cfg_path.read_text())
        out = Path(captured["config"]["adapter_path"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text("{}")
        (out / "adapters.safetensors").write_bytes(b"mlx-fake")
        class Result:
            returncode = 0
        return Result()
    out = tmp_path / "adapter"
    out.mkdir()
    runner = MlxLoraCandidateRunner(model="/models/qwen", dataset=ds, python_executable="python", run=fake_run)
    runner(p, out)
    assert captured["cmd"][:4] == ["python", "-m", "mlx_lm", "lora"]
    cfg = captured["config"]
    assert cfg["seed"] == p.seed
    assert cfg["learning_rate"] == p.learning_rate
    assert cfg["lora_parameters"] == {"rank": p.rank, "dropout": p.dropout, "scale": p.scale}
    assert (out / "adapters.safetensors").is_file()
