from __future__ import annotations

from pathlib import Path
import hashlib

import pytest

from egai.common.canonical import digest
from minagi.integration.qw3_state import RuntimeStateMismatch, ServedArtifactManifest
from minagi.v15.artifact_binding import materialize_canonical_artifact
from minagi.v15.runtime_closure import (
    MeasuredGovernedServingContract,
    MeasuredRuntimeClosure,
    PhysicalArtifactPaths,
    sha256_file,
    sha256_runtime_path,
    tokenizer_identity_sha256_v155,
)

ZERO = "0" * 64


def _write_hf_model(root: Path) -> None:
    root.mkdir()
    (root / "tokenizer.json").write_text('{"model":{"vocab":{"a":0},"merges":[]}}')
    (root / "config.json").write_text('{"text_config":{"bos_token_id":0,"eos_token_id":0}}')
    (root / "tokenizer_config.json").write_text('{"eos_token":"a"}')
    (root / "weights.safetensors").write_bytes(b"weights-v155")


def _manifest(model: Path, binary: Path, adapter: Path, retrieval: Path, skill: Path) -> ServedArtifactManifest:
    return ServedArtifactManifest(
        epoch_digest=digest({"epoch": 155}),
        runtime_manifest_digest=digest({"runtime": 155}),
        foundation_digest=sha256_runtime_path(model),
        tokenizer_digest=tokenizer_identity_sha256_v155(model),
        kv_archive_root=ZERO,
        adapter_set_root=sha256_file(adapter),
        retrieval_policy_root=sha256_file(retrieval),
        skill_policy_root=sha256_file(skill),
        runtime_binary_digest=sha256_file(binary),
        native_adapter_bundle_root=ZERO,
    )


def test_physical_closure_measures_all_non_model_artifacts(tmp_path):
    model = tmp_path / "model"
    _write_hf_model(model)
    binary = tmp_path / "qw3"
    binary.write_bytes(b"qw3-v155")
    adapter = tmp_path / "adapter-set.bin"; adapter.write_bytes(b"adapter-set")
    retrieval = tmp_path / "retrieval.json"; retrieval.write_bytes(b"retrieval-policy")
    skill = tmp_path / "skill.json"; skill.write_bytes(b"skill-policy")
    manifest = _manifest(model, binary, adapter, retrieval, skill)
    paths = PhysicalArtifactPaths(str(adapter), str(retrieval), str(skill))
    closure = MeasuredRuntimeClosure.from_files(
        manifest=manifest, model_path=model, runtime_binary_path=binary, artifacts=paths,
    )
    assert closure.tokenizer_digest == manifest.tokenizer_digest
    assert closure.adapter_set_root == manifest.adapter_set_root
    assert closure.retrieval_policy_root == manifest.retrieval_policy_root
    assert closure.skill_policy_root == manifest.skill_policy_root
    args = closure.qw3_args(manifest=manifest, artifacts=paths)
    assert "--state-adapter-set-artifact" in args
    assert str(adapter.resolve()) in args
    assert "--state-retrieval-policy-artifact" in args
    assert "--state-skill-policy-artifact" in args


def test_physical_closure_rejects_policy_tamper(tmp_path):
    model = tmp_path / "model"; _write_hf_model(model)
    binary = tmp_path / "qw3"; binary.write_bytes(b"qw3-v155")
    adapter = tmp_path / "adapter"; adapter.write_bytes(b"a")
    retrieval = tmp_path / "retrieval"; retrieval.write_bytes(b"r")
    skill = tmp_path / "skill"; skill.write_bytes(b"s")
    manifest = _manifest(model, binary, adapter, retrieval, skill)
    skill.write_bytes(b"tampered")
    with pytest.raises(RuntimeStateMismatch, match="skill_policy_root"):
        MeasuredRuntimeClosure.from_files(
            manifest=manifest, model_path=model, runtime_binary_path=binary,
            artifacts=PhysicalArtifactPaths(str(adapter), str(retrieval), str(skill)),
        )


def test_zero_root_cannot_hide_supplied_artifact(tmp_path):
    model = tmp_path / "model.gguf"; model.write_bytes(b"fake-gguf")
    binary = tmp_path / "qw3"; binary.write_bytes(b"qw3")
    extra = tmp_path / "unexpected"; extra.write_bytes(b"x")
    manifest = ServedArtifactManifest(
        epoch_digest=digest({"epoch": "zero"}),
        runtime_manifest_digest=digest({"runtime": "zero"}),
        foundation_digest=sha256_file(model), tokenizer_digest=sha256_file(model),
        kv_archive_root=ZERO, adapter_set_root=ZERO, retrieval_policy_root=ZERO,
        skill_policy_root=ZERO, runtime_binary_digest=sha256_file(binary),
    )
    with pytest.raises(RuntimeStateMismatch, match="zero-root"):
        MeasuredRuntimeClosure.from_files(
            manifest=manifest, model_path=model, runtime_binary_path=binary,
            artifacts=PhysicalArtifactPaths(skill_policy_artifact=str(extra)),
        )


def test_tokenizer_identity_tracks_only_native_tokenizer_inputs(tmp_path):
    model = tmp_path / "model"; _write_hf_model(model)
    before = tokenizer_identity_sha256_v155(model)
    (model / "weights.safetensors").write_bytes(b"different-weights")
    assert tokenizer_identity_sha256_v155(model) == before
    (model / "tokenizer_config.json").write_text('{"eos_token":"changed"}')
    assert tokenizer_identity_sha256_v155(model) != before


def test_tokenizer_identity_rejects_symlink_input(tmp_path):
    model = tmp_path / "model"; _write_hf_model(model)
    target = model / "real-tokenizer.json"
    target.write_bytes((model / "tokenizer.json").read_bytes())
    (model / "tokenizer.json").unlink()
    (model / "tokenizer.json").symlink_to(target.name)
    with pytest.raises(ValueError, match="regular tokenizer.json"):
        tokenizer_identity_sha256_v155(model)


def test_materialized_canonical_artifact_has_governance_root(tmp_path):
    value = {"schema": "test-policy-v1", "rules": ["a", "b"]}
    out = materialize_canonical_artifact(value, tmp_path / "policy.canonical")
    assert out.canonical_digest == digest(value)
    assert out.root_hex == digest(value).split(":", 1)[1]
    assert sha256_file(out.path) == out.root_hex


def test_measured_contract_requires_every_reported_physical_root(tmp_path):
    model = tmp_path / "model"; _write_hf_model(model)
    binary = tmp_path / "qw3"; binary.write_bytes(b"qw3-v155")
    adapter = tmp_path / "adapter"; adapter.write_bytes(b"a")
    retrieval = tmp_path / "retrieval"; retrieval.write_bytes(b"r")
    skill = tmp_path / "skill"; skill.write_bytes(b"s")
    manifest = _manifest(model, binary, adapter, retrieval, skill)
    paths = PhysicalArtifactPaths(str(adapter), str(retrieval), str(skill))
    closure = MeasuredRuntimeClosure.from_files(
        manifest=manifest, model_path=model, runtime_binary_path=binary, artifacts=paths,
    )
    class Lease:
        epoch_digest = manifest.epoch_digest
    state = {
        "governed": True,
        "epoch_id": manifest.epoch_digest,
        "manifest_digest": manifest.manifest_digest,
        "artifact_root": manifest.artifact_root,
        "adapter_set_root": manifest.adapter_set_root,
        "native_adapter_bundle_root": None,
        "native_adapter_loaded": False,
        "runtime_closure_verified": True,
        "runtime_closure_digest": closure.digest,
        "measured_foundation_digest": manifest.foundation_digest,
        "measured_tokenizer_digest": manifest.tokenizer_digest,
        "measured_runtime_binary_digest": manifest.runtime_binary_digest,
        "measured_kvmem_archive_root": manifest.kv_archive_root,
        "measured_adapter_set_root": manifest.adapter_set_root,
        "measured_retrieval_policy_root": manifest.retrieval_policy_root,
        "measured_skill_policy_root": manifest.skill_policy_root,
    }
    contract = MeasuredGovernedServingContract(base_url="http://unused", fetch_json=lambda _: state)
    headers = contract.request_headers(lease=Lease(), manifest=manifest)
    assert headers["X-MiniAGI-Tokenizer-Digest"] == manifest.tokenizer_digest
    assert headers["X-MiniAGI-Retrieval-Policy-Root"] == manifest.retrieval_policy_root
    assert headers["X-MiniAGI-Skill-Policy-Root"] == manifest.skill_policy_root
    bad = dict(state); bad["measured_retrieval_policy_root"] = "f" * 64
    with pytest.raises(RuntimeStateMismatch, match="measured-runtime"):
        MeasuredGovernedServingContract(
            base_url="http://unused", fetch_json=lambda _: bad,
        ).request_headers(lease=Lease(), manifest=manifest)


def test_manifest_builder_derives_every_served_root_from_bytes(tmp_path):
    from minagi.v15.artifact_binding import build_physically_bound_served_manifest
    model = tmp_path / "model"; _write_hf_model(model)
    binary = tmp_path / "qw3"; binary.write_bytes(b"qw3-v155-builder")
    adapter = tmp_path / "adapter"; adapter.write_bytes(b"adapter")
    retrieval = tmp_path / "retrieval"; retrieval.write_bytes(b"retrieval")
    skill = tmp_path / "skill"; skill.write_bytes(b"skill")
    manifest = build_physically_bound_served_manifest(
        epoch_digest=digest({"epoch": "builder"}),
        runtime_manifest_digest=digest({"runtime": "builder"}),
        model_path=model,
        runtime_binary_path=binary,
        adapter_set_artifact=adapter,
        retrieval_policy_artifact=retrieval,
        skill_policy_artifact=skill,
    )
    assert manifest.foundation_digest == sha256_runtime_path(model)
    assert manifest.tokenizer_digest == tokenizer_identity_sha256_v155(model)
    assert manifest.adapter_set_root == sha256_file(adapter)
    assert manifest.retrieval_policy_root == sha256_file(retrieval)
    assert manifest.skill_policy_root == sha256_file(skill)
