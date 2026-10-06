from __future__ import annotations

from pathlib import Path
import json
import struct

import numpy as np
import pytest

from egai.common.canonical import digest
from minagi.integration.qw3_state import GovernedServingContract, RuntimeStateMismatch, ServedArtifactManifest
from minagi.v15.adapter_learning import (
    AdapterArtifactBuilder,
    AdapterSetBundle,
    AdapterTrainingPlan,
    QualifiedAdapterSetCandidate,
)
from minagi.v15.native_adapter import NativeQW3AdapterCompiler


def d(label: str) -> str:
    return digest({"label": label})


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    offset = 0
    header = {}
    payload = bytearray()
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr.astype("<f4"))
        raw = arr.tobytes(order="C")
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        payload.extend(raw)
        offset += len(raw)
    h = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + payload)


def build_source(tmp_path: Path, *, include_unsupported: bool = False):
    plan = AdapterTrainingPlan(
        foundation_model_digest=d("foundation-v153"),
        dataset_digest=d("dataset-v153"),
        trainer_identity_digest=d("trainer-v153"),
        backend="external",
        rank=2,
        scale=0.5,
        iterations=1,
        target_modules=("lm_head",),
    )
    out = tmp_path / "adapter"
    out.mkdir(parents=True)
    tensors = {
        "lm_head.lora_A.weight": np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        "lm_head.lora_B.weight": np.array([[1, 0], [0, 1], [1, 1], [-1, 2]], dtype=np.float32),
    }
    if include_unsupported:
        tensors["model.layers.0.self_attn.q_proj.lora_A.weight"] = np.ones((2, 3), dtype=np.float32)
        tensors["model.layers.0.self_attn.q_proj.lora_B.weight"] = np.ones((3, 2), dtype=np.float32)
    write_safetensors(out / "adapter.safetensors", tensors)
    adapter = AdapterArtifactBuilder().collect(output_dir=out, plan=plan)
    aset = AdapterSetBundle.from_adapters(plan.foundation_model_digest, (adapter,))
    q = QualifiedAdapterSetCandidate(
        training_plan_digest=plan.digest,
        foundation_model_digest=plan.foundation_model_digest,
        adapter_manifest_digest=adapter.digest,
        adapter_payload_binding_digest=d("payload-binding"),
        adapter_set_bundle_digest=aset.digest,
        sealed_experiment_digest=d("sealed-experiment"),
        frozen_baseline_report_digest=d("baseline-report"),
        adapter_set_root=aset.root_hex,
        decision="PASS",
    )
    return plan, adapter, aset, q, out


def base_manifest() -> ServedArtifactManifest:
    return ServedArtifactManifest(
        epoch_digest=d("epoch-v153"),
        runtime_manifest_digest=d("runtime-v153"),
        foundation_digest=d("foundation-v153").split(":", 1)[1],
        tokenizer_digest="2" * 64,
        kv_archive_root="3" * 64,
        adapter_set_root="4" * 64,
        retrieval_policy_root="5" * 64,
        skill_policy_root="6" * 64,
        runtime_binary_digest="7" * 64,
    )


def test_native_compiler_materializes_full_lm_head_bundle(tmp_path):
    plan, adapter, aset, q, source = build_source(tmp_path)
    native = NativeQW3AdapterCompiler().compile(
        plan=plan, adapter=adapter, adapter_set=aset, qualified=q,
        artifact_dir=source, output_dir=tmp_path / "native",
        model_sha256="a" * 64, input_features=3, output_features=4,
    )
    assert native.adapter_set_root == aset.root_hex
    assert len(native.bundle_root) == 64
    assert len(native.tensors) == 1
    tensor = native.tensors[0]
    assert tensor.target == "output.weight"
    assert tensor.rank == 2
    assert tensor.a.bytes == 2 * 3 * 4
    assert tensor.b.bytes == 4 * 2 * 4
    assert (Path(native.directory) / "manifest.json").is_file()


def test_native_compiler_refuses_partial_all_linear_bundle(tmp_path):
    plan, adapter, aset, q, source = build_source(tmp_path, include_unsupported=True)
    with pytest.raises(ValueError, match="refuses partial adapter application"):
        NativeQW3AdapterCompiler().compile(
            plan=plan, adapter=adapter, adapter_set=aset, qualified=q,
            artifact_dir=source, output_dir=tmp_path / "native",
            model_sha256="a" * 64, input_features=3, output_features=4,
        )


def test_native_bundle_root_becomes_part_of_served_artifact_root(tmp_path):
    plan, adapter, aset, q, source = build_source(tmp_path)
    native = NativeQW3AdapterCompiler().compile(
        plan=plan, adapter=adapter, adapter_set=aset, qualified=q,
        artifact_dir=source, output_dir=tmp_path / "native",
        model_sha256="b" * 64, input_features=3, output_features=4,
    )
    old = base_manifest()
    new = NativeQW3AdapterCompiler.bind_served_manifest(
        base=old, qualified=q, native_bundle=native,
    )
    assert new.adapter_set_root == aset.root_hex
    assert new.native_adapter_bundle_root == native.bundle_root
    assert new.artifact_root != old.artifact_root
    assert new.manifest_digest != old.manifest_digest


def test_serving_contract_requires_loaded_native_bundle_identity(tmp_path):
    plan, adapter, aset, q, source = build_source(tmp_path)
    native = NativeQW3AdapterCompiler().compile(
        plan=plan, adapter=adapter, adapter_set=aset, qualified=q,
        artifact_dir=source, output_dir=tmp_path / "native",
        model_sha256="c" * 64, input_features=3, output_features=4,
    )
    manifest = NativeQW3AdapterCompiler.bind_served_manifest(
        base=base_manifest(), qualified=q, native_bundle=native,
    )

    class Lease:
        epoch_digest = manifest.epoch_digest

    good = {
        "governed": True,
        "epoch_id": manifest.epoch_digest,
        "manifest_digest": manifest.manifest_digest,
        "artifact_root": manifest.artifact_root,
        "adapter_set_root": manifest.adapter_set_root,
        "native_adapter_bundle_root": manifest.native_adapter_bundle_root,
        "native_adapter_loaded": True,
    }
    c = GovernedServingContract(base_url="http://unused", fetch_json=lambda path: good)
    headers = c.request_headers(lease=Lease(), manifest=manifest)
    assert headers["X-MiniAGI-Native-Adapter-Bundle-Root"] == native.bundle_root

    bad = dict(good)
    bad["native_adapter_bundle_root"] = "d" * 64
    c = GovernedServingContract(base_url="http://unused", fetch_json=lambda path: bad)
    with pytest.raises(RuntimeStateMismatch):
        c.request_headers(lease=Lease(), manifest=manifest)
