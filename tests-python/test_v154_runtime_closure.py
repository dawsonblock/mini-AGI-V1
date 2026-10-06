from __future__ import annotations

from pathlib import Path
import hashlib

import pytest

from egai.common.canonical import digest
from minagi.integration.qw3_state import RuntimeStateMismatch, ServedArtifactManifest
from minagi.v15.runtime_closure import (
    MeasuredGovernedServingContract,
    MeasuredRuntimeClosure,
    sha256_file,
)

ZERO = "0" * 64


def h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_manifest(model: Path, binary: Path) -> ServedArtifactManifest:
    return ServedArtifactManifest(
        epoch_digest=digest({"epoch": 154}),
        runtime_manifest_digest=digest({"runtime": 154}),
        foundation_digest=sha256_file(model),
        tokenizer_digest=sha256_file(model),
        kv_archive_root=ZERO,
        adapter_set_root=ZERO,
        retrieval_policy_root=ZERO,
        skill_policy_root=ZERO,
        runtime_binary_digest=sha256_file(binary),
        native_adapter_bundle_root=ZERO,
    )


def test_measured_runtime_closure_hashes_real_files(tmp_path):
    model = tmp_path / "model.gguf"
    binary = tmp_path / "qw3"
    model.write_bytes(b"model-v154")
    binary.write_bytes(b"binary-v154")
    manifest = make_manifest(model, binary)
    closure = MeasuredRuntimeClosure.from_files(
        manifest=manifest, model_path=model, runtime_binary_path=binary,
    )
    assert closure.foundation_digest == h(b"model-v154")
    assert closure.runtime_binary_digest == h(b"binary-v154")
    assert closure.kvmem_archive_root == ZERO
    args = closure.qw3_args(manifest=manifest)
    assert "--state-foundation-digest" in args
    assert manifest.runtime_binary_digest in args


def test_measured_runtime_closure_rejects_model_tamper(tmp_path):
    model = tmp_path / "model.gguf"
    binary = tmp_path / "qw3"
    model.write_bytes(b"model-v154")
    binary.write_bytes(b"binary-v154")
    manifest = make_manifest(model, binary)
    model.write_bytes(b"tampered")
    with pytest.raises(RuntimeStateMismatch, match="model bytes"):
        MeasuredRuntimeClosure.from_files(
            manifest=manifest, model_path=model, runtime_binary_path=binary,
        )


def test_measured_runtime_closure_rejects_binary_tamper(tmp_path):
    model = tmp_path / "model.gguf"
    binary = tmp_path / "qw3"
    model.write_bytes(b"model-v154")
    binary.write_bytes(b"binary-v154")
    manifest = make_manifest(model, binary)
    binary.write_bytes(b"tampered")
    with pytest.raises(RuntimeStateMismatch, match="binary bytes"):
        MeasuredRuntimeClosure.from_files(
            manifest=manifest, model_path=model, runtime_binary_path=binary,
        )


def test_measured_contract_requires_server_measurements(tmp_path):
    model = tmp_path / "model.gguf"
    binary = tmp_path / "qw3"
    model.write_bytes(b"model-v154")
    binary.write_bytes(b"binary-v154")
    manifest = make_manifest(model, binary)
    closure = MeasuredRuntimeClosure.from_files(
        manifest=manifest, model_path=model, runtime_binary_path=binary,
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
    c = MeasuredGovernedServingContract(
        base_url="http://unused", fetch_json=lambda _: state,
    )
    c.request_headers(lease=Lease(), manifest=manifest)
    bad = dict(state)
    bad["measured_runtime_binary_digest"] = "f" * 64
    c = MeasuredGovernedServingContract(
        base_url="http://unused", fetch_json=lambda _: bad,
    )
    with pytest.raises(RuntimeStateMismatch, match="measured-runtime"):
        c.request_headers(lease=Lease(), manifest=manifest)

def test_runtime_directory_digest_is_deterministic(tmp_path):
    from minagi.v15.runtime_closure import sha256_runtime_path
    root = tmp_path / "model"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_bytes(b"A")
    (root / "sub" / "b.bin").write_bytes(b"BC")
    assert sha256_runtime_path(root) == "c36e8f47adf9a053fad7795d89b75e1c3c9a634b99141423331f9d8d100913c2"
