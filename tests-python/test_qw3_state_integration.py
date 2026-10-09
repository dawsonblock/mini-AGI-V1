import pytest

from minagi.integration.qw3_state import (
    GovernedServingContract,
    RuntimeStateMismatch,
    ServedArtifactManifest,
)

D = {
    "epoch_digest": "sha256:" + "1"*64,
    "runtime_manifest_digest": "sha256:" + "2"*64,
    "foundation_digest": "3"*64,
    "tokenizer_digest": "4"*64,
    "kv_archive_root": "5"*64,
    "adapter_set_root": "6"*64,
    "retrieval_policy_root": "7"*64,
    "skill_policy_root": "8"*64,
    "runtime_binary_digest": "9"*64,
}

class Lease:
    epoch_digest = "sha256:" + "1"*64


def manifest():
    return ServedArtifactManifest(**D)


def test_manifest_artifact_root_changes_when_kv_changes():
    a = manifest()
    b = ServedArtifactManifest(**{**D, "kv_archive_root": "a"*64})
    assert a.artifact_root != b.artifact_root
    assert a.manifest_digest != b.manifest_digest


def test_contract_accepts_exact_loaded_state():
    m = manifest()
    def fetch(path):
        return {
            "governed": True,
            "epoch_id": m.epoch_digest,
            "manifest_digest": m.manifest_digest,
            "artifact_root": m.artifact_root,
            "adapter_set_root": m.adapter_set_root,
            "model_id": "qwen",
        }
    c = GovernedServingContract(base_url="http://unused", fetch_json=fetch)
    headers = c.request_headers(lease=Lease(), manifest=m)
    assert headers["X-MiniAGI-State-Epoch"] == m.epoch_digest
    assert headers["X-MiniAGI-Manifest-Digest"] == m.manifest_digest
    assert headers["X-MiniAGI-Artifact-Root"] == m.artifact_root
    assert headers["X-MiniAGI-Adapter-Set-Root"] == m.adapter_set_root


@pytest.mark.parametrize("field,bad", [
    ("epoch_id", "a"*64),
    ("manifest_digest", "b"*64),
    ("artifact_root", "c"*64),
    ("adapter_set_root", "d"*64),
])
def test_contract_rejects_any_loaded_state_mismatch(field, bad):
    m = manifest()
    state = {
        "governed": True,
        "epoch_id": m.epoch_digest,
        "manifest_digest": m.manifest_digest,
        "artifact_root": m.artifact_root,
        "adapter_set_root": m.adapter_set_root,
    }
    state[field] = bad
    c = GovernedServingContract(base_url="http://unused", fetch_json=lambda path: state)
    with pytest.raises(RuntimeStateMismatch):
        c.request_headers(lease=Lease(), manifest=m)


def test_contract_rejects_ungoverned_server():
    m = manifest()
    c = GovernedServingContract(base_url="http://unused", fetch_json=lambda path: {"governed": False})
    with pytest.raises(RuntimeStateMismatch):
        c.request_headers(lease=Lease(), manifest=m)


def test_contract_rejects_lease_manifest_epoch_mismatch():
    m = manifest()
    class OtherLease:
        epoch_digest = "sha256:" + "a"*64
    c = GovernedServingContract(base_url="http://unused", fetch_json=lambda path: {})
    with pytest.raises(RuntimeStateMismatch):
        c.request_headers(lease=OtherLease(), manifest=m)
