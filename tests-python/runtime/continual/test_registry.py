import pytest
from kvcontinual.continual.registry import AdapterRegistry
from kvcontinual.continual.qualification import QualificationPolicy


def make_candidate(reg, tmp_path, payload: bytes):
    f = tmp_path / (str(len(payload)) + ".safetensors")
    f.write_bytes(payload)
    return reg.register_candidate(str(f), "sha256:base", "sha256:data", {"lr": 1e-5})


def test_legacy_registry_is_fail_closed_adapter_to_execution_authority(tmp_path):
    with pytest.raises(PermissionError, match="unsigned"):
        AdapterRegistry(str(tmp_path / "bad"), require_signed_promotions=False)
    reg = AdapterRegistry(str(tmp_path / "registry"))
    c = make_candidate(reg, tmp_path, b"a")
    q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
    with pytest.raises(RuntimeError, match="qualification bundle"):
        reg.write_qualification(q)
    with pytest.raises(RuntimeError, match="no qualification record"):
        reg.promote(c.candidate_id)
