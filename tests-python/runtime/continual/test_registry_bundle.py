import pytest
from kvcontinual.continual.registry import AdapterRegistry


def test_directory_adapter_requires_content_bound_packaging(tmp_path):
    bundle = tmp_path / "mlx-adapter"; bundle.mkdir()
    (bundle / "adapters.safetensors").write_bytes(b"weights")
    reg = AdapterRegistry(str(tmp_path / "reg"))
    with pytest.raises(ValueError, match="regular non-symlink file"):
        reg.register_candidate(str(bundle), "sha256:base", "sha256:data", {"lr": 1e-5})


def test_directory_symlink_surface_is_not_accepted_as_candidate(tmp_path):
    bundle = tmp_path / "mlx-adapter"; bundle.mkdir(); (bundle / "weights").write_bytes(b"good")
    reg = AdapterRegistry(str(tmp_path / "reg"))
    with pytest.raises(ValueError):
        reg.register_candidate(str(bundle), "sha256:base", "sha256:data", {})
