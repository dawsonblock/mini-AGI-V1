from kvcontinual.continual.config import load_runtime_settings


def test_repository_config_is_loaded():
    s = load_runtime_settings("configs")
    assert s.memory_db == "data/memory.sqlite3"
    assert s.model["model"]["immutable_base"] is True
    assert s.runtime["reconstruction"]["default_mode"] == "BALANCED"
