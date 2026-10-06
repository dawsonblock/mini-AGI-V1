from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RuntimeSettings:
    config_dir: Path
    memory: dict[str, Any]
    model: dict[str, Any]
    runtime: dict[str, Any]
    training: dict[str, Any]

    @property
    def memory_db(self) -> str:
        env = os.getenv("KVCONTINUAL_MEMORY_DB")
        if env:
            return env
        return str(self.memory.get("persistent_memory", {}).get("sqlite_path", "data/memory.sqlite3"))

    @property
    def registry_root(self) -> str:
        return os.getenv("KVCONTINUAL_REGISTRY_ROOT", "artifacts/adapters")

    @property
    def experience_db(self) -> str:
        env = os.getenv("KVCONTINUAL_EXPERIENCE_DB")
        if env:
            return env
        return str(self.memory.get("experience", {}).get("sqlite_path", "data/experience.sqlite3"))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return data


def load_runtime_settings(config_dir: str | None = None) -> RuntimeSettings:
    root = Path(config_dir or os.getenv("KVCONTINUAL_CONFIG_DIR", "configs"))
    return RuntimeSettings(
        config_dir=root,
        memory=_read_yaml(root / "memory.yaml"),
        model=_read_yaml(root / "model.yaml"),
        runtime=_read_yaml(root / "runtime.yaml"),
        training=_read_yaml(root / "training.yaml"),
    )
