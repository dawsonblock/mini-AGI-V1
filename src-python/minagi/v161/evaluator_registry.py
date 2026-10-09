from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib
import inspect
import json
from typing import Any, Callable, Mapping
from egai.common.canonical import canonical_bytes, digest, validate_digest


def _file_sha256(path: str | Path) -> str:
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise ValueError(f"evaluator artifact must be a regular file: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


@dataclass(frozen=True)
class EvaluatorArtifact:
    evaluator_id: str
    module: str
    symbol: str
    implementation_digest: str
    config: Mapping[str, Any]
    config_digest: str
    schema_version: int = 1
    schema: str = "mini-agi-v16.1-evaluator-artifact-v1"

    def __post_init__(self) -> None:
        if not self.evaluator_id or not self.module or not self.symbol:
            raise ValueError("evaluator_id/module/symbol required")
        validate_digest(self.implementation_digest)
        validate_digest(self.config_digest)
        object.__setattr__(self, "config", dict(self.config))
        if digest(dict(self.config)) != self.config_digest:
            raise ValueError("config_digest does not match evaluator config")

    @property
    def digest(self) -> str:
        return digest(self)

    @classmethod
    def from_callable(cls, evaluator_id: str, fn: Callable[..., Any], *, config: Mapping[str, Any] | None = None):
        module = inspect.getmodule(fn)
        source_file = inspect.getsourcefile(fn)
        if module is None or source_file is None:
            raise ValueError("evaluator must resolve to a source-backed Python callable")
        cfg = dict(config or {})
        return cls(
            evaluator_id=evaluator_id,
            module=module.__name__,
            symbol=fn.__qualname__,
            implementation_digest=_file_sha256(source_file),
            config=cfg,
            config_digest=digest(cfg),
        )

    def resolve(self) -> Callable[..., Any]:
        module = importlib.import_module(self.module)
        value: Any = module
        for part in self.symbol.split("."):
            value = getattr(value, part)
        if not callable(value):
            raise TypeError("registered evaluator symbol is not callable")
        source_file = inspect.getsourcefile(value)
        if source_file is None or _file_sha256(source_file) != self.implementation_digest:
            raise PermissionError("evaluator implementation digest mismatch")
        return value


class EvaluatorRegistry:
    """Immutable content-addressed evaluator registry.

    Campaigns bind artifact.digest, never a human-readable evaluator name alone.
    """
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, artifact_digest: str) -> Path:
        validate_digest(artifact_digest)
        return self.root / (artifact_digest.split(":", 1)[1] + ".json")

    def register(self, artifact: EvaluatorArtifact) -> str:
        path = self._path(artifact.digest)
        payload = canonical_bytes(artifact)
        if path.exists():
            if path.read_bytes() != payload:
                raise RuntimeError("evaluator digest collision")
        else:
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(payload)
            tmp.replace(path)
        return artifact.digest

    def get(self, artifact_digest: str) -> EvaluatorArtifact:
        path = self._path(artifact_digest)
        if not path.is_file():
            raise KeyError(artifact_digest)
        doc = json.loads(path.read_text())
        value = doc.get("value", doc)
        artifact = EvaluatorArtifact(**value)
        if artifact.digest != artifact_digest:
            raise RuntimeError("stored evaluator artifact digest mismatch")
        return artifact

    def resolve(self, artifact_digest: str) -> Callable[..., Any]:
        return self.get(artifact_digest).resolve()
