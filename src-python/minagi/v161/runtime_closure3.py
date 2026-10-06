from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Iterable
from egai.common.canonical import digest, validate_digest

ZERO = "sha256:" + "0" * 64


def sha256_path(path: str | Path) -> str:
    p = Path(path)
    if p.is_symlink():
        raise ValueError(f"runtime artifact cannot be a symlink: {p}")
    if p.is_file():
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return "sha256:" + h.hexdigest()
    if not p.is_dir():
        raise ValueError(f"runtime artifact path missing: {p}")
    rows = []
    for f in sorted(x for x in p.rglob("*") if x.is_file() and not x.is_symlink()):
        rows.append((f.relative_to(p).as_posix(), f.stat().st_size, sha256_path(f)))
    return digest(rows)


@dataclass(frozen=True)
class RuntimeArtifactSpec:
    name: str
    path: str
    expected_digest: str
    required: bool = True
    schema: str = "mini-agi-v16.1-runtime-artifact-spec-v1"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("runtime artifact name required")
        validate_digest(self.expected_digest)

    def measure(self) -> str:
        p = Path(self.path)
        if not p.exists():
            if self.required:
                raise FileNotFoundError(self.path)
            return ZERO
        actual = sha256_path(p)
        if actual != self.expected_digest:
            raise PermissionError(f"physical runtime artifact mismatch: {self.name}")
        return actual


@dataclass(frozen=True)
class RuntimeClosureV161:
    candidate_digest: str
    promotion_digest: str
    epoch_digest: str
    artifacts: tuple[RuntimeArtifactSpec, ...]
    generation_config_digest: str
    environment_digest: str
    schema: str = "mini-agi-v16.1-runtime-closure-v1"

    def __post_init__(self) -> None:
        for value in (self.candidate_digest, self.promotion_digest, self.epoch_digest,
                      self.generation_config_digest, self.environment_digest):
            validate_digest(value)
        if not self.artifacts:
            raise ValueError("at least one physical runtime artifact is required")
        if len({x.name for x in self.artifacts}) != len(self.artifacts):
            raise ValueError("duplicate runtime artifact names")

    @property
    def digest(self) -> str:
        return digest(self)

    def verify_physical(self) -> dict[str, str]:
        return {spec.name: spec.measure() for spec in self.artifacts}
