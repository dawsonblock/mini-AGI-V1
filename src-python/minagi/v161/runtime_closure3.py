"""v16.1 runtime artifact closure — now backed by the v16.4.1 canonical
closure (`minagi.v161.artifact_closure`).

`sha256_path` keeps its historical digest semantics (sorted
(relpath, size, sha256) rows for directories, content digest for
files), so evidence recorded before v16.4.1 still verifies — but the
audit defect is closed: symbolic links and unsupported special files
inside an authorized tree are REFUSED instead of being silently
skipped, and the full authorized file listing is available to
manifests via `TreeClosure.manifest()`.

New callers should use `artifact_closure.close_tree` directly (it
returns the entries as well as the digest).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from egai.common.canonical import digest, validate_digest

from .artifact_closure import (TreeClosure,
                               close_tree)

ZERO = "sha256:" + "0" * 64


def sha256_path(path: str | Path) -> str:
    """Canonical digest of a file or directory tree. Symlinks and
    special files anywhere in the tree are refused (v16.4.1)."""
    return close_tree(path).digest


def close_path(path: str | Path) -> TreeClosure:
    """Digest plus the complete authorized entry listing."""
    return close_tree(path)


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
