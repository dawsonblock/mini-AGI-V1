from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import tempfile
from typing import Any

from egai.common.canonical import canonical_bytes, digest


@dataclass(frozen=True)
class MaterializedCanonicalArtifact:
    path: str
    root_hex: str
    bytes: int
    canonical_digest: str
    schema: str = "mini-agi-v15.5-materialized-canonical-artifact-v1"


def materialize_canonical_artifact(value: Any, path: str | Path) -> MaterializedCanonicalArtifact:
    """Write one canonical governance object so file SHA-256 equals its root hex.

    `egai.common.canonical.digest(value)` is sha256(canonical_bytes(value)), so the
    materialized file becomes a physical byte representation of the exact object
    already named by the governance digest. Existing files are verified rather
    than overwritten; symlinks are rejected.
    """
    target = Path(path)
    if target.is_symlink():
        raise ValueError(f"canonical artifact target cannot be a symlink: {target}")
    data = canonical_bytes(value)
    canonical_digest = digest(value)
    root_hex = hashlib.sha256(data).hexdigest()
    if canonical_digest != "sha256:" + root_hex:
        raise RuntimeError("canonical digest implementation mismatch")
    if target.exists():
        if not target.is_file() or target.read_bytes() != data:
            raise FileExistsError(f"canonical artifact path already exists with different bytes: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", dir=str(target.parent))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    return MaterializedCanonicalArtifact(
        path=str(target.resolve()), root_hex=root_hex, bytes=len(data),
        canonical_digest=canonical_digest,
    )


def build_physically_bound_served_manifest(
    *,
    epoch_digest: str,
    runtime_manifest_digest: str,
    model_path: str | Path,
    runtime_binary_path: str | Path,
    kvmem_archive_dir: str | Path | None = None,
    adapter_set_artifact: str | Path | None = None,
    retrieval_policy_artifact: str | Path | None = None,
    skill_policy_artifact: str | Path | None = None,
    native_adapter_bundle_root: str = "0" * 64,
):
    """Derive a ServedArtifactManifest only from physical artifact bytes.

    The governance epoch/runtime-manifest identities are inputs because they are
    issued by the authority plane. Every served component identity is measured
    here, eliminating hand-copied model/policy roots from deployment config.
    """
    from minagi.integration.qw3_state import ServedArtifactManifest
    from minagi.v15.runtime_closure import (
        sha256_file, sha256_runtime_path, tokenizer_identity_sha256_v155,
    )

    model = Path(model_path)
    binary = Path(runtime_binary_path)
    if kvmem_archive_dir is None:
        kv_root = "0" * 64
    else:
        kv_root = sha256_file(Path(kvmem_archive_dir) / "manifest.json")

    def optional_root(path: str | Path | None) -> str:
        return "0" * 64 if path is None else sha256_runtime_path(path)

    return ServedArtifactManifest(
        epoch_digest=epoch_digest,
        runtime_manifest_digest=runtime_manifest_digest,
        foundation_digest=sha256_runtime_path(model),
        tokenizer_digest=tokenizer_identity_sha256_v155(model),
        kv_archive_root=kv_root,
        adapter_set_root=optional_root(adapter_set_artifact),
        retrieval_policy_root=optional_root(retrieval_policy_artifact),
        skill_policy_root=optional_root(skill_policy_artifact),
        runtime_binary_digest=sha256_file(binary),
        native_adapter_bundle_root=native_adapter_bundle_root,
    )
