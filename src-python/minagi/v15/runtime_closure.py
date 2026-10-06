from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
from typing import Any, Mapping

from minagi.integration.qw3_state import (
    GovernedServingContract,
    QW3RuntimeState,
    RuntimeStateMismatch,
    ServedArtifactManifest,
    canonical_digest,
)

_ZERO_ROOT = "0" * 64


def sha256_file(path: str | Path) -> str:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"runtime-closure path cannot be a symlink: {raw}")
    p = raw.resolve()
    if not p.is_file():
        raise ValueError(f"runtime-closure path must be a regular file: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_runtime_path(path: str | Path) -> str:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"runtime identity path cannot be a symlink: {raw}")
    p = raw.resolve()
    if p.is_file():
        return sha256_file(p)
    if not p.is_dir():
        raise ValueError(f"runtime identity path is neither file nor directory: {p}")
    rows: list[bytes] = [b"mini-agi-v15.4-runtime-tree-v1\n"]
    entries = sorted(p.rglob("*"), key=lambda x: x.relative_to(p).as_posix())
    files: list[Path] = []
    for entry in entries:
        if entry.is_symlink():
            raise ValueError(f"runtime identity directory contains symlink: {entry}")
        if entry.is_file():
            files.append(entry)
    for f in files:
        rel = f.relative_to(p).as_posix().encode()
        rows.extend((rel, b"\0", str(f.stat().st_size).encode(), b"\0", sha256_file(f).encode(), b"\n"))
    return hashlib.sha256(b"".join(rows)).hexdigest()


def tokenizer_identity_sha256_v155(model_path: str | Path) -> str:
    """Hash exactly the physical tokenizer inputs QW3 uses.

    GGUF embeds tokenizer metadata, so the whole GGUF digest is the conservative
    tokenizer identity. HF directories bind tokenizer.json + config.json and
    the same optional EOS source the native tokenizer selects.
    """
    raw = Path(model_path)
    if raw.is_symlink():
        raise ValueError(f"tokenizer identity source cannot be a symlink: {raw}")
    root = raw.resolve()
    if root.is_file():
        return sha256_file(root)
    if not root.is_dir():
        raise ValueError("tokenizer identity source is neither GGUF nor HF directory")
    tokenizer = root / "tokenizer.json"
    config = root / "config.json"
    for p in (tokenizer, config):
        if p.is_symlink() or not p.is_file():
            raise ValueError(f"HF tokenizer identity requires regular {p.name}")
    files = [tokenizer, config]
    tok_cfg = root / "tokenizer_config.json"
    gen_cfg = root / "generation_config.json"
    if tok_cfg.exists():
        if tok_cfg.is_symlink() or not tok_cfg.is_file():
            raise ValueError("unsafe tokenizer_config.json")
        files.append(tok_cfg)
    elif gen_cfg.exists():
        if gen_cfg.is_symlink() or not gen_cfg.is_file():
            raise ValueError("unsafe generation_config.json")
        files.append(gen_cfg)
    files.sort(key=lambda x: x.relative_to(root).as_posix())
    rows: list[bytes] = [b"mini-agi-v15.5-tokenizer-identity-v1\n"]
    for f in files:
        rel = f.relative_to(root).as_posix().encode()
        rows.extend((rel, b"\0", str(f.stat().st_size).encode(), b"\0", sha256_file(f).encode(), b"\n"))
    return hashlib.sha256(b"".join(rows)).hexdigest()


@dataclass(frozen=True)
class PhysicalArtifactPaths:
    adapter_set_artifact: str | None = None
    retrieval_policy_artifact: str | None = None
    skill_policy_artifact: str | None = None

    def for_root(self, name: str, expected_root: str) -> str | None:
        path = getattr(self, name)
        if expected_root == _ZERO_ROOT:
            if path is not None:
                raise RuntimeStateMismatch(f"{name} supplied for zero-root artifact")
            return None
        if path is None:
            raise RuntimeStateMismatch(f"{name} is required for nonzero governed root")
        return path


def _measure_optional_artifact(path: str | None, expected_root: str, name: str) -> str:
    if expected_root == _ZERO_ROOT:
        if path is not None:
            raise RuntimeStateMismatch(f"{name} is zero-root but a physical path was supplied")
        return _ZERO_ROOT
    if path is None:
        raise RuntimeStateMismatch(f"{name} is nonzero but no physical artifact was supplied")
    measured = sha256_runtime_path(path)
    if measured != expected_root:
        raise RuntimeStateMismatch(f"physical {name} bytes do not match authorized root")
    return measured


@dataclass(frozen=True)
class MeasuredRuntimeClosure:
    """v15.5 deployment proof over every served-artifact component QW3 can measure.

    This remains deployment verification, never promotion authority. Model,
    tokenizer, executable, KVMem, adapter-set, retrieval-policy, and skill-policy
    identities are derived from physical bytes before governed serving starts.
    """

    epoch_id: str
    manifest_digest: str
    artifact_root: str
    foundation_digest: str
    tokenizer_digest: str
    runtime_binary_digest: str
    kvmem_archive_root: str
    adapter_set_root: str
    retrieval_policy_root: str
    skill_policy_root: str
    native_adapter_bundle_root: str
    schema: str = "mini-agi-v15.5-runtime-closure-v1"

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))

    @classmethod
    def from_files(
        cls,
        *,
        manifest: ServedArtifactManifest,
        model_path: str | Path,
        runtime_binary_path: str | Path,
        kvmem_archive_dir: str | Path | None = None,
        artifacts: PhysicalArtifactPaths | None = None,
    ) -> "MeasuredRuntimeClosure":
        artifacts = artifacts or PhysicalArtifactPaths()
        model_sha = sha256_runtime_path(model_path)
        tokenizer_sha = tokenizer_identity_sha256_v155(model_path)
        runtime_sha = sha256_file(runtime_binary_path)
        if kvmem_archive_dir is None:
            kv_root = _ZERO_ROOT
        else:
            archive_manifest = Path(kvmem_archive_dir) / "manifest.json"
            kv_root = sha256_file(archive_manifest)
        if model_sha != manifest.foundation_digest:
            raise RuntimeStateMismatch("model bytes do not match manifest.foundation_digest")
        if tokenizer_sha != manifest.tokenizer_digest:
            raise RuntimeStateMismatch("tokenizer inputs do not match manifest.tokenizer_digest")
        if runtime_sha != manifest.runtime_binary_digest:
            raise RuntimeStateMismatch("QW3 binary bytes do not match manifest.runtime_binary_digest")
        if kv_root != manifest.kv_archive_root:
            raise RuntimeStateMismatch("KVMem manifest bytes do not match manifest.kv_archive_root")
        adapter_root = _measure_optional_artifact(
            artifacts.adapter_set_artifact, manifest.adapter_set_root, "adapter_set_root")
        retrieval_root = _measure_optional_artifact(
            artifacts.retrieval_policy_artifact, manifest.retrieval_policy_root, "retrieval_policy_root")
        skill_root = _measure_optional_artifact(
            artifacts.skill_policy_artifact, manifest.skill_policy_root, "skill_policy_root")
        return cls(
            epoch_id=manifest.epoch_digest,
            manifest_digest=manifest.manifest_digest,
            artifact_root=manifest.artifact_root,
            foundation_digest=model_sha,
            tokenizer_digest=tokenizer_sha,
            runtime_binary_digest=runtime_sha,
            kvmem_archive_root=kv_root,
            adapter_set_root=adapter_root,
            retrieval_policy_root=retrieval_root,
            skill_policy_root=skill_root,
            native_adapter_bundle_root=manifest.native_adapter_bundle_root,
        )

    def qw3_args(
        self, *, manifest: ServedArtifactManifest,
        artifacts: PhysicalArtifactPaths | None = None,
    ) -> tuple[str, ...]:
        artifacts = artifacts or PhysicalArtifactPaths()
        expected = (
            manifest.epoch_digest, manifest.manifest_digest, manifest.artifact_root,
            manifest.foundation_digest, manifest.tokenizer_digest, manifest.runtime_binary_digest,
            manifest.kv_archive_root, manifest.adapter_set_root,
            manifest.retrieval_policy_root, manifest.skill_policy_root,
            manifest.native_adapter_bundle_root,
        )
        actual = (
            self.epoch_id, self.manifest_digest, self.artifact_root,
            self.foundation_digest, self.tokenizer_digest, self.runtime_binary_digest,
            self.kvmem_archive_root, self.adapter_set_root,
            self.retrieval_policy_root, self.skill_policy_root,
            self.native_adapter_bundle_root,
        )
        if actual != expected:
            raise RuntimeStateMismatch("runtime closure does not match served manifest")
        # Recheck path presence/zero-root semantics when producing launch args.
        adapter_path = artifacts.for_root("adapter_set_artifact", manifest.adapter_set_root)
        retrieval_path = artifacts.for_root("retrieval_policy_artifact", manifest.retrieval_policy_root)
        skill_path = artifacts.for_root("skill_policy_artifact", manifest.skill_policy_root)
        args = [
            "--state-epoch-id", manifest.epoch_digest,
            "--state-manifest-digest", manifest.manifest_digest,
            "--state-artifact-root", manifest.artifact_root,
            "--state-adapter-set-root", manifest.adapter_set_root,
            "--state-foundation-digest", manifest.foundation_digest,
            "--state-tokenizer-digest", manifest.tokenizer_digest,
            "--state-kvmem-archive-root", manifest.kv_archive_root,
            "--state-retrieval-policy-root", manifest.retrieval_policy_root,
            "--state-skill-policy-root", manifest.skill_policy_root,
            "--state-runtime-binary-digest", manifest.runtime_binary_digest,
        ]
        if adapter_path is not None:
            args += ["--state-adapter-set-artifact", str(Path(adapter_path).resolve())]
        if retrieval_path is not None:
            args += ["--state-retrieval-policy-artifact", str(Path(retrieval_path).resolve())]
        if skill_path is not None:
            args += ["--state-skill-policy-artifact", str(Path(skill_path).resolve())]
        if manifest.native_adapter_bundle_root != _ZERO_ROOT:
            args += ["--state-native-adapter-bundle-root", manifest.native_adapter_bundle_root]
        return tuple(args)


class MeasuredGovernedServingContract(GovernedServingContract):
    """v15.5 contract requiring QW3's measured physical-artifact closure."""

    def verify_loaded_state(self, *, lease, manifest: ServedArtifactManifest) -> QW3RuntimeState:
        state = super().verify_loaded_state(lease=lease, manifest=manifest)
        if not state.runtime_closure_verified:
            raise RuntimeStateMismatch("QW3 has not verified measured runtime closure")
        expected_closure = MeasuredRuntimeClosure(
            epoch_id=manifest.epoch_digest,
            manifest_digest=manifest.manifest_digest,
            artifact_root=manifest.artifact_root,
            foundation_digest=manifest.foundation_digest,
            tokenizer_digest=manifest.tokenizer_digest,
            runtime_binary_digest=manifest.runtime_binary_digest,
            kvmem_archive_root=manifest.kv_archive_root,
            adapter_set_root=manifest.adapter_set_root,
            retrieval_policy_root=manifest.retrieval_policy_root,
            skill_policy_root=manifest.skill_policy_root,
            native_adapter_bundle_root=manifest.native_adapter_bundle_root,
        )
        expected = (
            expected_closure.digest,
            manifest.foundation_digest,
            manifest.tokenizer_digest,
            manifest.runtime_binary_digest,
            manifest.kv_archive_root,
            manifest.adapter_set_root,
            manifest.retrieval_policy_root,
            manifest.skill_policy_root,
        )
        actual = (
            state.runtime_closure_digest,
            state.measured_foundation_digest,
            state.measured_tokenizer_digest,
            state.measured_runtime_binary_digest,
            state.measured_kvmem_archive_root,
            state.measured_adapter_set_root,
            state.measured_retrieval_policy_root,
            state.measured_skill_policy_root,
        )
        if actual != expected:
            raise RuntimeStateMismatch(
                "QW3 measured-runtime closure mismatch: expected " + repr(expected) +
                " got " + repr(actual)
            )
        return state
