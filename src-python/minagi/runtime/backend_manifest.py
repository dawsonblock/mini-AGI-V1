"""v16.4.3 runtime backend manifest (HARDENING_PLAN WP6 / SEC-206).

A grant that names ``hf-peft`` must not authorize an arbitrary object
claiming to be hf-peft. `RuntimeBackendManifestV1` binds a backend id to
a measured implementation: the digest of the backend's source module(s),
the dependency closure's lock digest, the runtime image digest where one
exists, the model/adapter formats the backend declares, and the policy
epoch the measurement was made under. The `admission` (or `runtime`)
authority signs the manifest; grants carry its digest as
``backend_binary_digest``, and the supervisor re-measures the installed
backend before PREPARED — a valid signature for hf-peft cannot authorize
a mutated implementation.

Scope of the measurement: for a Python backend this is the digests of
the backend module files plus the resolved versions of its declared
import dependencies (``importlib.metadata``), digested together. GPU
drivers and host firmware are environment evidence recorded by the
manifest's ``environment`` field; exact binary immutability of the host
OS is out of scope.
"""
from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import Ed25519Signer, SignedEnvelope

BACKEND_MANIFEST_SCHEMA = "mini-agi-v16.4.3-backend-manifest-v1"


class BackendRefused(PermissionError):
    """The installed backend does not match its authorized identity."""


@dataclass(frozen=True)
class RuntimeBackendManifestV1:
    backend_id: str
    implementation_digest: str
    dependency_lock_digest: str
    supported_model_formats: tuple[str, ...]
    supported_adapter_formats: tuple[str, ...]
    policy_epoch: int
    runtime_image_digest: str = ""
    environment: dict | None = None
    schema: str = BACKEND_MANIFEST_SCHEMA

    def __post_init__(self):
        if not self.backend_id:
            raise ValueError("backend_id required")
        validate_digest(self.implementation_digest)
        validate_digest(self.dependency_lock_digest)
        if self.runtime_image_digest:
            validate_digest(self.runtime_image_digest)
        if isinstance(self.policy_epoch, bool) or \
                not isinstance(self.policy_epoch, int):
            raise ValueError("policy_epoch must be an integer")

    def to_body(self) -> dict:
        return {"schema": self.schema, "backend_id": self.backend_id,
                "implementation_digest": self.implementation_digest,
                "dependency_lock_digest": self.dependency_lock_digest,
                "runtime_image_digest": self.runtime_image_digest,
                "supported_model_formats":
                    list(self.supported_model_formats),
                "supported_adapter_formats":
                    list(self.supported_adapter_formats),
                "policy_epoch": self.policy_epoch,
                "environment": dict(self.environment or {})}

    def to_doc(self, *, signer: Ed25519Signer) -> dict:
        body = self.to_body()
        env = signer.sign(body)
        return {"value": body, "digest": digest(body),
                "signer_key_id": env.key_id,
                "signature_b64": env.signature_b64}

    @classmethod
    def from_value(cls, value: dict) -> "RuntimeBackendManifestV1":
        return cls(
            backend_id=str(value["backend_id"]),
            implementation_digest=str(value["implementation_digest"]),
            dependency_lock_digest=str(value["dependency_lock_digest"]),
            runtime_image_digest=str(
                value.get("runtime_image_digest") or ""),
            supported_model_formats=tuple(
                str(f) for f in value.get("supported_model_formats") or ()),
            supported_adapter_formats=tuple(
                str(f) for f in value.get("supported_adapter_formats") or ()),
            policy_epoch=int(value.get("policy_epoch", 0)),
            environment=dict(value.get("environment") or {}),
            schema=str(value["schema"]))


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def measure_backend(module_names, *, dependency_packages=(),
                    backend_id: str = "", policy_epoch: int = 0,
                    environment: dict | None = None
                    ) -> RuntimeBackendManifestV1:
    """Measure the installed backend: digest the source files of
    `module_names` plus the resolved versions of `dependency_packages`
    (importlib.metadata), producing an unsigned manifest callers sign."""
    files: dict[str, str] = {}
    for name in module_names:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or \
                spec.origin.endswith(".pyc"):
            raise BackendRefused(
                f"backend module {name!r} cannot be resolved to a "
                "source file — an unmeasurable backend is not a "
                "qualified backend")
        files[name] = _hash_file(Path(spec.origin))
    implementation_digest = digest(dict(sorted(files.items())))

    deps: dict[str, str] = {}
    for pkg in dependency_packages:
        try:
            from importlib.metadata import version
            deps[pkg] = version(pkg)
        except Exception:  # noqa: BLE001 - distribution name vs module
            deps[pkg] = "unresolved"
    dependency_lock_digest = digest(dict(sorted(deps.items())))

    return RuntimeBackendManifestV1(
        backend_id=backend_id or module_names[0].rsplit(".", 1)[-1],
        implementation_digest=implementation_digest,
        dependency_lock_digest=dependency_lock_digest,
        supported_model_formats=(), supported_adapter_formats=(),
        policy_epoch=int(policy_epoch), environment=environment)


def verify_backend_manifest(doc, registry, *, role: str = "admission",
                            now=None) -> RuntimeBackendManifestV1:
    """Verify a signed backend manifest end to end: envelope digest,
    signer authorized for `role`, signature, and schema."""
    if not isinstance(doc, dict) or "value" not in doc:
        raise BackendRefused("backend manifest: signed envelope required")
    value = doc["value"]
    if doc.get("digest") != digest(value):
        raise BackendRefused("backend manifest: envelope digest mismatch")
    if value.get("schema") != BACKEND_MANIFEST_SCHEMA:
        raise BackendRefused(
            f"backend manifest schema {value.get('schema')!r} unknown")
    kid = str(doc.get("signer_key_id", ""))
    if not kid:
        raise BackendRefused("backend manifest is unsigned")
    if not registry.is_authorized(role, kid, now=now):
        raise BackendRefused(
            f"backend manifest signer {kid!r} is not an authorized "
            f"{role} authority")
    if not registry.verifier(now=now).verify(
            value, SignedEnvelope(kid, str(doc.get("signature_b64", "")))):
        raise BackendRefused("backend manifest signature invalid")
    return RuntimeBackendManifestV1.from_value(value)


def verify_installed_backend(manifest: RuntimeBackendManifestV1, *,
                             module_names, dependency_packages=(),
                             min_policy_epoch: int = 0) -> None:
    """Re-measure the installed backend and compare against the signed
    manifest — a grant cannot authorize an implementation that changed
    after admission, and a manifest under a superseded policy epoch
    refuses."""
    if manifest.policy_epoch < int(min_policy_epoch):
        raise BackendRefused(
            f"backend manifest policy epoch {manifest.policy_epoch} is "
            f"older than the operative epoch {min_policy_epoch} — "
            "superseded authorization refused")
    measured = measure_backend(
        module_names, dependency_packages=dependency_packages,
        backend_id=manifest.backend_id,
        policy_epoch=manifest.policy_epoch)
    if measured.implementation_digest != manifest.implementation_digest:
        raise BackendRefused(
            "installed backend implementation digest does not match "
            "the authorized manifest — the backend changed after "
            "admission")
    if measured.dependency_lock_digest != \
            manifest.dependency_lock_digest:
        raise BackendRefused(
            "installed dependency closure differs from the authorized "
            "manifest")
