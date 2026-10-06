from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen


_HEX64 = set("0123456789abcdef")
_ZERO_ROOT = "0" * 64


def _governance_digest(value: str, name: str) -> str:
    value = str(value).lower()
    if value.startswith("sha256:"):
        hexpart = value.split(":", 1)[1]
        if len(hexpart) == 64 and all(c in _HEX64 for c in hexpart):
            return "sha256:" + hexpart
    raise ValueError(f"{name} must be a canonical sha256: digest")


def _digest_value(value: str, name: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(c not in _HEX64 for c in value):
        raise ValueError(f"{name} must be a lowercase sha256 digest")
    return value


def canonical_digest(obj: Any) -> str:
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ServedArtifactManifest:
    """The exact state a QW3 process is authorized to serve.

    `artifact_root` closes over mutable/replaceable serving artifacts.  The
    individual roots are retained so operators can explain *what* changed.
    """

    epoch_digest: str
    runtime_manifest_digest: str
    foundation_digest: str
    tokenizer_digest: str
    kv_archive_root: str
    adapter_set_root: str
    retrieval_policy_root: str
    skill_policy_root: str
    runtime_binary_digest: str
    native_adapter_bundle_root: str = _ZERO_ROOT
    schema: str = "mini-agi-v15-served-artifact-manifest-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch_digest", _governance_digest(self.epoch_digest, "epoch_digest"))
        object.__setattr__(self, "runtime_manifest_digest", _governance_digest(self.runtime_manifest_digest, "runtime_manifest_digest"))
        for name in (
            "foundation_digest", "tokenizer_digest", "kv_archive_root", "adapter_set_root",
            "retrieval_policy_root", "skill_policy_root", "runtime_binary_digest",
            "native_adapter_bundle_root",
        ):
            object.__setattr__(self, name, _digest_value(getattr(self, name), name))

    @property
    def artifact_root(self) -> str:
        # Deliberately exclude epoch/runtime-manifest identity: this is the root
        # of the *served artifacts*, not of the governance envelope.
        body = {
            "schema": "mini-agi-v15-served-artifact-root-v1",
            "foundation_digest": self.foundation_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "kv_archive_root": self.kv_archive_root,
            "adapter_set_root": self.adapter_set_root,
            "retrieval_policy_root": self.retrieval_policy_root,
            "skill_policy_root": self.skill_policy_root,
            "runtime_binary_digest": self.runtime_binary_digest,
        }
        # Backward-compatible closure: pre-v15.3 manifests keep their exact
        # artifact root when no compiled native adapter bundle is present.
        if self.native_adapter_bundle_root != _ZERO_ROOT:
            body["native_adapter_bundle_root"] = self.native_adapter_bundle_root
        return canonical_digest(body)

    @property
    def manifest_digest(self) -> str:
        body = asdict(self)
        body["artifact_root"] = self.artifact_root
        return canonical_digest(body)

    def to_dict(self) -> dict[str, str]:
        out = asdict(self)
        out["artifact_root"] = self.artifact_root
        out["manifest_digest"] = self.manifest_digest
        return out


@dataclass(frozen=True)
class QW3RuntimeState:
    governed: bool
    epoch_id: str | None
    manifest_digest: str | None
    artifact_root: str | None
    adapter_set_root: str | None = None
    native_adapter_bundle_root: str | None = None
    native_adapter_loaded: bool = False
    runtime_closure_verified: bool = False
    runtime_closure_digest: str | None = None
    measured_foundation_digest: str | None = None
    measured_tokenizer_digest: str | None = None
    measured_runtime_binary_digest: str | None = None
    measured_kvmem_archive_root: str | None = None
    measured_adapter_set_root: str | None = None
    measured_retrieval_policy_root: str | None = None
    measured_skill_policy_root: str | None = None
    model_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QW3RuntimeState":
        return cls(
            governed=bool(value.get("governed", False)),
            epoch_id=value.get("epoch_id"),
            manifest_digest=value.get("manifest_digest"),
            artifact_root=value.get("artifact_root"),
            adapter_set_root=value.get("adapter_set_root"),
            native_adapter_bundle_root=value.get("native_adapter_bundle_root"),
            native_adapter_loaded=bool(value.get("native_adapter_loaded", False)),
            runtime_closure_verified=bool(value.get("runtime_closure_verified", False)),
            runtime_closure_digest=value.get("runtime_closure_digest"),
            measured_foundation_digest=value.get("measured_foundation_digest"),
            measured_tokenizer_digest=value.get("measured_tokenizer_digest"),
            measured_runtime_binary_digest=value.get("measured_runtime_binary_digest"),
            measured_kvmem_archive_root=value.get("measured_kvmem_archive_root"),
            measured_adapter_set_root=value.get("measured_adapter_set_root"),
            measured_retrieval_policy_root=value.get("measured_retrieval_policy_root"),
            measured_skill_policy_root=value.get("measured_skill_policy_root"),
            model_id=value.get("model_id"),
        )


class RuntimeStateMismatch(PermissionError):
    pass


class GovernedServingContract:
    """Fail-closed bridge between a SERVABLE StateEpoch and QW3.

    This class does not create promotion authority. It verifies that the native
    server reports the exact state authorized by an already-issued epoch lease,
    then emits headers QW3 independently enforces on generation routes.
    """

    def __init__(self, *, base_url: str, timeout: float = 5.0,
                 fetch_json: Callable[[str], Mapping[str, Any]] | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self._fetch_json = fetch_json or self._default_fetch_json

    def _default_fetch_json(self, path: str) -> Mapping[str, Any]:
        with urlopen(self.base_url + path, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def runtime_state(self) -> QW3RuntimeState:
        return QW3RuntimeState.from_mapping(self._fetch_json("/v1/runtime/state"))

    def verify_loaded_state(self, *, lease, manifest: ServedArtifactManifest) -> QW3RuntimeState:
        # StateEpochLeaseV145 uses `epoch_digest`. Keep structural typing so the
        # bridge does not create a second authority dependency.
        leased_epoch = str(getattr(lease, "epoch_digest"))
        if leased_epoch != manifest.epoch_digest:
            raise RuntimeStateMismatch("lease epoch does not match served-artifact manifest")
        state = self.runtime_state()
        if not state.governed:
            raise RuntimeStateMismatch("QW3 is not running in governed-state mode")
        expected = (manifest.epoch_digest, manifest.manifest_digest, manifest.artifact_root, manifest.adapter_set_root)
        actual = (state.epoch_id, state.manifest_digest, state.artifact_root, state.adapter_set_root)
        if actual != expected:
            raise RuntimeStateMismatch(
                "QW3 loaded-state mismatch: expected " + repr(expected) + " got " + repr(actual)
            )
        if manifest.native_adapter_bundle_root != _ZERO_ROOT:
            if not state.native_adapter_loaded:
                raise RuntimeStateMismatch("QW3 did not load the governed native adapter bundle")
            if state.native_adapter_bundle_root != manifest.native_adapter_bundle_root:
                raise RuntimeStateMismatch(
                    "QW3 native-adapter bundle mismatch: expected " +
                    manifest.native_adapter_bundle_root + " got " +
                    repr(state.native_adapter_bundle_root)
                )
        return state

    def request_headers(self, *, lease, manifest: ServedArtifactManifest) -> dict[str, str]:
        self.verify_loaded_state(lease=lease, manifest=manifest)
        headers = {
            "X-MiniAGI-State-Epoch": manifest.epoch_digest,
            "X-MiniAGI-Manifest-Digest": manifest.manifest_digest,
            "X-MiniAGI-Artifact-Root": manifest.artifact_root,
            "X-MiniAGI-Adapter-Set-Root": manifest.adapter_set_root,
            "X-MiniAGI-Tokenizer-Digest": manifest.tokenizer_digest,
            "X-MiniAGI-Retrieval-Policy-Root": manifest.retrieval_policy_root,
            "X-MiniAGI-Skill-Policy-Root": manifest.skill_policy_root,
        }
        if manifest.native_adapter_bundle_root != _ZERO_ROOT:
            headers["X-MiniAGI-Native-Adapter-Bundle-Root"] = manifest.native_adapter_bundle_root
        return headers

    def post_json(self, *, path: str, payload: Mapping[str, Any], lease,
                  manifest: ServedArtifactManifest) -> Mapping[str, Any]:
        headers = self.request_headers(lease=lease, manifest=manifest)
        req = Request(
            self.base_url + path,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
