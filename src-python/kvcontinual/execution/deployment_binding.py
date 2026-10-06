from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any

from kvcontinual.execution.hardware_qualification import AccelerationQualificationCertificate


def _digest_json(payload: Any) -> str:
    raw=json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
    return "sha256:"+hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class AccelerationDeploymentBinding:
    """Portable, tamper-evident handoff from qualification to deployment.

    The binding intentionally duplicates the certificate's deployment-critical
    identities. Operators can stage a namespace from one JSON object without
    manually copying six independent digests and accidentally mixing releases.
    """
    binding_version: int
    certificate_digest: str
    execution_identity_digest: str
    backend_fingerprint: str
    hardware_runtime_digest: str
    release_manifest_digest: str
    qualification_ledger_head: str
    qualification_suite_digest: str
    binding_digest: str = ""

    def _payload(self) -> dict[str, Any]:
        d=asdict(self); d.pop("binding_digest",None); return d

    def expected_digest(self) -> str:
        return _digest_json(self._payload())

    @property
    def reusable(self) -> bool:
        return (
            self.binding_version == 1
            and all((
                self.certificate_digest.startswith("sha256:"),
                bool(self.execution_identity_digest),
                bool(self.backend_fingerprint),
                self.hardware_runtime_digest.startswith("sha256:"),
                self.release_manifest_digest.startswith("sha256:"),
                self.qualification_ledger_head.startswith("sha256:"),
                self.qualification_suite_digest.startswith("sha256:"),
            ))
            and self.binding_digest == self.expected_digest()
        )

    def validate_certificate(self, certificate: AccelerationQualificationCertificate) -> None:
        if not self.reusable:
            raise ValueError("deployment binding is not reusable")
        if not certificate.reusable:
            raise ValueError("qualification certificate is not reusable")
        checks={
            "certificate digest": (certificate.certificate_digest,self.certificate_digest),
            "execution identity": (certificate.execution_identity_digest,self.execution_identity_digest),
            "backend fingerprint": (certificate.backend_fingerprint,self.backend_fingerprint),
            "hardware runtime": (certificate.hardware_runtime_digest,self.hardware_runtime_digest),
            "release manifest": (certificate.release_manifest_digest,self.release_manifest_digest),
            "qualification ledger": (certificate.qualification_ledger_head,self.qualification_ledger_head),
            "qualification suite": (certificate.qualification_suite_digest,self.qualification_suite_digest),
        }
        for label,(actual,expected) in checks.items():
            if actual != expected:
                raise ValueError(f"deployment binding {label} mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(),"binding_digest":self.binding_digest}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AccelerationDeploymentBinding":
        return cls(
            binding_version=int(raw["binding_version"]),
            certificate_digest=str(raw["certificate_digest"]),
            execution_identity_digest=str(raw["execution_identity_digest"]),
            backend_fingerprint=str(raw["backend_fingerprint"]),
            hardware_runtime_digest=str(raw["hardware_runtime_digest"]),
            release_manifest_digest=str(raw["release_manifest_digest"]),
            qualification_ledger_head=str(raw["qualification_ledger_head"]),
            qualification_suite_digest=str(raw["qualification_suite_digest"]),
            binding_digest=str(raw.get("binding_digest","")),
        )


def build_deployment_binding(
    certificate: AccelerationQualificationCertificate,
    *,
    backend_fingerprint: str,
    hardware_runtime_digest: str,
    release_manifest_digest: str,
    qualification_ledger_head: str,
    qualification_suite_digest: str,
) -> AccelerationDeploymentBinding:
    if not certificate.reusable:
        raise ValueError("qualification certificate is not reusable")
    expected={
        "backend fingerprint": (certificate.backend_fingerprint,backend_fingerprint),
        "hardware runtime": (certificate.hardware_runtime_digest,hardware_runtime_digest),
        "release manifest": (certificate.release_manifest_digest,release_manifest_digest),
        "qualification ledger": (certificate.qualification_ledger_head,qualification_ledger_head),
        "qualification suite": (certificate.qualification_suite_digest,qualification_suite_digest),
    }
    for label,(actual,wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"cannot build deployment binding: {label} mismatch")
    b=AccelerationDeploymentBinding(
        binding_version=1,
        certificate_digest=certificate.certificate_digest,
        execution_identity_digest=certificate.execution_identity_digest,
        backend_fingerprint=backend_fingerprint,
        hardware_runtime_digest=hardware_runtime_digest,
        release_manifest_digest=release_manifest_digest,
        qualification_ledger_head=qualification_ledger_head,
        qualification_suite_digest=qualification_suite_digest,
    )
    return replace(b,binding_digest=b.expected_digest())
