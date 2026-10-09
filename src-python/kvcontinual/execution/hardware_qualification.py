from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any

from kvcontinual.execution.platforms.macos import detect_macos_capabilities
from kvcontinual.execution.types import ExecutionIdentity


def _digest_json(payload: Any) -> str:
    # Qualification evidence must be canonical JSON. Reject NaN/Inf rather than
    # silently emitting Python's non-standard JSON tokens into a certificate.
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unavailable"


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


@dataclass(frozen=True)
class HardwareRuntimeFingerprint:
    """Exact host/runtime identity used for Mac acceleration qualification.

    This is intentionally more specific than ModelIdentity. A mathematically
    equivalent model on a different Metal/PyTorch/Transformers stack is a new
    qualification target until proven otherwise.
    """
    system: str
    machine: str
    os_version: str
    python_version: str
    torch_version: str
    transformers_version: str
    mlx_version: str
    metal_toolchain: str
    mps_built: bool
    mps_available: bool
    backend_fingerprint: str
    hardware_model: str = "unknown"
    chip_identity: str = "unknown"
    physical_memory_bytes: int = 0

    @property
    def digest(self) -> str:
        return _digest_json(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "digest": self.digest}


def detect_hardware_runtime_fingerprint(*, backend_fingerprint: str) -> HardwareRuntimeFingerprint:
    caps = detect_macos_capabilities()
    metal = _cmd(["xcrun", "metal", "--version"]) if caps.is_macos else ""
    if not metal and caps.is_macos:
        metal = _cmd(["xcrun", "--version"])
    return HardwareRuntimeFingerprint(
        system=platform.system(),
        machine=platform.machine(),
        os_version=caps.macos_version or platform.platform(),
        python_version=platform.python_version(),
        torch_version=_version("torch"),
        transformers_version=_version("transformers"),
        mlx_version=_version("mlx"),
        metal_toolchain=metal or "unavailable",
        mps_built=caps.torch_mps_built,
        mps_available=caps.torch_mps_available,
        backend_fingerprint=backend_fingerprint,
        hardware_model=(_cmd(["sysctl", "-n", "hw.model"]) if caps.is_macos else platform.node()) or "unknown",
        chip_identity=(_cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) if caps.is_macos else platform.processor()) or "unknown",
        physical_memory_bytes=int((_cmd(["sysctl", "-n", "hw.memsize"]) if caps.is_macos else "0") or 0),
    )


@dataclass(frozen=True)
class QualificationSuiteCase:
    """Content-bound descriptor for one qualification case.

    payload_digest binds the certificate to the actual prompt/token payload used
    for the paired exact/accelerated run. Metrics alone are not sufficient
    evidence because the same case_id can otherwise be reused for new content.
    """
    case_id: str
    payload_digest: str
    topology: str = "ARBITRARY"
    segment_count: int = 0

    def validate(self) -> None:
        if not self.case_id:
            raise ValueError("qualification suite case_id is required")
        if not self.payload_digest.startswith("sha256:") or len(self.payload_digest) != 71:
            raise ValueError(f"invalid qualification payload digest for {self.case_id}")
        if self.segment_count < 0:
            raise ValueError("qualification segment_count cannot be negative")


@dataclass(frozen=True)
class QualificationSuiteManifest:
    suite_version: int
    name: str
    cases: tuple[QualificationSuiteCase, ...]
    suite_digest: str = ""

    def _payload(self) -> dict[str, Any]:
        return {
            "suite_version": self.suite_version,
            "name": self.name,
            "cases": [asdict(case) for case in self.cases],
        }

    def expected_digest(self) -> str:
        return _digest_json(self._payload())

    @property
    def reusable(self) -> bool:
        if self.suite_version != 1 or not self.name or not self.cases:
            return False
        try:
            for case in self.cases:
                case.validate()
        except ValueError:
            return False
        return self.suite_digest == self.expected_digest()

    def validate_observations(self, observations: list["AccelerationQualificationObservation"]) -> None:
        if not self.reusable:
            raise ValueError("qualification suite manifest is not reusable")
        expected = [(c.case_id, c.topology, c.segment_count) for c in self.cases]
        actual = [(o.case_id, o.topology, o.segment_count) for o in observations]
        if expected != actual:
            raise ValueError("qualification observations do not match suite case order/topology")

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "suite_digest": self.suite_digest}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QualificationSuiteManifest":
        cases = tuple(QualificationSuiteCase(**case) for case in raw["cases"])
        return cls(int(raw["suite_version"]), str(raw["name"]), cases, str(raw.get("suite_digest", "")))


def build_qualification_suite_manifest(name: str, cases: list[QualificationSuiteCase]) -> QualificationSuiteManifest:
    if not cases:
        raise ValueError("qualification suite requires at least one case")
    seen: set[str] = set()
    for case in cases:
        case.validate()
        if case.case_id in seen:
            raise ValueError(f"duplicate qualification case_id: {case.case_id}")
        seen.add(case.case_id)
    manifest = QualificationSuiteManifest(1, name, tuple(cases))
    return replace(manifest, suite_digest=manifest.expected_digest())


def qualification_payload_digest(payload: Any) -> str:
    """Digest a JSON-compatible qualification payload for suite binding."""
    return _digest_json(payload)


@dataclass(frozen=True)
class AccelerationQualificationThresholds:
    max_logit_kl: float = 0.02
    min_top1_agreement: float = 0.995
    max_failure_rate: float = 0.01
    min_case_count: int = 100
    min_median_ttft_speedup: float = 1.0


@dataclass(frozen=True)
class AccelerationQualificationObservation:
    case_id: str
    logit_kl: float
    top1_agreement: float
    accepted: bool
    exact_ttft_ms: float = 0.0
    accelerated_ttft_ms: float = 0.0
    topology: str = "ARBITRARY"
    segment_count: int = 0

    @property
    def ttft_speedup(self) -> float:
        if self.accelerated_ttft_ms <= 0.0 or self.exact_ttft_ms <= 0.0:
            return 0.0
        return self.exact_ttft_ms / self.accelerated_ttft_ms


@dataclass(frozen=True)
class AccelerationQualificationSummary:
    case_count: int
    max_logit_kl: float
    mean_top1_agreement: float
    failure_rate: float
    median_ttft_speedup: float
    quality_pass: bool
    performance_pass: bool
    passed: bool
    observations_digest: str


def summarize_acceleration_observations(
    observations: list[AccelerationQualificationObservation],
    thresholds: AccelerationQualificationThresholds | None = None,
) -> AccelerationQualificationSummary:
    if not observations:
        raise ValueError("hardware qualification requires observations")
    t = thresholds or AccelerationQualificationThresholds()
    if t.min_case_count <= 0:
        raise ValueError("min_case_count must be positive")
    for o in observations:
        values = (o.logit_kl, o.top1_agreement, o.exact_ttft_ms, o.accelerated_ttft_ms)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f"non-finite qualification metric in case {o.case_id}")
        if o.logit_kl < 0.0:
            raise ValueError(f"negative KL divergence in case {o.case_id}")
        if not 0.0 <= o.top1_agreement <= 1.0:
            raise ValueError(f"top1_agreement out of range in case {o.case_id}")
        if o.exact_ttft_ms < 0.0 or o.accelerated_ttft_ms < 0.0:
            raise ValueError(f"negative TTFT in case {o.case_id}")
    ordered = sorted((float(o.ttft_speedup) for o in observations if o.ttft_speedup > 0.0))
    if ordered:
        n = len(ordered)
        median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    else:
        median = 0.0
    count = len(observations)
    max_kl = max(float(o.logit_kl) for o in observations)
    mean_top1 = sum(float(o.top1_agreement) for o in observations) / count
    failures = sum(1 for o in observations if not o.accepted)
    failure_rate = failures / count
    quality = (
        count >= t.min_case_count
        and max_kl <= t.max_logit_kl
        and mean_top1 >= t.min_top1_agreement
        and failure_rate <= t.max_failure_rate
    )
    performance = bool(ordered) and median >= t.min_median_ttft_speedup
    payload = {
        "thresholds": asdict(t),
        "observations": [asdict(o) for o in observations],
    }
    return AccelerationQualificationSummary(
        case_count=count,
        max_logit_kl=max_kl,
        mean_top1_agreement=mean_top1,
        failure_rate=failure_rate,
        median_ttft_speedup=median,
        quality_pass=quality,
        performance_pass=performance,
        passed=quality and performance,
        observations_digest=_digest_json(payload),
    )


@dataclass(frozen=True)
class AccelerationQualificationCertificate:
    certificate_version: int
    execution_identity_digest: str
    backend_fingerprint: str
    hardware_runtime_digest: str
    release_manifest_digest: str
    qualification_suite_digest: str
    thresholds: AccelerationQualificationThresholds
    summary: AccelerationQualificationSummary
    qualification_ledger_head: str
    created_at: str
    certificate_digest: str = ""

    def _payload(self) -> dict[str, Any]:
        return {
            "certificate_version": self.certificate_version,
            "execution_identity_digest": self.execution_identity_digest,
            "backend_fingerprint": self.backend_fingerprint,
            "hardware_runtime_digest": self.hardware_runtime_digest,
            "release_manifest_digest": self.release_manifest_digest,
            "qualification_suite_digest": self.qualification_suite_digest,
            "thresholds": asdict(self.thresholds),
            "summary": asdict(self.summary),
            "qualification_ledger_head": self.qualification_ledger_head,
            "created_at": self.created_at,
        }

    def expected_digest(self) -> str:
        return _digest_json(self._payload())

    @property
    def reusable(self) -> bool:
        return (
            self.certificate_version == 2
            and self.summary.passed
            and bool(self.execution_identity_digest)
            and bool(self.backend_fingerprint)
            and bool(self.hardware_runtime_digest)
            and bool(self.release_manifest_digest)
            and self.qualification_suite_digest.startswith("sha256:")
            and bool(self.qualification_ledger_head)
            and self.certificate_digest == self.expected_digest()
        )

    def validate(
        self,
        *,
        identity: ExecutionIdentity,
        backend_fingerprint: str,
        hardware_runtime_digest: str,
        release_manifest_digest: str,
        qualification_suite_digest: str | None = None,
        qualification_ledger_head: str | None = None,
    ) -> None:
        if not self.reusable:
            raise ValueError("hardware acceleration certificate is not reusable")
        if self.execution_identity_digest != identity.digest:
            raise ValueError("hardware qualification execution identity mismatch")
        if self.backend_fingerprint != backend_fingerprint:
            raise ValueError("hardware qualification backend fingerprint mismatch")
        if self.hardware_runtime_digest != hardware_runtime_digest:
            raise ValueError("hardware qualification host/runtime mismatch")
        if self.release_manifest_digest != release_manifest_digest:
            raise ValueError("hardware qualification release manifest mismatch")
        if qualification_suite_digest is not None and self.qualification_suite_digest != qualification_suite_digest:
            raise ValueError("hardware qualification suite mismatch")
        if qualification_ledger_head is not None and self.qualification_ledger_head != qualification_ledger_head:
            raise ValueError("hardware qualification ledger head mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "certificate_digest": self.certificate_digest}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AccelerationQualificationCertificate":
        return cls(
            certificate_version=int(raw["certificate_version"]),
            execution_identity_digest=str(raw["execution_identity_digest"]),
            backend_fingerprint=str(raw["backend_fingerprint"]),
            hardware_runtime_digest=str(raw["hardware_runtime_digest"]),
            release_manifest_digest=str(raw["release_manifest_digest"]),
            qualification_suite_digest=str(raw.get("qualification_suite_digest", "")),
            thresholds=AccelerationQualificationThresholds(**raw["thresholds"]),
            summary=AccelerationQualificationSummary(**raw["summary"]),
            qualification_ledger_head=str(raw["qualification_ledger_head"]),
            created_at=str(raw["created_at"]),
            certificate_digest=str(raw.get("certificate_digest", "")),
        )


def build_acceleration_qualification_certificate(
    *,
    identity: ExecutionIdentity,
    backend_fingerprint: str,
    hardware_runtime: HardwareRuntimeFingerprint,
    release_manifest_digest: str,
    observations: list[AccelerationQualificationObservation],
    qualification_suite: QualificationSuiteManifest | None = None,
    qualification_ledger_head: str,
    thresholds: AccelerationQualificationThresholds | None = None,
) -> AccelerationQualificationCertificate:
    if not backend_fingerprint:
        raise ValueError("backend_fingerprint is required")
    if hardware_runtime.backend_fingerprint != backend_fingerprint:
        raise ValueError("hardware runtime fingerprint/backend mismatch")
    if not release_manifest_digest.startswith("sha256:"):
        raise ValueError("release_manifest_digest must be sha256")
    if not qualification_ledger_head.startswith("sha256:"):
        raise ValueError("qualification_ledger_head must be sha256")
    if qualification_suite is None:
        # Backwards-compatible construction still gets a deterministic suite
        # binding, but production scripts should provide a payload-bound suite.
        synthetic_cases = [QualificationSuiteCase(
            o.case_id,
            _digest_json({"case_id": o.case_id, "topology": o.topology, "segment_count": o.segment_count}),
            o.topology, o.segment_count,
        ) for o in observations]
        qualification_suite = build_qualification_suite_manifest("observation-metadata-fallback", synthetic_cases)
    qualification_suite.validate_observations(observations)
    t = thresholds or AccelerationQualificationThresholds()
    summary = summarize_acceleration_observations(observations, t)
    c = AccelerationQualificationCertificate(
        certificate_version=2,
        execution_identity_digest=identity.digest,
        backend_fingerprint=backend_fingerprint,
        hardware_runtime_digest=hardware_runtime.digest,
        release_manifest_digest=release_manifest_digest,
        qualification_suite_digest=qualification_suite.suite_digest,
        thresholds=t,
        summary=summary,
        qualification_ledger_head=qualification_ledger_head,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return replace(c, certificate_digest=c.expected_digest())


class HardwareQualificationStore:
    """Atomic local certificate store keyed by certificate digest."""
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, certificate: AccelerationQualificationCertificate) -> Path:
        if certificate.certificate_digest != certificate.expected_digest():
            raise ValueError("qualification certificate digest mismatch")
        path = self.root / f"{certificate.certificate_digest.split(':',1)[1]}.json"
        payload = certificate.to_dict()
        with tempfile.NamedTemporaryFile("w", dir=self.root, delete=False, encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
            tmp = Path(f.name)
        tmp.replace(path)
        return path

    def get(self, certificate_digest: str) -> AccelerationQualificationCertificate | None:
        if not certificate_digest.startswith("sha256:"):
            return None
        path = self.root / f"{certificate_digest.split(':',1)[1]}.json"
        if not path.exists():
            return None
        try:
            cert = AccelerationQualificationCertificate.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if cert.certificate_digest != certificate_digest or cert.expected_digest() != certificate_digest:
                return None
            return cert
        except Exception:
            return None
