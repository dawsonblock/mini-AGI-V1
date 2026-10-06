from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from kvcontinual.execution.qualification_harness import (
    MacQualificationThresholds,
    QualificationObservation,
    summarize_observations,
)


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def sha256_json_file(path: str | Path) -> str:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return _sha256_bytes(_canonical(obj))


def _require_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise ValueError(f"{name} must be sha256:<64 lowercase hex>")
    tail = value[7:]
    if any(c not in "0123456789abcdef" for c in tail):
        raise ValueError(f"{name} must be sha256:<64 lowercase hex>")


def _finite_number(name: str, value: object, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} is below minimum")
    if maximum is not None and out > maximum:
        raise ValueError(f"{name} is above maximum")
    return out


@dataclass(frozen=True)
class QualificationBundle:
    schema_version: int
    hardware_toolchain_fingerprint: str
    hardware_probe_digest: str
    capture_manifest_digest: str
    model_weights_digest: str
    tokenizer_digest: str
    kernel_build_digest: str
    execution_identity_digest: str
    runtime_build_digest: str
    oracle_observations_digest: str
    qualification_context_digest: str
    thresholds: dict
    summary: dict
    bundle_digest: str


def build_qualification_bundle(
    *,
    hardware_probe: dict,
    capture_manifest: dict,
    observations: Iterable[QualificationObservation],
    model_weights_digest: str,
    tokenizer_digest: str,
    kernel_build_digest: str,
    execution_identity_digest: str,
    runtime_build_digest: str,
    thresholds: MacQualificationThresholds | None = None,
    require_apple_silicon: bool = True,
) -> QualificationBundle:
    for name, value in (
        ("model_weights_digest", model_weights_digest),
        ("tokenizer_digest", tokenizer_digest),
        ("kernel_build_digest", kernel_build_digest),
        ("execution_identity_digest", execution_identity_digest),
        ("runtime_build_digest", runtime_build_digest),
    ):
        _require_digest(name, value)
    hardware_fp = hardware_probe.get("hardware_toolchain_fingerprint")
    _require_digest("hardware_toolchain_fingerprint", hardware_fp)
    if require_apple_silicon:
        if hardware_probe.get("apple_silicon") is not True:
            raise ValueError("qualification probe is not from Apple Silicon")
        if hardware_probe.get("metal_available") is not True:
            raise ValueError("Metal compiler/toolchain is not available")
        if hardware_probe.get("mps_available") is not True:
            raise ValueError("PyTorch MPS is not available")

    obs = list(observations)
    for x in obs:
        if not math.isfinite(float(x.logit_kl)) or float(x.logit_kl) < 0:
            raise ValueError("qualification observations contain invalid logit_kl")
        if not math.isfinite(float(x.top1_agreement)) or not 0 <= float(x.top1_agreement) <= 1:
            raise ValueError("qualification observations contain invalid top1_agreement")
        if not isinstance(x.accepted, bool):
            raise ValueError("qualification observations contain non-boolean acceptance")
    t = thresholds or MacQualificationThresholds()
    summary = summarize_observations(obs, t)
    observations_payload = [asdict(x) for x in obs]
    hardware_probe_digest = _sha256_bytes(_canonical(hardware_probe))
    capture_manifest_digest = _sha256_bytes(_canonical(capture_manifest))
    oracle_observations_digest = _sha256_bytes(_canonical(observations_payload))
    thresholds_payload = asdict(t)
    summary_payload = asdict(summary)
    qualification_context = {
        "hardware_toolchain_fingerprint": hardware_fp,
        "hardware_probe_digest": hardware_probe_digest,
        "capture_manifest_digest": capture_manifest_digest,
        "model_weights_digest": model_weights_digest,
        "tokenizer_digest": tokenizer_digest,
        "kernel_build_digest": kernel_build_digest,
        "execution_identity_digest": execution_identity_digest,
        "runtime_build_digest": runtime_build_digest,
        "oracle_observations_digest": oracle_observations_digest,
        "thresholds": thresholds_payload,
    }
    qualification_context_digest = _sha256_bytes(_canonical(qualification_context))
    body = {
        "schema_version": 2,
        **qualification_context,
        "qualification_context_digest": qualification_context_digest,
        "summary": summary_payload,
    }
    bundle_digest = _sha256_bytes(_canonical(body))
    return QualificationBundle(**body, bundle_digest=bundle_digest)


def _verify_v1(bundle: dict) -> bool:
    """Read-only compatibility for RC11.3 qualification bundles."""
    expected = dict(bundle)
    bundle_digest = expected.pop("bundle_digest")
    _require_digest("bundle_digest", bundle_digest)
    for name in (
        "hardware_toolchain_fingerprint", "hardware_probe_digest",
        "capture_manifest_digest", "model_weights_digest", "tokenizer_digest",
        "kernel_build_digest", "oracle_observations_digest",
    ):
        _require_digest(name, expected[name])
    summary = expected.get("summary")
    if not isinstance(summary, dict) or summary.get("passed") is not True:
        return False
    return bundle_digest == _sha256_bytes(_canonical(expected))


def verify_qualification_bundle(bundle: dict) -> bool:
    try:
        if not isinstance(bundle, dict):
            return False
        schema = bundle.get("schema_version")
        if schema == 1:
            return _verify_v1(bundle)
        if schema != 2:
            return False
        expected = dict(bundle)
        bundle_digest = expected.pop("bundle_digest")
        _require_digest("bundle_digest", bundle_digest)
        for name in (
            "hardware_toolchain_fingerprint", "hardware_probe_digest",
            "capture_manifest_digest", "model_weights_digest", "tokenizer_digest",
            "kernel_build_digest", "execution_identity_digest", "runtime_build_digest",
            "oracle_observations_digest", "qualification_context_digest",
        ):
            _require_digest(name, expected[name])
        thresholds = expected.get("thresholds")
        summary = expected.get("summary")
        if not isinstance(thresholds, dict) or not isinstance(summary, dict):
            return False
        max_logit_kl = _finite_number("threshold.max_logit_kl", thresholds.get("max_logit_kl"), minimum=0)
        min_top1 = _finite_number("threshold.min_top1_agreement", thresholds.get("min_top1_agreement"), minimum=0, maximum=1)
        max_failure = _finite_number("threshold.max_failure_rate", thresholds.get("max_failure_rate"), minimum=0, maximum=1)
        min_cases = thresholds.get("min_cases")
        if not isinstance(min_cases, int) or isinstance(min_cases, bool) or min_cases <= 0:
            return False
        case_count = summary.get("case_count")
        if not isinstance(case_count, int) or isinstance(case_count, bool) or case_count < min_cases:
            return False
        _require_digest("summary.receipt_digest", summary.get("receipt_digest"))
        observed_max_kl = _finite_number("summary.max_logit_kl", summary.get("max_logit_kl"), minimum=0)
        observed_top1 = _finite_number("summary.mean_top1_agreement", summary.get("mean_top1_agreement"), minimum=0, maximum=1)
        observed_failure = _finite_number("summary.failure_rate", summary.get("failure_rate"), minimum=0, maximum=1)
        if summary.get("passed") is not True:
            return False
        if observed_max_kl > max_logit_kl or observed_top1 < min_top1 or observed_failure > max_failure:
            return False
        context = {
            "hardware_toolchain_fingerprint": expected["hardware_toolchain_fingerprint"],
            "hardware_probe_digest": expected["hardware_probe_digest"],
            "capture_manifest_digest": expected["capture_manifest_digest"],
            "model_weights_digest": expected["model_weights_digest"],
            "tokenizer_digest": expected["tokenizer_digest"],
            "kernel_build_digest": expected["kernel_build_digest"],
            "execution_identity_digest": expected["execution_identity_digest"],
            "runtime_build_digest": expected["runtime_build_digest"],
            "oracle_observations_digest": expected["oracle_observations_digest"],
            "thresholds": thresholds,
        }
        if expected["qualification_context_digest"] != _sha256_bytes(_canonical(context)):
            return False
        return bundle_digest == _sha256_bytes(_canonical(expected))
    except (KeyError, TypeError, ValueError):
        return False
