import hashlib
import json
import pytest

from kvcontinual.execution.qualification_bundle import build_qualification_bundle, verify_qualification_bundle
from kvcontinual.execution.qualification_harness import MacQualificationThresholds, QualificationObservation


def _d(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _probe():
    body = {
        "schema_version": 2,
        "platform": "darwin",
        "machine": "arm64",
        "apple_silicon": True,
        "metal_available": True,
        "mps_available": True,
        "xcodebuild": "Xcode test",
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["hardware_toolchain_fingerprint"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    return body


def test_qualification_bundle_is_deterministic_and_verified():
    obs = [QualificationObservation(f"c{i}", 0.001, 1.0, True) for i in range(4)]
    thresholds = MacQualificationThresholds(min_cases=4)
    kwargs = dict(
        hardware_probe=_probe(), capture_manifest={"layers": [1, 2]}, observations=obs,
        model_weights_digest=_d("model"), tokenizer_digest=_d("tok"),
        kernel_build_digest=_d("kernel"), execution_identity_digest=_d("exec"),
        runtime_build_digest=_d("runtime"), thresholds=thresholds,
    )
    a = build_qualification_bundle(**kwargs)
    b = build_qualification_bundle(**kwargs)
    assert a == b
    assert a.summary["passed"] is True
    assert verify_qualification_bundle(a.__dict__)


def test_qualification_bundle_fails_closed_without_mps():
    probe = _probe(); probe["mps_available"] = False
    with pytest.raises(ValueError, match="MPS"):
        build_qualification_bundle(
            hardware_probe=probe, capture_manifest={},
            observations=[QualificationObservation("c", 0.0, 1.0, True)],
            model_weights_digest=_d("model"), tokenizer_digest=_d("tok"), kernel_build_digest=_d("k"),
            execution_identity_digest=_d("exec"), runtime_build_digest=_d("runtime"),
            thresholds=MacQualificationThresholds(min_cases=1),
        )


def test_qualification_bundle_rejects_tampering():
    obs = [QualificationObservation("c", 0.0, 1.0, True)]
    bundle = build_qualification_bundle(
        hardware_probe=_probe(), capture_manifest={}, observations=obs,
        model_weights_digest=_d("model"), tokenizer_digest=_d("tok"), kernel_build_digest=_d("k"),
        execution_identity_digest=_d("exec"), runtime_build_digest=_d("runtime"),
        thresholds=MacQualificationThresholds(min_cases=1),
    )
    payload = dict(bundle.__dict__)
    payload["summary"] = dict(payload["summary"])
    payload["summary"]["case_count"] = 999
    assert not verify_qualification_bundle(payload)


def test_schema_v1_bundle_remains_readable_for_migration():
    probe = _probe()
    hardware_probe_digest = _d(json.dumps(probe, sort_keys=True, separators=(",", ":")))
    capture_digest = _d(json.dumps({}, sort_keys=True, separators=(",", ":")))
    observations = [{"case_id": "c", "logit_kl": 0.0, "top1_agreement": 1.0, "accepted": True}]
    obs_digest = _d(json.dumps(observations, sort_keys=True, separators=(",", ":")))
    # RC11.3 verification only required a passing summary and canonical body digest.
    body = {
        "schema_version": 1,
        "hardware_toolchain_fingerprint": probe["hardware_toolchain_fingerprint"],
        "hardware_probe_digest": hardware_probe_digest,
        "capture_manifest_digest": capture_digest,
        "model_weights_digest": _d("model"),
        "tokenizer_digest": _d("tok"),
        "kernel_build_digest": _d("kernel"),
        "oracle_observations_digest": obs_digest,
        "thresholds": {"max_logit_kl": 0.02, "min_top1_agreement": 0.995, "max_failure_rate": 0.01, "min_cases": 1},
        "summary": {"case_count": 1, "max_logit_kl": 0.0, "mean_top1_agreement": 1.0, "failure_rate": 0.0, "passed": True, "receipt_digest": _d("legacy")},
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    body["bundle_digest"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    assert verify_qualification_bundle(body)


def test_schema_v2_binds_execution_and_runtime_identity():
    bundle = build_qualification_bundle(
        hardware_probe=_probe(), capture_manifest={},
        observations=[QualificationObservation("c", 0.0, 1.0, True)],
        model_weights_digest=_d("model"), tokenizer_digest=_d("tok"), kernel_build_digest=_d("kernel"),
        execution_identity_digest=_d("exec"), runtime_build_digest=_d("runtime"),
        thresholds=MacQualificationThresholds(min_cases=1),
    )
    payload = dict(bundle.__dict__)
    payload["runtime_build_digest"] = _d("different-runtime")
    assert not verify_qualification_bundle(payload)
