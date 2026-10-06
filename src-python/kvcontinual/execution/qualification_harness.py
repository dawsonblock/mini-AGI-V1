from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import hashlib
import json

from kvcontinual.execution.cache.block import ArtifactQualification


@dataclass(frozen=True)
class AssemblyCase:
    name: str
    selected: tuple[str, ...]


ADVERSARIAL_ASSEMBLIES = (
    AssemblyCase("canonical", tuple("ABCDEFGH")),
    AssemblyCase("deletion", ("A", "D", "F", "G")),
    AssemblyCase("reorder", ("A", "F", "D", "G")),
    AssemblyCase("extreme_relocation", ("H", "A", "D")),
    AssemblyCase("duplication", ("A", "A", "D", "D")),
    AssemblyCase("isolated", ("D",)),
    AssemblyCase("reverseish", ("G", "B", "E", "A")),
)


def qualification_matrix(segment_sizes=(128, 512, 2048, 8192), model_classes=("dense_ffn", "moe"), adapter_versions=("base", "adapter")):
    for case, seg, model, adapter in product(ADVERSARIAL_ASSEMBLIES, segment_sizes, model_classes, adapter_versions):
        yield {"case": case.name, "selected": case.selected, "segment_size": seg, "model_class": model, "adapter_version": adapter}


@dataclass(frozen=True)
class MacQualificationThresholds:
    max_logit_kl: float = 0.02
    min_top1_agreement: float = 0.995
    max_failure_rate: float = 0.01
    min_cases: int = 100


@dataclass(frozen=True)
class QualificationObservation:
    case_id: str
    logit_kl: float
    top1_agreement: float
    accepted: bool


@dataclass(frozen=True)
class QualificationSummary:
    case_count: int
    max_logit_kl: float
    mean_top1_agreement: float
    failure_rate: float
    passed: bool
    receipt_digest: str


def summarize_observations(
    observations: list[QualificationObservation],
    thresholds: MacQualificationThresholds | None = None,
) -> QualificationSummary:
    t = thresholds or MacQualificationThresholds()
    if not observations:
        raise ValueError("qualification requires observations")
    count = len(observations)
    max_kl = max(float(x.logit_kl) for x in observations)
    top1 = sum(float(x.top1_agreement) for x in observations) / count
    failures = sum(1 for x in observations if not x.accepted)
    failure_rate = failures / count
    passed = count >= t.min_cases and max_kl <= t.max_logit_kl and top1 >= t.min_top1_agreement and failure_rate <= t.max_failure_rate
    payload = {
        "thresholds": asdict(t),
        "observations": [asdict(x) for x in observations],
        "summary": {
            "case_count": count,
            "max_logit_kl": max_kl,
            "mean_top1_agreement": top1,
            "failure_rate": failure_rate,
            "passed": passed,
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    receipt = "sha256:" + hashlib.sha256(raw).hexdigest()
    return QualificationSummary(count, max_kl, top1, failure_rate, passed, receipt)


def artifact_qualification_from_summary(
    summary: QualificationSummary,
    *,
    same_hidden_input_verified: bool,
    conv_boundary_complete: bool,
    attention_relocation_complete: bool,
    hardware_fingerprint: str,
    capture_manifest_digest: str,
) -> ArtifactQualification:
    return ArtifactQualification(
        same_hidden_input_verified=same_hidden_input_verified,
        conv_boundary_complete=conv_boundary_complete,
        attention_relocation_complete=attention_relocation_complete,
        oracle_qualified=summary.passed,
        max_logit_kl=summary.max_logit_kl,
        min_top1_agreement=summary.mean_top1_agreement,
        hardware_fingerprint=hardware_fingerprint,
        capture_manifest_digest=capture_manifest_digest,
        receipt_digest=summary.receipt_digest,
    )
