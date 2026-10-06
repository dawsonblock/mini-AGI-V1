from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from minagi.egai.canonical import atomic_write_json, sha256_json
from .models import EvidenceStrength, PermanenceLevel, require_digest


class _Artifact:
    kind: str
    @property
    def digest(self) -> str:
        return sha256_json({"kind": self.kind, "body": asdict(self)})


@dataclass(frozen=True)
class CandidateManifestRC14(_Artifact):
    proposal_digest: str
    candidate_artifact_digest: str
    requested_permanence: int
    policy_digest: str
    kind: str = "candidate_manifest"
    def __post_init__(self):
        require_digest(self.proposal_digest, field_name="proposal_digest")
        require_digest(self.candidate_artifact_digest, field_name="candidate_artifact_digest")
        require_digest(self.policy_digest, field_name="policy_digest")
        PermanenceLevel(int(self.requested_permanence))


@dataclass(frozen=True)
class BuildManifestRC14(_Artifact):
    candidate_digest: str
    environment_digest: str
    output_digest: str
    kind: str = "build_manifest"
    def __post_init__(self):
        for n in ("candidate_digest", "environment_digest", "output_digest"):
            require_digest(getattr(self, n), field_name=n)


@dataclass(frozen=True)
class EvaluationBundleRC14(_Artifact):
    candidate_digest: str
    build_digest: str
    falsification_plan_digest: str
    falsification_run_digest: str
    environment_digest: str
    kind: str = "evaluation_bundle"
    def __post_init__(self):
        for n in ("candidate_digest", "build_digest", "falsification_plan_digest", "falsification_run_digest", "environment_digest"):
            require_digest(getattr(self, n), field_name=n)


@dataclass(frozen=True)
class QualificationEvidenceRC14(_Artifact):
    """Derived evaluation facts used by QualificationRecordRC14.

    These fields are intended to be produced by QualificationEngineRC14 from a sealed
    preregistered plan and an immutable recorded run, not supplied directly by a
    candidate. The privileged QUALIFIED transition still requires an independent
    transition signature over the resulting record digest.
    """
    candidate_digest: str
    evaluation_digest: str
    falsification_plan_digest: str
    falsification_run_digest: str
    pass_rate: float
    mandatory_classes_passed: bool
    retention_passed: bool
    negative_controls_passed: bool
    security_passed: bool
    fresh_ood_validated: bool
    fresh_hidden_passed: bool
    evaluator_id: str
    environment_digest: str
    task_set_digest: str
    attestation_digests: tuple[str, ...] = ()
    independently_verified: bool = False
    replicated: bool = False
    independently_reproduced: bool = False
    operator_approved: bool = False
    continual_experiment_evidence_digest: str = ""
    kind: str = "qualification_evidence"

    def __post_init__(self):
        for n in (
            "candidate_digest", "evaluation_digest", "falsification_plan_digest",
            "falsification_run_digest", "environment_digest", "task_set_digest",
        ):
            require_digest(getattr(self, n), field_name=n)
        if not 0.0 <= float(self.pass_rate) <= 1.0:
            raise ValueError("pass_rate must be in [0,1]")
        if not self.evaluator_id:
            raise ValueError("evaluator_id is required")
        for digest in self.attestation_digests:
            require_digest(digest, field_name="attestation_digest")
        if self.continual_experiment_evidence_digest:
            require_digest(self.continual_experiment_evidence_digest, field_name="continual_experiment_evidence_digest")
        object.__setattr__(self, "attestation_digests", tuple(str(x) for x in self.attestation_digests))


_ALLOWED_ATTESTATION_SCOPES = {
    "independent_verification",
    "replication",
    "independent_reproduction",
    "operator_approval",
}


@dataclass(frozen=True)
class EvidenceAttestationRC14(_Artifact):
    """Signed external evidence claim consumed by QualificationEngineRC14."""
    scope: str
    candidate_digest: str
    evaluation_digest: str
    evidence_digest: str
    issuer_id: str
    receipt: Mapping[str, Any]
    kind: str = "evidence_attestation"

    def __post_init__(self):
        if self.scope not in _ALLOWED_ATTESTATION_SCOPES:
            raise ValueError("unsupported evidence attestation scope")
        for n in ("candidate_digest", "evaluation_digest", "evidence_digest"):
            require_digest(getattr(self, n), field_name=n)
        if not self.issuer_id:
            raise ValueError("issuer_id is required")
        if not isinstance(self.receipt, Mapping) or not isinstance(self.receipt.get("body"), Mapping):
            raise ValueError("signed evidence attestation receipt required")
        object.__setattr__(self, "receipt", dict(self.receipt))

    @property
    def signed_body(self) -> dict[str, Any]:
        return {
            "schema": "egai-rc14-evidence-attestation-v1",
            "scope": self.scope,
            "candidate_digest": self.candidate_digest,
            "evaluation_digest": self.evaluation_digest,
            "evidence_digest": self.evidence_digest,
            "issuer_id": self.issuer_id,
        }


@dataclass(frozen=True)
class ExternalWitnessRC14(_Artifact):
    """Independently signed observation of the locally committed epoch head."""
    epoch_digest: str
    transition_head_digest: str
    authority_state_digest: str
    witness_id: str
    observed_ns: int
    receipt: Mapping[str, Any]
    kind: str = "external_witness"

    def __post_init__(self):
        for n in ("epoch_digest", "transition_head_digest", "authority_state_digest"):
            require_digest(getattr(self, n), field_name=n)
        if not self.witness_id or int(self.observed_ns) <= 0:
            raise ValueError("witness_id and positive observed_ns are required")
        if not isinstance(self.receipt, Mapping) or not isinstance(self.receipt.get("body"), Mapping):
            raise ValueError("signed external-witness receipt required")
        object.__setattr__(self, "receipt", dict(self.receipt))

    @property
    def signed_body(self) -> dict[str, Any]:
        return {
            "schema": "egai-rc14-external-witness-v1",
            "epoch_digest": self.epoch_digest,
            "transition_head_digest": self.transition_head_digest,
            "authority_state_digest": self.authority_state_digest,
            "witness_id": self.witness_id,
            "observed_ns": int(self.observed_ns),
        }


@dataclass(frozen=True)
class RuntimeAttestationRC14(_Artifact):
    """Independently signed measurement of the runtime closure for one epoch."""
    epoch_digest: str
    witness_digest: str
    foundation_model_digest: str
    tokenizer_digest: str
    adapter_set_digest: str
    isolated_neural_memory_root: str
    execution_manifest_digest: str
    runtime_config_digest: str
    source_tree_digest: str
    dependency_lock_digest: str
    build_provenance_digest: str
    target_platform_policy_digest: str
    runtime_attestation_policy_digest: str
    authority_state_digest: str
    attestor_id: str
    attested_ns: int
    receipt: Mapping[str, Any]
    kind: str = "runtime_attestation"

    def __post_init__(self):
        for n in (
            "epoch_digest", "witness_digest", "foundation_model_digest", "tokenizer_digest",
            "adapter_set_digest", "isolated_neural_memory_root", "execution_manifest_digest",
            "runtime_config_digest", "source_tree_digest", "dependency_lock_digest",
            "build_provenance_digest", "target_platform_policy_digest",
            "runtime_attestation_policy_digest", "authority_state_digest",
        ):
            require_digest(getattr(self, n), field_name=n)
        if not self.attestor_id or int(self.attested_ns) <= 0:
            raise ValueError("attestor_id and positive attested_ns are required")
        if not isinstance(self.receipt, Mapping) or not isinstance(self.receipt.get("body"), Mapping):
            raise ValueError("signed runtime-attestation receipt required")
        object.__setattr__(self, "receipt", dict(self.receipt))

    @property
    def signed_body(self) -> dict[str, Any]:
        return {
            "schema": "egai-rc14-runtime-attestation-v2",
            "epoch_digest": self.epoch_digest,
            "witness_digest": self.witness_digest,
            "foundation_model_digest": self.foundation_model_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "adapter_set_digest": self.adapter_set_digest,
            "isolated_neural_memory_root": self.isolated_neural_memory_root,
            "execution_manifest_digest": self.execution_manifest_digest,
            "runtime_config_digest": self.runtime_config_digest,
            "source_tree_digest": self.source_tree_digest,
            "dependency_lock_digest": self.dependency_lock_digest,
            "build_provenance_digest": self.build_provenance_digest,
            "target_platform_policy_digest": self.target_platform_policy_digest,
            "runtime_attestation_policy_digest": self.runtime_attestation_policy_digest,
            "authority_state_digest": self.authority_state_digest,
            "attestor_id": self.attestor_id,
            "attested_ns": int(self.attested_ns),
        }




@dataclass(frozen=True)
class ContinualExperimentEvidenceRC14(_Artifact):
    """Replayable authority-store binding from a preregistered experiment to L6 evidence."""
    candidate_digest: str
    proposal_digest: str
    plan_digest: str
    ablation_certificate_digest: str
    neural_memory_manifest_digest: str
    frozen_foundation_proof_digest: str
    neural_memory_qualification_digest: str
    experiment_protocol_head_digest: str
    neural_memory_manifest_body: Mapping[str, Any]
    frozen_foundation_proof_body: Mapping[str, Any]
    neural_memory_qualification_body: Mapping[str, Any]
    qualified: bool
    kind: str = "continual_experiment_evidence"

    def __post_init__(self):
        for n in (
            "candidate_digest", "proposal_digest", "plan_digest", "ablation_certificate_digest",
            "neural_memory_manifest_digest", "frozen_foundation_proof_digest",
            "neural_memory_qualification_digest", "experiment_protocol_head_digest",
        ):
            require_digest(getattr(self, n), field_name=n)
        for field_name in ("neural_memory_manifest_body", "frozen_foundation_proof_body", "neural_memory_qualification_body"):
            if not isinstance(getattr(self, field_name), Mapping):
                raise ValueError(f"{field_name} must be a mapping")
            object.__setattr__(self, field_name, dict(getattr(self, field_name)))
        if sha256_json(dict(self.neural_memory_manifest_body)) != self.neural_memory_manifest_digest:
            raise ValueError("neural-memory manifest body/digest mismatch")
        if sha256_json(dict(self.frozen_foundation_proof_body)) != self.frozen_foundation_proof_digest:
            raise ValueError("frozen-foundation proof body/digest mismatch")
        if sha256_json(dict(self.neural_memory_qualification_body)) != self.neural_memory_qualification_digest:
            raise ValueError("neural-memory qualification body/digest mismatch")
        if bool(self.neural_memory_qualification_body.get("qualified")) != bool(self.qualified):
            raise ValueError("experiment evidence qualification flag mismatch")


@dataclass(frozen=True)
class IndependentReproductionEvidenceRC14(_Artifact):
    candidate_digest: str
    source_plan_digest: str
    reproduction_plan_digest: str
    reproduction_certificate_digest: str
    reproduction_protocol_head_digest: str
    reproduction_certificate_body: Mapping[str, Any]
    qualified: bool
    kind: str = "independent_reproduction_evidence"
    def __post_init__(self):
        for n in ("candidate_digest","source_plan_digest","reproduction_plan_digest","reproduction_certificate_digest","reproduction_protocol_head_digest"):
            require_digest(getattr(self,n),field_name=n)
        if not isinstance(self.reproduction_certificate_body,Mapping): raise ValueError("reproduction_certificate_body must be a mapping")
        object.__setattr__(self,"reproduction_certificate_body",dict(self.reproduction_certificate_body))
        if sha256_json(dict(self.reproduction_certificate_body)) != self.reproduction_certificate_digest: raise ValueError("independent reproduction certificate body/digest mismatch")
        if bool(self.reproduction_certificate_body.get("eligible")) != bool(self.qualified): raise ValueError("independent reproduction qualification flag mismatch")

@dataclass(frozen=True)
class QualificationRecordRC14(_Artifact):
    proposal_digest: str
    candidate_digest: str
    evaluation_digest: str
    policy_digest: str
    passed: bool
    qualification_evidence_digest: str = ""
    internally_consistent: bool = True
    independently_verified: bool = False
    replicated: bool = False
    fresh_ood_validated: bool = False
    independently_reproduced: bool = False
    retention_passed: bool = False
    negative_controls_passed: bool = False
    security_passed: bool = False
    operator_approved: bool = False
    continual_experiment_evidence_digest: str = ""
    reasons: tuple[str, ...] = ()
    kind: str = "qualification_record"
    def __post_init__(self):
        for n in ("proposal_digest", "candidate_digest", "evaluation_digest", "policy_digest"):
            require_digest(getattr(self, n), field_name=n)
        if self.qualification_evidence_digest:
            require_digest(self.qualification_evidence_digest, field_name="qualification_evidence_digest")
        if self.continual_experiment_evidence_digest:
            require_digest(self.continual_experiment_evidence_digest, field_name="continual_experiment_evidence_digest")
        if self.independently_reproduced and not self.fresh_ood_validated:
            raise ValueError("E5 requires E4")
        if self.fresh_ood_validated and not self.replicated:
            raise ValueError("E4 requires E3")
        if self.replicated and not self.independently_verified:
            raise ValueError("E3 requires E2")
        if self.independently_verified and not self.internally_consistent:
            raise ValueError("E2 requires E1")
        object.__setattr__(self, "reasons", tuple(str(x) for x in self.reasons))
    @property
    def evidence_strength(self) -> EvidenceStrength:
        if self.independently_reproduced:
            return EvidenceStrength.E5_INDEPENDENTLY_REPRODUCED
        if self.fresh_ood_validated:
            return EvidenceStrength.E4_FRESH_OOD_VALIDATED
        if self.replicated:
            return EvidenceStrength.E3_REPLICATED
        if self.independently_verified:
            return EvidenceStrength.E2_INDEPENDENTLY_VERIFIED
        if self.internally_consistent:
            return EvidenceStrength.E1_INTERNALLY_CONSISTENT
        return EvidenceStrength.E0_UNVERIFIED


@dataclass(frozen=True)
class PromotionDecisionRC14(_Artifact):
    candidate_digest: str
    qualification_digest: str
    policy_digest: str
    approved: bool
    reasons: tuple[str, ...] = ()
    kind: str = "promotion_decision"
    def __post_init__(self):
        for n in ("candidate_digest", "qualification_digest", "policy_digest"):
            require_digest(getattr(self, n), field_name=n)
        object.__setattr__(self, "reasons", tuple(str(x) for x in self.reasons))


@dataclass(frozen=True)
class PromotionAuthorizationRC14(_Artifact):
    candidate_digest: str
    qualification_digest: str
    promotion_decision_digest: str
    policy_digest: str
    authority_state_digest: str
    kind: str = "promotion_authorization"
    def __post_init__(self):
        for n in ("candidate_digest", "qualification_digest", "promotion_decision_digest", "policy_digest", "authority_state_digest"):
            require_digest(getattr(self, n), field_name=n)


_TYPES = {
    c.kind: c for c in (
        CandidateManifestRC14,
        BuildManifestRC14,
        EvaluationBundleRC14,
        QualificationEvidenceRC14,
        EvidenceAttestationRC14,
        ExternalWitnessRC14,
        RuntimeAttestationRC14,
        ContinualExperimentEvidenceRC14,
        IndependentReproductionEvidenceRC14,
        QualificationRecordRC14,
        PromotionDecisionRC14,
        PromotionAuthorizationRC14,
    )
}


class AuthorityArtifactStore:
    """Immutable content-addressed typed authority artifacts."""
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        require_digest(digest)
        return self.objects / f"{digest[7:]}.json"

    def put(self, artifact: _Artifact) -> str:
        digest = artifact.digest
        doc = {"kind": artifact.kind, "body": asdict(artifact), "digest": digest}
        p = self._path(digest)
        if p.exists():
            if json.loads(p.read_text(encoding="utf-8")) != doc:
                raise RuntimeError("authority artifact digest collision")
        else:
            atomic_write_json(p, doc)
        return digest

    def get(self, digest: str, *, expected_kind: str | None = None) -> _Artifact:
        doc = json.loads(self._path(digest).read_text(encoding="utf-8"))
        kind = str(doc.get("kind"))
        cls = _TYPES.get(kind)
        if cls is None:
            raise RuntimeError("unknown authority artifact kind")
        obj = cls(**doc["body"])
        if obj.digest != digest or doc.get("digest") != digest:
            raise RuntimeError("authority artifact digest mismatch")
        if expected_kind is not None and kind != expected_kind:
            raise PermissionError(f"expected {expected_kind}, got {kind}")
        return obj

    def exists(self, digest: str) -> bool:
        try:
            self.get(digest)
            return True
        except (FileNotFoundError, RuntimeError, ValueError, KeyError):
            return False

    def verify(self) -> dict[str, int]:
        n = 0
        for p in self.objects.glob("*.json"):
            doc = json.loads(p.read_text(encoding="utf-8"))
            self.get(str(doc["digest"]))
            n += 1
        return {"artifacts": n}
