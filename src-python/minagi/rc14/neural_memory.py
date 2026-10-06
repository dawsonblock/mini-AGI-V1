from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

from minagi.egai.canonical import sha256_json
from .experiment_protocol import AblationQualificationCertificateRC14, PreregisteredContinualExperimentPlanRC14
from .models import require_digest


@dataclass(frozen=True)
class IsolatedNeuralMemoryManifestRC14:
    candidate_digest: str
    parent_epoch_digest: str
    base_model_digest: str
    tokenizer_digest: str
    memory_artifact_digest: str
    training_dataset_digest: str
    training_config_digest: str
    trainable_scope_digest: str
    parameter_count: int
    capacity_bytes: int
    schema: str = "egai-rc14-isolated-neural-memory-manifest-v1"

    def __post_init__(self) -> None:
        for name in ("candidate_digest", "parent_epoch_digest", "base_model_digest", "tokenizer_digest", "memory_artifact_digest",
                     "training_dataset_digest", "training_config_digest", "trainable_scope_digest"):
            require_digest(getattr(self, name), field_name=name)
        if int(self.parameter_count) <= 0 or int(self.capacity_bytes) <= 0:
            raise ValueError("isolated neural memory must have positive bounded capacity")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class FrozenFoundationProofRC14:
    base_model_digest_before: str
    base_model_digest_after: str
    tokenizer_digest_before: str
    tokenizer_digest_after: str
    frozen_parameter_root_before: str
    frozen_parameter_root_after: str
    allowed_trainable_parameters: tuple[str, ...]
    observed_changed_parameters: tuple[str, ...]
    schema: str = "egai-rc14-frozen-foundation-proof-v1"

    def __post_init__(self) -> None:
        for name in ("base_model_digest_before", "base_model_digest_after", "tokenizer_digest_before", "tokenizer_digest_after",
                     "frozen_parameter_root_before", "frozen_parameter_root_after"):
            require_digest(getattr(self, name), field_name=name)
        allowed = tuple(str(x) for x in self.allowed_trainable_parameters)
        changed = tuple(str(x) for x in self.observed_changed_parameters)
        if not allowed or not changed or len(set(allowed)) != len(allowed) or len(set(changed)) != len(changed):
            raise ValueError("trainable/changed parameter sets must be unique and non-empty")
        object.__setattr__(self, "allowed_trainable_parameters", allowed)
        object.__setattr__(self, "observed_changed_parameters", changed)

    @property
    def foundation_unchanged(self) -> bool:
        return (self.base_model_digest_before == self.base_model_digest_after and
                self.tokenizer_digest_before == self.tokenizer_digest_after and
                self.frozen_parameter_root_before == self.frozen_parameter_root_after)

    @property
    def changes_confined(self) -> bool:
        return set(self.observed_changed_parameters).issubset(set(self.allowed_trainable_parameters))

    @property
    def trainable_scope_digest(self) -> str:
        return sha256_json(sorted(self.allowed_trainable_parameters))

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class IsolatedNeuralMemoryQualificationRC14:
    candidate_digest: str
    plan_digest: str
    ablation_certificate_digest: str
    manifest_digest: str
    frozen_foundation_proof_digest: str
    qualified: bool
    reasons: tuple[str, ...]
    schema: str = "egai-rc14-isolated-neural-memory-qualification-v1"

    def __post_init__(self) -> None:
        for name in ("candidate_digest", "plan_digest", "ablation_certificate_digest", "manifest_digest", "frozen_foundation_proof_digest"):
            require_digest(getattr(self, name), field_name=name)
        object.__setattr__(self, "reasons", tuple(str(x) for x in self.reasons))

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


class IsolatedNeuralMemoryQualificationEngineRC14:
    """Qualifies L6 evidence only; it cannot promote or activate a candidate."""

    can_promote = False
    can_activate = False

    def __init__(self, *, max_capacity_bytes: int = 2 * 1024 * 1024 * 1024, max_parameters: int = 50_000_000):
        self.max_capacity_bytes = int(max_capacity_bytes)
        self.max_parameters = int(max_parameters)
        if self.max_capacity_bytes <= 0 or self.max_parameters <= 0:
            raise ValueError("qualification budgets must be positive")

    def qualify(self, *, plan: PreregisteredContinualExperimentPlanRC14,
                certificate: AblationQualificationCertificateRC14,
                manifest: IsolatedNeuralMemoryManifestRC14,
                proof: FrozenFoundationProofRC14) -> IsolatedNeuralMemoryQualificationRC14:
        reasons: list[str] = []
        if certificate.plan_digest != plan.digest or certificate.candidate_digest != plan.candidate_digest:
            reasons.append("ablation_plan_binding")
        if manifest.candidate_digest != plan.candidate_digest:
            reasons.append("manifest_candidate_binding")
        if manifest.base_model_digest != plan.base_model_digest or manifest.tokenizer_digest != plan.tokenizer_digest:
            reasons.append("manifest_model_binding")
        if not certificate.complete: reasons.append("ablation_incomplete")
        if not certificate.eligible: reasons.append("ablation_not_eligible")
        if not proof.foundation_unchanged: reasons.append("foundation_mutated")
        if not proof.changes_confined: reasons.append("trainable_scope_escape")
        if proof.base_model_digest_before != manifest.base_model_digest or proof.tokenizer_digest_before != manifest.tokenizer_digest:
            reasons.append("proof_model_binding")
        if manifest.trainable_scope_digest != proof.trainable_scope_digest:
            reasons.append("trainable_scope_digest_binding")
        if manifest.capacity_bytes > self.max_capacity_bytes: reasons.append("capacity_budget")
        if manifest.parameter_count > self.max_parameters: reasons.append("parameter_budget")
        return IsolatedNeuralMemoryQualificationRC14(
            candidate_digest=manifest.candidate_digest, plan_digest=plan.digest,
            ablation_certificate_digest=certificate.digest, manifest_digest=manifest.digest,
            frozen_foundation_proof_digest=proof.digest, qualified=not reasons, reasons=tuple(dict.fromkeys(reasons)),
        )
