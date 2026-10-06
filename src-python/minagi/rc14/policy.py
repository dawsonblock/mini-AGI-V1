from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from minagi.egai.canonical import sha256_json
from .authority_artifacts import QualificationRecordRC14
from .models import EvidenceStrength, LearningProposalRC14, PermanenceLevel


@dataclass(frozen=True)
class GovernanceRule:
    permanence: PermanenceLevel
    min_evidence: EvidenceStrength
    activatable: bool
    require_retention: bool = True
    require_negative_controls: bool = True
    require_security: bool = True
    require_fresh_ood: bool = False
    require_independent_reproduction: bool = False
    require_operator_approval: bool = False
    min_falsification_pass_rate: float = 1.0

    def __post_init__(self):
        if not 0.0 <= float(self.min_falsification_pass_rate) <= 1.0:
            raise ValueError("min_falsification_pass_rate must be in [0,1]")


DEFAULT_RULES: tuple[GovernanceRule, ...] = (
    GovernanceRule(PermanenceLevel.L0_WORKING_CONTEXT, EvidenceStrength.E0_UNVERIFIED, True, False, False, False),
    GovernanceRule(PermanenceLevel.L1_RAW_EVIDENCE, EvidenceStrength.E0_UNVERIFIED, True, False, False, False),
    GovernanceRule(PermanenceLevel.L2_EPISODIC_MEMORY, EvidenceStrength.E1_INTERNALLY_CONSISTENT, True),
    GovernanceRule(PermanenceLevel.L3_SEMANTIC_BELIEF, EvidenceStrength.E2_INDEPENDENTLY_VERIFIED, True),
    GovernanceRule(PermanenceLevel.L4_REUSABLE_SKILL, EvidenceStrength.E3_REPLICATED, True),
    GovernanceRule(PermanenceLevel.L5_ROUTING_COMPOSITION, EvidenceStrength.E3_REPLICATED, True),
    GovernanceRule(PermanenceLevel.L6_ISOLATED_NEURAL_MEMORY, EvidenceStrength.E4_FRESH_OOD_VALIDATED, True, require_fresh_ood=True),
    GovernanceRule(PermanenceLevel.L7_SHARED_ADAPTER, EvidenceStrength.E4_FRESH_OOD_VALIDATED, False, require_fresh_ood=True, require_operator_approval=True),
    GovernanceRule(PermanenceLevel.L8_NEW_MODULE, EvidenceStrength.E5_INDEPENDENTLY_REPRODUCED, False, require_fresh_ood=True, require_independent_reproduction=True, require_operator_approval=True),
    GovernanceRule(PermanenceLevel.L9_FOUNDATION_CONSOLIDATION, EvidenceStrength.E5_INDEPENDENTLY_REPRODUCED, False, require_fresh_ood=True, require_independent_reproduction=True, require_operator_approval=True),
    GovernanceRule(PermanenceLevel.L10_ARCHITECTURE_CHANGE, EvidenceStrength.E5_INDEPENDENTLY_REPRODUCED, False, require_fresh_ood=True, require_independent_reproduction=True, require_operator_approval=True),
)


@dataclass(frozen=True)
class GovernancePolicyRC14:
    policy_generation: int = 1
    authority_generation: int = 1
    rules: tuple[GovernanceRule, ...] = DEFAULT_RULES
    schema: str = "egai-rc14-governance-policy-v2"

    def __post_init__(self) -> None:
        if self.policy_generation < 1 or self.authority_generation < 1: raise ValueError("policy and authority generations must be positive")
        levels = [r.permanence for r in self.rules]
        if set(levels) != set(PermanenceLevel) or len(levels) != len(set(levels)): raise ValueError("policy must define exactly one rule for every permanence level")

    @property
    def digest(self) -> str:
        return sha256_json({"schema": self.schema, "policy_generation": self.policy_generation, "authority_generation": self.authority_generation,
            "rules": [{**asdict(r), "permanence": int(r.permanence), "min_evidence": int(r.min_evidence)} for r in sorted(self.rules, key=lambda x: int(x.permanence))]})

    def rule_for(self, level: PermanenceLevel) -> GovernanceRule:
        for rule in self.rules:
            if rule.permanence == level: return rule
        raise KeyError(level)

    def evaluate(self, proposal: LearningProposalRC14) -> tuple[bool, tuple[str, ...]]:
        """Advisory proposal screening only. Evidence claims never authorize persistence."""
        rule = self.rule_for(proposal.coordinates.permanence); reasons: list[str] = []
        if not rule.activatable: reasons.append(f"permanence level {proposal.coordinates.permanence.name} remains proposal-only")
        if rule.min_evidence > EvidenceStrength.E0_UNVERIFIED: reasons.append("verified qualification evidence required; proposal-declared evidence strength is non-authoritative")
        return not reasons, tuple(reasons)

    def evaluate_verified(self, proposal: LearningProposalRC14, qualification: QualificationRecordRC14) -> tuple[bool, tuple[str, ...]]:
        rule = self.rule_for(proposal.coordinates.permanence); reasons: list[str] = []
        if qualification.proposal_digest != proposal.digest: reasons.append("qualification proposal mismatch")
        if qualification.policy_digest != self.digest: reasons.append("qualification policy mismatch")
        if not qualification.passed: reasons.append("qualification did not pass")
        strength = qualification.evidence_strength
        if strength < rule.min_evidence: reasons.append(f"verified evidence strength {strength.name} below required {rule.min_evidence.name}")
        if rule.require_retention and not qualification.retention_passed: reasons.append("retention requirement failed")
        if rule.require_negative_controls and not qualification.negative_controls_passed: reasons.append("negative-control requirement failed")
        if rule.require_security and not qualification.security_passed: reasons.append("security requirement failed")
        if rule.require_fresh_ood and not qualification.fresh_ood_validated: reasons.append("fresh OOD requirement failed")
        if rule.require_independent_reproduction and not qualification.independently_reproduced: reasons.append("independent reproduction required")
        if rule.require_operator_approval and not qualification.operator_approved: reasons.append("operator approval required")
        if proposal.coordinates.permanence == PermanenceLevel.L6_ISOLATED_NEURAL_MEMORY and not qualification.continual_experiment_evidence_digest:
            reasons.append("preregistered continual-experiment evidence required for L6")
        if not rule.activatable: reasons.append(f"permanence level {proposal.coordinates.permanence.name} remains proposal-only")
        return not reasons, tuple(reasons)

    def require_verified(self, proposal: LearningProposalRC14, qualification: QualificationRecordRC14) -> GovernanceRule:
        ok, reasons = self.evaluate_verified(proposal, qualification)
        if not ok: raise PermissionError("; ".join(reasons))
        return self.rule_for(proposal.coordinates.permanence)

    @classmethod
    def with_overrides(cls, *, base: "GovernancePolicyRC14" | None = None, rules: Iterable[GovernanceRule], **kwargs):
        parent = base or cls(); merged = {r.permanence: r for r in parent.rules}
        for rule in rules: merged[rule.permanence] = rule
        return cls(policy_generation=kwargs.get("policy_generation", parent.policy_generation), authority_generation=kwargs.get("authority_generation", parent.authority_generation), rules=tuple(merged[x] for x in PermanenceLevel))
