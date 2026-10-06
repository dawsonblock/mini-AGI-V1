from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .models import EvidenceStrength, GovernanceCoordinates, LearningMechanism, LearningProposalV14, PermanenceLevel


DEFAULT_REQUIRED_EVIDENCE: dict[PermanenceLevel, EvidenceStrength] = {
    PermanenceLevel.L0_WORKING_CONTEXT: EvidenceStrength.E0_UNVERIFIED,
    PermanenceLevel.L1_RAW_EVIDENCE: EvidenceStrength.E0_UNVERIFIED,
    PermanenceLevel.L2_EPISODIC_MEMORY: EvidenceStrength.E1_INTERNALLY_CONSISTENT,
    PermanenceLevel.L3_SEMANTIC_BELIEF: EvidenceStrength.E2_INDEPENDENTLY_VERIFIED,
    PermanenceLevel.L4_REUSABLE_SKILL: EvidenceStrength.E3_REPLICATED,
    PermanenceLevel.L5_ROUTING_COMPOSITION: EvidenceStrength.E3_REPLICATED,
    PermanenceLevel.L6_ISOLATED_NEURAL_MEMORY: EvidenceStrength.E4_FRESH_OOD_VALIDATED,
    PermanenceLevel.L7_SHARED_ADAPTER: EvidenceStrength.E4_FRESH_OOD_VALIDATED,
    PermanenceLevel.L8_NEW_MODULE: EvidenceStrength.E5_INDEPENDENTLY_REPRODUCED,
    PermanenceLevel.L9_FOUNDATION_CONSOLIDATION: EvidenceStrength.E5_INDEPENDENTLY_REPRODUCED,
    PermanenceLevel.L10_ARCHITECTURE_CHANGE: EvidenceStrength.E5_INDEPENDENTLY_REPRODUCED,
}


@dataclass(frozen=True)
class GovernancePolicyV14:
    """Static governance policy for the frozen-foundation V14 release.

    Higher-permanence mechanisms exist in the schema so the research plane can
    express them, but policy keeps them proposal-only until explicitly enabled.
    """

    policy_generation: int = 1
    authority_generation: int = 1
    max_activatable_level: PermanenceLevel = PermanenceLevel.L6_ISOLATED_NEURAL_MEMORY
    required_evidence: Mapping[PermanenceLevel, EvidenceStrength] = field(default_factory=lambda: dict(DEFAULT_REQUIRED_EVIDENCE))
    allow_shared_adapter_activation: bool = False
    allow_new_module_activation: bool = False
    allow_foundation_consolidation: bool = False
    allow_architecture_mutation: bool = False

    def required_strength(self, level: PermanenceLevel) -> EvidenceStrength:
        try:
            return EvidenceStrength(self.required_evidence[level])
        except KeyError as exc:
            raise ValueError(f"no evidence policy for {level!r}") from exc

    def evaluate(self, proposal: LearningProposalV14) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        p = proposal.coordinates.permanence
        e = proposal.coordinates.evidence_strength
        required = self.required_strength(p)
        if e < required:
            reasons.append(f"evidence strength {e.name} below required {required.name}")
        if p > self.max_activatable_level:
            reasons.append(f"permanence level {p.name} remains proposal-only")
        if p == PermanenceLevel.L7_SHARED_ADAPTER and not self.allow_shared_adapter_activation:
            reasons.append("shared adapter activation disabled")
        if p == PermanenceLevel.L8_NEW_MODULE and not self.allow_new_module_activation:
            reasons.append("new module activation disabled")
        if p == PermanenceLevel.L9_FOUNDATION_CONSOLIDATION and not self.allow_foundation_consolidation:
            reasons.append("foundation consolidation disabled")
        if p == PermanenceLevel.L10_ARCHITECTURE_CHANGE and not self.allow_architecture_mutation:
            reasons.append("architecture mutation disabled")
        return not reasons, tuple(reasons)

    def require(self, proposal: LearningProposalV14) -> None:
        ok, reasons = self.evaluate(proposal)
        if not ok:
            raise PermissionError("; ".join(reasons))
