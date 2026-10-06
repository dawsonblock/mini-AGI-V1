from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Any

from .models import LearningProposal, ProposalOption


@dataclass(frozen=True)
class PlasticityDecisionContext:
    novelty: float
    predicted_interference: float
    available_capacity: float
    expected_transfer_gain: float
    risk: float
    metadata: Mapping[str, Any]


class LeastPermanentPlasticityProposer:
    """Pure proposal policy implementing the permanence-escalation invariant.

    It deliberately has no reference to evidence ledgers, skill stores,
    qualification registries, or production authority. It can only return a
    :class:`LearningProposal`.
    """

    def __init__(self, *, proposer_id: str, minimum_utility: float = 0.0):
        self.proposer_id = str(proposer_id)
        self.minimum_utility = float(minimum_utility)

    def select_option(self, options: Iterable[ProposalOption]) -> ProposalOption:
        feasible = [x for x in options if x.feasible and x.utility >= self.minimum_utility]
        if not feasible:
            raise ValueError("no feasible plasticity option clears the utility floor")
        # Permanence dominates utility: use the least permanent mechanism that
        # solves the problem. Utility is only the tie-breaker within a level.
        min_level = min(int(x.level) for x in feasible)
        same_level = [x for x in feasible if int(x.level) == min_level]
        return max(same_level, key=lambda x: (x.utility, -x.risk, x.action.value))

    def propose(
        self,
        *,
        options: Iterable[ProposalOption],
        target: str,
        evidence_digests: list[str] | tuple[str, ...],
        production_identity_digest: str,
        context: PlasticityDecisionContext,
    ) -> LearningProposal:
        chosen = self.select_option(options)
        rationale = {
            "rule": "least-permanent-capable-mechanism",
            "selected_level": int(chosen.level),
            "selected_action": chosen.action.value,
            "context": {
                "novelty": context.novelty,
                "predicted_interference": context.predicted_interference,
                "available_capacity": context.available_capacity,
                "expected_transfer_gain": context.expected_transfer_gain,
                "risk": context.risk,
                "metadata": dict(context.metadata),
            },
            "selected_reason": chosen.reason,
        }
        return LearningProposal.create(
            action=chosen.action,
            level=chosen.level,
            target=target,
            evidence_digests=evidence_digests,
            rationale=rationale,
            proposer_id=self.proposer_id,
            production_identity_digest=production_identity_digest,
            expected_forward_transfer=chosen.expected_forward_transfer,
            expected_interference=chosen.expected_interference,
            expected_compute_cost=chosen.expected_compute_cost,
            expected_capacity_growth=chosen.expected_capacity_growth,
            risk=chosen.risk,
        )
