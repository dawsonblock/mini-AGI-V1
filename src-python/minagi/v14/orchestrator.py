from __future__ import annotations

from dataclasses import dataclass

from .models import LearningProposalV14
from .policy import GovernancePolicyV14
from .state import CandidateStateStore


@dataclass
class GovernedContinualOrchestrator:
    """Canonical authority-aware lifecycle coordinator.

    This class deliberately coordinates artifacts; it does not evaluate its own
    candidates, mint verification receipts, or hold promotion signing keys.
    """

    state: CandidateStateStore
    policy: GovernancePolicyV14

    def register(self, proposal: LearningProposalV14, *, candidate_digest: str, actor: str = "candidate-builder"):
        # Policy blocks unsupported permanence before the candidate can enter a
        # production-directed state machine. Research-only proposals remain
        # serializable outside this orchestrator.
        self.policy.require(proposal)
        return self.state.transition(
            candidate_digest=candidate_digest,
            new_state="REGISTERED",
            actor=actor,
            authority_generation=self.policy.authority_generation,
            policy_generation=self.policy.policy_generation,
        )

    def built(self, candidate_digest: str, *, actor: str = "builder"):
        return self._plain(candidate_digest, "BUILT", actor)

    def evaluated(self, candidate_digest: str, *, actor: str = "independent-evaluator"):
        return self._plain(candidate_digest, "EVALUATED", actor)

    def qualified(self, candidate_digest: str, *, actor: str = "qualifier"):
        return self._plain(candidate_digest, "QUALIFIED", actor)

    def authorized(self, candidate_digest: str, *, authorization_digest: str, actor: str = "promotion-authority"):
        return self._privileged(candidate_digest, "AUTHORIZED", authorization_digest, actor)

    def active(self, candidate_digest: str, *, authorization_digest: str, actor: str = "persistence-gateway"):
        return self._privileged(candidate_digest, "ACTIVE", authorization_digest, actor)

    def rolled_back(self, candidate_digest: str, *, authorization_digest: str, actor: str = "promotion-authority"):
        return self._privileged(candidate_digest, "ROLLED_BACK", authorization_digest, actor)

    def rejected(self, candidate_digest: str, *, actor: str = "qualifier"):
        return self._plain(candidate_digest, "REJECTED", actor)

    def _plain(self, candidate_digest: str, new_state: str, actor: str):
        return self.state.transition(
            candidate_digest=candidate_digest,
            new_state=new_state,
            actor=actor,
            authority_generation=self.policy.authority_generation,
            policy_generation=self.policy.policy_generation,
        )

    def _privileged(self, candidate_digest: str, new_state: str, authorization_digest: str, actor: str):
        return self.state.transition(
            candidate_digest=candidate_digest,
            new_state=new_state,
            actor=actor,
            authority_generation=self.policy.authority_generation,
            policy_generation=self.policy.policy_generation,
            authorization_digest=authorization_digest,
        )
