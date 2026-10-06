from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .authority_artifacts import (
    AuthorityArtifactStore, BuildManifestRC14, CandidateManifestRC14, EvaluationBundleRC14,
    PromotionAuthorizationRC14, PromotionDecisionRC14, QualificationRecordRC14,
)
from .candidate_state import TransactionalCandidateStateStore
from .models import CandidateState, LearningProposalRC14
from .policy import GovernancePolicyRC14


@dataclass
class GovernedContinualOrchestratorRC14:
    """Artifact-bound learning coordinator with no signing or activation authority of its own."""
    state: TransactionalCandidateStateStore
    artifacts: AuthorityArtifactStore
    policy: GovernancePolicyRC14

    can_sign = False
    can_promote = False
    can_activate = False

    def register(self, proposal: LearningProposalRC14, *, candidate_artifact_digest: str, actor: str = "candidate-builder"):
        manifest = CandidateManifestRC14(proposal.digest, candidate_artifact_digest, int(proposal.coordinates.permanence), self.policy.digest)
        candidate_digest = self.artifacts.put(manifest)
        out = self.state.transition(candidate_digest=candidate_digest, new_state=CandidateState.REGISTERED, actor=actor,
            authority_generation=self.policy.authority_generation, policy_generation=self.policy.policy_generation,
            stage_artifact_digest=candidate_digest, reason="artifact-bound candidate registration")
        return candidate_digest, out

    def built(self, manifest: BuildManifestRC14, *, actor: str = "builder"):
        digest = self.artifacts.put(manifest)
        return self.state.transition(candidate_digest=manifest.candidate_digest, new_state=CandidateState.BUILT, actor=actor,
            authority_generation=self.policy.authority_generation, policy_generation=self.policy.policy_generation,
            stage_artifact_digest=digest, reason="artifact-bound build")

    def evaluated(self, bundle: EvaluationBundleRC14, *, actor: str = "evaluator"):
        digest = self.artifacts.put(bundle)
        return self.state.transition(candidate_digest=bundle.candidate_digest, new_state=CandidateState.EVALUATED, actor=actor,
            authority_generation=self.policy.authority_generation, policy_generation=self.policy.policy_generation,
            stage_artifact_digest=digest, reason="artifact-bound evaluation")

    def qualified(self, proposal: LearningProposalRC14, record: QualificationRecordRC14, *, actor: str,
                  authority_state_digest: str, receipt: dict[str, Any]):
        self.policy.require_verified(proposal, record)
        digest = self.artifacts.put(record)
        return self.state.transition(candidate_digest=record.candidate_digest, new_state=CandidateState.QUALIFIED, actor=actor,
            authority_generation=self.policy.authority_generation, policy_generation=self.policy.policy_generation,
            stage_artifact_digest=digest, qualification_digest=digest, authority_state_digest=authority_state_digest,
            receipt=receipt, reason="independently verified qualification")

    def authorized(self, decision: PromotionDecisionRC14, authorization: PromotionAuthorizationRC14, *, actor: str,
                   authority_state_digest: str, receipt: dict[str, Any]):
        if not decision.approved: raise PermissionError("rejected promotion decision cannot be authorized")
        decision_digest = self.artifacts.put(decision)
        if authorization.promotion_decision_digest != decision_digest: raise PermissionError("promotion authorization decision mismatch")
        auth_digest = self.artifacts.put(authorization)
        return self.state.transition(candidate_digest=authorization.candidate_digest, new_state=CandidateState.AUTHORIZED, actor=actor,
            authority_generation=self.policy.authority_generation, policy_generation=self.policy.policy_generation,
            stage_artifact_digest=auth_digest, qualification_digest=authorization.qualification_digest,
            authorization_digest=auth_digest, authority_state_digest=authority_state_digest, receipt=receipt,
            reason="signed promotion authorization")

    def privileged(self, candidate_digest: str, new_state: CandidateState, *, actor: str, authorization_digest: str,
                   authority_state_digest: str, receipt: dict[str, Any], reason: str = ""):
        if new_state in {CandidateState.REGISTERED, CandidateState.BUILT, CandidateState.EVALUATED, CandidateState.QUALIFIED, CandidateState.AUTHORIZED}:
            raise PermissionError("use the artifact-specific lifecycle method for this transition")
        return self.state.transition(candidate_digest=candidate_digest, new_state=new_state, actor=actor,
            authority_generation=self.policy.authority_generation, policy_generation=self.policy.policy_generation,
            authorization_digest=authorization_digest, authority_state_digest=authority_state_digest, receipt=receipt, reason=reason)
