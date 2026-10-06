from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
from typing import Any, Iterable
import uuid

from .canonical import sha256_json
from .evidence import EvidenceLedger
from .improvement import ImprovementLedger, ImprovementRecord
from .models import AuthorityPlane, LearningProposal, PromotionVerdict
from .qualification import IndependentQualificationGate, PromotionDecision, QualificationBundle
from .skills import SkillRepository
from .beliefs import BeliefRepository


@dataclass(frozen=True)
class PromotionAuthorizationBody:
    authorization_id: str
    proposal_digest: str
    candidate_digest: str
    qualification_digest: str
    decision_digest: str
    verdict: str
    issued_at: float
    research_chain_digest: str = ""
    schema: str = "mini-agi-egai-promotion-authorization-v2"

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


class CognitivePlane:
    """Read/use plane. It can solve tasks but owns no promotion capability."""

    plane = AuthorityPlane.COGNITIVE

    def __init__(self, *, evidence: EvidenceLedger, skills: SkillRepository, beliefs: BeliefRepository | None = None):
        self.evidence = evidence
        self.skills = skills
        self.beliefs = beliefs


class LearningPlane:
    """Discovery plane. It can create candidates/proposals but cannot promote."""

    plane = AuthorityPlane.LEARNING
    can_promote = False

    def __init__(self, *, skills: SkillRepository, beliefs: BeliefRepository | None = None):
        self.skills = skills
        self.beliefs = beliefs
        self._proposals: dict[str, LearningProposal] = {}

    def stage_proposal(self, proposal: LearningProposal) -> str:
        self._proposals[proposal.digest] = proposal
        return proposal.digest

    def get_proposal(self, digest: str) -> LearningProposal:
        return self._proposals[str(digest)]


class PromotionAuthority:
    """The only EGAI plane allowed to issue persistence authorization."""

    plane = AuthorityPlane.PROMOTION

    def __init__(self, *, gate: IndependentQualificationGate, improvement_ledger: ImprovementLedger, signer: Any):
        self.gate = gate
        self.improvement_ledger = improvement_ledger
        self.signer = signer

    def decide(
        self,
        *,
        proposal: LearningProposal,
        bundle: QualificationBundle,
        research_chain=None,
        origin_evidence: Iterable[str],
        affected_components: Iterable[str],
        production_identity_before: str,
        rollback_target: str = "",
    ) -> tuple[PromotionDecision, dict[str, Any] | None, ImprovementRecord]:
        decision = self.gate.evaluate(proposal, bundle, research_chain)
        if int(proposal.level)>=6:
            decision=replace(decision, verdict=PromotionVerdict.REJECT,
                reasons=decision.reasons + ('neural plasticity remains proposal-only in the frozen-foundation experiment',))
        if decision.verdict is PromotionVerdict.APPROVE:
            self.gate.chain_verifier.consume(proposal, bundle, research_chain)
        metrics = asdict(bundle.metrics)
        improvement = ImprovementRecord(
            improvement_id="I-" + uuid.uuid4().hex,
            proposal_digest=proposal.digest,
            candidate_digest=bundle.candidate_digest,
            qualification_digest=bundle.digest,
            promotion_decision_digest=decision.digest,
            verdict=decision.verdict,
            level=proposal.level,
            origin_evidence=tuple(str(x) for x in origin_evidence),
            affected_components=tuple(str(x) for x in affected_components),
            metrics=metrics,
            production_identity_before=str(production_identity_before),
            rollback_target=str(rollback_target),
        )
        self.improvement_ledger.append_improvement(improvement)
        if decision.verdict is not PromotionVerdict.APPROVE:
            return decision, None, improvement
        body = PromotionAuthorizationBody(
            authorization_id="A-" + uuid.uuid4().hex,
            proposal_digest=proposal.digest,
            candidate_digest=bundle.candidate_digest,
            qualification_digest=bundle.digest,
            decision_digest=decision.digest,
            verdict=decision.verdict.value,
            issued_at=time.time(),
            research_chain_digest=research_chain.digest,
        )
        return decision, self.signer.issue(asdict(body)), improvement
