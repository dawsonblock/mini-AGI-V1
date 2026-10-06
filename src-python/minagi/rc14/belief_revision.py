from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Literal

from minagi.egai.canonical import sha256_json
from minagi.egai.models import Belief, BeliefStatus

Stance = Literal["support", "contradict"]


@dataclass(frozen=True)
class BeliefEvidenceSignal:
    """Promotion-eligible evidence interpreted for one semantic claim.

    `source_id` is intentionally separate from the evidence digest so repeated
    correlated records from one producer cannot overwhelm independent sources.
    """

    evidence_digest: str
    source_id: str
    stance: Stance
    confidence: float
    observed_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.evidence_digest.startswith("sha256:"):
            raise ValueError("evidence_digest must be sha256")
        if not self.source_id.strip():
            raise ValueError("source_id is required")
        if self.stance not in {"support", "contradict"}:
            raise ValueError("stance must be support or contradict")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be finite and in [0,1]")


@dataclass(frozen=True)
class BeliefRevisionProposalRC14:
    belief_id: str
    statement_digest: str
    predecessor_digest: str
    support_digests: tuple[str, ...]
    contradiction_digests: tuple[str, ...]
    distinct_support_sources: int
    distinct_contradiction_sources: int
    posterior_confidence: float
    disposition: str
    candidate: Belief
    schema: str = "egai-rc14-belief-revision-proposal-v1"

    @property
    def digest(self) -> str:
        body = asdict(self)
        body["candidate"] = asdict(self.candidate)
        body["candidate"]["status"] = self.candidate.status.value
        return sha256_json(body)


class SemanticBeliefRevisionEngine:
    """Deterministic proposal-only semantic belief revision.

    The engine deliberately does not write the belief repository. It aggregates
    independent-source evidence, detects contradictions and emits an immutable
    candidate that still has to pass the normal qualification/promotion path.
    """

    can_promote = False
    can_activate = False

    def __init__(self, *, prior_strength: float = 2.0, contested_margin: float = 0.15,
                 min_distinct_sources: int = 2):
        if prior_strength <= 0 or contested_margin < 0 or min_distinct_sources < 1:
            raise ValueError("invalid belief revision configuration")
        self.prior_strength = float(prior_strength)
        self.contested_margin = float(contested_margin)
        self.min_distinct_sources = int(min_distinct_sources)

    @staticmethod
    def _dedupe(signals: Iterable[BeliefEvidenceSignal]) -> tuple[BeliefEvidenceSignal, ...]:
        # One source gets at most one vote per stance; keep its strongest record.
        best: dict[tuple[str, str], BeliefEvidenceSignal] = {}
        for signal in signals:
            key = (signal.source_id, signal.stance)
            current = best.get(key)
            if current is None or (signal.confidence, signal.evidence_digest) > (current.confidence, current.evidence_digest):
                best[key] = signal
        return tuple(best[k] for k in sorted(best))

    def propose(self, *, belief_id: str, statement_digest: str, production_identity_digest: str,
                signals: Iterable[BeliefEvidenceSignal], current: Belief | None = None) -> BeliefRevisionProposalRC14:
        if not belief_id.strip() or not statement_digest.startswith("sha256:"):
            raise ValueError("belief_id and sha256 statement_digest are required")
        if not production_identity_digest.startswith("sha256:"):
            raise ValueError("production_identity_digest must be sha256")
        if current is not None:
            if current.belief_id != belief_id or current.statement_digest != statement_digest:
                raise ValueError("current belief identity/statement mismatch")
            prior = float(current.confidence)
            predecessor = current.digest
        else:
            prior = 0.5
            predecessor = ""

        xs = self._dedupe(signals)
        if not xs:
            raise ValueError("at least one evidence signal is required")
        source_count = len({x.source_id for x in xs})
        if source_count < self.min_distinct_sources:
            raise PermissionError("belief revision requires independent-source support")

        support = [x for x in xs if x.stance == "support"]
        contradict = [x for x in xs if x.stance == "contradict"]
        support_mass = sum(x.confidence for x in support)
        contradiction_mass = sum(x.confidence for x in contradict)
        alpha = self.prior_strength * prior + support_mass
        beta = self.prior_strength * (1.0 - prior) + contradiction_mass
        posterior = alpha / max(1e-12, alpha + beta)

        total_mass = support_mass + contradiction_mass
        if total_mass <= 0:
            disposition = "insufficient"
        elif support and contradict and abs(support_mass - contradiction_mass) / total_mass <= self.contested_margin:
            disposition = "contested"
        elif support_mass > contradiction_mass:
            disposition = "support"
        else:
            disposition = "contradict"

        candidate = Belief(
            belief_id=belief_id,
            statement_digest=statement_digest,
            supporting_evidence=tuple(sorted(x.evidence_digest for x in support)),
            contradicting_evidence=tuple(sorted(x.evidence_digest for x in contradict)),
            confidence=posterior,
            production_identity_digest=production_identity_digest,
            status=BeliefStatus.CANDIDATE,
        )
        return BeliefRevisionProposalRC14(
            belief_id=belief_id,
            statement_digest=statement_digest,
            predecessor_digest=predecessor,
            support_digests=candidate.supporting_evidence,
            contradiction_digests=candidate.contradicting_evidence,
            distinct_support_sources=len({x.source_id for x in support}),
            distinct_contradiction_sources=len({x.source_id for x in contradict}),
            posterior_confidence=posterior,
            disposition=disposition,
            candidate=candidate,
        )
