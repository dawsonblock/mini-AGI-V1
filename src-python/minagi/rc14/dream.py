from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

from minagi.egai.canonical import sha256_json


@dataclass(frozen=True)
class DreamCandidate:
    candidate_id: str
    policy_digest: str
    expected_gain: float
    expected_transfer: float
    expected_interference: float
    expected_compute_cost: float
    expected_security_risk: float
    expected_falsification_reuse: float
    replay_world_digests: tuple[str, ...]

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class DreamWeights:
    gain: float = 1.0
    transfer: float = 0.75
    falsification_reuse: float = 0.5
    interference: float = 1.0
    compute: float = 0.25
    security: float = 1.5


class DreamPolicyResearcher:
    """Proposal-only offline researcher. It has no promotion, signing or activation authority."""

    can_qualify = False
    can_promote = False
    can_activate = False
    can_commit = False

    def __init__(self, weights: DreamWeights | None = None):
        self.weights = weights or DreamWeights()

    def utility(self, candidate: DreamCandidate) -> float:
        vals = (
            candidate.expected_gain, candidate.expected_transfer, candidate.expected_interference,
            candidate.expected_compute_cost, candidate.expected_security_risk, candidate.expected_falsification_reuse,
        )
        if any(not math.isfinite(float(v)) for v in vals):
            raise ValueError("dream metrics must be finite")
        if not candidate.replay_world_digests:
            raise ValueError("dream candidate requires grounded replay worlds")
        w = self.weights
        return (
            w.gain * candidate.expected_gain
            + w.transfer * candidate.expected_transfer
            + w.falsification_reuse * candidate.expected_falsification_reuse
            - w.interference * candidate.expected_interference
            - w.compute * candidate.expected_compute_cost
            - w.security * candidate.expected_security_risk
        )

    def rank(self, candidates: Iterable[DreamCandidate]) -> tuple[DreamCandidate, ...]:
        rows = tuple(candidates)
        return tuple(sorted(rows, key=lambda c: (self.utility(c), c.digest), reverse=True))

    def propose(self, candidates: Iterable[DreamCandidate]) -> DreamCandidate:
        ranked = self.rank(candidates)
        if not ranked:
            raise ValueError("no dream candidates")
        return ranked[0]
