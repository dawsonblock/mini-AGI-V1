from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Iterable

from minagi.egai.canonical import sha256_json


@dataclass(frozen=True)
class DreamCandidate:
    candidate_id: str
    policy_digest: str
    expected_gain: float
    expected_risk: float
    replay_world_digests: tuple[str, ...]

    @property
    def utility(self) -> float:
        return float(self.expected_gain) - float(self.expected_risk)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


class DreamPolicyResearcher:
    """Offline proposal engine only.

    It intentionally exposes no persistence, activation, promotion, registry or
    signing API. It can rank candidate search policies from replay worlds and
    return a research proposal for an independent authority pipeline.
    """

    can_promote = False
    can_activate = False

    def rank(self, candidates: Iterable[DreamCandidate]) -> tuple[DreamCandidate, ...]:
        rows = tuple(candidates)
        for c in rows:
            if not math.isfinite(c.expected_gain) or not math.isfinite(c.expected_risk):
                raise ValueError("dream metrics must be finite")
            if not c.replay_world_digests:
                raise ValueError("dream candidate requires grounded replay worlds")
        return tuple(sorted(rows, key=lambda c: (c.utility, c.digest), reverse=True))

    def propose(self, candidates: Iterable[DreamCandidate]) -> DreamCandidate:
        ranked = self.rank(candidates)
        if not ranked:
            raise ValueError("no dream candidates")
        return ranked[0]
