from __future__ import annotations

"""NOT THE LIVE CONTROL PLANE.

This module implements a competing qualification engine, which no live path reaches. It is retained
for provenance and for packaging coverage, but it is NOT the authority
that governs promotions.

The single live authority is `minagi.v161.authority`; the live promotion
path is `scripts/promote.py` -> `minagi.v161.authority`. The successor to
this module is `scripts/validation/qualify_campaign1.py`.

See CONTROL_PLANE_MAP.md and CONTROL_PLANE_MANIFEST.json. Do not import
this module into a live path without recording that decision.
"""


from dataclasses import asdict
from datetime import datetime, timezone

from .canonical import sha256_digest
from .evaluator import ReplayEvaluator
from .models import CandidatePolicy, QualificationRecord, ReplayWorld
from .policy import ExplorationPolicy


class QualificationError(RuntimeError):
    pass


class Qualifier:
    def __init__(
        self,
        evaluator: ReplayEvaluator,
        min_holdout_worlds: int = 3,
        max_score_regression: float = 0.0,
        min_support_ratio: float = 0.75,
        max_uncertainty: float = 10.0,
    ):
        self.evaluator = evaluator
        self.min_holdout_worlds = int(min_holdout_worlds)
        self.max_score_regression = float(max_score_regression)
        self.min_support_ratio = float(min_support_ratio)
        self.max_uncertainty = float(max_uncertainty)

    def qualify(
        self,
        candidate_meta: CandidatePolicy,
        candidate: ExplorationPolicy,
        baseline: ExplorationPolicy,
        selection_worlds: list[ReplayWorld],
        holdout_worlds: list[ReplayWorld],
    ) -> QualificationRecord:
        if len(holdout_worlds) < self.min_holdout_worlds:
            raise QualificationError("insufficient sealed holdout worlds")
        selection, _ = self.evaluator.evaluate(selection_worlds or holdout_worlds, candidate)
        holdout, _ = self.evaluator.evaluate(holdout_worlds, candidate)
        baseline_holdout, _ = self.evaluator.evaluate(holdout_worlds, baseline)
        gates = {
            "holdout_count": holdout.worlds >= self.min_holdout_worlds,
            "no_holdout_score_regression": holdout.score + self.max_score_regression >= baseline_holdout.score,
            "support_ratio": holdout.mean_support_ratio >= self.min_support_ratio,
            "uncertainty_bound": holdout.mean_uncertainty <= self.max_uncertainty,
            "finite_score": abs(holdout.score) != float("inf") and holdout.score == holdout.score,
        }
        record_without_digest = {
            "candidate": asdict(candidate_meta),
            "baseline_policy_id": baseline.policy_id,
            "selection": asdict(selection),
            "holdout": asdict(holdout),
            "baseline_holdout": asdict(baseline_holdout),
            "gates": gates,
            "passed": all(gates.values()),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        digest = sha256_digest(record_without_digest)
        return QualificationRecord(
            candidate=candidate_meta,
            baseline_policy_id=baseline.policy_id,
            selection=selection,
            holdout=holdout,
            baseline_holdout=baseline_holdout,
            gates=gates,
            passed=all(gates.values()),
            evidence_digest=digest,
            created_at=record_without_digest["created_at"],
        )
