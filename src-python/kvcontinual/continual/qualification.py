from __future__ import annotations

from dataclasses import dataclass
from kvcontinual.continual.types import PromotionDecision, QualificationRecord


@dataclass
class QualificationPolicy:
    min_target_gain: float = 0.03
    max_general_regression: float = 0.01

    def evaluate(self, candidate_id: str, target_gain: float, general_regression: float, critical_pass: bool, cache_quality_pass: bool, metrics: dict[str, float] | None = None) -> QualificationRecord:
        metrics = metrics or {}
        if not critical_pass or not cache_quality_pass:
            decision = PromotionDecision.REJECT
            reason = "critical or cache-quality gate failed"
        elif general_regression > self.max_general_regression:
            decision = PromotionDecision.REJECT
            reason = "general regression exceeds budget"
        elif target_gain >= self.min_target_gain:
            decision = PromotionDecision.PROMOTE
            reason = "target gain met with regression constraints satisfied"
        else:
            decision = PromotionDecision.QUARANTINE
            reason = "safe but target gain insufficient"
        return QualificationRecord(
            candidate_id=candidate_id,
            decision=decision,
            target_gain=target_gain,
            general_regression=general_regression,
            critical_pass=critical_pass,
            cache_quality_pass=cache_quality_pass,
            metrics=metrics,
            reason=reason,
        )
