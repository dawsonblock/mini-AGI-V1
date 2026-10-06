from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from kvcontinual.execution.types import OracleMetrics, PromotionDecision, QualificationRecord


@dataclass
class QualificationPolicy:
    min_target_gain: float = 0.03
    max_general_regression: float = 0.01
    max_cache_warmup_cost: float = 1.0

    def evaluate(self, candidate_id: str, target_gain: float, general_regression: float, critical_pass: bool, cache_quality_pass: bool, metrics: dict[str, float] | None = None, cache_warmup_cost: float = 0.0) -> QualificationRecord:
        metrics = metrics or {}
        if not critical_pass or not cache_quality_pass:
            decision, reason = PromotionDecision.REJECT, "critical or cache-quality gate failed"
        elif general_regression > self.max_general_regression:
            decision, reason = PromotionDecision.REJECT, "general regression exceeds budget"
        elif cache_warmup_cost > self.max_cache_warmup_cost:
            decision, reason = PromotionDecision.QUARANTINE, "cache warm-up cost exceeds promotion budget"
        elif target_gain >= self.min_target_gain:
            decision, reason = PromotionDecision.PROMOTE, "gain met with regression and cache-cost constraints satisfied"
        else:
            decision, reason = PromotionDecision.QUARANTINE, "safe but target gain insufficient"
        return QualificationRecord(candidate_id, decision, target_gain, general_regression, critical_pass, cache_quality_pass, cache_warmup_cost, metrics, reason)


def relative_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a-b) / (np.linalg.norm(b) + eps))


def angle_deg(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    x, y = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx <= eps or ny <= eps:
        return 0.0 if nx <= eps and ny <= eps else 90.0
    c = float(np.dot(x, y) / (nx * ny))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def categorical_kl(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    p, q = np.asarray(p_logits, dtype=np.float64), np.asarray(q_logits, dtype=np.float64)
    p = p - np.max(p); q = q - np.max(q)
    p = np.exp(p); q = np.exp(q); p /= p.sum(); q /= q.sum()
    return float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30))))


def measure_oracle(approx_state: np.ndarray, exact_state: np.ndarray, approx_logits: np.ndarray | None = None, exact_logits: np.ndarray | None = None) -> OracleMetrics:
    m = OracleMetrics(state_rel_l2=relative_l2(approx_state, exact_state), state_angle_deg=angle_deg(approx_state, exact_state))
    if approx_logits is not None and exact_logits is not None:
        m.logit_kl = categorical_kl(exact_logits, approx_logits)
        m.top1_agreement = float(np.argmax(approx_logits) == np.argmax(exact_logits))
    return m
