from __future__ import annotations

from dataclasses import dataclass
from kvcontinual.continual.types import CoherenceMetrics, ReconstructionAction, ReconstructionMode


@dataclass(frozen=True)
class CoherenceThresholds:
    max_state_rel_l2: float = 0.03
    max_hidden_rel_l2: float = 0.03
    max_logit_kl: float = 0.02
    max_routing_disagreement: float = 0.02


class CoherenceGate:
    def __init__(self, thresholds: CoherenceThresholds | None = None):
        self.thresholds = thresholds or CoherenceThresholds()

    def acceptable(self, m: CoherenceMetrics) -> bool:
        t = self.thresholds
        return (
            m.state_rel_l2 <= t.max_state_rel_l2
            and m.hidden_rel_l2 <= t.max_hidden_rel_l2
            and m.logit_kl <= t.max_logit_kl
            and m.routing_disagreement <= t.max_routing_disagreement
        )

    def choose(self, mode: ReconstructionMode, metrics: CoherenceMetrics | None, seam_tokens: int) -> ReconstructionAction:
        if mode == ReconstructionMode.EXACT:
            return ReconstructionAction.EXACT_REPLAY
        if metrics is None:
            return ReconstructionAction.SEAM if mode == ReconstructionMode.BALANCED else ReconstructionAction.COMPOSE
        if self.acceptable(metrics):
            return ReconstructionAction.COMPOSE if mode == ReconstructionMode.FAST else ReconstructionAction.SEAM
        if seam_tokens < 128:
            return ReconstructionAction.SEAM
        if metrics.logit_kl <= self.thresholds.max_logit_kl * 4:
            return ReconstructionAction.SUFFIX
        return ReconstructionAction.EXACT_REPLAY
