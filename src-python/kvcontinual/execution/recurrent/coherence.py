from __future__ import annotations

from dataclasses import dataclass
from kvcontinual.execution.types import RuntimeRiskSignals


@dataclass(frozen=True)
class RuntimeRiskThresholds:
    max_seam_hidden_rel_l2: float = 0.08
    max_seam_attention_rel_l2: float = 0.08
    max_routing_disagreement: float = 0.10
    max_conv_correction_rel_l2: float = 0.15
    max_reorder_distance: float = 1.0
    max_historical_failure_rate: float = 0.05
    max_joins: int = 16


class RuntimeRiskGate:
    """Initial boring router: only cheap seam-derived features are used online."""
    def __init__(self, thresholds: RuntimeRiskThresholds | None = None):
        self.thresholds = thresholds or RuntimeRiskThresholds()

    def acceptable(self, x: RuntimeRiskSignals) -> bool:
        t = self.thresholds
        return (
            x.seam_hidden_rel_l2 <= t.max_seam_hidden_rel_l2
            and x.seam_attention_rel_l2 <= t.max_seam_attention_rel_l2
            and x.seam_routing_disagreement <= t.max_routing_disagreement
            and x.conv_correction_rel_l2 <= t.max_conv_correction_rel_l2
            and x.reorder_distance <= t.max_reorder_distance
            and x.historical_failure_rate <= t.max_historical_failure_rate
            and x.join_count <= t.max_joins
        )


# Compatibility shim: older code imported CoherenceGate. It now evaluates only
# attributes that can be known online when possible.
CoherenceGate = RuntimeRiskGate
