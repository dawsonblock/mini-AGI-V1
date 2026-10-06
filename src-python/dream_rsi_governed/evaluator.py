from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .models import AggregateMetrics, ReplayTrace, ReplayWorld
from .policy import ExplorationPolicy
from .replay import ReplayEngine


@dataclass(frozen=True)
class Objective:
    quality_weight: float = 1.0
    cost_weight: float = 0.01
    probe_weight: float = 0.005
    support_weight: float = 0.20
    uncertainty_weight: float = 0.05

    def score(self, quality: float, cost: float, probes: float, support: float, uncertainty: float) -> float:
        return (
            self.quality_weight * quality
            - self.cost_weight * cost
            - self.probe_weight * probes
            + self.support_weight * support
            - self.uncertainty_weight * uncertainty
        )


class ReplayEvaluator:
    def __init__(self, engine: ReplayEngine, objective: Objective | None = None):
        self.engine = engine
        self.objective = objective or Objective()

    def evaluate(self, worlds: list[ReplayWorld], policy: ExplorationPolicy) -> tuple[AggregateMetrics, list[ReplayTrace]]:
        if not worlds:
            raise ValueError("at least one world is required")
        traces = [self.engine.run(w, policy) for w in worlds]
        q = mean(t.best_quality for t in traces)
        c = mean(t.total_cost for t in traces)
        p = mean(t.probes for t in traces)
        s = mean(t.support_ratio for t in traces)
        u = mean(t.uncertainty for t in traces)
        return AggregateMetrics(q, c, p, s, u, self.objective.score(q, c, p, s, u), len(traces)), traces
