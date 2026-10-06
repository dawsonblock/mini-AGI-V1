from __future__ import annotations

from dataclasses import dataclass
from math import sqrt, log
from typing import Protocol, Sequence

from .models import Action, Outcome, ReplayDecision


@dataclass(frozen=True)
class PolicyContext:
    step: int
    budget_remaining: int
    max_parallelism: int
    legal_actions: tuple[Action, ...]
    history: tuple[Outcome, ...]
    sampled_counts: dict[str, int]
    empirical_means: dict[str, float]
    failure_counts: dict[str, int]


class ExplorationPolicy(Protocol):
    policy_id: str
    def decide(self, context: PolicyContext) -> ReplayDecision: ...


class FixedBreadthPolicy:
    policy_id = "fixed-breadth-v1"

    def decide(self, context: PolicyContext) -> ReplayDecision:
        if not context.legal_actions or context.budget_remaining <= 0:
            return ReplayDecision((), stop=True, rationale="no budget or legal actions")
        n = min(context.max_parallelism, context.budget_remaining, len(context.legal_actions))
        actions = sorted(context.legal_actions, key=lambda a: (a.branch, a.id))[:n]
        return ReplayDecision(tuple(a.id for a in actions), rationale="deterministic breadth baseline")


class RiskAwareUCBPolicy:
    """A deterministic adaptive scheduler that does not need an LLM at runtime.

    It favors underexplored/high-value branches, discounts repeated failures, and
    penalizes actions whose historical support is known to be sparse via metadata.
    """

    def __init__(self, beta: float = 1.0, failure_penalty: float = 0.15, support_penalty: float = 0.25):
        self.beta = float(beta)
        self.failure_penalty = float(failure_penalty)
        self.support_penalty = float(support_penalty)
        self.policy_id = f"risk-aware-ucb-beta-{self.beta:g}"

    def _score(self, a: Action, c: PolicyContext) -> float:
        count = c.sampled_counts.get(a.id, 0)
        mean = c.empirical_means.get(a.id, 0.0)
        failures = c.failure_counts.get(a.id, 0)
        total = max(1, sum(c.sampled_counts.values()))
        exploration = self.beta * sqrt(log(total + 1.0) / (count + 1.0))
        support = float(a.metadata.get("support", 1.0))
        novelty = 1.0 / (1.0 + count)
        return mean + exploration + 0.05 * novelty - self.failure_penalty * failures - self.support_penalty * (1.0 - support)

    def decide(self, context: PolicyContext) -> ReplayDecision:
        if not context.legal_actions or context.budget_remaining <= 0:
            return ReplayDecision((), stop=True, rationale="no budget or legal actions")
        ranked = sorted(context.legal_actions, key=lambda a: (-self._score(a, context), a.id))
        n = min(context.max_parallelism, context.budget_remaining, len(ranked))
        return ReplayDecision(tuple(a.id for a in ranked[:n]), rationale="risk-aware UCB ranking")
