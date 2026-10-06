from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean

from .evaluator import ReplayEvaluator
from .models import ReplayWorld
from .policy import ExplorationPolicy
from .replay import ReplayEngine


@dataclass(frozen=True)
class ReplicatedEstimate:
    metric: str
    mean: float
    ci_low: float
    ci_high: float
    runs: int
    seeds: tuple[int, ...]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("empty sample")
    ys = sorted(values)
    x = (len(ys)-1) * p
    lo = int(x); hi = min(len(ys)-1, lo+1); frac = x-lo
    return ys[lo] * (1-frac) + ys[hi] * frac


def replicated_score(worlds: list[ReplayWorld], policy: ExplorationPolicy, seeds: list[int], bootstrap_samples: int = 1000, bootstrap_seed: int = 0) -> ReplicatedEstimate:
    if len(seeds) < 2:
        raise ValueError("at least two independent seeds are required")
    scores = []
    for seed in seeds:
        metrics, _ = ReplayEvaluator(ReplayEngine(seed=seed)).evaluate(worlds, policy)
        scores.append(metrics.score)
    rng = random.Random(bootstrap_seed)
    boots = [mean(rng.choice(scores) for _ in scores) for _ in range(bootstrap_samples)]
    return ReplicatedEstimate("objective_score", mean(scores), _percentile(boots, 0.025), _percentile(boots, 0.975), len(scores), tuple(seeds))
