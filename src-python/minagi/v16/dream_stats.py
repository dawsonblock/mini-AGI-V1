from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import mean
import random
from typing import Callable, Sequence

from dream_rsi_governed.canonical import sha256_digest
from dream_rsi_governed.evaluator import ReplayEvaluator
from dream_rsi_governed.experiment import replicated_score
from dream_rsi_governed.models import CandidatePolicy, ReplayWorld
from dream_rsi_governed.policy import ExplorationPolicy


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        raise ValueError("empty sample")
    ys = sorted(float(x) for x in values)
    pos = (len(ys) - 1) * p
    lo = int(pos)
    hi = min(len(ys) - 1, lo + 1)
    frac = pos - lo
    return ys[lo] * (1.0 - frac) + ys[hi] * frac


def _paired_bootstrap_delta(candidate: Sequence[float], baseline: Sequence[float], *, samples: int, seed: int):
    if len(candidate) != len(baseline) or len(candidate) < 2:
        raise ValueError("paired candidate/baseline samples with n>=2 required")
    deltas = [float(c) - float(b) for c, b in zip(candidate, baseline)]
    rng = random.Random(seed)
    boots = [mean(rng.choice(deltas) for _ in deltas) for _ in range(samples)]
    return mean(deltas), _percentile(boots, 0.025), _percentile(boots, 0.975)


@dataclass(frozen=True)
class ReplicatedDreamQualificationV160:
    candidate_policy_id: str
    baseline_policy_id: str
    candidate_meta_digest: str
    holdout_world_count: int
    seeds: tuple[int, ...]
    candidate_mean: float
    baseline_mean: float
    mean_delta: float
    delta_ci_low: float
    delta_ci_high: float
    candidate_support_ratio: float
    candidate_uncertainty: float
    passed: bool
    gates: dict[str, bool]
    created_at: str
    schema: str = "mini-agi-v16-dream-statistical-qualification-v1"

    @property
    def digest(self) -> str:
        return sha256_digest(asdict(self))


class StatisticalDreamQualifierV160:
    """Replicated, paired qualification for replay policy changes.

    Unlike Dream-RSI v1's aggregate one-pass gate, this evaluates candidate and
    baseline on the same holdout worlds across independent replay seeds and
    requires the lower bootstrap bound on paired score gain to meet the allowed
    regression threshold.
    """

    def __init__(
        self,
        evaluator_factory: Callable[[int], ReplayEvaluator],
        *,
        min_holdout_worlds: int = 3,
        min_seeds: int = 4,
        max_score_regression: float = 0.0,
        min_support_ratio: float = 0.75,
        max_uncertainty: float = 10.0,
        bootstrap_samples: int = 2000,
        bootstrap_seed: int = 0,
    ):
        self.evaluator_factory = evaluator_factory
        self.min_holdout_worlds = int(min_holdout_worlds)
        self.min_seeds = int(min_seeds)
        self.max_score_regression = float(max_score_regression)
        self.min_support_ratio = float(min_support_ratio)
        self.max_uncertainty = float(max_uncertainty)
        self.bootstrap_samples = int(bootstrap_samples)
        self.bootstrap_seed = int(bootstrap_seed)

    def qualify(
        self,
        *,
        candidate_meta: CandidatePolicy,
        candidate: ExplorationPolicy,
        baseline: ExplorationPolicy,
        holdout_worlds: list[ReplayWorld],
        seeds: Sequence[int],
    ) -> ReplicatedDreamQualificationV160:
        seeds = tuple(int(s) for s in seeds)
        if len(holdout_worlds) < self.min_holdout_worlds:
            raise ValueError("insufficient sealed holdout worlds")
        if len(seeds) < self.min_seeds or len(set(seeds)) != len(seeds):
            raise ValueError("independent unique replay seeds required")

        cand_scores, base_scores = [], []
        supports, uncertainties = [], []
        for seed in seeds:
            evaluator = self.evaluator_factory(seed)
            cm, _ = evaluator.evaluate(holdout_worlds, candidate)
            bm, _ = evaluator.evaluate(holdout_worlds, baseline)
            cand_scores.append(cm.score)
            base_scores.append(bm.score)
            supports.append(cm.mean_support_ratio)
            uncertainties.append(cm.mean_uncertainty)

        delta, ci_low, ci_high = _paired_bootstrap_delta(
            cand_scores, base_scores, samples=self.bootstrap_samples, seed=self.bootstrap_seed
        )
        support = mean(supports)
        uncertainty = mean(uncertainties)
        gates = {
            "holdout_count": len(holdout_worlds) >= self.min_holdout_worlds,
            "replicated_seeds": len(seeds) >= self.min_seeds,
            "paired_ci_no_regression": ci_low + self.max_score_regression >= 0.0,
            "mean_no_regression": delta + self.max_score_regression >= 0.0,
            "support_ratio": support >= self.min_support_ratio,
            "uncertainty_bound": uncertainty <= self.max_uncertainty,
        }
        created = datetime.now(timezone.utc).isoformat()
        return ReplicatedDreamQualificationV160(
            candidate_policy_id=candidate.policy_id,
            baseline_policy_id=baseline.policy_id,
            candidate_meta_digest=sha256_digest(asdict(candidate_meta)),
            holdout_world_count=len(holdout_worlds),
            seeds=seeds,
            candidate_mean=mean(cand_scores),
            baseline_mean=mean(base_scores),
            mean_delta=delta,
            delta_ci_low=ci_low,
            delta_ci_high=ci_high,
            candidate_support_ratio=support,
            candidate_uncertainty=uncertainty,
            passed=all(gates.values()),
            gates=gates,
            created_at=created,
        )


@dataclass(frozen=True)
class ReplicatedCanaryAttestationV160:
    candidate_policy_id: str
    baseline_policy_id: str
    runs: int
    candidate_mean_score: float
    baseline_mean_score: float
    score_delta: float
    score_delta_ci_low: float
    score_delta_ci_high: float
    candidate_mean_cost: float
    baseline_mean_cost: float
    passed: bool
    environment_digest: str
    run_digest: str
    created_at: str
    schema: str = "mini-agi-v16-replicated-live-canary-v1"


def compare_replicated_live_canary_v160(
    *,
    candidate_policy_id: str,
    baseline_policy_id: str,
    run_candidate: Callable[[int], tuple[float, float]],
    run_baseline: Callable[[int], tuple[float, float]],
    environment_digest: str,
    seeds: Sequence[int],
    max_quality_regression: float = 0.0,
    max_cost_multiplier: float = 1.25,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> ReplicatedCanaryAttestationV160:
    seeds = tuple(int(s) for s in seeds)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three unique canary seeds are required")
    cscore, bscore, ccost, bcost = [], [], [], []
    for seed in seeds:
        cq, cc = run_candidate(seed)
        bq, bc = run_baseline(seed)
        cscore.append(float(cq)); ccost.append(float(cc))
        bscore.append(float(bq)); bcost.append(float(bc))
    delta, ci_low, ci_high = _paired_bootstrap_delta(cscore, bscore, samples=bootstrap_samples, seed=bootstrap_seed)
    quality_ok = ci_low + max_quality_regression >= 0.0
    cost_ok = mean(ccost) <= max_cost_multiplier * max(mean(bcost), 1e-12)
    created = datetime.now(timezone.utc).isoformat()
    body = {
        "candidate_policy_id": candidate_policy_id,
        "baseline_policy_id": baseline_policy_id,
        "runs": len(seeds),
        "candidate_mean_score": mean(cscore),
        "baseline_mean_score": mean(bscore),
        "score_delta": delta,
        "score_delta_ci_low": ci_low,
        "score_delta_ci_high": ci_high,
        "candidate_mean_cost": mean(ccost),
        "baseline_mean_cost": mean(bcost),
        "environment_digest": environment_digest,
        "created_at": created,
    }
    return ReplicatedCanaryAttestationV160(
        **body,
        passed=quality_ok and cost_ok,
        run_digest=sha256_digest(body),
    )
