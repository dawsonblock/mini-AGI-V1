from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable

from egai.bench.statistics import paired_bootstrap
from egai.bench.forward_transfer import one_sided_sign_test_positive
from egai.common.canonical import digest


@dataclass(frozen=True)
class GroundedOutcome:
    action_id: str
    score: float
    evidence_digest: str
    metadata: dict

    def __post_init__(self):
        if not self.evidence_digest:
            raise ValueError("grounded outcomes require evidence_digest")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be in [0,1]")


@dataclass(frozen=True)
class ReplayRound:
    t: int
    visible_prefix_digest: str
    legal_actions: tuple[str, ...]
    outcomes: tuple[GroundedOutcome, ...]

    def __post_init__(self):
        if self.t < 0:
            raise ValueError("negative replay time")
        if len(set(self.legal_actions)) != len(self.legal_actions):
            raise ValueError("duplicate legal action")
        by_id = {o.action_id for o in self.outcomes}
        if set(self.legal_actions) != by_id:
            raise ValueError("every legal action must have exactly one grounded outcome")


@dataclass(frozen=True)
class GroundedReplayEpisode:
    episode_id: str
    rounds: tuple[ReplayRound, ...]
    budget: float
    episode_digest: str = ""

    def __post_init__(self):
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        times = [r.t for r in self.rounds]
        if times != list(range(len(times))):
            raise ValueError("rounds must be contiguous from t=0")
        expected = digest({"episode_id": self.episode_id, "rounds": self.rounds, "budget": self.budget})
        if self.episode_digest and self.episode_digest != expected:
            raise ValueError("episode digest mismatch")
        object.__setattr__(self, "episode_digest", expected)


@dataclass(frozen=True)
class PolicyView:
    episode_id: str
    t: int
    visible_prefix_digest: str
    legal_actions: tuple[str, ...]
    observed_results: tuple[GroundedOutcome, ...]
    remaining_budget: float


@dataclass(frozen=True)
class PolicyDecision:
    action_ids: tuple[str, ...] = ()
    stop: bool = False


@dataclass(frozen=True)
class ReplayObjective:
    attempt_cost: float = 0.01
    round_cost: float = 0.001
    budget_overrun_penalty: float = 1.0

    def score(self, best_quality: float, attempts: int, rounds: int, overrun: float = 0.0) -> float:
        return (
            float(best_quality)
            - self.attempt_cost * attempts
            - self.round_cost * rounds
            - self.budget_overrun_penalty * max(0.0, overrun)
        )


@dataclass(frozen=True)
class EpisodePolicyResult:
    episode_id: str
    objective_score: float
    best_quality: float
    attempts: int
    rounds: int
    spent: float
    stopped: bool
    chosen_actions: tuple[str, ...]
    evidence_digests: tuple[str, ...]


@dataclass(frozen=True)
class PolicyComparison:
    baseline_scores: tuple[float, ...]
    candidate_scores: tuple[float, ...]
    mean_gain: float
    ci_low: float
    ci_high: float
    sign_test_p: float
    win_rate: float
    loss_rate: float
    baseline_mean_cost: float
    candidate_mean_cost: float
    cost_delta: float
    evidence_closed: bool


class PrefixOnlyPolicyEvaluator:
    """Grounded historical replay evaluator.

    A policy receives only a PolicyView for the current historical round. It never receives
    the episode object, future outcomes, or outcomes of unselected actions. Outcomes are
    disclosed only after an action was selected, and each disclosed outcome must bind to
    authoritative evidence via evidence_digest.
    """

    def __init__(self, objective: ReplayObjective | None = None, attempt_unit_cost: float = 1.0):
        self.objective = objective or ReplayObjective()
        self.attempt_unit_cost = float(attempt_unit_cost)
        if self.attempt_unit_cost <= 0:
            raise ValueError("attempt_unit_cost must be positive")

    def evaluate_episode(
        self,
        episode: GroundedReplayEpisode,
        policy: Callable[[PolicyView], PolicyDecision],
    ) -> EpisodePolicyResult:
        observed: list[GroundedOutcome] = []
        chosen: list[str] = []
        attempts = 0
        rounds = 0
        spent = 0.0
        stopped = False
        best = 0.0
        evidence: list[str] = []
        for r in episode.rounds:
            if spent >= episode.budget:
                break
            rounds += 1
            view = PolicyView(
                episode.episode_id,
                r.t,
                r.visible_prefix_digest,
                r.legal_actions,
                tuple(observed),
                max(0.0, episode.budget - spent),
            )
            decision = policy(view)
            if not isinstance(decision, PolicyDecision):
                raise TypeError("policy must return PolicyDecision")
            if decision.stop:
                stopped = True
                break
            if len(set(decision.action_ids)) != len(decision.action_ids):
                raise ValueError("policy selected duplicate actions")
            illegal = [a for a in decision.action_ids if a not in r.legal_actions]
            if illegal:
                raise PermissionError(f"illegal replay action(s): {illegal}")
            by_id = {o.action_id: o for o in r.outcomes}
            for aid in decision.action_ids:
                if spent + self.attempt_unit_cost > episode.budget:
                    break
                outcome = by_id[aid]
                attempts += 1
                spent += self.attempt_unit_cost
                chosen.append(aid)
                observed.append(outcome)
                evidence.append(outcome.evidence_digest)
                best = max(best, float(outcome.score))
        score = self.objective.score(best, attempts, rounds, max(0.0, spent - episode.budget))
        return EpisodePolicyResult(
            episode.episode_id,
            score,
            best,
            attempts,
            rounds,
            spent,
            stopped,
            tuple(chosen),
            tuple(evidence),
        )

    def compare(
        self,
        episodes: Iterable[GroundedReplayEpisode],
        baseline,
        candidate,
        bootstrap_samples: int = 2000,
        alpha: float = 0.05,
        seed: int = 0,
    ) -> tuple[PolicyComparison, tuple[EpisodePolicyResult, ...], tuple[EpisodePolicyResult, ...]]:
        eps = tuple(episodes)
        if not eps:
            raise ValueError("at least one replay episode is required")
        b = tuple(self.evaluate_episode(e, baseline) for e in eps)
        c = tuple(self.evaluate_episode(e, candidate) for e in eps)
        bs = [x.objective_score for x in b]
        cs = [x.objective_score for x in c]
        est = paired_bootstrap(bs, cs, samples=bootstrap_samples, seed=seed, alpha=alpha)
        p = one_sided_sign_test_positive(bs, cs)
        wins = sum(y > x for x, y in zip(bs, cs)) / len(bs)
        losses = sum(y < x for x, y in zip(bs, cs)) / len(bs)
        bm = sum(x.spent for x in b) / len(b)
        cm = sum(x.spent for x in c) / len(c)
        closed = all(x.evidence_digests for x in b + c if x.attempts > 0)
        comparison = PolicyComparison(
            tuple(bs), tuple(cs), est.mean_gain, est.ci_low, est.ci_high, p,
            wins, losses, bm, cm, cm - bm, closed,
        )
        return comparison, b, c
