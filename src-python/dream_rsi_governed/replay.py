from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict

from .models import Outcome, ReplayStep, ReplayTrace, ReplayWorld
from .policy import ExplorationPolicy, PolicyContext


class ReplayError(RuntimeError):
    pass


class ReplayEngine:
    """Prefix-only replay over recorded outcomes.

    Unlike one-child deterministic replay, each action may contain multiple historical
    outcomes. A stable seed samples one realization per probe. The engine exposes no
    unrevealed outcome to the policy.
    """

    def __init__(self, seed: int = 0, support_floor: int = 2):
        self.seed = int(seed)
        self.support_floor = int(support_floor)

    def _rng(self, world_id: str, policy_id: str) -> random.Random:
        h = hashlib.sha256(f"{self.seed}|{world_id}|{policy_id}".encode()).digest()
        return random.Random(int.from_bytes(h[:8], "big"))

    def run(self, world: ReplayWorld, policy: ExplorationPolicy) -> ReplayTrace:
        nodes = world.node_map()
        history: list[Outcome] = []
        steps: list[ReplayStep] = []
        sampled_counts: Counter[str] = Counter()
        sums: defaultdict[str, float] = defaultdict(float)
        failure_counts: Counter[str] = Counter()
        cumulative_cost = 0.0
        best_quality = float("-inf")
        unsupported = 0
        rng = self._rng(world.world_id, policy.policy_id)
        stopped = False

        for step_idx in range(world.budget):
            legal = tuple(n.action for n in world.nodes if n.outcomes)
            if not legal:
                break
            empirical = {k: sums[k] / sampled_counts[k] for k in sampled_counts if sampled_counts[k]}
            ctx = PolicyContext(
                step=step_idx,
                budget_remaining=world.budget - len(history),
                max_parallelism=world.max_parallelism,
                legal_actions=legal,
                history=tuple(history),
                sampled_counts=dict(sampled_counts),
                empirical_means=empirical,
                failure_counts=dict(failure_counts),
            )
            decision = policy.decide(ctx)
            if decision.stop:
                stopped = True
                break
            chosen = tuple(dict.fromkeys(decision.action_ids))
            if len(chosen) > min(world.max_parallelism, ctx.budget_remaining):
                raise ReplayError("policy requested more actions than allowed")
            observed: list[Outcome] = []
            for aid in chosen:
                if aid not in nodes:
                    unsupported += 1
                    continue
                node = nodes[aid]
                if not node.outcomes:
                    unsupported += 1
                    continue
                if len(node.outcomes) < self.support_floor:
                    unsupported += 1
                outcome = rng.choice(node.outcomes)
                observed.append(outcome)
                history.append(outcome)
                sampled_counts[aid] += 1
                sums[aid] += outcome.quality
                if not outcome.success:
                    failure_counts[aid] += 1
                cumulative_cost += outcome.cost
                best_quality = max(best_quality, outcome.quality)
                if len(history) >= world.budget:
                    break
            steps.append(
                ReplayStep(
                    index=step_idx,
                    chosen=chosen,
                    observed=tuple(observed),
                    best_quality=best_quality if history else float("-inf"),
                    cumulative_cost=cumulative_cost,
                    unsupported_requests=unsupported,
                )
            )
            if len(history) >= world.budget:
                break

        supported_probes = max(0, len(history) - unsupported)
        support_ratio = supported_probes / max(1, len(history))
        variances = []
        for node in world.nodes:
            qs = [o.quality for o in node.outcomes]
            if len(qs) >= 2:
                m = sum(qs) / len(qs)
                variances.append(sum((q-m)**2 for q in qs) / (len(qs)-1))
            elif qs:
                variances.append(1.0)
        uncertainty = math.sqrt(sum(variances) / max(1, len(variances)))
        if best_quality == float("-inf"):
            best_quality = 0.0
        return ReplayTrace(
            world_id=world.world_id,
            policy_id=policy.policy_id,
            steps=tuple(steps),
            best_quality=best_quality,
            total_cost=cumulative_cost,
            probes=len(history),
            support_ratio=support_ratio,
            uncertainty=uncertainty,
            stopped=stopped,
        )
