"""Replay worlds for cheap optimization of exploration policies.

A discovery run is stored as an immutable tree.  Alternative policies can then
be evaluated by revealing already-recorded children rather than rerunning the
expensive underlying agent/evaluator.  This is the control-plane idea used by
v4 for policy experiments; replay never fabricates unobserved outcomes.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiscoveryNode:
    id: str
    parent: str | None
    score: float
    cost: float = 1.0
    observation: dict = field(default_factory=dict)


class DiscoveryTree:
    def __init__(self, nodes):
        self.nodes = {n.id: n for n in nodes}
        roots = [n.id for n in nodes if n.parent is None]
        if len(roots) != 1:
            raise ValueError("a replay world must have exactly one root")
        self.root = roots[0]
        self.children = {k: [] for k in self.nodes}
        for n in nodes:
            if n.parent is not None:
                if n.parent not in self.nodes: raise ValueError(f"missing parent {n.parent}")
                self.children[n.parent].append(n.id)

    def child(self, node_id, revealed):
        for c in self.children.get(node_id, []):
            if c not in revealed: return c
        return None


@dataclass
class ReplayResult:
    value: float
    best_score: float
    attempts: int
    rounds: int
    revealed: tuple[str, ...]


class ReplaySimulator:
    def __init__(self, beta_cost=0.0, beta_parallel=0.0, max_rounds=100):
        self.beta_cost = float(beta_cost)
        self.beta_parallel = float(beta_parallel)
        self.max_rounds = int(max_rounds)

    def evaluate(self, tree: DiscoveryTree, policy, workers=1):
        revealed = {tree.root}; frontier = {tree.root}
        attempts = 0; rounds = 0; best = float(tree.nodes[tree.root].score)
        while rounds < self.max_rounds:
            eligible = sorted(frontier | {tree.root})
            chosen = list(policy(eligible, tree, revealed, workers) or [])[:workers]
            if not chosen: break
            if any(x not in eligible for x in chosen):
                raise ValueError("policy selected an ineligible node")
            added = []
            for v in chosen:
                c = tree.child(v, revealed)
                if c is not None:
                    revealed.add(c); added.append(c); attempts += 1
                    best = max(best, float(tree.nodes[c].score))
                    if v != tree.root: frontier.discard(v)
            frontier.update(added)
            rounds += 1
            if not added: break
        cost = sum(float(tree.nodes[x].cost) for x in revealed if x != tree.root)
        parallel = attempts / max(1, rounds)
        value = best - self.beta_cost * cost + self.beta_parallel * parallel
        return ReplayResult(value, best, attempts, rounds, tuple(sorted(revealed)))


def best_policy(policies, worlds, simulator: ReplaySimulator, workers=1):
    scored = []
    for name, p in policies.items():
        rs = [simulator.evaluate(w, p, workers=workers) for w in worlds]
        mean = sum(r.value for r in rs) / max(1, len(rs))
        scored.append((mean, name, rs))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0] if scored else None


@dataclass
class PolicyQualification:
    name: str
    train_mean: float
    holdout_mean: float
    holdout_delta: float
    delta_lcb95: float
    promoted: bool
    reason: str


def qualify_policy(policies, train_worlds, holdout_worlds,
                   simulator: ReplaySimulator, *, baseline: str,
                   workers=1, min_holdout_gain=0.0, require_lcb=True):
    """Select on replay-training worlds, qualify on untouched replay worlds.

    DREAM-style replay makes trying many exploration policies cheap, which also
    makes overfitting the historical worlds cheap.  v4.1 therefore separates
    policy *selection* from *qualification*.  The best policy is chosen only by
    ``train_worlds``.  Its promotion is then decided by paired differences
    against a named baseline on ``holdout_worlds``.

    The reported 95% lower confidence bound uses a normal approximation over
    paired world deltas. It is a conservative gate, not a claim of a full
    statistical analysis for tiny world counts.
    """
    import math
    if baseline not in policies:
        raise ValueError("baseline policy is missing")
    if not train_worlds or not holdout_worlds:
        raise ValueError("both replay train and holdout worlds are required")

    train_scores = {}
    for name, p in policies.items():
        vals = [simulator.evaluate(w, p, workers=workers).value
                for w in train_worlds]
        train_scores[name] = sum(vals) / len(vals)
    name = max(train_scores, key=train_scores.get)

    chosen = policies[name]
    base = policies[baseline]
    cv = [simulator.evaluate(w, chosen, workers=workers).value
          for w in holdout_worlds]
    bv = [simulator.evaluate(w, base, workers=workers).value
          for w in holdout_worlds]
    deltas = [a - b for a, b in zip(cv, bv)]
    mean_delta = sum(deltas) / len(deltas)
    if len(deltas) > 1:
        mu = mean_delta
        var = sum((x - mu) ** 2 for x in deltas) / (len(deltas) - 1)
        se = math.sqrt(var / len(deltas))
        lcb = mu - 1.96 * se
    else:
        lcb = float("-inf") if require_lcb else mean_delta
    hold_mean = sum(cv) / len(cv)
    gain_ok = mean_delta >= float(min_holdout_gain)
    lcb_ok = (lcb >= float(min_holdout_gain)) if require_lcb else True
    promoted = bool(name != baseline and gain_ok and lcb_ok)
    if name == baseline:
        reason = "training selection did not beat the baseline policy"
    elif not gain_ok:
        reason = (f"holdout gain {mean_delta:.6g} below required "
                  f"{float(min_holdout_gain):.6g}")
    elif not lcb_ok:
        reason = (f"holdout gain is not stable: 95% lower bound {lcb:.6g} "
                  f"below {float(min_holdout_gain):.6g}")
    else:
        reason = (f"selected on replay-train and passed independent holdout: "
                  f"delta={mean_delta:.6g}, lcb95={lcb:.6g}")
    return PolicyQualification(
        name=name, train_mean=float(train_scores[name]),
        holdout_mean=float(hold_mean), holdout_delta=float(mean_delta),
        delta_lcb95=float(lcb), promoted=promoted, reason=reason)
