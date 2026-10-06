from __future__ import annotations

import random
from .models import Action, Outcome, ReplayNode, ReplayWorld


def make_demo_worlds(n: int = 30, seed: int = 7) -> list[ReplayWorld]:
    rng = random.Random(seed)
    worlds = []
    for wi in range(n):
        nodes = []
        for i, base in enumerate((0.35, 0.50, 0.65, 0.75)):
            aid = f"w{wi}-a{i}"
            outcomes = []
            for j in range(4):
                q = min(1.0, max(0.0, base + rng.gauss(0, 0.07)))
                success = rng.random() > (0.12 if i < 3 else 0.22)
                outcomes.append(Outcome(aid, q if success else q * 0.4, cost=1.0 + 0.15*i, success=success, failure_class=None if success else "repairable"))
            action = Action(aid, branch=f"b{i}", metadata={"support": min(1.0, len(outcomes)/4)})
            nodes.append(ReplayNode(action, tuple(outcomes)))
        worlds.append(ReplayWorld(f"world-{wi:03d}", f"task-{wi%6}", tuple(nodes), budget=8, max_parallelism=2))
    return worlds
