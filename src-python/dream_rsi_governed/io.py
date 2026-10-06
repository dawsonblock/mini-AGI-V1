from __future__ import annotations

import json
from pathlib import Path

from .models import Action, Outcome, ReplayNode, ReplayWorld


def world_from_dict(d: dict) -> ReplayWorld:
    nodes = []
    for n in d["nodes"]:
        a = n["action"]
        action = Action(
            id=a["id"], branch=a.get("branch", a["id"]), parent_id=a.get("parent_id"),
            estimated_cost=float(a.get("estimated_cost", 1.0)), metadata=dict(a.get("metadata", {})),
        )
        outcomes = tuple(
            Outcome(
                action_id=action.id, quality=float(o["quality"]), cost=float(o.get("cost", 1.0)),
                terminal=bool(o.get("terminal", False)), success=bool(o.get("success", True)),
                failure_class=o.get("failure_class"), payload=dict(o.get("payload", {})),
            ) for o in n.get("outcomes", [])
        )
        nodes.append(ReplayNode(action, outcomes))
    return ReplayWorld(
        world_id=d["world_id"], task_id=d.get("task_id", d["world_id"]), nodes=tuple(nodes),
        budget=int(d.get("budget", 8)), max_parallelism=int(d.get("max_parallelism", 2)), metadata=dict(d.get("metadata", {})),
    )


def load_worlds(path: str | Path) -> list[ReplayWorld]:
    d = json.loads(Path(path).read_text())
    entries = d if isinstance(d, list) else d.get("worlds", [d])
    return [world_from_dict(x) for x in entries]
