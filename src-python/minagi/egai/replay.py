from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Iterable, Mapping, Sequence

from .canonical import sha256_json


class WorldKind(str, Enum):
    GROUNDED = "grounded"
    IMAGINED = "imagined"


@dataclass(frozen=True)
class ExperienceNode:
    node_id: str
    parent_id: str | None
    observation: Mapping[str, object]
    outcome_digest: str
    score: float = 0.0
    cost: float = 1.0
    creation_index: int = 0

    def __post_init__(self) -> None:
        if not self.node_id or not self.outcome_digest.startswith("sha256:"):
            raise ValueError("node_id and outcome_digest are required")
        if self.creation_index < 0:
            raise ValueError("creation_index cannot be negative")
        object.__setattr__(self, "observation", dict(self.observation))

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class ReplayWorld:
    world_id: str
    nodes: tuple[ExperienceNode, ...]
    source_evidence_digests: tuple[str, ...]
    kind: WorldKind = WorldKind.GROUNDED
    schema: str = "mini-agi-egai-replay-world-v2"

    def __post_init__(self) -> None:
        if self.schema != "mini-agi-egai-replay-world-v2":
            raise ValueError("unsupported replay world schema")
        if not self.nodes:
            raise ValueError("replay world requires at least one node")
        ids = {n.node_id for n in self.nodes}
        roots = [n for n in self.nodes if n.parent_id is None]
        if len(roots) != 1:
            raise ValueError("replay world requires exactly one root")
        if len(ids) != len(self.nodes):
            raise ValueError("duplicate replay node id")
        for node in self.nodes:
            if node.parent_id is not None and node.parent_id not in ids:
                raise ValueError("replay node references missing parent")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "source_evidence_digests", tuple(self.source_evidence_digests))

    @property
    def digest(self) -> str:
        body = asdict(self)
        body["kind"] = self.kind.value
        return sha256_json(body)

    @property
    def promotion_evidence_eligible(self) -> bool:
        return self.kind is WorldKind.GROUNDED


@dataclass(frozen=True)
class VisibleReplayNode:
    node_id: str
    parent_id: str | None
    observation: Mapping[str, object]
    score: float
    cost: float


class PrefixReplayView:
    """Only the prefix revealed to a replay policy.

    There is deliberately no reference to the source ReplayWorld and no method
    for looking up hidden nodes. This makes prefix observability structural.
    """

    __slots__ = ("_visible", "_root_id", "round_index")

    def __init__(self, visible: Sequence[VisibleReplayNode], *, root_id: str, round_index: int):
        self._visible = tuple(visible)
        self._root_id = str(root_id)
        self.round_index = int(round_index)

    @property
    def nodes(self) -> tuple[VisibleReplayNode, ...]:
        return self._visible

    @property
    def root_id(self) -> str:
        return self._root_id

    @property
    def eligible_ids(self) -> tuple[str, ...]:
        parent_ids = {x.parent_id for x in self._visible if x.parent_id is not None}
        leaves = {x.node_id for x in self._visible if x.node_id not in parent_ids}
        return tuple(sorted(leaves | {self._root_id}))


@dataclass(frozen=True)
class ReplayResult:
    world_digest: str
    revealed_node_ids: tuple[str, ...]
    best_score: float
    represented_attempts: int
    rounds: int
    total_cost: float


ReplayPolicy = Callable[[PrefixReplayView, int], Iterable[str]]


class GroundedReplayEngine:
    def __init__(self, *, max_rounds: int = 100):
        self.max_rounds = max(1, int(max_rounds))

    def evaluate(self, world: ReplayWorld, policy: ReplayPolicy, *, workers: int = 1) -> ReplayResult:
        if world.kind is not WorldKind.GROUNDED:
            raise ValueError("GroundedReplayEngine accepts grounded worlds only")
        workers = max(1, int(workers))
        root = next(x for x in world.nodes if x.parent_id is None)
        children: dict[str, list[ExperienceNode]] = {x.node_id: [] for x in world.nodes}
        for node in world.nodes:
            if node.parent_id is not None:
                children[node.parent_id].append(node)
        for group in children.values():
            group.sort(key=lambda x: (x.creation_index, x.node_id))

        visible: dict[str, ExperienceNode] = {root.node_id: root}
        revealed_children: dict[str, int] = {x.node_id: 0 for x in world.nodes}
        rounds = 0
        for round_index in range(self.max_rounds):
            view_nodes = tuple(
                VisibleReplayNode(
                    node_id=x.node_id,
                    parent_id=x.parent_id,
                    observation=dict(x.observation),
                    score=float(x.score),
                    cost=float(x.cost),
                )
                for x in sorted(visible.values(), key=lambda n: (n.creation_index, n.node_id))
            )
            view = PrefixReplayView(view_nodes, root_id=root.node_id, round_index=round_index)
            selected = list(policy(view, workers) or [])[:workers]
            if not selected:
                break
            if len(set(selected)) != len(selected):
                raise ValueError("replay policy selected duplicate node")
            eligible = set(view.eligible_ids)
            if any(node_id not in eligible for node_id in selected):
                raise ValueError("replay policy selected hidden or ineligible node")
            changed = False
            for node_id in selected:
                idx = revealed_children[node_id]
                if idx >= len(children[node_id]):
                    continue
                child = children[node_id][idx]
                revealed_children[node_id] += 1
                visible[child.node_id] = child
                changed = True
            rounds += 1
            if not changed:
                break

        non_root = [x for x in visible.values() if x.node_id != root.node_id]
        return ReplayResult(
            world_digest=world.digest,
            revealed_node_ids=tuple(sorted(visible)),
            best_score=max(float(x.score) for x in visible.values()),
            represented_attempts=len(non_root),
            rounds=rounds,
            total_cost=sum(float(x.cost) for x in non_root),
        )


@dataclass(frozen=True)
class SimulationRecord:
    simulator_id: str
    input_digest: str
    predicted_outcome: Mapping[str, object]
    uncertainty: float
    schema: str = "mini-agi-egai-simulation-v2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicted_outcome", dict(self.predicted_outcome))
        if not 0.0 <= float(self.uncertainty) <= 1.0:
            raise ValueError("uncertainty must be in [0,1]")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))

    @property
    def promotion_evidence_eligible(self) -> bool:
        return False


class ReplayWorldCompiler:
    @staticmethod
    def compile_grounded(
        *, world_id: str, nodes: Sequence[ExperienceNode], source_evidence_digests: Sequence[str]
    ) -> ReplayWorld:
        if not source_evidence_digests:
            raise ValueError("grounded replay world requires source evidence")
        return ReplayWorld(
            world_id=str(world_id),
            nodes=tuple(nodes),
            source_evidence_digests=tuple(str(x) for x in source_evidence_digests),
            kind=WorldKind.GROUNDED,
        )
