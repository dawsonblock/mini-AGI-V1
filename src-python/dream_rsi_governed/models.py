from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DatasetSplit(str, Enum):
    DEVELOPMENT = "development"
    SELECTION = "selection"
    HOLDOUT = "holdout"


@dataclass(frozen=True)
class Action:
    id: str
    branch: str
    parent_id: str | None = None
    estimated_cost: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    action_id: str
    quality: float
    cost: float = 1.0
    terminal: bool = False
    success: bool = True
    failure_class: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayNode:
    action: Action
    outcomes: tuple[Outcome, ...]


@dataclass(frozen=True)
class ReplayWorld:
    world_id: str
    task_id: str
    nodes: tuple[ReplayNode, ...]
    budget: int
    max_parallelism: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def node_map(self) -> dict[str, ReplayNode]:
        return {n.action.id: n for n in self.nodes}


@dataclass(frozen=True)
class ReplayDecision:
    action_ids: tuple[str, ...]
    stop: bool = False
    rationale: str = ""


@dataclass(frozen=True)
class ReplayStep:
    index: int
    chosen: tuple[str, ...]
    observed: tuple[Outcome, ...]
    best_quality: float
    cumulative_cost: float
    unsupported_requests: int = 0


@dataclass(frozen=True)
class ReplayTrace:
    world_id: str
    policy_id: str
    steps: tuple[ReplayStep, ...]
    best_quality: float
    total_cost: float
    probes: int
    support_ratio: float
    uncertainty: float
    stopped: bool


@dataclass(frozen=True)
class CandidatePolicy:
    policy_id: str
    version: str
    source_digest: str
    config: dict[str, Any] = field(default_factory=dict)
    parent_policy_id: str | None = None


@dataclass(frozen=True)
class AggregateMetrics:
    mean_best_quality: float
    mean_cost: float
    mean_probes: float
    mean_support_ratio: float
    mean_uncertainty: float
    score: float
    worlds: int


@dataclass(frozen=True)
class QualificationRecord:
    candidate: CandidatePolicy
    baseline_policy_id: str
    selection: AggregateMetrics
    holdout: AggregateMetrics
    baseline_holdout: AggregateMetrics
    gates: dict[str, bool]
    passed: bool
    evidence_digest: str
    created_at: str


@dataclass(frozen=True)
class PromotionManifest:
    candidate: CandidatePolicy
    qualification_digest: str
    artifact_root_digest: str
    issued_at: str
    signer_key_id: str
    signature_b64: str
    generation: int

@dataclass(frozen=True)
class CanaryAttestation:
    candidate_policy_id: str
    baseline_policy_id: str
    candidate_score: float
    baseline_score: float
    candidate_cost: float
    baseline_cost: float
    passed: bool
    environment_digest: str
    run_digest: str
    created_at: str


@dataclass(frozen=True)
class CalibrationReport:
    candidate_ids: tuple[str, ...]
    replay_scores: tuple[float, ...]
    live_scores: tuple[float, ...]
    spearman_rho: float
    passed: bool
    threshold: float
