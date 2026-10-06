from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping

from minagi.egai.canonical import sha256_json
from minagi.egai.models import SkillManifest


@dataclass(frozen=True)
class VerifiedSkillEpisodeRC14:
    episode_id: str
    task_family: str
    procedure_digest: str
    evidence_root: str
    verification_receipt_digest: str
    verifier_id: str
    score: float

    def __post_init__(self) -> None:
        if not self.episode_id or not self.task_family or not self.verifier_id:
            raise ValueError("episode_id, task_family and verifier_id are required")
        for name in ("procedure_digest", "evidence_root", "verification_receipt_digest"):
            if not getattr(self, name).startswith("sha256:"):
                raise ValueError(f"{name} must be sha256")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")


@dataclass(frozen=True)
class HeldOutSkillEvaluationRC14:
    evaluation_id: str
    task_family: str
    procedure_digest: str
    baseline_score: float
    candidate_score: float
    verification_receipt_digest: str
    verifier_id: str
    fresh: bool = True

    def __post_init__(self) -> None:
        if not self.evaluation_id or not self.task_family or not self.verifier_id:
            raise ValueError("evaluation identity is required")
        if not self.procedure_digest.startswith("sha256:") or not self.verification_receipt_digest.startswith("sha256:"):
            raise ValueError("procedure/evaluation receipt digests must be sha256")
        if not all(math.isfinite(float(v)) for v in (self.baseline_score, self.candidate_score)):
            raise ValueError("held-out scores must be finite")

    @property
    def transfer_gain(self) -> float:
        return float(self.candidate_score) - float(self.baseline_score)


@dataclass(frozen=True)
class SkillInductionCandidateRC14:
    procedure_digest: str
    supporting_episode_ids: tuple[str, ...]
    training_task_families: tuple[str, ...]
    heldout_task_families: tuple[str, ...]
    verifier_ids: tuple[str, ...]
    evidence_roots: tuple[str, ...]
    heldout_evaluation_receipts: tuple[str, ...]
    mean_training_score: float
    mean_heldout_transfer_gain: float
    min_heldout_transfer_gain: float
    schema: str = "egai-rc14-skill-induction-candidate-v1"

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))

    def to_skill_manifest(self, *, skill_id: str, version: int, name: str, activation_conditions: Iterable[str],
                          preconditions: Iterable[str] = (), contraindications: Iterable[str] = (),
                          permissions: Iterable[str] = (), resource_budget: Mapping[str, float] | None = None,
                          termination_conditions: Iterable[str] = (), rollback_target: str = "") -> SkillManifest:
        return SkillManifest(
            skill_id=skill_id,
            version=int(version),
            name=name,
            implementation_digest=self.procedure_digest,
            activation_conditions=tuple(activation_conditions),
            preconditions=tuple(preconditions),
            contraindications=tuple(contraindications),
            permissions=tuple(permissions),
            resource_budget=dict(resource_budget or {}),
            termination_conditions=tuple(termination_conditions),
            verifier_digest=sha256_json({"verifiers": self.verifier_ids, "heldout_receipts": self.heldout_evaluation_receipts}),
            supporting_evidence=self.evidence_roots,
            rollback_target=rollback_target,
        )


class CrossTaskSkillInductionEngine:
    """Proposal-only induction requiring both repeated support and fresh transfer."""

    can_promote = False
    can_activate = False

    def __init__(self, *, min_episodes: int = 3, min_training_families: int = 2,
                 min_independent_verifiers: int = 2, min_evidence_roots: int = 2,
                 min_heldout_families: int = 1, min_transfer_gain: float = 0.0):
        if min_episodes < 2 or min_training_families < 1 or min_independent_verifiers < 1 or min_evidence_roots < 1 or min_heldout_families < 1:
            raise ValueError("invalid skill induction thresholds")
        self.min_episodes = int(min_episodes)
        self.min_training_families = int(min_training_families)
        self.min_independent_verifiers = int(min_independent_verifiers)
        self.min_evidence_roots = int(min_evidence_roots)
        self.min_heldout_families = int(min_heldout_families)
        self.min_transfer_gain = float(min_transfer_gain)

    def induce(self, episodes: Iterable[VerifiedSkillEpisodeRC14], heldout: Iterable[HeldOutSkillEvaluationRC14]) -> SkillInductionCandidateRC14:
        eps = tuple(episodes)
        evals = tuple(heldout)
        if len(eps) < self.min_episodes:
            raise PermissionError("insufficient verified episode support")
        procedures = {e.procedure_digest for e in eps}
        if len(procedures) != 1:
            raise ValueError("skill induction episodes do not share one procedure digest")
        procedure = next(iter(procedures))
        train_families = {e.task_family for e in eps}
        if len(train_families) < self.min_training_families:
            raise PermissionError("insufficient cross-task-family training support")
        if len({e.verifier_id for e in eps}) < self.min_independent_verifiers:
            raise PermissionError("insufficient independent verifiers")
        if len({e.evidence_root for e in eps}) < self.min_evidence_roots:
            raise PermissionError("insufficient independent evidence roots")
        if not evals:
            raise PermissionError("fresh held-out transfer evaluation is required")
        if any(not e.fresh for e in evals):
            raise PermissionError("held-out skill evaluation must be fresh")
        if any(e.procedure_digest != procedure for e in evals):
            raise ValueError("held-out evaluation procedure mismatch")
        heldout_families = {e.task_family for e in evals}
        if heldout_families & train_families:
            raise PermissionError("held-out task families must be disjoint from induction families")
        if len(heldout_families) < self.min_heldout_families:
            raise PermissionError("insufficient held-out task-family coverage")
        gains = [e.transfer_gain for e in evals]
        if min(gains) < self.min_transfer_gain:
            raise PermissionError("held-out transfer gain below required threshold")

        return SkillInductionCandidateRC14(
            procedure_digest=procedure,
            supporting_episode_ids=tuple(sorted(e.episode_id for e in eps)),
            training_task_families=tuple(sorted(train_families)),
            heldout_task_families=tuple(sorted(heldout_families)),
            verifier_ids=tuple(sorted({e.verifier_id for e in eps} | {e.verifier_id for e in evals})),
            evidence_roots=tuple(sorted({e.evidence_root for e in eps})),
            heldout_evaluation_receipts=tuple(sorted(e.verification_receipt_digest for e in evals)),
            mean_training_score=sum(float(e.score) for e in eps) / len(eps),
            mean_heldout_transfer_gain=sum(gains) / len(gains),
            min_heldout_transfer_gain=min(gains),
        )
