from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum, IntEnum
import math
import time
from typing import Any, Mapping, Sequence
import uuid

from .canonical import sha256_json


class AuthorityPlane(str, Enum):
    COGNITIVE = "cognitive"
    LEARNING = "learning"
    PROMOTION = "promotion"


class EvidenceOrigin(str, Enum):
    ENVIRONMENT = "environment"
    DETERMINISTIC_TOOL = "deterministic_tool"
    EXTERNAL_SOURCE = "external_source"
    HUMAN = "human"
    MODEL_INFERENCE = "model_inference"
    GROUNDED_REPLAY = "grounded_replay"
    SIMULATION = "simulation"


class EvidenceClass(str, Enum):
    OBSERVED = "observed"
    DERIVED = "derived"
    SIMULATED = "simulated"


class LearningLevel(IntEnum):
    L0_WORKING_STATE = 0
    L1_EVIDENCE_MEMORY = 1
    L2_BELIEF_UPDATE = 2
    L3_PROCEDURAL_SKILL = 3
    L4_POLICY_MODULATION = 4
    L5_MODULE_COMPOSITION = 5
    L6_MODULE_ADAPTATION = 6
    L7_NEW_MODULE = 7
    L8_SHARED_CONSOLIDATION = 8
    L9_ARCHITECTURE_CHANGE = 9


class LearningAction(str, Enum):
    IGNORE = "ignore"
    REMEMBER = "remember"
    REVISE_BELIEF = "revise_belief"
    CREATE_PROCEDURE = "create_procedure"
    REUSE = "reuse"
    COMPOSE = "compose"
    ADAPT = "adapt"
    EXPAND = "expand"
    CONSOLIDATE = "consolidate"
    ARCHITECTURE_CHANGE = "architecture_change"


class PromotionVerdict(str, Enum):
    REJECT = "reject"
    APPROVE = "approve"


class BeliefStatus(str, Enum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    REVOKED = "revoked"


class HypothesisStatus(str, Enum):
    OPEN = "open"
    FALSIFIED = "falsified"
    SUPPORTED = "supported"


TRANSFER_RINGS: tuple[str, ...] = (
    "episode",
    "procedure",
    "reusable_skill",
    "composition",
    "task_family",
    "new_domain",
    "extrapolation",
)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    origin_class: EvidenceOrigin
    evidence_class: EvidenceClass
    producer: str
    payload_digest: str
    provenance_digest: str
    production_identity_digest: str
    observed_at: float
    valid_from: float | None = None
    valid_until: float | None = None
    confidence: float = 1.0
    parent_digests: tuple[str, ...] = ()
    source_locator: str = ""
    summary: str = ""
    schema: str = "mini-agi-egai-evidence-v2"

    def __post_init__(self) -> None:
        if self.schema != "mini-agi-egai-evidence-v2":
            raise ValueError("unsupported evidence schema")
        if not self.evidence_id or not self.producer:
            raise ValueError("evidence_id and producer are required")
        if not self.payload_digest.startswith("sha256:"):
            raise ValueError("payload_digest must be a sha256 digest")
        if not self.provenance_digest.startswith("sha256:"):
            raise ValueError("provenance_digest must be a sha256 digest")
        if not self.production_identity_digest.startswith("sha256:"):
            raise ValueError("production_identity_digest must be a sha256 digest")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be finite and in [0,1]")
        if self.valid_from is not None and self.valid_until is not None and self.valid_until < self.valid_from:
            raise ValueError("valid_until cannot precede valid_from")
        if self.origin_class is EvidenceOrigin.SIMULATION and self.evidence_class is not EvidenceClass.SIMULATED:
            raise ValueError("simulation origin must be classified as simulated")
        if self.evidence_class is EvidenceClass.SIMULATED and self.origin_class is not EvidenceOrigin.SIMULATION:
            # Grounded replay is derived from recorded evidence, not simulated.
            raise ValueError("simulated evidence class is reserved for simulation origin")
        object.__setattr__(self, "parent_digests", tuple(str(x) for x in self.parent_digests))

    @property
    def digest(self) -> str:
        body = asdict(self)
        body["origin_class"] = self.origin_class.value
        body["evidence_class"] = self.evidence_class.value
        return sha256_json(body)

    @property
    def promotion_eligible(self) -> bool:
        return self.evidence_class is not EvidenceClass.SIMULATED


@dataclass(frozen=True)
class Belief:
    belief_id: str
    statement_digest: str
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    confidence: float
    production_identity_digest: str
    status: BeliefStatus = BeliefStatus.CANDIDATE
    valid_from: float | None = None
    valid_until: float | None = None
    qualification_digest: str = ""
    schema: str = "mini-agi-egai-belief-v2"

    def __post_init__(self) -> None:
        if self.schema != "mini-agi-egai-belief-v2":
            raise ValueError("unsupported belief schema")
        if not self.belief_id or not self.statement_digest.startswith("sha256:"):
            raise ValueError("belief identity/statement digest are required")
        if not self.production_identity_digest.startswith("sha256:"):
            raise ValueError("production identity digest must be sha256")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("belief confidence must be in [0,1]")
        if self.valid_from is not None and self.valid_until is not None and self.valid_until < self.valid_from:
            raise ValueError("belief validity interval is inverted")
        if self.status is BeliefStatus.PROMOTED and not self.qualification_digest.startswith("sha256:"):
            raise ValueError("promoted belief requires qualification digest")
        object.__setattr__(self, "supporting_evidence", tuple(self.supporting_evidence))
        object.__setattr__(self, "contradicting_evidence", tuple(self.contradicting_evidence))

    @property
    def digest(self) -> str:
        body = asdict(self)
        body["status"] = self.status.value
        return sha256_json(body)


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    claim_digest: str
    evidence_digests: tuple[str, ...]
    falsification_tests: tuple[str, ...]
    predicted_transfer_rings: tuple[str, ...]
    production_identity_digest: str
    status: HypothesisStatus = HypothesisStatus.OPEN
    notes: str = ""
    schema: str = "mini-agi-egai-hypothesis-v2"

    def __post_init__(self) -> None:
        if self.schema != "mini-agi-egai-hypothesis-v2":
            raise ValueError("unsupported hypothesis schema")
        if not self.hypothesis_id or not self.claim_digest.startswith("sha256:"):
            raise ValueError("hypothesis identity/claim digest are required")
        if not self.evidence_digests or not self.falsification_tests:
            raise ValueError("hypothesis requires evidence and falsification tests")
        if not self.production_identity_digest.startswith("sha256:"):
            raise ValueError("production identity digest must be sha256")
        unknown = set(self.predicted_transfer_rings) - set(TRANSFER_RINGS)
        if unknown:
            raise ValueError(f"unknown predicted transfer rings: {sorted(unknown)}")
        object.__setattr__(self, "evidence_digests", tuple(self.evidence_digests))
        object.__setattr__(self, "falsification_tests", tuple(self.falsification_tests))
        object.__setattr__(self, "predicted_transfer_rings", tuple(self.predicted_transfer_rings))

    @property
    def digest(self) -> str:
        body = asdict(self)
        body["status"] = self.status.value
        return sha256_json(body)


@dataclass(frozen=True)
class ProposalOption:
    action: LearningAction
    level: LearningLevel
    expected_forward_transfer: float
    expected_interference: float
    expected_compute_cost: float = 0.0
    expected_capacity_growth: float = 0.0
    risk: float = 0.0
    feasible: bool = True
    reason: str = ""

    @property
    def utility(self) -> float:
        return (
            float(self.expected_forward_transfer)
            - float(self.expected_interference)
            - float(self.expected_compute_cost)
            - float(self.expected_capacity_growth)
            - float(self.risk)
        )


@dataclass(frozen=True)
class LearningProposal:
    proposal_id: str
    action: LearningAction
    level: LearningLevel
    target: str
    evidence_digests: tuple[str, ...]
    rationale_digest: str
    proposer_id: str
    production_identity_digest: str
    expected_forward_transfer: float
    expected_interference: float
    expected_compute_cost: float
    expected_capacity_growth: float
    risk: float
    created_at: float = 0.0
    schema: str = "mini-agi-egai-learning-proposal-v2"

    def __post_init__(self) -> None:
        if self.schema != "mini-agi-egai-learning-proposal-v2":
            raise ValueError("unsupported learning proposal schema")
        if not self.proposal_id or not self.target or not self.proposer_id:
            raise ValueError("proposal_id, target and proposer_id are required")
        if not self.evidence_digests:
            raise ValueError("learning proposal requires evidence")
        if not self.production_identity_digest.startswith("sha256:"):
            raise ValueError("production_identity_digest must be sha256")
        if not self.rationale_digest.startswith("sha256:"):
            raise ValueError("rationale_digest must be sha256")
        object.__setattr__(self, "evidence_digests", tuple(str(x) for x in self.evidence_digests))
        if not self.created_at:
            object.__setattr__(self, "created_at", float(time.time()))

    @property
    def digest(self) -> str:
        body = asdict(self)
        body["action"] = self.action.value
        body["level"] = int(self.level)
        return sha256_json(body)

    @classmethod
    def create(
        cls,
        *,
        action: LearningAction,
        level: LearningLevel,
        target: str,
        evidence_digests: Sequence[str],
        rationale: Mapping[str, Any] | str,
        proposer_id: str,
        production_identity_digest: str,
        expected_forward_transfer: float,
        expected_interference: float,
        expected_compute_cost: float = 0.0,
        expected_capacity_growth: float = 0.0,
        risk: float = 0.0,
    ) -> "LearningProposal":
        return cls(
            proposal_id="P-" + uuid.uuid4().hex,
            action=action,
            level=level,
            target=str(target),
            evidence_digests=tuple(str(x) for x in evidence_digests),
            rationale_digest=sha256_json(rationale),
            proposer_id=str(proposer_id),
            production_identity_digest=str(production_identity_digest),
            expected_forward_transfer=float(expected_forward_transfer),
            expected_interference=float(expected_interference),
            expected_compute_cost=float(expected_compute_cost),
            expected_capacity_growth=float(expected_capacity_growth),
            risk=float(risk),
        )


@dataclass(frozen=True)
class SkillManifest:
    skill_id: str
    version: int
    name: str
    implementation_digest: str
    activation_conditions: tuple[str, ...]
    preconditions: tuple[str, ...]
    contraindications: tuple[str, ...]
    permissions: tuple[str, ...]
    resource_budget: Mapping[str, float]
    termination_conditions: tuple[str, ...]
    verifier_digest: str
    supporting_evidence: tuple[str, ...]
    known_failures: tuple[str, ...] = ()
    qualification_digest: str = ""
    rollback_target: str = ""
    schema: str = "mini-agi-egai-skill-manifest-v2"

    def __post_init__(self) -> None:
        if self.schema != "mini-agi-egai-skill-manifest-v2":
            raise ValueError("unsupported skill schema")
        if self.version < 1 or not self.skill_id or not self.name:
            raise ValueError("skill identity/version are required")
        if not self.implementation_digest.startswith("sha256:"):
            raise ValueError("implementation_digest must be sha256")
        if not self.verifier_digest.startswith("sha256:"):
            raise ValueError("verifier_digest must be sha256")
        object.__setattr__(self, "activation_conditions", tuple(self.activation_conditions))
        object.__setattr__(self, "preconditions", tuple(self.preconditions))
        object.__setattr__(self, "contraindications", tuple(self.contraindications))
        object.__setattr__(self, "permissions", tuple(self.permissions))
        object.__setattr__(self, "termination_conditions", tuple(self.termination_conditions))
        object.__setattr__(self, "supporting_evidence", tuple(self.supporting_evidence))
        object.__setattr__(self, "known_failures", tuple(self.known_failures))
        object.__setattr__(self, "resource_budget", {str(k): float(v) for k, v in dict(self.resource_budget).items()})

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))
