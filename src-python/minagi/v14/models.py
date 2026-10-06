from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
import math
import time
import uuid

from minagi.egai.canonical import sha256_json


class PermanenceLevel(IntEnum):
    """How difficult a learned change is to reverse.

    The values are deliberately independent from the legacy LearningLevel enum.
    V14 keeps the old API stable while making the complementary-memory hierarchy
    explicit and adding a distinct isolated-neural-memory tier.
    """

    L0_WORKING_CONTEXT = 0
    L1_RAW_EVIDENCE = 1
    L2_EPISODIC_MEMORY = 2
    L3_SEMANTIC_BELIEF = 3
    L4_REUSABLE_SKILL = 4
    L5_ROUTING_COMPOSITION = 5
    L6_ISOLATED_NEURAL_MEMORY = 6
    L7_SHARED_ADAPTER = 7
    L8_NEW_MODULE = 8
    L9_FOUNDATION_CONSOLIDATION = 9
    L10_ARCHITECTURE_CHANGE = 10


class EvidenceStrength(IntEnum):
    E0_UNVERIFIED = 0
    E1_INTERNALLY_CONSISTENT = 1
    E2_INDEPENDENTLY_VERIFIED = 2
    E3_REPLICATED = 3
    E4_FRESH_OOD_VALIDATED = 4
    E5_INDEPENDENTLY_REPRODUCED = 5


class LearningMechanism(str, Enum):
    NONE = "none"
    EPISODIC_STORE = "episodic_store"
    BELIEF_UPDATE = "belief_update"
    SKILL = "skill"
    ROUTING_POLICY = "routing_policy"
    COMPOSITION = "composition"
    SPARSE_EPISODIC_ADAPTER = "sparse_episodic_adapter"
    LORA = "lora"
    REPLAY_LORA = "replay_lora"
    SELECTIVE_DECORRELATION_ADAPTER = "selective_decorrelation_adapter"
    ADAPTER_COMPOSITION = "adapter_composition"
    NEW_MODULE = "new_module"
    FOUNDATION_CONSOLIDATION = "foundation_consolidation"
    ARCHITECTURE_PROPOSAL = "architecture_proposal"


@dataclass(frozen=True)
class GovernanceCoordinates:
    permanence: PermanenceLevel
    evidence_strength: EvidenceStrength

    @property
    def digest(self) -> str:
        return sha256_json({"permanence": int(self.permanence), "evidence_strength": int(self.evidence_strength)})


@dataclass(frozen=True)
class LearningProposalV14:
    proposal_id: str
    target: str
    mechanism: LearningMechanism
    coordinates: GovernanceCoordinates
    evidence_digests: tuple[str, ...]
    rationale_digest: str
    proposer_id: str
    production_identity_digest: str
    expected_gain: float
    expected_interference: float
    expected_compute_cost: float = 0.0
    expected_capacity_growth: float = 0.0
    risk: float = 0.0
    created_at: float = 0.0
    schema: str = "mini-agi-v14-learning-proposal-v1"

    def __post_init__(self) -> None:
        if self.schema != "mini-agi-v14-learning-proposal-v1":
            raise ValueError("unsupported v14 proposal schema")
        if not self.proposal_id or not self.target or not self.proposer_id:
            raise ValueError("proposal_id, target and proposer_id are required")
        if not self.evidence_digests:
            raise ValueError("proposal requires evidence")
        for d in self.evidence_digests:
            if not str(d).startswith("sha256:"):
                raise ValueError("evidence digests must be sha256")
        if not self.rationale_digest.startswith("sha256:"):
            raise ValueError("rationale_digest must be sha256")
        if not self.production_identity_digest.startswith("sha256:"):
            raise ValueError("production_identity_digest must be sha256")
        for name in ("expected_gain", "expected_interference", "expected_compute_cost", "expected_capacity_growth", "risk"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "evidence_digests", tuple(str(x) for x in self.evidence_digests))
        if not self.created_at:
            object.__setattr__(self, "created_at", float(time.time()))

    @property
    def digest(self) -> str:
        body = asdict(self)
        body["mechanism"] = self.mechanism.value
        body["coordinates"] = {
            "permanence": int(self.coordinates.permanence),
            "evidence_strength": int(self.coordinates.evidence_strength),
        }
        return sha256_json(body)

    @classmethod
    def create(cls, **kwargs) -> "LearningProposalV14":
        kwargs.setdefault("proposal_id", "LP14-" + uuid.uuid4().hex)
        return cls(**kwargs)


@dataclass(frozen=True)
class CandidateStateEvent:
    candidate_digest: str
    previous_state: str | None
    new_state: str
    actor: str
    authority_generation: int
    policy_generation: int
    authorization_digest: str = ""
    created_at: float = field(default_factory=time.time)
    schema: str = "mini-agi-v14-candidate-state-event-v1"

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))
