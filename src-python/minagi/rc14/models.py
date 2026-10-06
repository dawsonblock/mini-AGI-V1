from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
import math
import time
import uuid

from minagi.egai.canonical import sha256_json


def require_digest(value: str, *, field_name: str = "digest") -> str:
    value = str(value)
    if not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{field_name} must be a sha256 digest")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a sha256 digest") from exc
    return value


class PermanenceLevel(IntEnum):
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


class CandidateState(str, Enum):
    REGISTERED = "REGISTERED"
    BUILT = "BUILT"
    EVALUATED = "EVALUATED"
    QUALIFIED = "QUALIFIED"
    AUTHORIZED = "AUTHORIZED"
    EPOCH_PREPARED = "EPOCH_PREPARED"
    COMMITTED = "COMMITTED"
    WITNESSED = "WITNESSED"
    ATTESTED = "ATTESTED"
    SERVABLE = "SERVABLE"
    REJECTED = "REJECTED"
    REVOKED = "REVOKED"
    ROLLED_BACK = "ROLLED_BACK"
    RETIRED = "RETIRED"


class EpochState(str, Enum):
    PREPARED = "PREPARED"
    AUTHORIZED = "AUTHORIZED"
    LOCALLY_COMMITTED = "LOCALLY_COMMITTED"
    EXTERNALLY_WITNESSED = "EXTERNALLY_WITNESSED"
    ATTESTED = "ATTESTED"
    SERVABLE = "SERVABLE"
    RETIRED = "RETIRED"


@dataclass(frozen=True)
class GovernanceCoordinates:
    permanence: PermanenceLevel
    evidence_strength: EvidenceStrength

    @property
    def digest(self) -> str:
        return sha256_json({"permanence": int(self.permanence), "evidence_strength": int(self.evidence_strength)})


@dataclass(frozen=True)
class LearningProposalRC14:
    target: str
    mechanism: LearningMechanism
    coordinates: GovernanceCoordinates
    evidence_digests: tuple[str, ...]
    rationale_digest: str
    proposer_id: str
    production_identity_digest: str
    expected_gain: float
    expected_transfer: float = 0.0
    expected_interference: float = 0.0
    expected_compute_cost: float = 0.0
    expected_capacity_growth: float = 0.0
    expected_security_risk: float = 0.0
    source_epoch_digest: str = ""
    source_request_id: str = ""
    proposal_id: str = field(default_factory=lambda: "LP14-" + uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    schema: str = "egai-rc14-learning-proposal-v2"

    def __post_init__(self) -> None:
        if self.schema not in {"egai-rc14-learning-proposal-v1", "egai-rc14-learning-proposal-v2"}:
            raise ValueError("unsupported RC14 proposal schema")
        if not self.target or not self.proposer_id or not self.proposal_id:
            raise ValueError("proposal_id, target and proposer_id are required")
        if not self.evidence_digests:
            raise ValueError("proposal requires evidence")
        for d in self.evidence_digests:
            require_digest(d, field_name="evidence_digest")
        require_digest(self.rationale_digest, field_name="rationale_digest")
        require_digest(self.production_identity_digest, field_name="production_identity_digest")
        if bool(self.source_epoch_digest) != bool(self.source_request_id):
            raise ValueError("source_epoch_digest and source_request_id must be supplied together")
        if self.source_epoch_digest:
            require_digest(self.source_epoch_digest, field_name="source_epoch_digest")
        for name in (
            "expected_gain", "expected_transfer", "expected_interference", "expected_compute_cost",
            "expected_capacity_growth", "expected_security_risk",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "evidence_digests", tuple(str(x) for x in self.evidence_digests))

    @property
    def digest(self) -> str:
        body = asdict(self)
        body["mechanism"] = self.mechanism.value
        body["coordinates"] = {
            "permanence": int(self.coordinates.permanence),
            "evidence_strength": int(self.coordinates.evidence_strength),
        }
        return sha256_json(body)


@dataclass(frozen=True)
class ExecutionContext:
    request_id: str
    lease_id: str
    epoch_digest: str
    epoch_generation: int
    acquired_ns: int
    expires_ns: int

    def __post_init__(self) -> None:
        require_digest(self.epoch_digest, field_name="epoch_digest")
        if self.epoch_generation < 0:
            raise ValueError("epoch_generation must be non-negative")
