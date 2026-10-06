from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class ReconstructionMode(str, Enum):
    FAST = "FAST"
    BALANCED = "BALANCED"
    EXACT = "EXACT"


class ReconstructionAction(str, Enum):
    COMPOSE = "COMPOSE"
    SEAM = "SEAM"
    SUFFIX = "SUFFIX"
    EXACT_REPLAY = "EXACT_REPLAY"


class Compatibility(str, Enum):
    EXACT_COMPATIBLE = "EXACT_COMPATIBLE"
    APPROX_COMPATIBLE = "APPROX_COMPATIBLE"
    REPLAY_REQUIRED = "REPLAY_REQUIRED"
    INVALID = "INVALID"


class PromotionDecision(str, Enum):
    PROMOTE = "PROMOTE"
    QUARANTINE = "QUARANTINE"
    REJECT = "REJECT"


@dataclass(frozen=True)
class CacheIdentity:
    base_model_digest: str
    adapter_set_digest: str
    tokenizer_digest: str
    layer_layout_digest: str
    position_scheme: str
    recurrence_impl: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoherenceMetrics:
    state_rel_l2: float = 0.0
    hidden_rel_l2: float = 0.0
    logit_kl: float = 0.0
    routing_disagreement: float = 0.0
    transition_condition: float = 1.0
    retrieval_discontinuity: float = 0.0


@dataclass
class QualificationRecord:
    candidate_id: str
    decision: PromotionDecision
    target_gain: float
    general_regression: float
    critical_pass: bool
    cache_quality_pass: bool
    metrics: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    # Filled/bound by AdapterRegistry.write_qualification. Promotion verifies
    # these again immediately before changing production state.
    candidate_manifest_digest: str = ""
    adapter_digest: str = ""
    evaluation_digest: str = ""
    evaluator_digest: str = ""
