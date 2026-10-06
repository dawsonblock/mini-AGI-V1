from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any


class ReconstructionMode(str, Enum):
    FAST = "FAST"
    BALANCED = "BALANCED"
    EXACT = "EXACT"


class ReconstructionAction(str, Enum):
    HYPIC_SEAM8 = "HYPIC_SEAM8"
    SINGLE_STATE_INIT = "SINGLE_STATE_INIT"
    EXACT_PREFIX_REPLAY = "EXACT_PREFIX_REPLAY"
    EXACT_SELECTED_REPLAY = "EXACT_SELECTED_REPLAY"
    # Backward-compatible aliases from the original RC10 skeleton.
    COMPOSE = "HYPIC_SEAM8"
    SEAM = "HYPIC_SEAM8"
    EXACT_REPLAY = "EXACT_SELECTED_REPLAY"


class Compatibility(str, Enum):
    EXACT_COMPATIBLE = "EXACT_COMPATIBLE"
    NUMERICALLY_COMPATIBLE = "NUMERICALLY_COMPATIBLE"
    REPLAY_REQUIRED = "REPLAY_REQUIRED"
    INVALID = "INVALID"


class PromotionDecision(str, Enum):
    PROMOTE = "PROMOTE"
    QUARANTINE = "QUARANTINE"
    REJECT = "REJECT"


class TransitionOrientation(str, Enum):
    LEFT_MULTIPLY = "LEFT_MULTIPLY"
    RIGHT_MULTIPLY = "RIGHT_MULTIPLY"


class AssemblyTopology(str, Enum):
    CANONICAL = "CANONICAL"
    EXACT_PREFIX = "EXACT_PREFIX"
    ARBITRARY = "ARBITRARY"
    UNKNOWN = "UNKNOWN"


class CacheTier(str, Enum):
    NVME = "NVME"
    DRAM = "DRAM"
    HBM = "HBM"


def _digest_dict(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ModelIdentity:
    base_weights: str
    adapter_manifest: str
    tokenizer: str
    architecture: str
    position_config: str
    quantization_config: str = "none"

    @property
    def digest(self) -> str:
        return _digest_dict(asdict(self))


@dataclass(frozen=True)
class ExecutionIdentity:
    model: ModelIdentity
    kernel_abi: str
    recurrence_impl: str
    tensor_layout: str
    activation_dtype: str
    cache_dtype: str
    runtime_learning_state: str = "frozen"

    @property
    def digest(self) -> str:
        return _digest_dict({"model": asdict(self.model), **{k: v for k, v in asdict(self).items() if k != "model"}})

    def to_dict(self) -> dict[str, Any]:
        return {"model": asdict(self.model), **{k: v for k, v in asdict(self).items() if k != "model"}, "digest": self.digest}


@dataclass(frozen=True)
class CacheIdentity:
    """Legacy RC10 identity retained for compatibility with older callers.

    New code should use ModelIdentity + ExecutionIdentity. Any adapter change remains
    replay-required by default.
    """
    base_model_digest: str
    adapter_set_digest: str
    tokenizer_digest: str
    layer_layout_digest: str
    position_scheme: str
    recurrence_impl: str

    def to_execution_identity(self) -> ExecutionIdentity:
        model = ModelIdentity(
            base_weights=self.base_model_digest,
            adapter_manifest=self.adapter_set_digest,
            tokenizer=self.tokenizer_digest,
            architecture=self.layer_layout_digest,
            position_config=self.position_scheme,
        )
        return ExecutionIdentity(
            model=model,
            kernel_abi="legacy",
            recurrence_impl=self.recurrence_impl,
            tensor_layout="legacy",
            activation_dtype="unknown",
            cache_dtype="unknown",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeRiskSignals:
    """Cheap online observables; none requires exact selected replay."""
    seam_hidden_rel_l2: float = 0.0
    seam_attention_rel_l2: float = 0.0
    seam_routing_disagreement: float = 0.0
    conv_correction_rel_l2: float = 0.0
    join_count: int = 0
    reorder_distance: float = 0.0
    predecessor_changed: bool = False
    historical_failure_rate: float = 0.0


@dataclass
class OracleMetrics:
    """Offline/shadow labels measured against exact selected replay."""
    state_rel_l2: float = 0.0
    state_angle_deg: float = 0.0
    hidden_rel_l2: float = 0.0
    attention_rel_l2: float = 0.0
    logit_kl: float = 0.0
    top1_agreement: float = 1.0
    topk_agreement: float = 1.0
    routing_disagreement: float = 0.0
    task_score_delta: float = 0.0


# Backward-compatible name used by the original skeleton.
CoherenceMetrics = OracleMetrics


@dataclass
class QualificationRecord:
    candidate_id: str
    decision: PromotionDecision
    target_gain: float
    general_regression: float
    critical_pass: bool
    cache_quality_pass: bool
    cache_warmup_cost: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    reason: str = ""
