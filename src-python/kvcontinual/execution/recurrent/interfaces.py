from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any

from kvcontinual.execution.cache.block import ExecutionArtifact
from kvcontinual.execution.checkpoints import ExactPrefixCheckpoint
from kvcontinual.execution.types import OracleMetrics, RuntimeRiskSignals


@dataclass
class ApproximateExecution:
    state: Any
    risk_signals: RuntimeRiskSignals
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BackendCapabilities:
    # Acceleration is fail-closed by default. Exact selected replay remains the
    # protocol's required baseline path.
    hypic_seam8: bool = False
    single_state_init: bool = False
    exact_prefix_checkpoint: bool = False
    exact_selected_replay: bool = True
    causal_conv_seam_repair: bool = False
    full_attention_relocation: bool = False
    requires_artifact_qualification: bool = False
    platform: str = "generic"
    device: str = "unknown"


def backend_capabilities(backend: Any) -> BackendCapabilities:
    fn = getattr(backend, "capabilities", None)
    if fn is None:
        return BackendCapabilities()
    caps = fn()
    if not isinstance(caps, BackendCapabilities):
        raise TypeError("backend capabilities() must return BackendCapabilities")
    return caps


class ReconstructionBackend(Protocol):
    def materialize(self, source_segment_ids: list[str], identity) -> list[ExecutionArtifact]: ...
    def hypic_seam8(self, artifacts: list[ExecutionArtifact]) -> ApproximateExecution: ...
    def single_state_init(self, artifacts: list[ExecutionArtifact]) -> ApproximateExecution: ...
    def exact_prefix_replay(self, source_segment_ids: list[str], checkpoint: ExactPrefixCheckpoint | None) -> Any: ...
    def exact_selected_replay(self, source_segment_ids: list[str]) -> Any: ...
    def compare_to_oracle(self, approximate: Any, exact: Any) -> OracleMetrics: ...
