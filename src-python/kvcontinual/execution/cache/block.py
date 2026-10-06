from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid
import numpy as np

from kvcontinual.execution.attention.relocation import FullAttentionRelocationMetadata
from kvcontinual.execution.recurrent.affine import AffineSummary
from kvcontinual.execution.recurrent.conv_boundary import ConvBoundaryState
from kvcontinual.execution.types import CacheTier, ExecutionIdentity


@dataclass
class BoundaryAnchors:
    """Original boundary observations used only as comparison anchors.

    Fresh seam values are always computed under the new predecessor; these anchors
    are never treated as authoritative after relocation.
    """
    leading_tokens: list[int] = field(default_factory=list)
    original_hidden: dict[int, np.ndarray] = field(default_factory=dict)
    original_attention_output: dict[int, np.ndarray] = field(default_factory=dict)
    original_route_ids: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class RecurrentTailArtifact:
    layer: int
    head: int
    seam_width: int
    # This summary represents source_tokens[seam_width:] ONLY.
    tail_summary: AffineSummary
    trailing_conv_payload: np.ndarray | None = None
    stats: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactQualification:
    """Evidence that a model-derived artifact is eligible for accelerated reuse.

    A backend may require this receipt before it advertises production HYPIC.
    Reference/experimental artifacts can still exist without a receipt; the runtime
    then fails closed to exact replay when qualification is required.
    """
    same_hidden_input_verified: bool = False
    conv_boundary_complete: bool = False
    attention_relocation_complete: bool = False
    oracle_qualified: bool = False
    max_logit_kl: float = float("inf")
    min_top1_agreement: float = 0.0
    hardware_fingerprint: str = ""
    capture_manifest_digest: str = ""
    receipt_digest: str = ""
    source_content_digest: str = ""
    execution_identity_digest: str = ""
    model_weights_digest: str = ""
    verifier_key_id: str = ""
    signature_b64: str = ""

    @property
    def reusable_for_hypic(self) -> bool:
        return (
            self.same_hidden_input_verified
            and self.conv_boundary_complete
            and self.attention_relocation_complete
            and self.oracle_qualified
            and bool(self.hardware_fingerprint)
            and bool(self.capture_manifest_digest)
            and bool(self.receipt_digest)
        )


@dataclass
class ExecutionArtifact:
    source_segment_id: str
    identity: ExecutionIdentity
    fixed_seam_width: int
    recurrent: dict[tuple[int, int], RecurrentTailArtifact]
    attention_kv_ref: str | None = None
    position_metadata: dict[str, Any] = field(default_factory=dict)
    boundary: BoundaryAnchors = field(default_factory=BoundaryAnchors)
    conv_boundaries: dict[int, ConvBoundaryState] = field(default_factory=dict)
    attention_relocation: dict[int, FullAttentionRelocationMetadata] = field(default_factory=dict)
    capture_manifest_digest: str = ""
    qualification: ArtifactQualification | None = None
    tier: CacheTier = CacheTier.DRAM
    byte_size: int = 0
    source_content_digest: str = ""
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def validate(self) -> None:
        if self.fixed_seam_width < 0:
            raise ValueError("fixed_seam_width must be >= 0")
        for key, item in self.recurrent.items():
            if key != (item.layer, item.head):
                raise ValueError(f"recurrent key {key} does not match layer/head")
            if item.seam_width != self.fixed_seam_width:
                raise ValueError("all recurrent tails must share the artifact seam contract")
        for layer, state in self.conv_boundaries.items():
            if layer < 0:
                raise ValueError("convolution layer index must be non-negative")
            state.validate()
        for layer, meta in self.attention_relocation.items():
            if layer != meta.layer:
                raise ValueError("attention relocation key must equal metadata layer")
            meta.validate()
        if self.qualification is not None:
            if self.capture_manifest_digest and self.qualification.capture_manifest_digest != self.capture_manifest_digest:
                raise ValueError("qualification capture manifest does not match execution artifact")
            qsrc = getattr(self.qualification, "source_content_digest", "")
            if qsrc and self.source_content_digest and qsrc != self.source_content_digest:
                raise ValueError("qualification source digest does not match execution artifact")

    @property
    def hypic_qualified(self) -> bool:
        return bool(self.qualification is not None and self.qualification.reusable_for_hypic)


# Legacy wrapper retained so old tests/tools can still load an RC10 block. New
# code should use SourceSegment + ExecutionArtifact instead.
@dataclass
class HybridMemoryBlock:
    block_id: str
    token_start: int
    token_end: int
    tokens: list[int]
    recurrence: dict[int, AffineSummary]
    cache_identity: object
    position_metadata: dict[str, Any] = field(default_factory=dict)
    attention_kv_ref: str | None = None
    entry_hidden: dict[int, np.ndarray] = field(default_factory=dict)
    exit_hidden: dict[int, np.ndarray] = field(default_factory=dict)
    conv_boundary_state: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start
