from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Iterable
from .rc10 import RC10BlockCache, RepairDecision, RiskFeatures, RiskRouter
from .generation import EffectiveModelGeneration

class AssemblyOp(str, Enum):
    REPLAY_BOUNDARY = "replay_boundary"
    APPEND_FRESH_BOUNDARY = "append_fresh_boundary_state"
    APPLY_CACHED_SUFFIX = "apply_cached_recurrent_suffix"
    APPEND_CACHED_FA_SUFFIX = "append_relocated_full_attention_suffix"
    SEGMENT_REPLAY = "segment_replay"
    EXACT_REPLAY = "exact_selected_replay"

@dataclass(frozen=True)
class PlannedOp:
    block_id: str
    op: AssemblyOp
    seam_width: int = 0

@dataclass(frozen=True)
class BlockAssemblyPlan:
    block_id: str
    decision: RepairDecision
    operations: tuple[PlannedOp, ...]

class RC10Assembler:
    """Produces an execution plan that enforces RC10's seam-before-reuse invariant.

    This class deliberately does not implement model-specific Qwen/GDN kernels.
    Those kernels consume this plan. Exact selected replay remains the oracle.
    """
    def __init__(self, generation: EffectiveModelGeneration, router: RiskRouter):
        self.generation = generation
        self.router = router

    def plan_block(self, block: RC10BlockCache, risk: RiskFeatures) -> BlockAssemblyPlan:
        block.compatibility.assert_compatible(self.generation)
        decision = self.router.decide(risk, block.supported_seams())
        if decision == RepairDecision.EXACT:
            return BlockAssemblyPlan(block.block_id, decision,
                                     (PlannedOp(block.block_id, AssemblyOp.EXACT_REPLAY),))
        if decision == RepairDecision.SEGMENT_REPLAY:
            return BlockAssemblyPlan(block.block_id, decision,
                                     (PlannedOp(block.block_id, AssemblyOp.SEGMENT_REPLAY),))
        width = {
            RepairDecision.REUSE_8: 8,
            RepairDecision.REUSE_32: 32,
            RepairDecision.REUSE_128: 128,
        }[decision]
        if block.variant(width) is None:
            # A risk router may request a repair width that this block was never
            # compiled for. Fail toward replay; never apply a mismatched summary.
            return BlockAssemblyPlan(block.block_id, RepairDecision.SEGMENT_REPLAY,
                                     (PlannedOp(block.block_id, AssemblyOp.SEGMENT_REPLAY),))
        ops = (
            PlannedOp(block.block_id, AssemblyOp.REPLAY_BOUNDARY, width),
            PlannedOp(block.block_id, AssemblyOp.APPEND_FRESH_BOUNDARY, width),
            PlannedOp(block.block_id, AssemblyOp.APPLY_CACHED_SUFFIX, width),
            PlannedOp(block.block_id, AssemblyOp.APPEND_CACHED_FA_SUFFIX, width),
        )
        return BlockAssemblyPlan(block.block_id, decision, ops)

    def plan(self, blocks: Iterable[tuple[RC10BlockCache, RiskFeatures]]) -> tuple[BlockAssemblyPlan, ...]:
        return tuple(self.plan_block(b, r) for b, r in blocks)
