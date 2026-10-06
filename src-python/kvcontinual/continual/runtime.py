from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kvcontinual.continual.cache.store import BlockStore, CompositionResult
from kvcontinual.continual.recurrent.coherence import CoherenceGate
from kvcontinual.continual.types import CacheIdentity, CoherenceMetrics, ReconstructionAction, ReconstructionMode


class RepairBackend(Protocol):
    def seam_replay(self, block_ids: list[str], seam_tokens: int) -> CoherenceMetrics: ...
    def suffix_replay(self, block_ids: list[str], suffix_tokens: int) -> CoherenceMetrics: ...
    def exact_replay(self, block_ids: list[str]) -> object: ...


@dataclass
class ReconstructionResult:
    action: ReconstructionAction
    composition: CompositionResult | None = None
    seam_tokens: int | None = None
    suffix_tokens: int | None = None
    metrics: CoherenceMetrics | None = None
    exact_state: object | None = None


class ReconstructionRuntime:
    """Reference policy for coherent hybrid reconstruction.

    FAST = affine composition plus fixed 8-token seam repair.
    BALANCED = composition plus adaptive seam/suffix escalation.
    EXACT = exact selected replay (the correctness oracle).
    """

    def __init__(self, store: BlockStore, backend: RepairBackend, gate: CoherenceGate | None = None):
        self.store = store
        self.backend = backend
        self.gate = gate or CoherenceGate()
        self.seams = [8, 16, 32, 64, 128]
        self.suffixes = [256, 512, 1024, 2048]

    def reconstruct(self, block_ids: list[str], identity: CacheIdentity, mode: ReconstructionMode = ReconstructionMode.BALANCED) -> ReconstructionResult:
        composition = self.store.compose(block_ids, identity)
        if mode == ReconstructionMode.EXACT:
            return ReconstructionResult(
                ReconstructionAction.EXACT_REPLAY,
                composition=composition,
                exact_state=self.backend.exact_replay(block_ids),
            )
        if mode == ReconstructionMode.FAST:
            seam = self.seams[0]
            metrics = self.backend.seam_replay(block_ids, seam)
            return ReconstructionResult(
                ReconstructionAction.SEAM,
                composition=composition,
                seam_tokens=seam,
                metrics=metrics,
            )

        last_metrics = None
        for seam in self.seams:
            last_metrics = self.backend.seam_replay(block_ids, seam)
            if self.gate.acceptable(last_metrics):
                return ReconstructionResult(
                    ReconstructionAction.SEAM,
                    composition=composition,
                    seam_tokens=seam,
                    metrics=last_metrics,
                )

        for suffix in self.suffixes:
            last_metrics = self.backend.suffix_replay(block_ids, suffix)
            if self.gate.acceptable(last_metrics):
                return ReconstructionResult(
                    ReconstructionAction.SUFFIX,
                    composition=composition,
                    suffix_tokens=suffix,
                    metrics=last_metrics,
                )

        return ReconstructionResult(
            ReconstructionAction.EXACT_REPLAY,
            composition=composition,
            metrics=last_metrics,
            exact_state=self.backend.exact_replay(block_ids),
        )
