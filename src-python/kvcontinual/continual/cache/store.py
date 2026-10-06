from __future__ import annotations

from dataclasses import dataclass

from kvcontinual.continual.cache.block import HybridMemoryBlock
from kvcontinual.continual.recurrent.affine import AffineSummary, compose_sequence
from kvcontinual.continual.types import CacheIdentity


@dataclass
class CompositionResult:
    selected_ids: list[str]
    per_layer: dict[int, AffineSummary]


class BlockStore:
    def __init__(self):
        self._blocks: dict[str, HybridMemoryBlock] = {}

    def put(self, block: HybridMemoryBlock) -> None:
        if not block.recurrence:
            raise ValueError("recurrent layer summary cannot be empty")
        self._blocks[block.block_id] = block

    def get(self, block_id: str) -> HybridMemoryBlock:
        return self._blocks[block_id]

    def compose(self, block_ids: list[str], expected_identity: CacheIdentity) -> CompositionResult:
        if not block_ids:
            raise ValueError("block_ids cannot be empty")
        blocks = [self.get(i) for i in block_ids]
        expected_layers = set(blocks[0].recurrence)
        if not expected_layers:
            raise ValueError("recurrent layer summary cannot be empty")
        for b in blocks:
            if b.cache_identity != expected_identity:
                raise ValueError(f"Cache identity mismatch for block {b.block_id}")
            if set(b.recurrence) != expected_layers:
                missing = sorted(expected_layers - set(b.recurrence))
                extra = sorted(set(b.recurrence) - expected_layers)
                raise ValueError(
                    f"Recurrent layer set mismatch for block {b.block_id}: "
                    f"missing={missing}, extra={extra}"
                )
        per_layer = {
            layer: compose_sequence([b.recurrence[layer] for b in blocks])
            for layer in sorted(expected_layers)
        }
        return CompositionResult(selected_ids=list(block_ids), per_layer=per_layer)
