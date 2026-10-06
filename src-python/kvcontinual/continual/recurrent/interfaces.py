from __future__ import annotations

from typing import Protocol
from kvcontinual.continual.cache.block import HybridMemoryBlock
from kvcontinual.continual.types import CoherenceMetrics


class BlockSummarizer(Protocol):
    def summarize(self, tokens: list[int]) -> HybridMemoryBlock: ...


class ExactReplayer(Protocol):
    def replay_selected(self, blocks: list[HybridMemoryBlock]) -> object: ...


class SeamReplayer(Protocol):
    def replay_seam(self, blocks: list[HybridMemoryBlock], seam_tokens: int) -> CoherenceMetrics: ...


class SuffixReplayer(Protocol):
    def replay_suffix(self, blocks: list[HybridMemoryBlock], suffix_tokens: int) -> CoherenceMetrics: ...


class CoherenceProbe(Protocol):
    def compare(self, approximate: object, exact: object) -> CoherenceMetrics: ...
