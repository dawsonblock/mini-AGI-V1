from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from kvcontinual.execution.cache.block import ExecutionArtifact, HybridMemoryBlock
from kvcontinual.execution.cache.identity import compare_execution_identity
from kvcontinual.execution.recurrent.affine import compose_sequence
from kvcontinual.execution.types import CacheIdentity, CacheTier, ExecutionIdentity


@dataclass
class CompositionResult:
    selected_ids: list[str]
    per_layer: dict[object, object]


class ExecutionArtifactStore:
    """Model-derived cache keyed by source alias + exact ExecutionIdentity.

    RC11 additionally binds each artifact to the immutable source-content digest.
    Legacy unbound artifacts may still be represented for migration, but a caller
    that supplies a source digest will never receive an artifact bound to different
    content.
    """
    def __init__(self):
        self._artifacts: dict[tuple[str, str], ExecutionArtifact] = {}
        self._bytes_by_tier: dict[CacheTier, int] = defaultdict(int)

    def put(self, artifact: ExecutionArtifact) -> None:
        artifact.validate()
        key = (artifact.source_segment_id, artifact.identity.digest)
        old = self._artifacts.get(key)
        if old is not None:
            if old.source_content_digest and artifact.source_content_digest and old.source_content_digest != artifact.source_content_digest:
                raise ValueError("source alias attempted to replace an execution artifact with different source content")
            self._bytes_by_tier[old.tier] -= old.byte_size
        self._artifacts[key] = artifact
        self._bytes_by_tier[artifact.tier] += artifact.byte_size

    def get(self, source_segment_id: str, identity: ExecutionIdentity, source_content_digest: str | None = None) -> ExecutionArtifact | None:
        art = self._artifacts.get((source_segment_id, identity.digest))
        if art is None:
            return None
        if source_content_digest is not None and art.source_content_digest and art.source_content_digest != source_content_digest:
            return None
        return art

    def resolve(self, source_segment_ids: list[str], identity: ExecutionIdentity,
                source_content_digests: dict[str, str] | None = None) -> tuple[list[ExecutionArtifact], list[str]]:
        hits, misses = [], []
        for sid in source_segment_ids:
            digest = source_content_digests.get(sid) if source_content_digests else None
            a = self.get(sid, identity, digest)
            (hits if a is not None else misses).append(a if a is not None else sid)
        return hits, misses

    def compatible_candidates(self, source_segment_id: str, identity: ExecutionIdentity):
        out = []
        for (sid, _), art in self._artifacts.items():
            if sid == source_segment_id:
                out.append((art, compare_execution_identity(art.identity, identity)))
        return out

    def move_tier(self, source_segment_id: str, identity: ExecutionIdentity, tier: CacheTier) -> None:
        art = self.get(source_segment_id, identity)
        if art is None:
            raise KeyError(source_segment_id)
        if art.tier == tier:
            return
        self._bytes_by_tier[art.tier] -= art.byte_size
        art.tier = tier
        self._bytes_by_tier[tier] += art.byte_size

    def evict_namespace(self, identity: ExecutionIdentity) -> int:
        doomed = [k for k, v in self._artifacts.items() if v.identity.digest == identity.digest]
        for key in doomed:
            art = self._artifacts.pop(key)
            self._bytes_by_tier[art.tier] -= art.byte_size
        return len(doomed)

    def bytes_by_tier(self) -> dict[str, int]:
        return {k.value: int(v) for k, v in self._bytes_by_tier.items()}

    def compose_fixed_input_tails(self, source_segment_ids: list[str], identity: ExecutionIdentity):
        """Research-only algebra check; not the arbitrary-PIC runtime path."""
        arts = [self.get(sid, identity) for sid in source_segment_ids]
        if any(a is None for a in arts):
            raise KeyError("missing execution artifact")
        common = set(arts[0].recurrent)
        for art in arts[1:]:
            common &= set(art.recurrent)
        return {key: compose_sequence([art.recurrent[key].tail_summary for art in arts]) for key in sorted(common)}


class BlockStore:
    """Legacy RC10 block store, preserved for migration tooling."""
    def __init__(self):
        self._blocks: dict[str, HybridMemoryBlock] = {}

    def put(self, block: HybridMemoryBlock) -> None:
        self._blocks[block.block_id] = block

    def get(self, block_id: str) -> HybridMemoryBlock:
        return self._blocks[block_id]

    def compose(self, block_ids: list[str], expected_identity: CacheIdentity) -> CompositionResult:
        if not block_ids:
            raise ValueError("block_ids cannot be empty")
        blocks = [self.get(i) for i in block_ids]
        for b in blocks:
            if b.cache_identity != expected_identity:
                raise ValueError(f"Cache identity mismatch for block {b.block_id}")
        common_layers = set(blocks[0].recurrence)
        for b in blocks[1:]:
            common_layers &= set(b.recurrence)
        per_layer = {layer: compose_sequence([b.recurrence[layer] for b in blocks]) for layer in sorted(common_layers)}
        return CompositionResult(selected_ids=block_ids, per_layer=per_layer)
