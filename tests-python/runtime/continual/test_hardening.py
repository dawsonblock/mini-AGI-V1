
import numpy as np
import pytest

from kvcontinual.continual.cache.block import HybridMemoryBlock
from kvcontinual.continual.cache.identity import compare_cache_identity
from kvcontinual.continual.cache.store import BlockStore
from kvcontinual.continual.memory.store import MemoryRecord, MemoryStore
from kvcontinual.continual.qualification import QualificationPolicy
from kvcontinual.continual.recurrent.affine import AffineSummary
from kvcontinual.continual.registry import AdapterRegistry
from kvcontinual.continual.replay import ReplayItem, ReplayStore
from kvcontinual.continual.runtime import ReconstructionRuntime
from kvcontinual.continual.types import CacheIdentity, Compatibility, ReconstructionAction, ReconstructionMode


def ident(**kw):
    values = dict(
        base_model_digest="base",
        adapter_set_digest="adapter",
        tokenizer_digest="tok",
        layer_layout_digest="layout",
        position_scheme="rope",
        recurrence_impl="gdn-v1",
    )
    values.update(kw)
    return CacheIdentity(**values)


def summary():
    return AffineSummary(np.eye(2), np.zeros((2, 2)))


def test_structural_cache_changes_fail_closed():
    base = ident()
    assert compare_cache_identity(base, ident(adapter_set_digest="new")) == Compatibility.REPLAY_REQUIRED
    assert compare_cache_identity(base, ident(position_scheme="rope-v2")) == Compatibility.INVALID
    assert compare_cache_identity(base, ident(layer_layout_digest="layout-v2")) == Compatibility.INVALID
    assert compare_cache_identity(base, ident(recurrence_impl="gdn-v2")) == Compatibility.INVALID


def test_missing_recurrent_layer_is_hard_error():
    store = BlockStore()
    i = ident()
    store.put(HybridMemoryBlock("A", 0, 1, [1], {0: summary(), 4: summary()}, i))
    store.put(HybridMemoryBlock("B", 1, 2, [2], {0: summary()}, i))
    with pytest.raises(ValueError, match="layer set mismatch"):
        store.compose(["A", "B"], i)


def test_legacy_registry_cannot_write_unsigned_qualification(tmp_path):
    reg = AdapterRegistry(str(tmp_path / "registry"))
    adapter = tmp_path / "adapter.safetensors"; adapter.write_bytes(b"qualified")
    manifest = reg.register_candidate(str(adapter), "sha256:base", "sha256:data", {"lr": 1e-5})
    q = QualificationPolicy().evaluate(manifest.candidate_id, .04, .001, True, True)
    with pytest.raises(RuntimeError, match="qualification bundle"):
        reg.write_qualification(q)


def test_registry_rejects_path_traversal(tmp_path):
    reg = AdapterRegistry(str(tmp_path / "registry"))
    with pytest.raises(ValueError, match="unsafe candidate_id|invalid candidate_id"):
        reg.promote("../escape")


def test_future_fact_not_current():
    s = MemoryStore(":memory:")
    r = MemoryRecord(
        "future",
        "fact",
        "test",
        "1",
        valid_from="2099-01-01T00:00:00+00:00",
    )
    s.put(r)
    assert r.id not in {x.id for x in s.list_current(at="2026-10-04T00:00:00+00:00")}
    assert r.id in {x.id for x in s.list_current(at="2100-01-01T00:00:00+00:00")}


def test_replay_sampling_is_without_replacement():
    store = ReplayStore()
    for i in range(5):
        store.add(ReplayItem({"id": i}))
    sample = store.sample(5, seed=7)
    assert len(sample) == 5
    assert len({x.payload["id"] for x in sample}) == 5


class _Backend:
    def seam_replay(self, block_ids, seam_tokens):
        from kvcontinual.continual.types import CoherenceMetrics
        return CoherenceMetrics()

    def suffix_replay(self, block_ids, suffix_tokens):
        from kvcontinual.continual.types import CoherenceMetrics
        return CoherenceMetrics()

    def exact_replay(self, block_ids):
        return {"ids": block_ids}


def test_fast_mode_returns_composition_and_fixed_seam():
    store = BlockStore()
    i = ident()
    store.put(HybridMemoryBlock("A", 0, 1, [1], {0: summary()}, i))
    result = ReconstructionRuntime(store, _Backend()).reconstruct(["A"], i, ReconstructionMode.FAST)
    assert result.action == ReconstructionAction.SEAM
    assert result.seam_tokens == 8
    assert result.composition is not None
    assert 0 in result.composition.per_layer
