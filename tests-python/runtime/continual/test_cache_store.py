import numpy as np
import pytest
from kvcontinual.continual.cache.block import HybridMemoryBlock
from kvcontinual.continual.cache.store import BlockStore
from kvcontinual.continual.recurrent.affine import AffineSummary
from kvcontinual.continual.types import CacheIdentity


def test_block_store_rejects_wrong_identity():
    a = CacheIdentity("b", "a", "t", "l", "rope", "r")
    b = CacheIdentity("b", "other", "t", "l", "rope", "r")
    store = BlockStore()
    store.put(HybridMemoryBlock("x", 0, 1, [1], {0: AffineSummary(np.eye(2), np.zeros((2,2)))}, a))
    with pytest.raises(ValueError):
        store.compose(["x"], b)
