import numpy as np
from kvcontinual.continual.cache.block import HybridMemoryBlock
from kvcontinual.continual.cache.store import BlockStore
from kvcontinual.continual.recurrent.affine import AffineSummary
from kvcontinual.continual.runtime import ReconstructionRuntime
from kvcontinual.continual.types import CacheIdentity, CoherenceMetrics, ReconstructionAction


class Backend:
    def seam_replay(self, block_ids, seam_tokens):
        e = 0.1 / seam_tokens
        return CoherenceMetrics(state_rel_l2=e, hidden_rel_l2=e, logit_kl=e, routing_disagreement=e)
    def suffix_replay(self, block_ids, suffix_tokens):
        return CoherenceMetrics(state_rel_l2=.001, hidden_rel_l2=.001, logit_kl=.001, routing_disagreement=.001)
    def exact_replay(self, block_ids):
        return {"exact": block_ids}


def test_adaptive_seam_escalates_until_acceptable():
    ident = CacheIdentity("b", "a", "t", "l", "rope", "r")
    store = BlockStore()
    eye = np.eye(2)
    for i in ["A", "D"]:
        store.put(HybridMemoryBlock(i, 0, 1, [1], {0: AffineSummary(eye, np.zeros((2,2)))}, ident))
    rt = ReconstructionRuntime(store, Backend())
    r = rt.reconstruct(["A", "D"], ident)
    assert r.action == ReconstructionAction.SEAM
    assert r.seam_tokens in (8, 16, 32, 64, 128)
