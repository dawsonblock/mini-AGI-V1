import torch
from minagi.expert_index import HierarchicalExpertIndex


def test_index_periodic_audit_records_recall():
    torch.manual_seed(1)
    w = torch.randn(128, 16)
    x = torch.randn(8, 16)
    idx = HierarchicalExpertIndex(group_size=16, top_groups=3,
                                  max_candidates=48, audit_every=1,
                                  min_recall=0.0)
    ids, cov = idx.candidates(x, w, 128, min_candidates=8)
    st = idx.stats()
    assert ids.numel() <= 48
    assert 0 < cov <= 1
    assert st["audits"] == 1
    assert st["recall_ema"] is not None


def test_index_fails_safe_to_full_pool_on_recall_miss():
    # Construct groups whose centroids hide one extreme row. Selecting only one
    # group can miss that row; a 100% recall floor must trigger exact fallback.
    w = torch.zeros(16, 2)
    w[0] = torch.tensor([100.0, 0.0])
    w[1:4, 0] = -40.0       # group-0 centroid becomes poor
    w[4:8, 0] = 1.0         # group-1 centroid wins coarse routing
    x = torch.tensor([[1.0, 0.0]])
    idx = HierarchicalExpertIndex(group_size=4, top_groups=1,
                                  max_candidates=4, audit_every=1,
                                  min_recall=1.0, fallback_calls=2)
    ids, cov = idx.candidates(x, w, 16, min_candidates=1)
    assert ids.numel() == 16 and cov == 1.0
    st = idx.stats()
    assert st["fallbacks"] == 1
    # Subsequent call remains exact for the bounded fail-safe window.
    ids2, cov2 = idx.candidates(x, w, 16, min_candidates=1)
    assert ids2.numel() == 16 and cov2 == 1.0
