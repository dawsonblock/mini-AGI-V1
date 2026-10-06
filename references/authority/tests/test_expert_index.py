import torch
from minagi.expert_index import HierarchicalExpertIndex


def test_hierarchical_index_bounds_candidates_and_keeps_best_group():
    torch.manual_seed(0)
    w=torch.randn(256,8)*0.01
    # Make expert 130 and therefore its group strongly match x.
    w[128:144].zero_(); w[130,0]=10.0
    x=torch.zeros(4,8); x[:,0]=1.0
    idx=HierarchicalExpertIndex(group_size=16, top_groups=2, refresh_every=999,
                                max_candidates=32)
    ids, z, coverage=idx.score(x,w,256,min_candidates=8)
    assert ids.numel() <= 32
    assert 130 in ids.tolist()
    assert z.shape == (4, ids.numel())
    assert 0.0 < coverage <= 1.0
    st=idx.stats(); assert st['groups']==16


def test_index_refreshes_after_resize():
    w=torch.randn(64,4); x=torch.randn(2,4)
    idx=HierarchicalExpertIndex(group_size=8, top_groups=2)
    ids,_,_=idx.score(x,w,64)
    w2=torch.randn(80,4)
    ids2,_,_=idx.score(x,w2,80)
    assert idx.stats()['experts']==80
    assert int(ids2.max()) < 80
