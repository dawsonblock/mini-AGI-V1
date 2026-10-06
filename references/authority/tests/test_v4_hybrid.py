import torch
from minagi.recur import RecurConfig, RecurCoder, HybridSparseMLP
from minagi.optim_groups import split_parameters


def test_v4_dense_sparse_branch_and_optimizer_ownership():
    c=RecurConfig(vocab_size=265,d_model=16,n_head=4,d_ff=32,block=16,
                  n_prelude=0,n_recur=1,n_coda=0,max_steps=1,
                  use_pool=True,pool_experts=4,pool_d_ff=32,pool_top_k=1,
                  pool_max=4,pool_dense_residual=True)
    m=RecurCoder(c)
    h=[x for x in m.modules() if isinstance(x,HybridSparseMLP)]
    assert len(h)==1
    sp=split_parameters(m)
    assert id(h[0].dense.w1.weight) in {id(p) for p in sp.trunk}
    assert id(h[0].sparse.router.weight) in {id(p) for p in sp.pool}
    x=torch.randint(0,265,(1,8))
    h[0].sparse_enabled=False
    y,_=m(x)
    assert y.shape==(1,8,265)
