import torch
from minagi.recur import RecurConfig, RecurCoder
from minagi.consolidation import Consolidator


def test_consolidator_updates_only_trunk():
    torch.manual_seed(0)
    c=RecurConfig(vocab_size=32,d_model=8,n_head=2,d_ff=16,block=8,
                  n_prelude=0,n_recur=1,n_coda=0,max_steps=1,
                  use_pool=True,pool_experts=2,pool_d_ff=16,pool_top_k=1,
                  pool_max=2,pool_dense_residual=True)
    m=RecurCoder(c)
    router=next(x.router.weight for x in m.modules() if x.__class__.__name__=='PooledMLP').detach().clone()
    x=torch.randint(0,32,(1,6)); y=torch.randint(0,32,(1,6))
    co=Consolidator(m,lr=1e-4)
    rec=co.step(x,y)
    router2=next(x.router.weight for x in m.modules() if x.__class__.__name__=='PooledMLP').detach()
    assert rec['step']==1 and rec['loss']>=0
    assert torch.equal(router,router2)
