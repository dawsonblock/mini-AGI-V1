import torch
from minagi.recur import RecurConfig, RecurCoder
from minagi.consolidation import ConsolidationCandidate
from minagi.optim_groups import split_parameters


def _model():
    c=RecurConfig(vocab_size=16,d_model=8,n_head=2,d_ff=16,block=8,
                  n_prelude=0,n_recur=1,n_coda=0,max_steps=1,
                  use_pool=True,pool_experts=2,pool_d_ff=16,pool_top_k=1,
                  pool_max=2,pool_dense_residual=True)
    return RecurCoder(c)


def test_candidate_guard_rolls_back_by_default():
    m=_model(); p=split_parameters(m).trunk[0]
    before=p.detach().clone()
    with ConsolidationCandidate(m):
        with torch.no_grad(): p.add_(1)
    assert torch.equal(before,p.detach())


def test_candidate_guard_commit_keeps_change():
    m=_model(); p=split_parameters(m).trunk[0]
    before=p.detach().clone()
    with ConsolidationCandidate(m) as c:
        with torch.no_grad(): p.add_(1)
        c.commit()
    assert not torch.equal(before,p.detach())
