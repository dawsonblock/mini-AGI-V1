from minagi.optim_groups import split_parameters
from minagi.recur import RecurCoder, RecurConfig
from minagi.pool import PooledMLP


def test_router_and_depth_embeddings_are_pool_owned():
    cfg = RecurConfig(vocab_size=32, d_model=16, n_head=4, d_ff=32, block=16,
                      use_pool=True, pool_experts=4, pool_d_ff=24, pool_top_k=2,
                      pool_max=8, n_prelude=1, n_recur=1, n_coda=0,
                      max_steps=1)
    model = RecurCoder(cfg)
    split = split_parameters(model)
    pool_ids = {id(p) for p in split.pool}
    sites = [m for m in model.modules() if isinstance(m, PooledMLP)]
    assert sites
    for site in sites:
        assert id(site.router.weight) in pool_ids
        assert id(site.depth_emb) in pool_ids
    assert id(model.pool.gate) in pool_ids
