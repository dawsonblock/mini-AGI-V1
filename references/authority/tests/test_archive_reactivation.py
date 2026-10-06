import numpy as np
import torch

from minagi.paged import PagedPool
from minagi.pool import PooledMLP


def _expert_file(root, i, d_model=4, d_ff=8):
    rng = np.random.default_rng(i + 10)
    np.savez(root / f"e{i:05d}.npz",
             w1=rng.normal(size=(d_ff, d_model)).astype("float32"),
             w3=rng.normal(size=(d_ff, d_model)).astype("float32"),
             w2=rng.normal(size=(d_model, d_ff)).astype("float32"))


def test_pruned_expert_can_be_reactivated_with_router_row(tmp_path):
    for i in range(3):
        _expert_file(tmp_path, i)
    pool = PagedPool(str(tmp_path), 4, 8, 3, resident=1, ram_capacity=3,
                     max_experts=8)
    site = PooledMLP(pool, 4, top_k=1, grad_checkpoint=False, capacity_factor=0)
    pool._sites = [site]
    with torch.no_grad():
        site.router.weight.copy_(torch.tensor([
            [1., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 3., 0.],
        ]))
    saved = site.router.weight[1].detach().clone()
    pool.segments = 100
    pool.last_seen[:] = torch.tensor([100., 0., 100.])
    pool.born[:] = 0
    assert pool.prune(step=100, survival=10) == 1
    assert 1 in pool.archived_uids()
    assert pool.n_experts() == 2

    assert pool.reactivate_archived(1, step=101)
    assert pool.n_experts() == 3
    assert int(pool.uid[-1]) == 1
    torch.testing.assert_close(site.router.weight[-1], saved)
    assert (tmp_path / "e00001.npz").exists()
