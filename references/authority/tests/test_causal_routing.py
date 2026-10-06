import numpy as np
import torch

from minagi.paged import PagedPool
from minagi.pool import PooledMLP


def _expert_file(root, i, d_model=4, d_ff=8):
    rng = np.random.default_rng(i + 1)
    np.savez(
        root / f"e{i:05d}.npz",
        w1=rng.normal(size=(d_ff, d_model)).astype("float32") * 0.1,
        w3=rng.normal(size=(d_ff, d_model)).astype("float32") * 0.1,
        w2=rng.normal(size=(d_model, d_ff)).astype("float32") * 0.1,
    )


def test_suffix_cannot_change_prefix_expert_admission(tmp_path):
    for i in range(4):
        _expert_file(tmp_path, i)
    pool = PagedPool(str(tmp_path), 4, 8, 4, resident=2, ram_capacity=4)
    pool.admission_mode = "prefix_causal"
    site = PooledMLP(pool, 4, top_k=1, grad_checkpoint=False,
                     capacity_factor=0)
    with torch.no_grad():
        site.router.weight.zero_()
        # first token [1,0,0,0] always wants expert 0
        site.router.weight[0, 0] = 5
        # future suffix can strongly request expert 2 or 3
        site.router.weight[2, 2] = 10
        site.router.weight[3, 3] = 10

    prefix = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    xa = torch.cat([prefix, torch.tensor([[[0.0, 0.0, 3.0, 0.0],
                                           [0.0, 0.0, 3.0, 0.0]]])], dim=1)
    xb = torch.cat([prefix, torch.tensor([[[0.0, 0.0, 0.0, 3.0],
                                           [0.0, 0.0, 0.0, 3.0]]])], dim=1)

    pool.begin_text(False)
    ya = site(xa)
    admitted_a = set(pool._admitted)

    pool.begin_text(False)
    yb = site(xb)
    admitted_b = set(pool._admitted)

    assert admitted_a == admitted_b
    torch.testing.assert_close(ya[:, 0], yb[:, 0], rtol=0, atol=1e-7)


def test_legacy_window_vote_is_explicit_compatibility_mode(tmp_path):
    for i in range(4):
        _expert_file(tmp_path, i)
    pool = PagedPool(str(tmp_path), 4, 8, 4, resident=2, ram_capacity=4)
    pool.admission_mode = "window_vote"
    assert pool.admission_mode == "window_vote"


def test_causal_prefix_vote_uses_prefix_but_ignores_later_suffix(tmp_path):
    for i in range(4):
        _expert_file(tmp_path, i)
    pool = PagedPool(str(tmp_path), 4, 8, 4, resident=2, ram_capacity=4)
    pool.admission_mode = "causal_prefix_vote"
    pool.admission_prefix_tokens = 2
    site = PooledMLP(pool, 4, top_k=1, grad_checkpoint=False,
                     capacity_factor=0)
    with torch.no_grad():
        site.router.weight.zero_()
        site.router.weight[0, 0] = 6
        site.router.weight[1, 1] = 6
        site.router.weight[2, 2] = 12
        site.router.weight[3, 3] = 12

    prefix = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0]]])
    xa = torch.cat([prefix, torch.tensor([[[0.0, 0.0, 4.0, 0.0]]])], dim=1)
    xb = torch.cat([prefix, torch.tensor([[[0.0, 0.0, 0.0, 4.0]]])], dim=1)
    pool.begin_text(False)
    site(xa)
    a = set(pool._admitted)
    pool.begin_text(False)
    site(xb)
    b = set(pool._admitted)
    assert a == b
    # Both prefix tokens can contribute; this is not v2's first-token-only path.
    assert 0 in a and 1 in a
