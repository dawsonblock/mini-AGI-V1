import numpy as np
import torch

from minagi.paged_adamw import PagedAdamW, mark_rowwise


def test_rowwise_adam_updates_only_active_rows_and_clocks():
    p = mark_rowwise(torch.nn.Parameter(torch.tensor([[1.0], [1.0], [1.0]])))
    opt = PagedAdamW([{"params": [p], "lr": 0.1, "weight_decay": 0.1}], lr=0.1)
    before = p.detach().clone()
    p.grad = torch.tensor([[0.0], [1.0], [0.0]])
    opt.step()
    st = opt.state[p]
    assert st["row_step"].tolist() == [0.0, 1.0, 0.0]
    assert torch.equal(p.detach()[0], before[0])
    assert torch.equal(p.detach()[2], before[2])
    assert not torch.equal(p.detach()[1], before[1])


def test_rowwise_clock_is_independent_across_steps():
    p = mark_rowwise(torch.nn.Parameter(torch.zeros(2, 2)))
    opt = PagedAdamW([p], lr=0.01, weight_decay=0.0)
    p.grad = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    opt.step()
    p.grad = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    opt.step()
    p.grad = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    opt.step()
    assert opt.state[p]["row_step"].tolist() == [1.0, 2.0]


def test_paged_expert_clock_survives_eviction_and_reload(tmp_path):
    from minagi.integrity import write_sidecar
    from minagi.paged import PagedPool

    d_model, d_ff = 2, 3
    for uid in (0, 1):
        f = tmp_path / f"e{uid:05d}.npz"
        np.savez(f,
                 w1=np.zeros((d_ff, d_model), dtype=np.float32),
                 w3=np.zeros((d_ff, d_model), dtype=np.float32),
                 w2=np.zeros((d_model, d_ff), dtype=np.float32))
        write_sidecar(f)
    pool = PagedPool(str(tmp_path), d_model, d_ff, n_experts=2,
                     resident=1, ram_capacity=2, device="cpu")
    opt = PagedAdamW([pool.w1, pool.w3, pool.w2], lr=0.01, weight_decay=0.0)
    pool.attach_optimiser(opt)

    def step_expert(e):
        pool.begin_forward(explore=True)
        pool._place([e])
        for p in (pool.w1, pool.w3, pool.w2):
            p.grad = torch.ones_like(p)
        opt.step(); opt.zero_grad(set_to_none=True)

    step_expert(0)
    step_expert(1)  # evicts expert 0 and writes its own t=1 state
    step_expert(0)  # restores t=1 before updating, so it becomes t=2
    pool.flush()
    z = np.load(tmp_path / "e00000.npz")
    assert float(z["adam_step"]) == 2.0


def test_nonpaged_expert_scalar_clock_survives_save_load(tmp_path):
    from torch import nn
    from minagi.pool import SharedPool
    from minagi import store

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = SharedPool(2, n_experts=2, d_ff=3, max_experts=4)

    model = Tiny()
    opt = PagedAdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    for _ in range(3):
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        opt.step()
        opt.zero_grad(set_to_none=True)
    p0 = next(model.pool.experts[0].parameters())
    assert float(opt.state[p0]["step"]) == 3.0

    store.save(model, str(tmp_path), step=3, opt=opt, cfg={})
    restored = Tiny()
    restored_opt = PagedAdamW(restored.parameters(), lr=0.01, weight_decay=0.0)
    store.load(restored, str(tmp_path), opt=restored_opt)
    p1 = next(restored.pool.experts[0].parameters())
    assert float(restored_opt.state[p1]["step"]) == 3.0


def test_shared_pool_gate_clock_survives_growth_resync():
    from types import SimpleNamespace
    from torch import nn
    from minagi.pool import SharedPool
    from train import _resync_opt

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = SharedPool(2, n_experts=2, d_ff=3, max_experts=4)

    model = Tiny()
    opt = PagedAdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    old_gate = model.pool.gate
    old_gate.grad = torch.ones_like(old_gate)
    opt.step(); opt.zero_grad(set_to_none=True)
    model.pool.add_experts(1, birth_gate=0.001)
    _resync_opt(opt, model, SimpleNamespace(lr=0.01, trunk_lr_mult=0.1, wd=0.0))
    assert opt.state[model.pool.gate]["row_step"].tolist() == [1.0, 1.0, 0.0]
