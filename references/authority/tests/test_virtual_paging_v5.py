import torch
import pytest

from minagi.expert_store import ExpertStore
from minagi.expert_cache import ExpertCache
from minagi.expert_grad_store import ExpertGradStore
from minagi.external_adamw import ExternalExpertAdamW
from minagi.graph_epoch import GraphEpoch
from minagi.paged_autograd import paged_expert
from minagi.routing_tape import RoutingTape
from minagi.virtual_pool import VirtualPagedPool
from minagi.pool import PooledMLP
from minagi.paged_adamw import build_adamw


def _state(d=4, h=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "w1": torch.randn(h, d, generator=g) * 0.2,
        "w3": torch.randn(h, d, generator=g) * 0.2,
        "w2": torch.randn(d, h, generator=g) * 0.2,
    }


def _dense(x, st):
    return torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(x, st["w1"]))
        * torch.nn.functional.linear(x, st["w3"]), st["w2"])


def test_virtual_paged_gradients_match_dense_with_one_cache_slot(tmp_path):
    store = ExpertStore(tmp_path, 4, 6)
    states = {}
    for uid in range(3):
        states[uid] = _state(seed=uid + 10)
        store.create(uid, from_state=states[uid])
    cache = ExpertCache(store, capacity=1, device="cpu")
    grads = ExpertGradStore()
    epoch = GraphEpoch(store, training=True)

    x = torch.randn(7, 4, generator=torch.Generator().manual_seed(9), requires_grad=True)
    # A -> B -> C -> A forces slot reuse while one graph is still alive.
    schedule = [0, 1, 2, 0, 2, 1, 0]
    ys = [paged_expert(x[i:i+1], uid, epoch, cache, grads) for i, uid in enumerate(schedule)]
    loss = torch.cat(ys).pow(2).sum()
    epoch.begin_backward(); loss.backward(); epoch.finish_backward()
    dx_virtual = x.grad.detach().clone()

    xr = x.detach().clone().requires_grad_(True)
    dense_states = {}
    for uid, st in states.items():
        dense_states[uid] = {k: v.detach().clone().requires_grad_(True) for k, v in st.items()}
    yr = torch.cat([_dense(xr[i:i+1], dense_states[uid]) for i, uid in enumerate(schedule)])
    yr.pow(2).sum().backward()

    assert cache.evictions >= 2
    assert torch.allclose(dx_virtual, xr.grad, atol=2e-6, rtol=2e-5)
    for uid in range(3):
        got = grads.get(uid)
        for name in ("w1", "w3", "w2"):
            assert torch.allclose(got[name], dense_states[uid][name].grad,
                                  atol=2e-6, rtol=2e-5)


def test_epoch_rejects_expert_mutation_before_backward(tmp_path):
    store = ExpertStore(tmp_path, 4, 6)
    store.create(0, from_state=_state(seed=1))
    cache = ExpertCache(store, capacity=1)
    grads = ExpertGradStore()
    epoch = GraphEpoch(store, training=True)
    x = torch.randn(2, 4, requires_grad=True)
    y = paged_expert(x, 0, epoch, cache, grads)
    # Illegally mutate durable expert state while the graph is alive.
    changed = store.load(0)
    changed["w1"].add_(0.01)
    store.write(0, changed)
    epoch.begin_backward()
    with pytest.raises(RuntimeError, match="changed during graph epoch|mutated inside graph"):
        y.sum().backward()


def test_external_adam_updates_only_touched_uid_and_advances_own_clock(tmp_path):
    store = ExpertStore(tmp_path, 4, 6)
    for uid in (0, 1): store.create(uid, from_state=_state(seed=uid + 3))
    cache = ExpertCache(store, capacity=1)
    grads = ExpertGradStore()
    epoch = GraphEpoch(store, training=True)
    x = torch.randn(3, 4, requires_grad=True)
    before0 = store.load(0); before1 = store.load(1)
    y = paged_expert(x, 0, epoch, cache, grads)
    epoch.begin_backward(); y.sum().backward(); epoch.finish_backward()
    opt = ExternalExpertAdamW(store, grads, cache, lr=1e-2)
    changed = opt.step(epoch)
    epoch.close()
    after0 = store.load(0); after1 = store.load(1)
    assert changed == [0]
    assert float(after0["adam_step"]) == 1.0
    assert float(after1["adam_step"]) == 0.0
    assert not torch.equal(after0["w1"], before0["w1"])
    assert torch.equal(after1["w1"], before1["w1"])


def test_routing_tape_replays_discrete_decisions():
    t = RoutingTape()
    a = torch.tensor([[2, 1], [0, 2]])
    assert torch.equal(t.choose(7, a), a)
    t.reset_replay()
    b = torch.tensor([[0, 0], [0, 0]])
    assert torch.equal(t.choose(7, b), a)


def test_routing_tape_replays_candidate_graph_before_topk():
    t = RoutingTape()
    cand = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    top = torch.tensor([[2], [6]])
    assert torch.equal(t.choose_candidates(3, cand), cand)
    assert torch.equal(t.choose(3, top), top)
    digest = t.digest()
    assert len(digest) == 64
    t.reset_replay()
    assert torch.equal(t.choose_candidates(3, torch.zeros_like(cand)), cand)
    assert torch.equal(t.choose(3, torch.zeros_like(top)), top)
    assert t.digest() == digest


def test_pooled_mlp_uses_external_experts_in_one_graph(tmp_path):
    store = ExpertStore(tmp_path, 4, 8)
    for uid in range(3): store.create(uid, from_state=_state(d=4, h=8, seed=20 + uid))
    pool = VirtualPagedPool(tmp_path, 4, 8, 3, resident=1, device="cpu")
    site = PooledMLP(pool, 4, top_k=2, site=3, grad_checkpoint=False,
                     capacity_factor=0.0)
    pool._sites = [site]
    opt = build_adamw([{"params": list(site.parameters()) + [pool.gate],
                        "name": "pool", "lr": 1e-3, "weight_decay": 0.0}], lr=1e-3)
    epoch = pool.open_graph(training=True)
    x = torch.randn(1, 9, 4, requires_grad=True)
    out = site(x)
    epoch.begin_backward(); out.pow(2).mean().backward(); pool.finish_backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert site.router.weight.grad is not None
    assert pool.gate.grad is not None
    assert pool.grad_store.uids()
    opt.step()
    changed = pool.external_step(lr=1e-3)
    assert changed
    assert pool.cache.evictions > 0  # one slot, multiple logical experts


def test_virtual_archive_reactivation_restores_uid_weights_and_router_row(tmp_path):
    store = ExpertStore(tmp_path, 4, 8)
    for uid in range(3):
        store.create(uid, from_state=_state(d=4, h=8, seed=50 + uid))
    pool = VirtualPagedPool(tmp_path, 4, 8, 3, resident=1, device="cpu")
    site = PooledMLP(pool, 4, top_k=1, site=0, grad_checkpoint=False)
    pool._sites = [site]
    opt = build_adamw([{"params": [site.router.weight, pool.gate],
                        "name": "pool", "lr": 1e-3, "weight_decay": 0.0}], lr=1e-3)
    pool.attach_optimiser(opt)
    # Seed nonzero optimizer state and distinctive selection state.
    site.router.weight.grad = torch.randn_like(site.router.weight)
    pool.gate.grad = torch.randn_like(pool.gate)
    opt.step(); opt.zero_grad(set_to_none=True)
    uid = int(pool.uid[1])
    before_w = store.load(uid)["w1"].clone()
    before_row = site.router.weight.detach()[1].clone()
    before_gate = float(pool.gate.detach()[1])
    assert pool.retire_uids([uid], step=9, reason="test") == 1
    assert uid in pool.archived_uids()
    assert pool.reactivate_archived(uid, step=10)
    assert int(pool.uid[-1]) == uid
    assert torch.equal(store.load(uid)["w1"], before_w)
    assert torch.allclose(site.router.weight.detach()[-1], before_row)
    assert abs(float(pool.gate.detach()[-1]) - before_gate) < 1e-6
