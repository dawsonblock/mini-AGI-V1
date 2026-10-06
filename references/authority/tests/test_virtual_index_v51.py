import torch

from minagi.expert_index import HierarchicalExpertIndex
from minagi.expert_store import ExpertStore
from minagi.expert_cache import ExpertCache
from minagi.virtual_pool import VirtualPagedPool
from minagi.pool import PooledMLP


def _state(d=4, h=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "w1": torch.randn(h, d, generator=g) * 0.05,
        "w3": torch.randn(h, d, generator=g) * 0.05,
        "w2": torch.randn(d, h, generator=g) * 0.05,
    }


def test_token_candidates_are_per_token_and_bounded():
    w = torch.zeros(16, 2)
    # Four groups, each group points in a distinct direction/sign.
    w[0:4, 0] = 2.0
    w[4:8, 1] = 2.0
    w[8:12, 0] = -2.0
    w[12:16, 1] = -2.0
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    idx = HierarchicalExpertIndex(group_size=4, top_groups=1,
                                  max_candidates=4, audit_every=0)
    cand, cov = idx.token_candidates(x, w, 16, min_candidates=1)
    assert cand.shape == (3, 4)
    assert cand[0].tolist() == [0, 1, 2, 3]
    assert cand[1].tolist() == [4, 5, 6, 7]
    assert cand[2].tolist() == [8, 9, 10, 11]
    assert torch.all((cov > 0) & (cov <= 1))


def test_token_candidate_audit_fails_safe_on_miss():
    # Group centroid hides the actual best row, exactly like the admission audit
    # regression but now evaluated per token.
    w = torch.zeros(16, 2)
    w[0] = torch.tensor([100.0, 0.0])
    w[1:4, 0] = -40.0
    w[4:8, 0] = 1.0
    x = torch.tensor([[1.0, 0.0]])
    idx = HierarchicalExpertIndex(group_size=4, top_groups=1,
                                  max_candidates=4, audit_every=1,
                                  min_recall=1.0, fallback_calls=2)
    cand, cov = idx.token_candidates(x, w, 16, min_candidates=1)
    assert cand is None
    assert float(cov[0]) == 1.0
    assert idx.stats()["fallbacks"] == 1
    cand2, _ = idx.token_candidates(x, w, 16, min_candidates=1)
    assert cand2 is None


def test_prefetch_uses_digest_bound_host_cache(tmp_path):
    store = ExpertStore(tmp_path, 4, 8)
    for uid in (0, 1):
        store.create(uid, from_state=_state(seed=uid + 1))
    cache = ExpertCache(store, capacity=1, host_capacity=2, device="cpu",
                        prefetch_workers=2)
    items = [(u, store.version(u).sha256) for u in (0, 1)]
    assert cache.prefetch(items) == 2
    cache.get(*items[0]); cache.get(*items[1])
    # Device capacity 1 evicts uid0, but the decoded host entry remains and is
    # reused without another file decode when uid0 is requested again.
    before = cache.report()["host_loads"]
    cache.get(*items[0])
    after = cache.report()
    assert after["host_loads"] == before
    assert after["host_hits"] >= 1
    assert after["evictions"] >= 1
    cache.close()


def test_virtual_router_uses_hierarchical_candidates_with_gradients(tmp_path):
    n = 16
    store = ExpertStore(tmp_path, 4, 8)
    for uid in range(n):
        store.create(uid, from_state=_state(seed=100 + uid))
    pool = VirtualPagedPool(tmp_path, 4, 8, n, resident=2, ram_capacity=8,
                            prefetch_workers=0, device="cpu")
    site = PooledMLP(pool, 4, top_k=1, site=0, grad_checkpoint=False,
                     capacity_factor=0.0, hierarchical_index=True,
                     index_group_size=4, index_top_groups=1,
                     index_max_candidates=4, index_audit_every=0)
    pool._sites = [site]
    # Make four coherent router groups so the coarse index has high recall.
    with torch.no_grad():
        site.router.weight.zero_()
        site.router.weight[0:4, 0] = 2.0
        site.router.weight[4:8, 1] = 2.0
        site.router.weight[8:12, 0] = -2.0
        site.router.weight[12:16, 1] = -2.0
    x = torch.tensor([[[1.0, 0.0, 0.2, -0.1],
                       [0.0, 1.0, -0.1, 0.2],
                       [-1.0, 0.0, 0.1, 0.1]]], requires_grad=True)
    epoch = pool.open_graph(training=True)
    y = site(x)
    epoch.begin_backward(); y.pow(2).sum().backward(); pool.finish_backward()
    assert site.index_last_candidates <= 4
    assert site.index_last_candidates < n
    assert site.router.weight.grad is not None
    assert torch.isfinite(site.router.weight.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert pool.grad_store.uids()
    pool.abort_graph()


def test_virtual_index_replay_uses_recorded_candidates_even_if_index_falls_back(tmp_path):
    n = 16
    store = ExpertStore(tmp_path, 4, 8)
    for uid in range(n):
        store.create(uid, from_state=_state(seed=200 + uid))
    pool = VirtualPagedPool(tmp_path, 4, 8, n, resident=2, ram_capacity=8,
                            prefetch_workers=0, device="cpu")
    site = PooledMLP(pool, 4, top_k=1, site=2, grad_checkpoint=False,
                     capacity_factor=0.0, hierarchical_index=True,
                     index_group_size=4, index_top_groups=1,
                     index_max_candidates=4, index_audit_every=0)
    pool._sites = [site]
    with torch.no_grad():
        site.router.weight.zero_()
        site.router.weight[0:4, 0] = 2.0
        site.router.weight[4:8, 1] = 2.0
        site.router.weight[8:12, 0] = -2.0
        site.router.weight[12:16, 1] = -2.0
    pool.eval(); site.eval()
    epoch = pool.open_graph(training=False)
    x = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    y1 = site(x)
    assert [e.get("kind", "topk") for e in epoch.routing.entries] == ["candidates", "topk"]
    # Force a fresh index call to exact fallback. Replay must not call it: it
    # consumes the recorded candidate graph first, then the recorded top-k.
    site.expert_index._force_exact_remaining = 5
    epoch.routing.reset_replay()
    y2 = site(x)
    assert torch.allclose(y1, y2, atol=0, rtol=0)
    assert epoch.routing.cursor == len(epoch.routing.entries)
    epoch.close()
