import torch
import torch.nn.functional as F

from minagi.expert_index import HierarchicalExpertIndex


def test_kmeans_partition_recovers_geometry_lost_by_uid_blocks():
    g = torch.Generator().manual_seed(0)
    d, clusters, per = 16, 8, 16
    n = clusters * per
    centers = F.normalize(torch.randn(clusters, d, generator=g), dim=-1)
    rows = []
    for c in range(clusters):
        rows.append(centers[c].repeat(per, 1) + 0.03 * torch.randn(per, d, generator=g))
    w = torch.cat(rows, 0)
    w = w[torch.randperm(n, generator=g)]  # deliberately destroy UID locality
    x = centers[0].view(1, -1)
    exact = int(torch.argmax((x @ w.T)[0]))

    contiguous = HierarchicalExpertIndex(group_size=16, top_groups=1,
        max_candidates=16, audit_every=0, strategy="contiguous")
    c_ids, _ = contiguous.candidates(x, w, n, min_candidates=1)
    geometry = HierarchicalExpertIndex(group_size=16, top_groups=1,
        max_candidates=16, audit_every=0, strategy="kmeans", kmeans_iters=8)
    k_ids, _ = geometry.candidates(x, w, n, min_candidates=1)

    assert exact not in c_ids.tolist()   # deterministic regression fixture
    assert exact in k_ids.tolist()
    assert geometry.stats()["strategy"] == "kmeans"


def test_kmeans_index_still_fails_safe_under_strict_audit():
    torch.manual_seed(3)
    w = torch.randn(96, 12)
    x = torch.randn(7, 12)
    idx = HierarchicalExpertIndex(group_size=12, top_groups=1,
        max_candidates=12, audit_every=1, min_recall=1.0,
        fallback_calls=1, strategy="kmeans", kmeans_iters=4)
    cand, coverage = idx.token_candidates(x, w, 96, min_candidates=4)
    # With a 100% floor a miss either returns exact (None) or proves this call
    # happened to retain every exact top-4. Both are valid; the audit must run.
    assert idx.stats()["audits"] == 1
    assert torch.all((coverage > 0) & (coverage <= 1))
