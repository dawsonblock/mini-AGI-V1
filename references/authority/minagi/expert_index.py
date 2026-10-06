"""Audited candidate retrieval for large logical expert pools.

v6 retains v5.1's exact-recall audit/fail-safe behavior and adds a geometry-
aware search partition.  The old implementation grouped experts only by
contiguous UID ranges; that is fast but has no reason to preserve routing
neighborhoods once expert IDs and router geometry diverge.  ``strategy='kmeans'``
clusters normalized router rows and stores group membership separately from the
stable expert UID.  ``strategy='contiguous'`` is retained for reproduction.

The index is an acceleration structure only.  Candidate construction is no-grad;
the real trainable router rows score candidates with gradients enabled, and
periodic exact scans can fail the same call safe to full-pool routing.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


class HierarchicalExpertIndex:
    def __init__(self, group_size=64, top_groups=4, refresh_every=128,
                 max_candidates=512, audit_every=256, min_recall=0.95,
                 fallback_calls=16, strategy="contiguous", kmeans_iters=6,
                 kmeans_seed=0):
        self.group_size = max(4, int(group_size))
        self.top_groups = max(1, int(top_groups))
        self.refresh_every = max(1, int(refresh_every))
        self.max_candidates = max(self.group_size, int(max_candidates))
        self.audit_every = max(0, int(audit_every))
        self.min_recall = min(1.0, max(0.0, float(min_recall)))
        self.fallback_calls = max(0, int(fallback_calls))
        self.strategy = str(strategy).lower()
        if self.strategy not in {"contiguous", "kmeans"}:
            raise ValueError("expert index strategy must be 'contiguous' or 'kmeans'")
        self.kmeans_iters = max(1, int(kmeans_iters))
        self.kmeans_seed = int(kmeans_seed)
        self._centroids = None
        self._members: list[torch.Tensor] = []
        self._n = 0
        self._calls = 0
        self._weight_version = -1
        self._audits = 0
        self._fallbacks = 0
        self._recall_ema = None
        self._force_exact_remaining = 0

    def invalidate(self):
        self._centroids = None
        self._members = []
        self._n = 0
        self._weight_version = -1

    @torch.no_grad()
    def _contiguous_partition(self, rows: torch.Tensor, n: int):
        gs = self.group_size
        g = math.ceil(n / gs)
        members = []
        centroids = []
        for gi in range(g):
            a, b = gi * gs, min(n, (gi + 1) * gs)
            ids = torch.arange(a, b, device=rows.device, dtype=torch.long)
            members.append(ids)
            centroids.append(rows[ids].mean(0))
        return torch.stack(centroids, 0), members

    @torch.no_grad()
    def _kmeans_partition(self, rows: torch.Tensor, n: int):
        """Deterministic spherical k-means over router rows.

        Expert UID remains the identity; only the acceleration index changes.
        Empty clusters are repaired with farthest-point rows.  Centroids used
        for coarse scoring are raw row means after clustering, which preserves
        the router's dot-product scale better than normalized means.
        """
        g = max(1, math.ceil(n / self.group_size))
        if g == 1:
            ids = torch.arange(n, device=rows.device, dtype=torch.long)
            return rows.mean(0, keepdim=True), [ids]
        norm = F.normalize(rows, dim=-1)
        # Deterministic farthest-point seeding gives much stabler clusters than
        # UID blocks without depending on global RNG state.
        first = int(self.kmeans_seed % n)
        seeds = [first]
        min_dist = 1.0 - (norm @ norm[first]).clamp(-1, 1)
        for _ in range(1, g):
            nxt = int(torch.argmax(min_dist).item())
            seeds.append(nxt)
            d = 1.0 - (norm @ norm[nxt]).clamp(-1, 1)
            min_dist = torch.minimum(min_dist, d)
        c = norm[torch.tensor(seeds, device=rows.device)].clone()
        labels = torch.zeros(n, device=rows.device, dtype=torch.long)
        for _ in range(self.kmeans_iters):
            labels = torch.argmax(norm @ c.T, dim=1)
            new = []
            used_repair = set()
            for gi in range(g):
                ids = torch.where(labels == gi)[0]
                if ids.numel() == 0:
                    # Repair empty cluster with a row least represented by its
                    # current assigned centroid.
                    assigned = (norm * c[labels]).sum(-1)
                    order = torch.argsort(assigned)
                    rid = next(int(v) for v in order.tolist() if int(v) not in used_repair)
                    used_repair.add(rid)
                    vec = norm[rid]
                else:
                    vec = F.normalize(norm[ids].mean(0), dim=0)
                new.append(vec)
            nc = torch.stack(new, 0)
            if torch.allclose(nc, c, atol=1e-5, rtol=0):
                c = nc
                break
            c = nc
        labels = torch.argmax(norm @ c.T, dim=1)
        members: list[torch.Tensor] = []
        raw_centroids = []
        for gi in range(g):
            ids = torch.where(labels == gi)[0]
            if ids.numel() == 0:
                # Final defensive repair; should be rare after the loop.
                rid = torch.argmax(1.0 - (norm @ c[gi]).abs()).reshape(1)
                ids = rid.to(rows.device)
            members.append(ids.sort().values)
            raw_centroids.append(rows[ids].mean(0))
        return torch.stack(raw_centroids, 0), members

    @torch.no_grad()
    def rebuild(self, weight, n):
        n = int(n)
        if n <= 0:
            self._centroids = weight.new_zeros((0, weight.shape[1]))
            self._members = []
            self._n = 0
            return
        rows = weight[:n].detach().float()
        if self.strategy == "kmeans":
            self._centroids, self._members = self._kmeans_partition(rows, n)
        else:
            self._centroids, self._members = self._contiguous_partition(rows, n)
        self._n = n
        self._weight_version = int(getattr(weight, "_version", 0))

    def _needs_refresh(self, weight, n):
        if self._centroids is None or self._n != int(n):
            return True
        if self._calls % self.refresh_every == 0:
            return True
        if self._centroids.shape[1] != weight.shape[1]:
            return True
        # Router parameter in-place updates bump Tensor._version.
        if int(getattr(weight, "_version", 0)) != self._weight_version:
            return True
        return False

    @staticmethod
    @torch.no_grad()
    def _proxy_top(ids, x, weight, k):
        q = x.detach().float().mean(0, keepdim=True)
        if ids is None:
            z = F.linear(q, weight.detach().float())[0]
            kk = min(max(1, int(k)), int(z.numel()))
            return torch.topk(z, kk).indices
        z = F.linear(q, weight[ids].detach().float())[0]
        kk = min(max(1, int(k)), int(z.numel()))
        return ids[torch.topk(z, kk).indices]

    @torch.no_grad()
    def _audit(self, cand, x, weight, n, k):
        self._audits += 1
        exact = self._proxy_top(None, x, weight[:n], k)
        cand_set = set(int(v) for v in cand.detach().cpu().tolist() if int(v) >= 0)
        hit = sum(int(v) in cand_set for v in exact.detach().cpu().tolist())
        recall = hit / max(1, int(exact.numel()))
        self._recall_ema = (recall if self._recall_ema is None
                            else 0.9 * self._recall_ema + 0.1 * recall)
        if recall + 1e-12 < self.min_recall:
            self._force_exact_remaining = max(self._force_exact_remaining,
                                              self.fallback_calls)
            self._fallbacks += 1
            return False, recall
        return True, recall

    def _union_groups(self, group_ids, device):
        parts = [self._members[int(gi)].to(device) for gi in group_ids]
        if not parts:
            return torch.empty(0, device=device, dtype=torch.long)
        return torch.unique(torch.cat(parts), sorted=True)

    @torch.no_grad()
    def candidates(self, x, weight, n, min_candidates=1):
        self._calls += 1
        n = int(n)
        if n <= 0:
            return torch.empty(0, device=weight.device, dtype=torch.long), 1.0
        if self._force_exact_remaining > 0:
            self._force_exact_remaining -= 1
            return torch.arange(n, device=weight.device), 1.0
        if self._needs_refresh(weight, n):
            self.rebuild(weight, n)
        g = int(self._centroids.shape[0])
        if g <= 1:
            return torch.arange(n, device=weight.device), 1.0
        z = F.linear(x.detach().float(), self._centroids)
        gp = F.softmax(z, -1)
        need_groups = max(self.top_groups,
                          math.ceil(max(1, int(min_candidates)) / self.group_size))
        need_groups = min(g, need_groups)
        score = gp.mean(0)
        chosen_g = torch.topk(score, need_groups).indices
        coverage = float(score[chosen_g].sum().clamp(0, 1))
        cand = self._union_groups(chosen_g.tolist(), weight.device)
        if cand.numel() > self.max_candidates:
            q = x.detach().float().mean(0, keepdim=True)
            fine = F.linear(q, weight[cand].detach().float())[0]
            take = torch.topk(fine, self.max_candidates).indices
            cand = cand[take].sort().values
        if self.audit_every and (self._calls % self.audit_every == 0):
            ok, _ = self._audit(cand, x, weight, n,
                                min(max(1, int(min_candidates)), n))
            if not ok:
                return torch.arange(n, device=weight.device), 1.0
        return cand, coverage

    @torch.no_grad()
    def token_candidates(self, x, weight, n, min_candidates=1):
        self._calls += 1
        n = int(n)
        if n <= 0:
            return x.new_empty((x.shape[0], 0), dtype=torch.long), x.new_ones((x.shape[0],))
        if self._force_exact_remaining > 0:
            self._force_exact_remaining -= 1
            return None, x.new_ones((x.shape[0],))
        if self._needs_refresh(weight, n):
            self.rebuild(weight, n)
        g = int(self._centroids.shape[0])
        if g <= 1:
            return None, x.new_ones((x.shape[0],))
        q = x.detach().float()
        coarse = F.linear(q, self._centroids)
        gp = F.softmax(coarse, -1)
        need_groups = max(self.top_groups,
                          math.ceil(max(1, int(min_candidates)) / self.group_size))
        need_groups = min(need_groups, g)
        chosen = torch.topk(gp, need_groups, dim=-1).indices
        coverage = gp.gather(1, chosen).sum(-1).clamp(0, 1)

        rows: list[torch.Tensor] = []
        max_len = 0
        for ti in range(q.shape[0]):
            ids = self._union_groups(chosen[ti].tolist(), weight.device)
            if ids.numel() > self.max_candidates:
                fine = F.linear(q[ti:ti+1], weight[ids].detach().float())[0]
                take = torch.topk(fine, self.max_candidates).indices
                ids = ids[take].sort().values
            rows.append(ids)
            max_len = max(max_len, int(ids.numel()))
        # Fixed-width per-token matrix with -1 mask sentinel.
        cand = torch.full((q.shape[0], max_len), -1, device=weight.device, dtype=torch.long)
        for ti, ids in enumerate(rows):
            cand[ti, :ids.numel()] = ids

        if self.audit_every and (self._calls % self.audit_every == 0):
            self._audits += 1
            m = min(32, q.shape[0])
            sample = (torch.linspace(0, q.shape[0] - 1, steps=m, device=q.device).long()
                      if m else torch.empty(0, dtype=torch.long, device=q.device))
            if m:
                exact_z = F.linear(q[sample], weight[:n].detach().float())
                kk = min(max(1, int(min_candidates)), n)
                exact = torch.topk(exact_z, kk, dim=-1).indices
                cc = cand[sample]
                hit = (exact.unsqueeze(-1) == cc.unsqueeze(1)).any(-1).float().mean()
                recall = float(hit)
            else:
                recall = 1.0
            self._recall_ema = (recall if self._recall_ema is None
                                else 0.9 * self._recall_ema + 0.1 * recall)
            if recall + 1e-12 < self.min_recall:
                self._force_exact_remaining = max(self._force_exact_remaining,
                                                  self.fallback_calls)
                self._fallbacks += 1
                return None, x.new_ones((x.shape[0],))
        return cand, coverage

    @torch.no_grad()
    def score(self, x, weight, n, min_candidates=1):
        ids, coverage = self.candidates(x, weight, n, min_candidates)
        return ids, F.linear(x, weight[ids]).float(), coverage

    def stats(self):
        sizes = [int(x.numel()) for x in self._members]
        return {
            "experts": self._n,
            "groups": 0 if self._centroids is None else int(self._centroids.shape[0]),
            "group_size": self.group_size,
            "group_size_min": min(sizes) if sizes else 0,
            "group_size_max": max(sizes) if sizes else 0,
            "strategy": self.strategy,
            "top_groups": self.top_groups,
            "max_candidates": self.max_candidates,
            "refresh_every": self.refresh_every,
            "audit_every": self.audit_every,
            "min_recall": self.min_recall,
            "audits": self._audits,
            "recall_ema": self._recall_ema,
            "fallbacks": self._fallbacks,
            "force_exact_remaining": self._force_exact_remaining,
            "calls": self._calls,
        }
