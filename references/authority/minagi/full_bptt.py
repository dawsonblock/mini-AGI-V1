"""One-graph training transaction for v5 virtual expert paging."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch


@dataclass
class FullBPTTReport:
    loss: float
    shared_grad_norm: float
    expert_grad_norm: float
    experts_updated: int
    graph_id: int


class FullBPTTStepper:
    """Coordinate normal autograd and external expert AdamW at one barrier."""
    def __init__(self, model, optimizer, *, clip=1.0, aux_weight=0.0,
                 expert_lr=None, expert_weight_decay=0.0,
                 expert_betas=(0.9, 0.95), expert_eps=1e-8):
        self.model = model; self.optimizer = optimizer
        self.clip = float(clip); self.aux_weight = float(aux_weight)
        self.expert_lr = expert_lr; self.expert_weight_decay = float(expert_weight_decay)
        self.expert_betas = expert_betas; self.expert_eps = float(expert_eps)

    def step(self, x, y, **forward_kw):
        pool = getattr(self.model, "pool", None)
        if pool is None or not getattr(pool, "external_autograd", False):
            raise TypeError("FullBPTTStepper requires VirtualPagedPool")
        epoch = pool.open_graph(training=True)
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        try:
            _logits, loss = self.model(x, y, **forward_kw)
            if self.aux_weight:
                loss = loss + self.aux_weight * self.model.pool_aux()
            if hasattr(self.model, "pool_balance"):
                loss = loss + self.model.pool_balance()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            epoch.begin_backward()
            loss.backward()
            pool.finish_backward()
            expert_norm = pool.grad_store.clip_(self.clip) if self.clip > 0 else pool.grad_store.squared_norm() ** 0.5
            shared_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)) if self.clip > 0 else 0.0
            self.optimizer.step()
            lr = self.expert_lr
            if lr is None:
                # Use the fast/pool group LR when available.
                groups = [g for g in self.optimizer.param_groups if str(g.get("name", "")).startswith("pool")]
                lr = float(groups[0]["lr"] if groups else self.optimizer.param_groups[-1]["lr"])
            changed = pool.external_step(lr=lr, weight_decay=self.expert_weight_decay,
                                         betas=self.expert_betas, eps=self.expert_eps)
            return FullBPTTReport(float(loss.detach()), shared_norm, float(expert_norm),
                                  len(changed), epoch.id)
        except Exception:
            pool.abort_graph(); self.optimizer.zero_grad(set_to_none=True)
            raise
