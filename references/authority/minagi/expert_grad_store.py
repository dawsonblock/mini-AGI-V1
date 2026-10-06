"""Gradient accumulation keyed by logical Expert UID."""
from __future__ import annotations
import torch


class ExpertGradStore:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self._grads: dict[int, dict[str, torch.Tensor]] = {}

    @torch.no_grad()
    def accumulate(self, uid: int, **grads):
        uid = int(uid)
        ent = self._grads.setdefault(uid, {})
        for name, grad in grads.items():
            if grad is None:
                continue
            g = grad.detach().to(self.device, dtype=torch.float32)
            if name in ent:
                ent[name].add_(g)
            else:
                ent[name] = g.clone()

    def get(self, uid: int):
        return self._grads.get(int(uid))

    def uids(self):
        return sorted(self._grads)

    def clear(self):
        self._grads.clear()

    def squared_norm(self) -> float:
        return float(sum(float(g.float().pow(2).sum())
                         for ent in self._grads.values() for g in ent.values()))

    @torch.no_grad()
    def clip_(self, max_norm: float):
        max_norm = float(max_norm)
        if max_norm <= 0:
            return self.squared_norm() ** 0.5
        norm = self.squared_norm() ** 0.5
        if norm > max_norm:
            scale = max_norm / (norm + 1e-12)
            for ent in self._grads.values():
                for g in ent.values():
                    g.mul_(scale)
        return norm
