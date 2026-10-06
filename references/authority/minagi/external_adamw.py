"""AdamW over expert files, keyed by logical UID rather than CUDA slot."""
from __future__ import annotations
import math
import torch


class ExternalExpertAdamW:
    def __init__(self, store, grad_store, cache=None, *, lr=3e-4,
                 betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        self.store = store
        self.grad_store = grad_store
        self.cache = cache
        self.lr = float(lr)
        self.betas = tuple(float(x) for x in betas)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.steps = 0

    @torch.no_grad()
    def step(self, epoch, *, lr=None, weight_decay=None):
        epoch.assert_step_allowed()
        lr = self.lr if lr is None else float(lr)
        wd = self.weight_decay if weight_decay is None else float(weight_decay)
        b1, b2 = self.betas
        changed = []
        for uid in self.grad_store.uids():
            digest = epoch.pin(uid)
            state = self.store.load(uid, expected_sha256=digest)
            grads = self.grad_store.get(uid)
            t = float(state.get("adam_step", torch.tensor(0.0))) + 1.0
            bc1 = 1.0 - b1 ** t
            bc2 = 1.0 - b2 ** t
            for nm in ("w1", "w3", "w2"):
                g = grads[nm].float()
                p = state[nm].float()
                m = state.get(nm + "_m", torch.zeros_like(p)).float()
                v = state.get(nm + "_v", torch.zeros_like(p)).float()
                if wd:
                    p.mul_(1.0 - lr * wd)
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                denom = v.sqrt().div_(math.sqrt(bc2)).add_(self.eps)
                p.addcdiv_(m, denom, value=-(lr / bc1))
                state[nm], state[nm + "_m"], state[nm + "_v"] = p, m, v
            state["adam_step"] = torch.tensor(t, dtype=torch.float32)
            self.store.write(uid, state)
            if self.cache is not None:
                self.cache.invalidate_uid(uid)
            changed.append(uid)
        self.steps += 1
        self.grad_store.clear()
        return changed
