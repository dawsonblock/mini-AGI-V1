"""Custom autograd boundary for virtual expert paging."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _expert_forward(x, w1, w3, w2):
    return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)


class PagedExpertFunction(torch.autograd.Function):
    """Expert weights are external state; only x participates in PyTorch params.

    Forward loads ``UID@digest`` into a cache buffer. Backward verifies and
    reloads the same version, recomputes the local expert, returns dX into the
    full shared graph, and accumulates dW by UID into ExpertGradStore.
    """
    @staticmethod
    def forward(ctx, x, uid: int, epoch, cache, grad_store):
        uid = int(uid)
        digest = epoch.pin(uid)
        w = cache.get(uid, digest)
        y = _expert_forward(x, w["w1"], w["w3"], w["w2"])
        ctx.uid = uid
        ctx.epoch = epoch
        ctx.cache = cache
        ctx.grad_store = grad_store
        ctx.digest = digest
        ctx.save_for_backward(x.detach())
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (x_saved,) = ctx.saved_tensors
        # pin() both verifies no mutation and documents the dependency.
        digest = ctx.epoch.pin(ctx.uid)
        if digest != ctx.digest:
            raise RuntimeError("expert version changed between forward and backward")
        w = ctx.cache.get(ctx.uid, digest)
        with torch.enable_grad():
            x = x_saved.detach().requires_grad_(True)
            w1 = w["w1"].detach().requires_grad_(True)
            w3 = w["w3"].detach().requires_grad_(True)
            w2 = w["w2"].detach().requires_grad_(True)
            y = _expert_forward(x, w1, w3, w2)
            dx, dw1, dw3, dw2 = torch.autograd.grad(
                y, (x, w1, w3, w2), grad_y, retain_graph=False,
                create_graph=False, allow_unused=False)
        ctx.grad_store.accumulate(ctx.uid, w1=dw1, w3=dw3, w2=dw2)
        return dx, None, None, None, None


def paged_expert(x, uid: int, epoch, cache, grad_store):
    return PagedExpertFunction.apply(x, int(uid), epoch, cache, grad_store)
