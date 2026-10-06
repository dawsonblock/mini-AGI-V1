"""AdamW with independent clocks for axis-0 expert/router rows.

PyTorch AdamW keeps one scalar ``step`` per Parameter.  That is correct for a
normal dense tensor and wrong for parameters whose rows are independent things
that are born, paged, pruned, or reactivated at different times.  mini-AGI has
three such cases:

* paged expert slot tensors (one resident expert per row),
* router matrices (one expert-selection row per expert), and
* the per-expert gate vector.

``PagedAdamW`` preserves ordinary AdamW semantics for every unmarked parameter.
For a Parameter marked ``_minagi_rowwise = True`` it keeps ``row_step`` and
updates only rows with a non-zero gradient.  Weight decay is likewise applied
only to rows that actually participated in the step.  For paged expert slots,
``PagedPool`` swaps each resident expert's moments *and row clock* together with
its weights, so an expert resumes exactly where its own optimiser history left
off rather than inheriting the slot's history.

The class intentionally implements only the AdamW features this repository
uses: dense gradients, no AMSGrad, no capturable/differentiable optimiser.
Keeping this implementation small makes its state semantics auditable.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import Optimizer


def mark_rowwise(param: torch.nn.Parameter, *, paged_expert: bool = False) -> torch.nn.Parameter:
    """Mark ``param`` as a collection of independent axis-0 optimiser rows."""
    setattr(param, "_minagi_rowwise", True)
    if paged_expert:
        setattr(param, "_minagi_paged_expert", True)
    return param


def is_rowwise(param: torch.nn.Parameter) -> bool:
    return bool(getattr(param, "_minagi_rowwise", False))


def is_paged_expert_param(param: torch.nn.Parameter) -> bool:
    return bool(getattr(param, "_minagi_paged_expert", False))


class PagedAdamW(Optimizer):
    """AdamW with per-row clocks for marked parameters.

    Unmarked parameters use a scalar ``step`` exactly as normal AdamW does.
    Marked parameters keep ``row_step`` with shape ``(parameter.shape[0],)``.
    A row is active iff any gradient element in that row is non-zero.  Inactive
    rows receive neither moment decay nor decoupled weight decay.
    """

    def __init__(self, params: Iterable, lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2):
        if lr < 0:
            raise ValueError(f"invalid lr: {lr}")
        if eps < 0:
            raise ValueError(f"invalid eps: {eps}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"invalid betas: {betas}")
        if weight_decay < 0:
            raise ValueError(f"invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        # PagedPool's step-pre-hook needs moment tensors to exist before the
        # first call to step(), because it fills them with each expert's own
        # persisted history.  Initialise only rowwise state eagerly; dense
        # state remains lazy like torch.optim.AdamW.
        for group in self.param_groups:
            for p in group["params"]:
                if is_rowwise(p):
                    self.ensure_state(p)

    @torch.no_grad()
    def ensure_state(self, p: torch.nn.Parameter):
        st = self.state[p]
        if "exp_avg" not in st:
            st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        if is_rowwise(p):
            if p.dim() == 0:
                raise ValueError("rowwise parameter must have at least one dimension")
            rs = st.get("row_step")
            if rs is None or tuple(rs.shape) != (p.shape[0],):
                st["row_step"] = torch.zeros(p.shape[0], dtype=torch.float32,
                                             device=p.device)
        elif "step" not in st:
            st["step"] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
        return st

    @staticmethod
    def _active_rows(grad: torch.Tensor) -> torch.Tensor:
        if grad.dim() == 1:
            return grad != 0
        return grad.reshape(grad.shape[0], -1).ne(0).any(dim=1)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group.get("weight_decay", 0.0))
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("PagedAdamW does not support sparse gradients")
                st = self.ensure_state(p)
                m = st["exp_avg"]
                v = st["exp_avg_sq"]

                if is_rowwise(p):
                    active = self._active_rows(grad)
                    if not bool(active.any()):
                        continue
                    g = grad[active]
                    mm = m[active]
                    vv = v[active]
                    pp = p.data[active]
                    steps = st["row_step"]
                    steps[active] += 1.0
                    tt = steps[active]

                    if wd:
                        pp = pp * (1.0 - lr * wd)
                    mm = mm.mul(beta1).add(g, alpha=1.0 - beta1)
                    vv = vv.mul(beta2).addcmul(g, g, value=1.0 - beta2)

                    bc1 = 1.0 - torch.pow(torch.full_like(tt, beta1), tt)
                    bc2 = 1.0 - torch.pow(torch.full_like(tt, beta2), tt)
                    shape = (tt.shape[0],) + (1,) * (p.dim() - 1)
                    step_size = (lr / bc1).view(shape)
                    denom = (vv.sqrt() / bc2.sqrt().view(shape)).add_(eps)
                    pp = pp - step_size * (mm / denom)

                    p.data[active] = pp
                    m[active] = mm
                    v[active] = vv
                else:
                    st["step"] += 1.0
                    t = float(st["step"])
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    bc1 = 1.0 - beta1 ** t
                    bc2 = 1.0 - beta2 ** t
                    step_size = lr / bc1
                    denom = v.sqrt().div_(math.sqrt(bc2)).add_(eps)
                    p.addcdiv_(m, denom, value=-step_size)
        return loss


def build_adamw(groups, *, lr: float, betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.0, **_ignored) -> PagedAdamW:
    """Repository-wide optimiser factory.

    ``fused`` and other torch.AdamW-only hints may be passed by older callers;
    they are intentionally ignored because rowwise clocks require this explicit
    implementation.
    """
    return PagedAdamW(groups, lr=lr, betas=betas, eps=eps,
                      weight_decay=weight_decay)
