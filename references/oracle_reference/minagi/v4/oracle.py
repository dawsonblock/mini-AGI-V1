from __future__ import annotations

"""Exact-oracle comparison metrics for RC10 qualification."""

from dataclasses import dataclass, asdict
from typing import Sequence
import math
import torch


@dataclass(frozen=True)
class TensorDivergence:
    relative_l2: float
    angle_degrees: float
    max_abs: float


@dataclass(frozen=True)
class OracleComparison:
    state: TensorDivergence
    logit_kl: float | None = None
    top1_agree: bool | None = None


def tensor_divergence(approx: torch.Tensor, exact: torch.Tensor, eps: float = 1e-12) -> TensorDivergence:
    a = approx.detach().float().reshape(-1)
    e = exact.detach().float().reshape(-1)
    if a.shape != e.shape:
        raise ValueError("shape mismatch")
    diff = a - e
    rel = float(torch.linalg.vector_norm(diff) / (torch.linalg.vector_norm(e) + eps))
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(e) + eps
    cos = float(torch.clamp(torch.dot(a, e) / denom, -1.0, 1.0))
    ang = math.degrees(math.acos(cos))
    return TensorDivergence(rel, ang, float(diff.abs().max()) if diff.numel() else 0.0)


def compare(
    approx_state: torch.Tensor,
    exact_state: torch.Tensor,
    *,
    approx_logits: torch.Tensor | None = None,
    exact_logits: torch.Tensor | None = None,
) -> OracleComparison:
    state = tensor_divergence(approx_state, exact_state)
    if approx_logits is None or exact_logits is None:
        return OracleComparison(state)
    if approx_logits.shape != exact_logits.shape:
        raise ValueError("logit shape mismatch")
    p = torch.log_softmax(approx_logits.float(), dim=-1)
    q = torch.log_softmax(exact_logits.float(), dim=-1)
    qprob = q.exp()
    kl = float(torch.sum(qprob * (q - p), dim=-1).mean())
    agree = bool(torch.equal(approx_logits.argmax(-1), exact_logits.argmax(-1)))
    return OracleComparison(state, kl, agree)
