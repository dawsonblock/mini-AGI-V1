"""Context-asymmetric speculative-style steering primitives.

This is an inference-engine building block, not a claim of exact speculative
sampling.  A lightweight drafter may inspect a richer/full context while the
expensive verifier operates on a compressed view.  The same drafter is evaluated
on both views; their logit difference isolates a context-induced signal that can
steer the compressed verifier.

The method is calibrated for greedy/controlled emission.  Because logits are
modified, it does *not* preserve the verifier's original target distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


def js_divergence_from_logits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence per position, using natural logs."""
    pa = F.softmax(a.float(), -1).clamp_min(1e-12)
    pb = F.softmax(b.float(), -1).clamp_min(1e-12)
    m = 0.5 * (pa + pb)
    return 0.5 * ((pa * (pa.log() - m.log())).sum(-1)
                  + (pb * (pb.log() - m.log())).sum(-1))


def context_gain(full_drafter_logits: torch.Tensor,
                 compressed_drafter_logits: torch.Tensor) -> torch.Tensor:
    if full_drafter_logits.shape != compressed_drafter_logits.shape:
        raise ValueError("drafter logit shapes must match")
    return full_drafter_logits - compressed_drafter_logits


def effective_threshold(gamma: float, divergence: torch.Tensor) -> torch.Tensor:
    g = float(gamma)
    if not 0 < g <= 1:
        raise ValueError("gamma must be in (0,1]")
    return g * torch.exp(-divergence.float())


@dataclass(frozen=True)
class AsymSpecDecision:
    accepted: torch.Tensor
    emitted: torch.Tensor
    divergence: torch.Tensor
    threshold: torch.Tensor


def asym_spec_decision(verifier_logits: torch.Tensor,
                       full_drafter_logits: torch.Tensor,
                       compressed_drafter_logits: torch.Tensor,
                       draft_tokens: torch.Tensor, *, gamma: float = 0.5,
                       beta: float = 1.0) -> AsymSpecDecision:
    """Evaluate one vectorized speculation position/batch.

    Inputs have shape ``[..., vocab]`` and ``draft_tokens`` has the leading
    shape. Accepted positions emit the draft token; rejected positions emit the
    argmax of verifier + beta*(full_drafter-compressed_drafter).
    """
    if verifier_logits.shape != full_drafter_logits.shape or verifier_logits.shape != compressed_drafter_logits.shape:
        raise ValueError("all logit tensors must have identical shape")
    if tuple(draft_tokens.shape) != tuple(verifier_logits.shape[:-1]):
        raise ValueError("draft token shape must match logit leading dimensions")
    if not 0 <= float(beta) <= 2.0:
        raise ValueError("beta must be in [0,2]")
    delta = context_gain(full_drafter_logits, compressed_drafter_logits)
    div = js_divergence_from_logits(full_drafter_logits, compressed_drafter_logits)
    thr = effective_threshold(gamma, div)
    vp = F.softmax(verifier_logits.float(), -1)
    bp = F.softmax(compressed_drafter_logits.float(), -1)
    d = draft_tokens.long().unsqueeze(-1)
    vpd = vp.gather(-1, d).squeeze(-1)
    bpd = bp.gather(-1, d).squeeze(-1)
    accepted = vpd > thr * bpd
    fused = (verifier_logits.float() + float(beta) * delta.float()).argmax(-1)
    emitted = torch.where(accepted, draft_tokens.long(), fused)
    return AsymSpecDecision(accepted=accepted, emitted=emitted,
                            divergence=div, threshold=thr)
