"""Safe segment-causal training runtime for paged expert-card evolution.

The ordinary high-throughput training path admits one resident expert card for
a multi-token forward. Autoregressive generation can update that card after
each new token. A single backward graph cannot safely swap the resident slot
parameters midway through the graph: the same slot tensor would represent
multiple expert identities before backward.

This runtime chooses the explicit alternative: make every routing segment its
own forward/backward/optimizer transaction and detach the KV cache between
segments. The expert card may therefore evolve between segments without slot
identity aliasing. ``segment_tokens=1`` gives token-by-token causal admission;
larger values trade routing fidelity for throughput.

The price is equally explicit: gradients do not cross segment boundaries, and
weights may update between segments. This is truncated-BPTT online training,
not a claim that exact card evolution has been embedded inside one large
autograd graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import math
import torch

from .stream import detach_caches


@dataclass
class SegmentedStepReport:
    tokens: int
    segments: int
    mean_loss: float
    max_grad_norm: float
    segment_tokens: int
    exact_token_admission: bool


@contextmanager
def _causal_segment_admission(model):
    pool = getattr(model, "pool", None)
    if pool is None or not hasattr(pool, "admission_mode"):
        yield
        return
    old_mode = pool.admission_mode
    old_prefix = getattr(pool, "admission_prefix_tokens", 1)
    # The first state in each segment is visible before all targets in that
    # segment and therefore cannot leak a future suffix into admission.
    pool.admission_mode = "prefix_causal"
    pool.admission_prefix_tokens = 1
    try:
        yield
    finally:
        pool.admission_mode = old_mode
        pool.admission_prefix_tokens = old_prefix


class SegmentedCausalStepper:
    def __init__(self, model, optimizer, *, segment_tokens=1, clip=1.0,
                 aux_weight=0.0, detach_between=True):
        self.model = model
        self.optimizer = optimizer
        self.segment_tokens = max(1, int(segment_tokens))
        self.clip = float(clip)
        self.aux_weight = float(aux_weight)
        if not detach_between:
            raise ValueError(
                "paged expert-card swaps require detached segment boundaries; "
                "use the ordinary forward for a single autograd graph")
        self.detach_between = True

    def step(self, x: torch.Tensor, y: torch.Tensor) -> SegmentedStepReport:
        if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
            raise ValueError("x and y must be equal-shaped [batch, time] tensors")
        if x.shape[1] <= 0:
            raise ValueError("segmented step requires at least one token")
        if x.shape[0] != 1:
            raise ValueError("segmented paged training currently requires batch size 1")
        if not hasattr(self.model, "empty_caches"):
            raise TypeError("model must provide empty_caches()")

        self.model.train()
        caches = self.model.empty_caches()
        losses = []
        grad_norms = []
        offset = 0
        T = int(x.shape[1])
        with _causal_segment_admission(self.model):
            while offset < T:
                end = min(T, offset + self.segment_tokens)
                xs = x[:, offset:end]
                ys = y[:, offset:end]
                self.optimizer.zero_grad(set_to_none=True)
                _logits, loss = self.model(xs, ys, caches=caches,
                                           pos_offset=offset)
                if self.aux_weight and getattr(self.model, "pool", None) is not None:
                    loss = loss + self.aux_weight * self.model.pool_aux()
                if hasattr(self.model, "pool_balance"):
                    loss = loss + self.model.pool_balance()
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss in segmented causal training")
                loss.backward()
                if self.clip > 0:
                    gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    grad_norms.append(float(gn))
                else:
                    grad_norms.append(0.0)
                self.optimizer.step()
                losses.append((float(loss.detach()), end - offset))
                detach_caches(caches)
                offset = end
        weighted = sum(v * n for v, n in losses) / max(1, sum(n for _, n in losses))
        if not math.isfinite(weighted):
            raise RuntimeError("non-finite aggregate segmented loss")
        return SegmentedStepReport(
            tokens=T,
            segments=len(losses),
            mean_loss=float(weighted),
            max_grad_norm=max(grad_norms) if grad_norms else 0.0,
            segment_tokens=self.segment_tokens,
            exact_token_admission=(self.segment_tokens == 1),
        )
