from __future__ import annotations

"""Reference Gated-DeltaNet algebra used by RC10 qualification.

This module intentionally operates on *captured layer inputs* (k, v, beta,
decay), not raw tokens.  That isolates the affine recurrent algebra from the
deeper-model hidden-state drift that remains after arbitrary segment
reassembly.  Production model adapters can map their native tensor layout into
these primitives without changing the generic RC10 cache ABI.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .rc10 import AffineSegmentSummary, TransitionOrientation, summary_from_steps


def _expand_scalar(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Expand a scalar/per-head tensor so it broadcasts over a matrix state."""
    while x.ndim < ndim:
        x = x.unsqueeze(-1)
    return x


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.unsqueeze(-1) * b.unsqueeze(-2)


@dataclass(frozen=True)
class GDNInputs:
    """One recurrent token's model-derived inputs.

    Shapes are ``[..., d_k]`` for ``key`` and ``[..., d_v]`` for ``value``.
    ``beta`` and ``decay`` may be scalars or tensors broadcastable to the batch
    and head prefix. ``decay`` is the multiplicative recurrent decay (usually
    exp(g) in implementations that store g in log space).
    """

    key: torch.Tensor
    value: torch.Tensor
    beta: torch.Tensor
    decay: torch.Tensor

    def validate(self) -> None:
        if self.key.ndim < 1 or self.value.ndim < 1:
            raise ValueError("key/value must have a feature dimension")
        if self.key.shape[:-1] != self.value.shape[:-1]:
            raise ValueError("key/value batch+head dimensions must match")
        if not torch.isfinite(self.key).all() or not torch.isfinite(self.value).all():
            raise ValueError("non-finite GDN key/value")
        if not torch.isfinite(self.beta).all() or not torch.isfinite(self.decay).all():
            raise ValueError("non-finite GDN beta/decay")


def affine_step(
    inputs: GDNInputs,
    *,
    orientation: TransitionOrientation = TransitionOrientation.LEFT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(T, U)`` for one Gated DeltaNet token.

    LEFT layout uses state ``[..., d_k, d_v]`` and
    ``S' = T @ S + U``.

    RIGHT layout uses state ``[..., d_v, d_k]`` and
    ``S' = S @ T + U``.

    With ``gamma=decay`` the affine transition is
    ``T = gamma * (I - beta * k k^T)`` and the write is the appropriately
    oriented ``beta * k v^T`` (or its transpose).  This is exactly equivalent
    to the delta update when k/v/beta/gamma are held fixed.
    """
    inputs.validate()
    k, v = inputs.key, inputs.value
    d_k = k.shape[-1]
    prefix = k.shape[:-1]
    eye = torch.eye(d_k, dtype=k.dtype, device=k.device)
    if prefix:
        eye = eye.expand(*prefix, d_k, d_k)
    kk = _outer(k, k)
    beta_m = _expand_scalar(inputs.beta.to(dtype=k.dtype, device=k.device), kk.ndim)
    decay_m = _expand_scalar(inputs.decay.to(dtype=k.dtype, device=k.device), kk.ndim)
    transition = decay_m * (eye - beta_m * kk)

    beta_u = _expand_scalar(inputs.beta.to(dtype=k.dtype, device=k.device), k.ndim)
    if orientation == TransitionOrientation.LEFT:
        write = _outer(beta_u * k, v)
    else:
        write = _outer(v, beta_u * k)
    return transition, write


def delta_step(
    state: torch.Tensor,
    inputs: GDNInputs,
    *,
    orientation: TransitionOrientation = TransitionOrientation.LEFT,
) -> torch.Tensor:
    """Direct delta-rule implementation, independent of ``affine_step``.

    Keeping a direct form gives RC10 an algebra oracle: tests can compare the
    affine transition against the actual decay/predict/residual/write update
    rather than testing one algebra implementation against itself.
    """
    inputs.validate()
    k, v = inputs.key, inputs.value
    decay = _expand_scalar(inputs.decay.to(state), state.ndim)
    beta = _expand_scalar(inputs.beta.to(state), v.ndim)
    decayed = state * decay

    if orientation == TransitionOrientation.LEFT:
        # state [..., K, V], prediction [..., V]
        pred = torch.einsum("...k,...kv->...v", k.to(state), decayed)
        residual = v.to(state) - pred
        return decayed + _outer(k.to(state), beta * residual)

    # state [..., V, K]
    pred = torch.einsum("...vk,...k->...v", decayed, k.to(state))
    residual = v.to(state) - pred
    return decayed + _outer(beta * residual, k.to(state))


def scan(
    steps: Sequence[GDNInputs],
    initial_state: torch.Tensor,
    *,
    orientation: TransitionOrientation = TransitionOrientation.LEFT,
) -> torch.Tensor:
    state = initial_state
    for step in steps:
        state = delta_step(state, step, orientation=orientation)
    return state


def compile_summary(
    steps: Sequence[GDNInputs],
    *,
    orientation: TransitionOrientation = TransitionOrientation.LEFT,
    seam_width: int = 0,
    layer: int | None = None,
    segment_id: str = "",
) -> AffineSegmentSummary:
    """Compile only the reusable suffix after ``seam_width`` fresh tokens."""
    seam_width = int(seam_width)
    if seam_width < 0 or seam_width >= len(steps):
        raise ValueError("seam_width must leave at least one reusable token")
    suffix = steps[seam_width:]
    transitions, writes = zip(*(affine_step(s, orientation=orientation) for s in suffix))
    summary = summary_from_steps(
        transitions, writes, orientation=orientation, segment_id=segment_id
    )
    return AffineSegmentSummary(
        transition=summary.transition,
        zero_state=summary.zero_state,
        orientation=summary.orientation,
        seam_width=seam_width,
        layer=layer,
        segment_id=segment_id,
    )


def zero_state_for(step: GDNInputs, *, orientation: TransitionOrientation) -> torch.Tensor:
    prefix = step.key.shape[:-1]
    dk, dv = step.key.shape[-1], step.value.shape[-1]
    shape = (*prefix, dk, dv) if orientation == TransitionOrientation.LEFT else (*prefix, dv, dk)
    return torch.zeros(shape, dtype=step.key.dtype, device=step.key.device)
