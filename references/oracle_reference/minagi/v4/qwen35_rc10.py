from __future__ import annotations

"""Qwen3.5-style Gated DeltaNet integration contract for RC10.

This is deliberately a capture/compile adapter, not a vendored Transformers
implementation.  It accepts the recurrent quantities produced by a Qwen3.5-like
layer and emits RC10 summaries plus causal-convolution boundary state.  That
keeps model-library versioning outside the canonical RC10 algebra.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence
import torch

from .conv_boundary import ConvBoundaryState, capture_boundary
from .gdn_reference import GDNInputs, compile_summary
from .rc10 import AffineSegmentSummary, RecurrentVariant, TransitionOrientation


@dataclass(frozen=True)
class CapturedGDNLayer:
    layer: int
    keys: torch.Tensor      # [time, heads, d_k] or [..., time, heads, d_k]
    values: torch.Tensor    # same prefix, d_v
    beta: torch.Tensor      # broadcastable per token/head
    decay: torch.Tensor     # multiplicative decay, broadcastable per token/head
    conv_inputs: torch.Tensor | None = None  # [..., time, channels]
    conv_kernel_size: int = 1

    def time_dim(self) -> int:
        return -3  # immediately before head and feature dimensions


@dataclass(frozen=True)
class CompiledQwen35Layer:
    layer: int
    variants: tuple[AffineSegmentSummary, ...]
    conv_boundary: ConvBoundaryState | None


def _time_slice(x: torch.Tensor, t: int, time_dim: int) -> torch.Tensor:
    return x.select(time_dim % x.ndim, t)


def _scalar_slice(x: torch.Tensor, t: int, time_len: int) -> torch.Tensor:
    # Prefer a time axis equal to time_len.  This supports common [T,H],
    # [B,T,H], [T,H,1], and [B,T,H,1] forms without imposing one HF version.
    axes = [i for i, size in enumerate(x.shape) if size == time_len]
    if not axes:
        return x
    return x.select(axes[0], t)


def captured_steps(capture: CapturedGDNLayer) -> list[GDNInputs]:
    td = capture.time_dim() % capture.keys.ndim
    tlen = capture.keys.shape[td]
    if capture.values.shape[td] != tlen:
        raise ValueError("key/value time mismatch")
    out: list[GDNInputs] = []
    for t in range(tlen):
        out.append(GDNInputs(
            _time_slice(capture.keys, t, td),
            _time_slice(capture.values, t, td),
            _scalar_slice(capture.beta, t, tlen),
            _scalar_slice(capture.decay, t, tlen),
        ))
    return out


def compile_layer(
    capture: CapturedGDNLayer,
    *,
    seams: Sequence[int] = (8,),
    orientation: TransitionOrientation = TransitionOrientation.LEFT,
    segment_id: str = "",
) -> CompiledQwen35Layer:
    steps = captured_steps(capture)
    variants = []
    for seam in sorted(set(map(int, seams))):
        if seam < len(steps):
            variants.append(compile_summary(
                steps,
                orientation=orientation,
                seam_width=seam,
                layer=capture.layer,
                segment_id=segment_id,
            ))
    if not variants:
        raise ValueError("no requested seam leaves a reusable suffix")
    conv = None
    if capture.conv_inputs is not None:
        conv = capture_boundary(capture.conv_inputs, capture.conv_kernel_size)
    return CompiledQwen35Layer(capture.layer, tuple(variants), conv)


def compile_recurrent_variant(
    captures: Sequence[CapturedGDNLayer],
    *,
    seam_width: int,
    orientation: TransitionOrientation = TransitionOrientation.LEFT,
    segment_id: str = "",
) -> RecurrentVariant:
    summaries = []
    for capture in captures:
        compiled = compile_layer(
            capture,
            seams=(int(seam_width),),
            orientation=orientation,
            segment_id=segment_id,
        )
        summaries.append(compiled.variants[0])
    return RecurrentVariant(int(seam_width), tuple(summaries))


def slice_captured_layer(capture: CapturedGDNLayer, start: int, end: int) -> CapturedGDNLayer:
    """Return a token-range view of a captured GDN layer.

    The helper keeps time-axis handling explicit because Transformers revisions
    have used several beta/decay layouts.
    """
    start, end = int(start), int(end)
    td = capture.time_dim() % capture.keys.ndim
    tlen = capture.keys.shape[td]
    if not (0 <= start < end <= tlen):
        raise ValueError("invalid captured-layer token slice")

    def slice_axis(x: torch.Tensor, axis: int, a: int, b: int) -> torch.Tensor:
        idx = [slice(None)] * x.ndim
        idx[axis] = slice(a, b)
        return x[tuple(idx)]

    keys = slice_axis(capture.keys, td, start, end)
    values = slice_axis(capture.values, td, start, end)

    def slice_scalar(x: torch.Tensor) -> torch.Tensor:
        axes = [i for i, size in enumerate(x.shape) if size == tlen]
        return x if not axes else slice_axis(x, axes[0], start, end)

    beta = slice_scalar(capture.beta)
    decay = slice_scalar(capture.decay)
    conv_inputs = None
    if capture.conv_inputs is not None:
        # capture conv inputs are [..., time, channels]
        conv_inputs = slice_axis(capture.conv_inputs, capture.conv_inputs.ndim - 2, start, end)
    return CapturedGDNLayer(
        int(capture.layer), keys, values, beta, decay,
        conv_inputs, int(capture.conv_kernel_size),
    )
