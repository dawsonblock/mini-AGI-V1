from __future__ import annotations

"""Reference causal-convolution boundary state for RC10 seam reconstruction."""

from dataclasses import dataclass
from typing import Sequence
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ConvBoundaryState:
    """Trailing pre-convolution inputs needed to continue a causal depthwise conv.

    ``history`` shape is ``[..., channels, kernel_size-1]``.
    """

    history: torch.Tensor
    kernel_size: int

    def validate(self) -> None:
        if self.kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        expected = max(0, self.kernel_size - 1)
        if self.history.shape[-1] != expected:
            raise ValueError("history length does not match kernel_size")


def capture_boundary(inputs: torch.Tensor, kernel_size: int) -> ConvBoundaryState:
    """Capture the exact trailing raw inputs before a future segment boundary.

    ``inputs`` is ``[..., time, channels]``.
    """
    if inputs.ndim < 2:
        raise ValueError("expected [..., time, channels]")
    k = int(kernel_size)
    if k < 1:
        raise ValueError("kernel_size must be positive")
    need = k - 1
    if need == 0:
        hist = inputs.new_zeros((*inputs.shape[:-2], inputs.shape[-1], 0))
    else:
        tail = inputs[..., -need:, :]
        if tail.shape[-2] < need:
            pad = inputs.new_zeros((*inputs.shape[:-2], need - tail.shape[-2], inputs.shape[-1]))
            tail = torch.cat([pad, tail], dim=-2)
        hist = tail.transpose(-1, -2).contiguous()
    return ConvBoundaryState(hist, k)


def causal_depthwise_conv(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    boundary: ConvBoundaryState | None = None,
) -> tuple[torch.Tensor, ConvBoundaryState]:
    """Reference depthwise causal convolution with explicit continuation state.

    inputs: ``[..., time, channels]``
    weight: ``[channels, kernel_size]``
    returns output in the same time/channel layout plus the new boundary state.
    """
    if inputs.ndim < 2 or weight.ndim != 2:
        raise ValueError("invalid convolution tensor ranks")
    channels, kernel_size = weight.shape
    if inputs.shape[-1] != channels:
        raise ValueError("channel mismatch")
    flat_prefix = inputs.shape[:-2]
    time = inputs.shape[-2]
    x = inputs.reshape(-1, time, channels).transpose(1, 2)
    if boundary is None:
        hist = x.new_zeros((x.shape[0], channels, kernel_size - 1))
    else:
        boundary.validate()
        if boundary.kernel_size != kernel_size:
            raise ValueError("convolution kernel mismatch")
        hist = boundary.history.reshape(-1, channels, kernel_size - 1).to(x)
        if hist.shape[0] not in (1, x.shape[0]):
            raise ValueError("boundary batch mismatch")
        if hist.shape[0] == 1 and x.shape[0] != 1:
            hist = hist.expand(x.shape[0], -1, -1)
    cat = torch.cat([hist, x], dim=-1)
    # grouped conv with no additional padding because explicit history is present
    y = F.conv1d(cat, weight.to(x).unsqueeze(1), None if bias is None else bias.to(x), groups=channels)
    new_hist = cat[..., -(kernel_size - 1):] if kernel_size > 1 else cat[..., :0]
    out = y.transpose(1, 2).reshape(*flat_prefix, time, channels)
    state = ConvBoundaryState(new_hist.reshape(*flat_prefix, channels, max(0, kernel_size - 1)).contiguous(), kernel_size)
    return out, state
