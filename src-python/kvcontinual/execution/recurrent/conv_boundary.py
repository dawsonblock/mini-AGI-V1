from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ConvBoundaryState:
    """Depthwise causal-convolution history immediately before a seam.

    history shape is [channels, kernel_size - 1], oldest sample first. This
    matches the bundled RC9.2 recurrent-convolution state convention.
    """
    history: np.ndarray
    kernel_size: int

    def validate(self) -> None:
        h = np.asarray(self.history)
        if self.kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        expected = max(0, self.kernel_size - 1)
        if h.ndim != 2 or h.shape[1] != expected:
            raise ValueError("history must be [channels, kernel_size-1]")


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def causal_depthwise_conv_seam(
    projected: np.ndarray,
    weights: np.ndarray,
    incoming: ConvBoundaryState,
    *,
    bias: np.ndarray | None = None,
    apply_silu: bool = True,
) -> tuple[np.ndarray, ConvBoundaryState]:
    """Replay a causal depthwise convolution under a new predecessor.

    projected: [tokens, channels]
    weights: [channels, kernel_size], oldest-history weight first and current
             token weight last, matching RC9.2's native kernel.
    """
    x = np.asarray(projected)
    w = np.asarray(weights)
    incoming.validate()
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError("projected and weights must be rank-2")
    t, c = x.shape
    if w.shape[0] != c or w.shape[1] != incoming.kernel_size:
        raise ValueError("convolution dimensions do not match boundary state")
    if incoming.history.shape[0] != c:
        raise ValueError("history channel count does not match projected input")
    if bias is not None and np.asarray(bias).shape != (c,):
        raise ValueError("bias must have shape [channels]")

    k = incoming.kernel_size
    history = np.asarray(incoming.history, dtype=np.result_type(x.dtype, w.dtype)).copy()
    out = np.empty((t, c), dtype=np.result_type(x.dtype, w.dtype))
    b = None if bias is None else np.asarray(bias)
    for i in range(t):
        acc = w[:, -1] * x[i]
        if k > 1:
            acc = acc + np.sum(w[:, :-1] * history, axis=1)
        if b is not None:
            acc = acc + b
        out[i] = _silu(acc) if apply_silu else acc
        if k > 1:
            if k > 2:
                history[:, :-1] = history[:, 1:]
            history[:, -1] = x[i]
    return out, ConvBoundaryState(history=history, kernel_size=k)
