from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from kvcontinual.execution.recurrent.affine import AffineSummary
from kvcontinual.execution.types import TransitionOrientation


@dataclass(frozen=True)
class ArrayBackendInfo:
    name: str
    accelerated: bool


class MacAffineExecutor:
    """Apple-Silicon-friendly affine executor.

    Uses MLX when installed; otherwise uses NumPy. This accelerates the RC10
    affine T/Z reference primitive only. It is deliberately not presented as a
    full Qwen Gated-DeltaNet/HYPIC kernel.
    """

    def __init__(self, prefer_mlx: bool = True):
        self.mx = None
        if prefer_mlx:
            try:
                import mlx.core as mx
                self.mx = mx
            except Exception:
                self.mx = None

    @property
    def info(self) -> ArrayBackendInfo:
        return ArrayBackendInfo("mlx" if self.mx is not None else "numpy", self.mx is not None)

    def apply(self, summary: AffineSummary, state: np.ndarray) -> np.ndarray:
        if self.mx is None:
            return summary.apply(state)
        mx = self.mx
        T = mx.array(np.asarray(summary.T))
        Z = mx.array(np.asarray(summary.Z))
        S = mx.array(np.asarray(state))
        if summary.orientation == TransitionOrientation.LEFT_MULTIPLY:
            out = T @ S + Z
        else:
            out = S @ T + Z
        mx.eval(out)
        return np.asarray(out)

    def compose(self, first: AffineSummary, second: AffineSummary) -> AffineSummary:
        if first.orientation != second.orientation:
            raise ValueError("Cannot compose summaries with different orientations")
        if self.mx is None:
            return first.then(second)
        mx = self.mx
        aT, aZ = mx.array(first.T), mx.array(first.Z)
        bT, bZ = mx.array(second.T), mx.array(second.Z)
        if first.orientation == TransitionOrientation.LEFT_MULTIPLY:
            T = bT @ aT
            Z = bT @ aZ + bZ
        else:
            T = aT @ bT
            Z = aZ @ bT + bZ
        mx.eval(T, Z)
        return AffineSummary(np.asarray(T), np.asarray(Z), first.orientation)
