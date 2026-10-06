from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from kvcontinual.execution.types import TransitionOrientation


@dataclass
class AffineSummary:
    """Affine recurrent summary with an explicit operator orientation.

    LEFT_MULTIPLY:  S_out = T @ S_in + Z
    RIGHT_MULTIPLY: S_out = S_in @ T + Z

    The orientation is part of the cache ABI because silently transposing a GDN
    state layout can produce numerically plausible but invalid results.
    """

    T: np.ndarray
    Z: np.ndarray
    orientation: TransitionOrientation = TransitionOrientation.LEFT_MULTIPLY

    def __post_init__(self) -> None:
        self.T = np.asarray(self.T)
        self.Z = np.asarray(self.Z)
        if self.T.ndim < 2 or self.T.shape[-1] != self.T.shape[-2]:
            raise ValueError("T must end in a square matrix")
        n = self.T.shape[-1]
        if self.Z.ndim == 0 or (self.Z.shape[-1] != n and (self.Z.ndim < 2 or self.Z.shape[-2] != n)):
            raise ValueError("Z shape is incompatible with T")

    def apply(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state)
        if self.orientation == TransitionOrientation.LEFT_MULTIPLY:
            return self.T @ state + self.Z
        return state @ self.T + self.Z

    def then(self, nxt: "AffineSummary") -> "AffineSummary":
        """Return the summary for this segment followed by ``nxt``."""
        if self.orientation != nxt.orientation:
            raise ValueError("Cannot compose summaries with different orientations")
        if self.orientation == TransitionOrientation.LEFT_MULTIPLY:
            T = nxt.T @ self.T
            Z = nxt.T @ self.Z + nxt.Z
        else:
            T = self.T @ nxt.T
            Z = self.Z @ nxt.T + nxt.Z
        return AffineSummary(T=T, Z=Z, orientation=self.orientation)

    @staticmethod
    def identity(n: int, dtype=np.float64, orientation: TransitionOrientation = TransitionOrientation.LEFT_MULTIPLY) -> "AffineSummary":
        return AffineSummary(T=np.eye(n, dtype=dtype), Z=np.zeros((n, n), dtype=dtype), orientation=orientation)


def compose_sequence(summaries: list[AffineSummary]) -> AffineSummary:
    if not summaries:
        raise ValueError("At least one summary is required")
    out = summaries[0]
    for nxt in summaries[1:]:
        out = out.then(nxt)
    return out
