from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AffineSummary:
    """Summary of an affine recurrent block: S_out = T @ S_in + Z."""

    T: np.ndarray
    Z: np.ndarray

    def __post_init__(self) -> None:
        self.T = np.asarray(self.T)
        self.Z = np.asarray(self.Z)
        if self.T.ndim < 2 or self.T.shape[-1] != self.T.shape[-2]:
            raise ValueError("T must end in a square matrix")
        if self.Z.shape[-2:] != self.T.shape[-2:]:
            # Matrix-state convention: T left-multiplies S; Z has same state shape.
            # For vector-state users, supply Nx1 arrays.
            if not (self.Z.ndim >= 1 and self.Z.shape[-1] == self.T.shape[-1]):
                raise ValueError("Z shape is incompatible with T")

    def apply(self, state: np.ndarray) -> np.ndarray:
        return self.T @ state + self.Z

    def then(self, nxt: "AffineSummary") -> "AffineSummary":
        """Return summary for this block followed by nxt."""
        T = nxt.T @ self.T
        Z = nxt.T @ self.Z + nxt.Z
        return AffineSummary(T=T, Z=Z)

    @staticmethod
    def identity(n: int, dtype=np.float64) -> "AffineSummary":
        return AffineSummary(T=np.eye(n, dtype=dtype), Z=np.zeros((n, n), dtype=dtype))


def compose_sequence(summaries: list[AffineSummary]) -> AffineSummary:
    if not summaries:
        raise ValueError("At least one summary is required")
    out = summaries[0]
    for nxt in summaries[1:]:
        out = out.then(nxt)
    return out
