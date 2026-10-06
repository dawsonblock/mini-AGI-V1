from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DenseAffineStorageEstimate:
    heads: int
    state_dim: int
    layers: int
    bytes_per_scalar: int

    @property
    def bytes_per_segment(self) -> int:
        # T and Z, each state_dim x state_dim, per head and recurrent layer.
        return 2 * self.layers * self.heads * self.state_dim * self.state_dim * self.bytes_per_scalar

    @property
    def mib_per_segment(self) -> float:
        return self.bytes_per_segment / (1024.0 * 1024.0)


def dense_affine_storage_estimate(*, heads: int, state_dim: int, layers: int, dtype: str = "fp16") -> DenseAffineStorageEstimate:
    widths = {"fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8}
    if dtype not in widths:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if min(heads, state_dim, layers) <= 0:
        raise ValueError("heads/state_dim/layers must be positive")
    return DenseAffineStorageEstimate(heads, state_dim, layers, widths[dtype])


def production_storage_allowed(estimate: DenseAffineStorageEstimate, *, max_mib_per_segment: float = 16.0) -> bool:
    """Fail closed on dense T/Z storage beyond a configured production budget.

    RC11 keeps dense affine summaries as correctness oracles. Production reuse
    should move to a validated structured/factored operator before lifting this
    budget rather than silently consuming unbounded memory.
    """
    return estimate.mib_per_segment <= max_mib_per_segment
