from __future__ import annotations

import numpy as np

from kvcontinual.execution.recurrent.affine import AffineSummary, compose_sequence
from kvcontinual.execution.types import TransitionOrientation


def gdn_token_affine(k: np.ndarray, v: np.ndarray, decay: float, beta: float) -> AffineSummary:
    """Reference affine form for the RC9.2 column-state GDN CUDA layout.

    For each state column the native kernel performs:
        s' = g*s + k * beta * (v_col - g * k^T s)
    which is equivalent to:
        S' = g(I - beta*k*k^T) S + beta*k*v^T

    This routine is intentionally CPU/NumPy and exists as a correctness oracle,
    not as a production kernel.
    """
    k = np.asarray(k, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if k.shape != v.shape:
        raise ValueError("reference implementation requires d_k == d_v")
    T = float(decay) * (np.eye(k.size, dtype=np.float64) - float(beta) * np.outer(k, k))
    Z = float(beta) * np.outer(k, v)
    return AffineSummary(T, Z, TransitionOrientation.LEFT_MULTIPLY)


def capture_segment_tail(k: np.ndarray, v: np.ndarray, decay: np.ndarray, beta: np.ndarray, seam_width: int = 8) -> AffineSummary:
    k = np.asarray(k); v = np.asarray(v); decay = np.asarray(decay); beta = np.asarray(beta)
    if k.ndim != 2 or v.shape != k.shape or decay.shape[0] != k.shape[0] or beta.shape[0] != k.shape[0]:
        raise ValueError("expected k/v [T,D], decay/beta [T]")
    if not 0 <= seam_width < k.shape[0]:
        raise ValueError("seam_width must leave a non-empty cached tail")
    parts = [gdn_token_affine(k[t], v[t], decay[t], beta[t]) for t in range(seam_width, k.shape[0])]
    return compose_sequence(parts)


def direct_recurrence(state: np.ndarray, k: np.ndarray, v: np.ndarray, decay: np.ndarray, beta: np.ndarray) -> np.ndarray:
    out = np.asarray(state, dtype=np.float64).copy()
    for t in range(len(k)):
        out = gdn_token_affine(k[t], v[t], decay[t], beta[t]).apply(out)
    return out
