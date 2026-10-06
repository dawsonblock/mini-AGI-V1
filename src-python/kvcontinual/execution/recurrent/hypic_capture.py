from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from kvcontinual.execution.recurrent.affine import AffineSummary
from kvcontinual.execution.types import TransitionOrientation


def _softplus_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def prepare_gdn_gates(
    alpha_raw: np.ndarray,
    beta_raw: np.ndarray,
    dt_bias: np.ndarray,
    ssm_a: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Qwen-style raw GDN controls into decay g and sigmoid(beta).

    Shapes:
      alpha_raw, beta_raw: [T, V]
      dt_bias, ssm_a:      [V]

    This mirrors the native RC9.2 CUDA prep path. It intentionally returns the
    multiplicative decay g rather than log(g), because the RC10 affine summary
    uses the recurrence actually applied to state.
    """
    a = np.asarray(alpha_raw, dtype=np.float64)
    b = np.asarray(beta_raw, dtype=np.float64)
    dt = np.asarray(dt_bias, dtype=np.float64).reshape(-1)
    ssm = np.asarray(ssm_a, dtype=np.float64).reshape(-1)
    if a.ndim != 2 or b.shape != a.shape:
        raise ValueError("alpha_raw/beta_raw must have shape [T,V]")
    if dt.shape != (a.shape[1],) or ssm.shape != (a.shape[1],):
        raise ValueError("dt_bias/ssm_a must have shape [V]")
    log_g = _softplus_stable(a + dt[None, :]) * ssm[None, :]
    decay = np.exp(log_g)
    sigmoid_beta = np.empty_like(b)
    positive = b >= 0.0
    sigmoid_beta[positive] = 1.0 / (1.0 + np.exp(-b[positive]))
    eb = np.exp(b[~positive])
    sigmoid_beta[~positive] = eb / (1.0 + eb)
    return decay, sigmoid_beta


def capture_head_tail(
    k: np.ndarray,
    v: np.ndarray,
    decay: np.ndarray,
    beta: np.ndarray,
    *,
    seam_width: int = 8,
) -> AffineSummary:
    """Capture one GDN head's fixed-input HYPIC tail summary.

    The returned affine map represents tokens ``[seam_width:]`` only:

        S_out = T_tail @ S_in + Z_tail

    for the RC9.2/Qwen column-state orientation. This is exact for the supplied
    k/v/decay/beta inputs; it does not claim that those inputs stay unchanged
    after relocating a segment behind a different predecessor.
    """
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    g = np.asarray(decay, dtype=np.float64).reshape(-1)
    b = np.asarray(beta, dtype=np.float64).reshape(-1)
    if k.ndim != 2 or v.shape != k.shape:
        raise ValueError("k/v must have shape [T,D] and match")
    if g.shape != (k.shape[0],) or b.shape != (k.shape[0],):
        raise ValueError("decay/beta must have shape [T]")
    if not 0 <= seam_width < k.shape[0]:
        raise ValueError("seam_width must leave a non-empty cached tail")

    d = k.shape[1]
    T = np.eye(d, dtype=np.float64)
    Z = np.zeros((d, d), dtype=np.float64)

    # Rank-1 form avoids materializing each token transition and mirrors the
    # intended Metal/CUDA implementation:
    #   T_i X = g_i (X - beta_i k_i (k_i^T X))
    for t in range(seam_width, k.shape[0]):
        kt = k[t]
        vt = v[t]
        gt = float(g[t])
        bt = float(b[t])
        kT = kt @ T
        kZ = kt @ Z
        T = gt * (T - bt * np.outer(kt, kT))
        Z = gt * (Z - bt * np.outer(kt, kZ)) + bt * np.outer(kt, vt)

    return AffineSummary(T, Z, TransitionOrientation.LEFT_MULTIPLY)


@dataclass(frozen=True)
class CapturedGDNHeads:
    seam_width: int
    head_dim: int
    num_k_heads: int
    num_v_heads: int
    summaries: tuple[AffineSummary, ...]

    def summary_for_v_head(self, head: int) -> AffineSummary:
        return self.summaries[head]


def capture_segment_tail_heads(
    k: np.ndarray,
    v: np.ndarray,
    decay: np.ndarray,
    beta: np.ndarray,
    *,
    seam_width: int = 8,
) -> CapturedGDNHeads:
    """Capture all v-head affine tails using RC9.2's v-head→k-head mapping.

    Shapes:
      k:     [T, K, D]
      v:     [T, V, D]
      decay: [T, V]
      beta:  [T, V]

    RC9.2 maps ``kh = vh % num_k_heads``. Qwen configurations should still be
    validated before capture; this routine rejects zero/mismatched dimensions.
    """
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    decay = np.asarray(decay, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if k.ndim != 3 or v.ndim != 3:
        raise ValueError("k/v must have shape [T,H,D]")
    if k.shape[0] != v.shape[0] or k.shape[2] != v.shape[2]:
        raise ValueError("k/v token count and head dimension must match")
    Tn, K, D = k.shape
    V = v.shape[1]
    if K <= 0 or V <= 0:
        raise ValueError("head counts must be positive")
    if decay.shape != (Tn, V) or beta.shape != (Tn, V):
        raise ValueError("decay/beta must have shape [T,V]")
    if not 0 <= seam_width < Tn:
        raise ValueError("seam_width must leave a non-empty cached tail")
    summaries = []
    for vh in range(V):
        kh = vh % K
        summaries.append(
            capture_head_tail(
                k[:, kh, :],
                v[:, vh, :],
                decay[:, vh],
                beta[:, vh],
                seam_width=seam_width,
            )
        )
    return CapturedGDNHeads(seam_width, D, K, V, tuple(summaries))


def direct_head_recurrence(
    state: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    decay: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Direct one-head recurrence oracle for fixed supplied GDN inputs."""
    S = np.asarray(state, dtype=np.float64).copy()
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    decay = np.asarray(decay, dtype=np.float64).reshape(-1)
    beta = np.asarray(beta, dtype=np.float64).reshape(-1)
    if k.ndim != 2 or v.shape != k.shape or S.shape != (k.shape[1], k.shape[1]):
        raise ValueError("state [D,D], k/v [T,D] required")
    for t in range(k.shape[0]):
        # Exact native recurrence: S' = g*S + k*beta*(v - g*k^T*S)
        predicted = k[t] @ S
        delta = (v[t] - float(decay[t]) * predicted) * float(beta[t])
        S = float(decay[t]) * S + np.outer(k[t], delta)
    return S
