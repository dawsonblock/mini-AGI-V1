from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class GDNToken:
    k: np.ndarray
    v: np.ndarray
    g: float
    beta: float

    def normalized(self) -> "GDNToken":
        k = np.asarray(self.k, dtype=np.float64)
        v = np.asarray(self.v, dtype=np.float64)
        if k.ndim != 1 or v.ndim != 1 or k.size == 0 or k.shape != v.shape:
            raise ValueError("k and v must be non-empty 1D vectors of equal length")
        if not np.all(np.isfinite(k)) or not np.all(np.isfinite(v)):
            raise ValueError("k/v must be finite")
        if not math.isfinite(self.g) or self.g < 0:
            raise ValueError("g must be finite and non-negative")
        if not math.isfinite(self.beta) or not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be in [0,1]")
        return GDNToken(k=k, v=v, g=float(self.g), beta=float(self.beta))


@dataclass
class GDNMatrixAffine:
    T: np.ndarray
    Z: np.ndarray

    def __post_init__(self) -> None:
        self.T = np.asarray(self.T, dtype=np.float64)
        self.Z = np.asarray(self.Z, dtype=np.float64)
        if self.T.ndim != 2 or self.T.shape[0] == 0 or self.T.shape[0] != self.T.shape[1]:
            raise ValueError("T must be a non-empty square matrix")
        if self.Z.shape != self.T.shape:
            raise ValueError("Z must have the same matrix shape as T")
        if not np.all(np.isfinite(self.T)) or not np.all(np.isfinite(self.Z)):
            raise ValueError("T/Z must be finite")

    @property
    def dim(self) -> int:
        return int(self.T.shape[0])

    @staticmethod
    def identity(dim: int) -> "GDNMatrixAffine":
        if dim <= 0:
            raise ValueError("dim must be positive")
        return GDNMatrixAffine(np.eye(dim, dtype=np.float64), np.zeros((dim, dim), dtype=np.float64))

    def apply(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != self.T.shape:
            raise ValueError("state shape mismatch")
        return self.T @ state + self.Z

    def then(self, nxt: "GDNMatrixAffine") -> "GDNMatrixAffine":
        if self.T.shape != nxt.T.shape:
            raise ValueError("affine dimensions differ")
        return GDNMatrixAffine(T=nxt.T @ self.T, Z=nxt.T @ self.Z + nxt.Z)


def sigmoid(x: float) -> float:
    if not math.isfinite(x):
        raise ValueError("sigmoid input must be finite")
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softplus(x: float) -> float:
    if not math.isfinite(x):
        raise ValueError("softplus input must be finite")
    if x > 40.0:
        return x
    if x < -40.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def decay_from_raw(alpha_raw: float, dt_bias: float, ssm_a: float) -> float:
    if not all(math.isfinite(x) for x in (alpha_raw, dt_bias, ssm_a)):
        raise ValueError("gate inputs must be finite")
    return math.exp(softplus(alpha_raw + dt_bias) * ssm_a)


def token_affine(token: GDNToken) -> GDNMatrixAffine:
    token = token.normalized()
    d = token.k.size
    T = token.g * (np.eye(d, dtype=np.float64) - token.beta * np.outer(token.k, token.k))
    Z = token.beta * np.outer(token.k, token.v)
    return GDNMatrixAffine(T=T, Z=Z)


def compose_tokens(tokens: list[GDNToken]) -> GDNMatrixAffine:
    if not tokens:
        raise ValueError("at least one token is required")
    first = tokens[0].normalized()
    out = GDNMatrixAffine.identity(first.k.size)
    for token in tokens:
        norm = token.normalized()
        if norm.k.size != out.dim:
            raise ValueError("token dimensions differ")
        out = out.then(token_affine(norm))
    return out


def replay(tokens: list[GDNToken], initial_state: np.ndarray) -> np.ndarray:
    state = np.asarray(initial_state, dtype=np.float64)
    if state.ndim != 2 or state.shape[0] == 0 or state.shape[0] != state.shape[1]:
        raise ValueError("initial_state must be a non-empty square matrix")
    d = state.shape[0]
    state = state.copy()
    for token in tokens:
        token = token.normalized()
        if token.k.size != d:
            raise ValueError("token dimension mismatch")
        kv = state.T @ token.k
        delta = token.beta * (token.v - token.g * kv)
        state = token.g * state + np.outer(token.k, delta)
    return state


def readout(state: np.ndarray, q: np.ndarray, scale: float = 1.0) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if state.ndim != 2 or state.shape[0] != state.shape[1] or q.shape != (state.shape[0],):
        raise ValueError("state/q dimension mismatch")
    if not math.isfinite(scale) or not np.all(np.isfinite(state)) or not np.all(np.isfinite(q)):
        raise ValueError("state/q/scale must be finite")
    return (state.T @ q) * scale


def state_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("state comparison shape mismatch")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return math.inf, math.inf
    diff = a - b
    return float(np.max(np.abs(diff))), float(np.linalg.norm(diff) / max(float(np.linalg.norm(b)), 1e-30))
