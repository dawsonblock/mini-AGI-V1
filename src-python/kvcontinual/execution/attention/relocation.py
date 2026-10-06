from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class RopeSpec:
    """Explicit RoPE geometry.

    ``rope_dim`` may be smaller than the attention head dimension, as in Qwen
    hybrid attention. ``pairing_layout='split_half'`` matches the common
    rotate-half convention; ``interleaved`` retains the legacy adjacent-pair
    reference behavior.
    """
    head_dim: int
    rope_dim: int
    pairing_layout: str = "split_half"
    theta: float = 10000.0
    scaling: str = "none"

    def validate(self) -> None:
        if self.head_dim <= 0 or self.rope_dim <= 0:
            raise ValueError("head_dim and rope_dim must be positive")
        if self.rope_dim > self.head_dim or self.rope_dim % 2:
            raise ValueError("rope_dim must be even and <= head_dim")
        if self.pairing_layout not in {"split_half", "interleaved"}:
            raise ValueError("unsupported RoPE pairing_layout")

    @property
    def digest(self) -> str:
        self.validate()
        raw = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class FullAttentionRelocationMetadata:
    layer: int
    token_count: int
    source_position_start: int
    keys_are_rope_rotated: bool
    rope_digest: str
    stores_pre_rope_keys: bool = False

    def validate(self) -> None:
        if self.layer < 0 or self.token_count < 0 or self.source_position_start < 0:
            raise ValueError("invalid attention relocation metadata")
        if not self.rope_digest:
            raise ValueError("rope_digest is required")


def rope_digest(inv_freq: np.ndarray, spec: RopeSpec | None = None) -> str:
    a = np.ascontiguousarray(inv_freq, dtype=np.float64)
    h = hashlib.sha256()
    h.update(a.tobytes())
    if spec is not None:
        h.update(spec.digest.encode("ascii"))
    return "sha256:" + h.hexdigest()


def _angles(positions: np.ndarray, inv_freq: np.ndarray, ndim: int):
    angles = positions[:, None] * inv_freq[None, :]
    expand = (slice(None),) + (None,) * (ndim - 2) + (slice(None),)
    return np.cos(angles)[expand], np.sin(angles)[expand]


def rope_apply(x: np.ndarray, positions: np.ndarray, inv_freq: np.ndarray, spec: RopeSpec | None = None) -> np.ndarray:
    """Apply RoPE to token-major keys/queries using explicit geometry when supplied.

    Legacy callers may omit ``spec``; that preserves full-head interleaved RoPE.
    With a spec, only ``rope_dim`` leading dimensions are rotated and the tail is
    copied exactly.
    """
    x = np.asarray(x)
    pos = np.asarray(positions, dtype=np.float64)
    inv = np.asarray(inv_freq, dtype=np.float64)
    if x.ndim < 2 or x.shape[0] != pos.shape[0]:
        raise ValueError("x must be token-major and match positions")
    d = x.shape[-1]
    if spec is None:
        spec = RopeSpec(head_dim=d, rope_dim=d, pairing_layout="interleaved")
    spec.validate()
    if spec.head_dim != d:
        raise ValueError(f"RoPE head_dim {spec.head_dim} does not match tensor dim {d}")
    if inv.shape != (spec.rope_dim // 2,):
        raise ValueError("invalid inv_freq shape for rope_dim")

    out = np.array(x, copy=True, dtype=np.result_type(x.dtype, np.float64))
    rot = x[..., :spec.rope_dim]
    c, s = _angles(pos, inv, x.ndim)
    half = spec.rope_dim // 2
    if spec.pairing_layout == "interleaved":
        a, b = rot[..., 0::2], rot[..., 1::2]
        out[..., :spec.rope_dim:2] = a * c - b * s
        out[..., 1:spec.rope_dim:2] = a * s + b * c
    else:  # split_half / rotate_half
        a, b = rot[..., :half], rot[..., half:spec.rope_dim]
        out[..., :half] = a * c - b * s
        out[..., half:spec.rope_dim] = a * s + b * c
    return out.astype(x.dtype, copy=False)


def relocate_rope_keys(rotated_keys: np.ndarray, old_positions: np.ndarray, new_positions: np.ndarray,
                       inv_freq: np.ndarray, spec: RopeSpec | None = None) -> np.ndarray:
    """Relocate already-RoPE-rotated keys via R(new-old)R(old)=R(new)."""
    old = np.asarray(old_positions)
    new = np.asarray(new_positions)
    if old.shape != new.shape:
        raise ValueError("old/new positions must have the same shape")
    return rope_apply(rotated_keys, new - old, inv_freq, spec=spec)
