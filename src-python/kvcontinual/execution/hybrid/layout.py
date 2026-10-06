from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Any


class HybridLayerKind(str, Enum):
    GATED_DELTA = "GATED_DELTA"
    FULL_ATTENTION = "FULL_ATTENTION"
    OTHER = "OTHER"


def _read(config: Any, *names: str, default=None):
    for name in names:
        if isinstance(config, dict) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _kind(name: str) -> HybridLayerKind:
    s = str(name).lower().replace("-", "_")
    if any(x in s for x in ("linear", "delta", "gdn", "recurrent")):
        return HybridLayerKind.GATED_DELTA
    if "attention" in s or "attn" in s:
        return HybridLayerKind.FULL_ATTENTION
    return HybridLayerKind.OTHER


@dataclass(frozen=True)
class HybridLayerSpec:
    index: int
    kind: HybridLayerKind
    q_heads: int
    kv_heads: int
    head_dim: int
    conv_kernel_size: int = 0


@dataclass(frozen=True)
class HybridModelLayout:
    architecture: str
    layers: tuple[HybridLayerSpec, ...]

    @property
    def digest(self) -> str:
        payload = {
            "architecture": self.architecture,
            "layers": [
                {**asdict(x), "kind": x.kind.value}
                for x in self.layers
            ],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    @property
    def gdn_layers(self) -> tuple[int, ...]:
        return tuple(x.index for x in self.layers if x.kind == HybridLayerKind.GATED_DELTA)

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(x.index for x in self.layers if x.kind == HybridLayerKind.FULL_ATTENTION)

    @classmethod
    def from_config(cls, config: Any) -> "HybridModelLayout":
        n = int(_read(config, "num_hidden_layers", "n_layer", default=0) or 0)
        if n <= 0:
            raise ValueError("model config must expose num_hidden_layers")

        layer_types = _read(config, "layer_types", "layers_block_type", default=None)
        if layer_types is not None:
            layer_types = list(layer_types)
            if len(layer_types) != n:
                raise ValueError("layer_types length does not match num_hidden_layers")
        else:
            interval = _read(config, "full_attention_interval", "attention_interval", default=None)
            if interval is None:
                raise ValueError(
                    "hybrid layout is ambiguous: provide layer_types or full_attention_interval"
                )
            interval = int(interval)
            if interval <= 0:
                raise ValueError("full_attention_interval must be positive")
            layer_types = [
                "full_attention" if (i + 1) % interval == 0 else "gated_delta"
                for i in range(n)
            ]

        q_heads = int(_read(config, "num_attention_heads", "num_heads", default=0) or 0)
        kv_heads = int(_read(config, "num_key_value_heads", "num_kv_heads", default=q_heads) or q_heads)
        hidden = int(_read(config, "hidden_size", "d_model", default=0) or 0)
        head_dim = int(_read(config, "head_dim", default=(hidden // q_heads if q_heads else 0)) or 0)
        if q_heads <= 0 or head_dim <= 0:
            raise ValueError("config must expose valid attention head counts/head_dim")

        linear_k = int(_read(config, "linear_num_key_heads", "linear_num_k_heads", default=kv_heads) or kv_heads)
        linear_v = int(_read(config, "linear_num_value_heads", "linear_num_v_heads", default=q_heads) or q_heads)
        conv_k = int(_read(config, "conv_kernel_size", "conv_kernel", default=4) or 0)
        architecture = str(_read(config, "model_type", "architectures", default="unknown"))

        specs: list[HybridLayerSpec] = []
        for i, raw_kind in enumerate(layer_types):
            kind = _kind(raw_kind)
            if kind == HybridLayerKind.GATED_DELTA:
                specs.append(HybridLayerSpec(i, kind, linear_v, linear_k, head_dim, conv_k))
            elif kind == HybridLayerKind.FULL_ATTENTION:
                specs.append(HybridLayerSpec(i, kind, q_heads, kv_heads, head_dim, 0))
            else:
                specs.append(HybridLayerSpec(i, kind, q_heads, kv_heads, head_dim, 0))
        return cls(architecture=architecture, layers=tuple(specs))
