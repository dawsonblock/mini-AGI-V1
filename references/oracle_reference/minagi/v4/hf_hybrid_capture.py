from __future__ import annotations

"""Version-tolerant Hugging Face hybrid-model instrumentation for RC10.

The module deliberately avoids importing private Transformers model classes.
Instead it detects the capabilities RC10 needs (Gated-Delta projections,
causal convolution, full-attention projection/cache fields) and adapts those
objects into the generic v4 cache ABI.

This keeps the generic RC10 algebra independent from a particular Transformers
release while still providing a concrete path for Qwen3.5-style models.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .conv_boundary import causal_depthwise_conv
from .pic import FAPICSegment
from .qwen35_rc10 import CapturedGDNLayer


class HybridLayerKind(str, Enum):
    GATED_DELTA = "gated_delta"
    FULL_ATTENTION = "full_attention"
    OTHER = "other"


@dataclass(frozen=True)
class LayerDescriptor:
    name: str
    layer_idx: int | None
    kind: HybridLayerKind
    class_name: str


@dataclass(frozen=True)
class HybridModelLayout:
    layers: tuple[LayerDescriptor, ...]

    @property
    def gated_delta_layers(self) -> tuple[LayerDescriptor, ...]:
        return tuple(x for x in self.layers if x.kind == HybridLayerKind.GATED_DELTA)

    @property
    def full_attention_layers(self) -> tuple[LayerDescriptor, ...]:
        return tuple(x for x in self.layers if x.kind == HybridLayerKind.FULL_ATTENTION)

    def assert_hybrid(self) -> None:
        if not self.gated_delta_layers:
            raise ValueError("no Gated-Delta-style layers discovered")
        if not self.full_attention_layers:
            raise ValueError("no full-attention layers discovered")


@dataclass(frozen=True)
class CapturedFullAttentionLayer:
    layer: int
    hidden_states: torch.Tensor
    position_ids: torch.Tensor | None = None
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass(frozen=True)
class ForwardCapture:
    recurrent: Mapping[int, CapturedGDNLayer]
    full_attention_inputs: Mapping[int, CapturedFullAttentionLayer]
    logits: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] = ()
    past_key_values: Any = None


def _is_gdn_module(module: Any) -> bool:
    required = (
        "in_proj_qkv", "in_proj_b", "in_proj_a", "A_log", "dt_bias", "conv1d",
        "key_dim", "value_dim", "head_k_dim", "head_v_dim", "num_v_heads", "num_k_heads",
    )
    return all(hasattr(module, x) for x in required)


def _is_attention_module(module: Any) -> bool:
    # Capability check first, class-name hint second.  Requiring q/k/v projections
    # prevents ordinary wrapper modules from being misidentified as attention.
    has_qkv = all(hasattr(module, x) for x in ("q_proj", "k_proj", "v_proj"))
    if not has_qkv:
        return False
    name = type(module).__name__.lower()
    return "attention" in name or "attn" in name


def discover_hybrid_layout(model: torch.nn.Module) -> HybridModelLayout:
    found: list[LayerDescriptor] = []
    for name, module in model.named_modules():
        if not name:
            continue
        idx = getattr(module, "layer_idx", None)
        idx = int(idx) if isinstance(idx, int) else None
        if _is_gdn_module(module):
            found.append(LayerDescriptor(name, idx, HybridLayerKind.GATED_DELTA, type(module).__name__))
        elif _is_attention_module(module):
            found.append(LayerDescriptor(name, idx, HybridLayerKind.FULL_ATTENTION, type(module).__name__))
    # Stable order: explicit layer index first, module traversal order as a tiebreaker.
    return HybridModelLayout(tuple(sorted(found, key=lambda x: (10**9 if x.layer_idx is None else x.layer_idx, x.name))))


def _activation(x: torch.Tensor, activation: Any) -> torch.Tensor:
    if activation is None:
        return x
    if callable(activation):
        return activation(x)
    name = str(activation).lower()
    if name in ("silu", "swish"):
        return F.silu(x)
    if name == "relu":
        return F.relu(x)
    if name == "gelu":
        return F.gelu(x)
    raise ValueError(f"unsupported causal-convolution activation: {activation!r}")


def _mask_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states
    if attention_mask.ndim != 2 or hidden_states.ndim != 3:
        # RC10 qualification intentionally refuses to guess how a vendor-specific
        # high-rank mask should be interpreted.
        raise ValueError("capture currently requires a 2D padding mask or no mask")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention mask shape mismatch")
    return hidden_states * attention_mask.to(hidden_states).unsqueeze(-1)


def fla_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match the FLA/Transformers key normalization used by Gated DeltaNet."""
    return x * torch.rsqrt((x.float() * x.float()).sum(dim=-1, keepdim=True) + eps).to(x.dtype)


def capture_gdn_from_hidden(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    layer_idx: int | None = None,
) -> CapturedGDNLayer:
    """Reconstruct the recurrent inputs consumed by a Qwen3.5-style GDN layer.

    This implements the documented Transformers prefill path using the module's
    own projections and convolution weights.  It is intentionally a *prefill*
    capture: no prior recurrent/conv cache may be involved.
    """
    if not _is_gdn_module(module):
        raise TypeError("module does not expose the Qwen3.5-style GDN capability contract")
    hs = _mask_hidden(hidden_states, attention_mask)
    if hs.ndim != 3:
        raise ValueError("expected hidden states [batch,time,hidden]")

    mixed_raw = module.in_proj_qkv(hs)  # [B,T,C]
    weight = module.conv1d.weight.squeeze(1)
    bias = module.conv1d.bias
    conv_out, _ = causal_depthwise_conv(mixed_raw, weight, bias)
    mixed = _activation(conv_out, getattr(module, "activation", None))

    query, key, value = torch.split(
        mixed,
        [int(module.key_dim), int(module.key_dim), int(module.value_dim)],
        dim=-1,
    )
    del query  # RC10 transition compilation needs only k/v/beta/decay.
    bsz, seq_len, _ = key.shape
    key = key.reshape(bsz, seq_len, -1, int(module.head_k_dim))
    value = value.reshape(bsz, seq_len, -1, int(module.head_v_dim))

    beta = module.in_proj_b(hs).sigmoid()
    a = module.in_proj_a(hs)
    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)

    v_heads = int(module.num_v_heads)
    k_heads = int(module.num_k_heads)
    if v_heads % k_heads != 0:
        raise ValueError("value-head count must be divisible by key-head count")
    if v_heads // k_heads > 1:
        key = key.repeat_interleave(v_heads // k_heads, dim=2)

    # Transformers passes use_qk_l2norm_in_kernel=True.  RC10 stores the key
    # actually used by the recurrent update, not the pre-normalization projection.
    key = fla_l2norm(key.float()).to(value.dtype)
    decay = torch.exp(g).to(value.dtype)
    beta = beta.to(value.dtype)

    layer = int(layer_idx if layer_idx is not None else getattr(module, "layer_idx", 0))
    return CapturedGDNLayer(
        layer=layer,
        keys=key,
        values=value,
        beta=beta,
        decay=decay,
        conv_inputs=mixed_raw,
        conv_kernel_size=int(weight.shape[-1]),
    )


def _extract_hidden_arg(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> torch.Tensor:
    if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
        return kwargs["hidden_states"]
    for arg in args:
        if torch.is_tensor(arg) and arg.ndim >= 3:
            return arg
    raise ValueError("unable to locate hidden_states in module call")


class HFHybridTraceCollector:
    """Capture recurrent/full-attention layer inputs during one exact model pass.

    Hooks are removed deterministically on exit.  The collector never mutates the
    model and can therefore sit under ``FrozenBaseGuard``.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layout = discover_hybrid_layout(model)
        self._handles: list[Any] = []
        self.recurrent: dict[int, CapturedGDNLayer] = {}
        self.full_attention: dict[int, CapturedFullAttentionLayer] = {}

    def __enter__(self) -> "HFHybridTraceCollector":
        self.recurrent.clear()
        self.full_attention.clear()
        modules = dict(self.model.named_modules())
        for desc in self.layout.layers:
            module = modules[desc.name]
            if desc.kind == HybridLayerKind.GATED_DELTA:
                def pre_hook(mod, args, kwargs, *, _idx=desc.layer_idx):
                    hs = _extract_hidden_arg(args, kwargs).detach()
                    mask = kwargs.get("attention_mask")
                    self.recurrent[int(_idx if _idx is not None else getattr(mod, "layer_idx", 0))] = capture_gdn_from_hidden(
                        mod, hs, attention_mask=mask, layer_idx=_idx
                    )
                self._handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
            elif desc.kind == HybridLayerKind.FULL_ATTENTION:
                def attn_hook(mod, args, kwargs, *, _idx=desc.layer_idx):
                    hs = _extract_hidden_arg(args, kwargs).detach()
                    pe = kwargs.get("position_embeddings")
                    pos = kwargs.get("position_ids")
                    self.full_attention[int(_idx if _idx is not None else getattr(mod, "layer_idx", 0))] = CapturedFullAttentionLayer(
                        int(_idx if _idx is not None else getattr(mod, "layer_idx", 0)),
                        hs,
                        None if pos is None else pos.detach(),
                        None if pe is None else tuple(x.detach() for x in pe),
                    )
                self._handles.append(module.register_forward_pre_hook(attn_hook, with_kwargs=True))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @torch.no_grad()
    def run(self, *args, **kwargs) -> ForwardCapture:
        # Caller controls output_hidden_states/use_cache because some model
        # versions have different output contracts.  We only normalize what is present.
        with self:
            output = self.model(*args, **kwargs)
        logits = getattr(output, "logits", None)
        hidden = getattr(output, "hidden_states", None) or ()
        past = getattr(output, "past_key_values", None)
        return ForwardCapture(
            recurrent=dict(self.recurrent),
            full_attention_inputs=dict(self.full_attention),
            logits=None if logits is None else logits.detach(),
            hidden_states=tuple(x.detach() for x in hidden),
            past_key_values=past,
        )


def _layer_kv_from_object(layer: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    candidates = (
        ("keys", "values"),
        ("key", "value"),
        ("key_cache", "value_cache"),
    )
    for kname, vname in candidates:
        if hasattr(layer, kname) and hasattr(layer, vname):
            k, v = getattr(layer, kname), getattr(layer, vname)
            if torch.is_tensor(k) and torch.is_tensor(v):
                return k, v
    return None


def extract_full_attention_pic(
    past_key_values: Any,
    layer_idx: int,
    *,
    positions: Sequence[int] | None = None,
    token_digest: str = "",
) -> FAPICSegment:
    """Extract a full-attention K/V plane from common Transformers cache layouts.

    Qwen hybrid-cache internals have changed across Transformers revisions.  The
    extractor therefore accepts several public-ish layouts and fails closed when
    none match instead of silently inventing K/V tensors.
    """
    if past_key_values is None:
        raise ValueError("past_key_values are required for full-attention PIC extraction")
    idx = int(layer_idx)
    layer = None
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[idx]
    elif isinstance(past_key_values, (tuple, list)):
        layer = past_key_values[idx]
    elif hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        k, v = past_key_values.key_cache[idx], past_key_values.value_cache[idx]
        layer = (k, v)

    kv = None
    if isinstance(layer, (tuple, list)) and len(layer) >= 2 and torch.is_tensor(layer[0]) and torch.is_tensor(layer[1]):
        kv = (layer[0], layer[1])
    elif layer is not None:
        kv = _layer_kv_from_object(layer)
    if kv is None:
        raise ValueError(f"unable to extract K/V for cache layer {idx}")
    key, value = kv

    # RC10's FAPICSegment uses sequence at -2.  Common HF caches are [B,H,T,D].
    if key.ndim < 2 or value.ndim != key.ndim:
        raise ValueError("unexpected K/V cache ranks")
    seq = key.shape[-2]
    if value.shape[-2] != seq:
        raise ValueError("K/V cache sequence mismatch")
    pos = tuple(range(seq)) if positions is None else tuple(map(int, positions))
    if len(pos) != seq:
        raise ValueError("position count does not match cache sequence")
    return FAPICSegment(key.detach(), value.detach(), pos, token_digest=token_digest)


__all__ = [
    "HybridLayerKind", "LayerDescriptor", "HybridModelLayout",
    "CapturedFullAttentionLayer", "ForwardCapture", "discover_hybrid_layout",
    "fla_l2norm", "capture_gdn_from_hidden", "HFHybridTraceCollector",
    "extract_full_attention_pic",
]
