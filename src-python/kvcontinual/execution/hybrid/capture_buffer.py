from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .instrumentation import HybridCaptureManifest
from .layout import HybridLayerKind


@dataclass
class GdnLayerCapture:
    k: Any
    v: Any
    decay: Any
    beta: Any
    conv_projected: Any
    incoming_conv_history: Any
    boundary_hidden: Any | None = None


@dataclass
class AttentionLayerCapture:
    pre_rope_keys: Any
    values: Any
    source_positions: Any
    boundary_hidden: Any | None = None


@dataclass
class HybridCaptureBuffer:
    """Version-agnostic capture boundary for real Qwen/Transformers hooks.

    The hook implementation is deliberately kept outside the cache object. A
    model integration must populate this buffer with tensors taken from the
    actual current execution before artifact materialization can proceed.
    """
    manifest: HybridCaptureManifest
    gdn: dict[int, GdnLayerCapture] = field(default_factory=dict)
    attention: dict[int, AttentionLayerCapture] = field(default_factory=dict)

    def put_gdn(self, layer: int, capture: GdnLayerCapture) -> None:
        self._require_kind(layer, HybridLayerKind.GATED_DELTA)
        self.gdn[layer] = capture

    def put_attention(self, layer: int, capture: AttentionLayerCapture) -> None:
        self._require_kind(layer, HybridLayerKind.FULL_ATTENTION)
        self.attention[layer] = capture

    def _require_kind(self, layer: int, kind: HybridLayerKind) -> None:
        matches = [r for r in self.manifest.requirements if r.layer == layer]
        if not matches:
            raise KeyError(f"layer {layer} is not in capture manifest")
        if matches[0].kind != kind:
            raise ValueError(f"layer {layer} is {matches[0].kind.value}, not {kind.value}")

    def missing_requirements(self) -> list[str]:
        missing: list[str] = []
        for r in self.manifest.requirements:
            if r.kind == HybridLayerKind.GATED_DELTA:
                c = self.gdn.get(r.layer)
                if c is None:
                    missing.append(f"layer {r.layer}:gdn")
                elif r.capture_conv_history and c.incoming_conv_history is None:
                    missing.append(f"layer {r.layer}:conv_history")
            elif r.kind == HybridLayerKind.FULL_ATTENTION:
                c = self.attention.get(r.layer)
                if c is None:
                    missing.append(f"layer {r.layer}:attention")
                elif r.capture_pre_rope_keys and c.pre_rope_keys is None:
                    missing.append(f"layer {r.layer}:pre_rope_keys")
        return missing

    @property
    def complete(self) -> bool:
        return not self.missing_requirements()
