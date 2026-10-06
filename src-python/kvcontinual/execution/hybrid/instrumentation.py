from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .layout import HybridLayerKind, HybridModelLayout


@dataclass(frozen=True)
class CaptureRequirement:
    layer: int
    kind: HybridLayerKind
    capture_boundary_hidden: bool
    capture_gdn_kv_decay_beta: bool
    capture_conv_history: bool
    capture_full_attention_kv: bool
    capture_pre_rope_keys: bool


@dataclass(frozen=True)
class HybridCaptureManifest:
    layout_digest: str
    seam_width: int
    requirements: tuple[CaptureRequirement, ...]

    @property
    def digest(self) -> str:
        payload = {
            "layout_digest": self.layout_digest,
            "seam_width": self.seam_width,
            "requirements": [
                {**asdict(x), "kind": x.kind.value}
                for x in self.requirements
            ],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    def required_gdn_layers(self) -> tuple[int, ...]:
        return tuple(x.layer for x in self.requirements if x.kind == HybridLayerKind.GATED_DELTA)

    def required_attention_layers(self) -> tuple[int, ...]:
        return tuple(x.layer for x in self.requirements if x.kind == HybridLayerKind.FULL_ATTENTION)


def build_capture_manifest(layout: HybridModelLayout, seam_width: int = 8) -> HybridCaptureManifest:
    if seam_width <= 0:
        raise ValueError("seam_width must be positive")
    reqs = []
    for layer in layout.layers:
        is_gdn = layer.kind == HybridLayerKind.GATED_DELTA
        is_fa = layer.kind == HybridLayerKind.FULL_ATTENTION
        reqs.append(CaptureRequirement(
            layer=layer.index,
            kind=layer.kind,
            capture_boundary_hidden=is_gdn or is_fa,
            capture_gdn_kv_decay_beta=is_gdn,
            capture_conv_history=is_gdn and layer.conv_kernel_size > 1,
            capture_full_attention_kv=is_fa,
            capture_pre_rope_keys=is_fa,
        ))
    return HybridCaptureManifest(layout.digest, seam_width, tuple(reqs))
