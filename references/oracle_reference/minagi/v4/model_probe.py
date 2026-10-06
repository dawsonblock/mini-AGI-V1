from __future__ import annotations

"""Model/version probing for the optional RC10 Transformers integration."""

from dataclasses import dataclass, asdict
from typing import Any
import hashlib
import json

from .hf_hybrid_capture import discover_hybrid_layout


@dataclass(frozen=True)
class ModelProbeReport:
    model_class: str
    model_type: str
    transformers_version: str
    gated_delta_layers: tuple[int, ...]
    full_attention_layers: tuple[int, ...]
    layer_classes: tuple[tuple[str, str], ...]
    config_digest: str

    def jsonable(self) -> dict[str, Any]:
        return asdict(self)


def _config_digest(model: Any) -> str:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return ""
    if hasattr(cfg, "to_dict"):
        payload = cfg.to_dict()
    else:
        payload = vars(cfg)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def probe_model(model: Any) -> ModelProbeReport:
    layout = discover_hybrid_layout(model)
    try:
        import transformers  # type: ignore
        version = str(transformers.__version__)
    except Exception:
        version = "unavailable"
    cfg = getattr(model, "config", None)
    model_type = str(getattr(cfg, "model_type", ""))
    return ModelProbeReport(
        model_class=type(model).__name__,
        model_type=model_type,
        transformers_version=version,
        gated_delta_layers=tuple(x.layer_idx for x in layout.gated_delta_layers if x.layer_idx is not None),
        full_attention_layers=tuple(x.layer_idx for x in layout.full_attention_layers if x.layer_idx is not None),
        layer_classes=tuple((x.name, x.class_name) for x in layout.layers),
        config_digest=_config_digest(model),
    )


__all__ = ["ModelProbeReport", "probe_model"]
