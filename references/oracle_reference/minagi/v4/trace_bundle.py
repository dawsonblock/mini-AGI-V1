from __future__ import annotations

"""Immutable RC10 trace bundles for reproducible whole-model qualification.

Trace bundles are disposable qualification artifacts, not canonical memory.
They are namespaced by the exact effective-model generation and include hashes
for the tensor payload so stale/tampered captures fail closed.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json

import torch

from .generation import EffectiveModelGeneration
from .hf_hybrid_capture import ForwardCapture, CapturedFullAttentionLayer
from .qwen35_rc10 import CapturedGDNLayer
from .model_probe import ModelProbeReport


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _token_digest(tokens: Sequence[int]) -> str:
    h = hashlib.sha256()
    for token in tokens:
        h.update(int(token).to_bytes(8, "little", signed=True))
    return h.hexdigest()


@dataclass(frozen=True)
class TraceBundleManifest:
    schema: str
    generation_digest: str
    token_digest: str
    token_count: int
    model_probe: Mapping[str, Any]
    recurrent_layers: tuple[int, ...]
    full_attention_layers: tuple[int, ...]
    tensor_payload: str
    tensor_payload_sha256: str

    def assert_generation(self, generation: EffectiveModelGeneration) -> None:
        if self.generation_digest != generation.digest:
            raise ValueError("trace bundle belongs to a different effective model generation")


def save_trace_bundle(
    directory: str | Path,
    *,
    capture: ForwardCapture,
    generation: EffectiveModelGeneration,
    probe: ModelProbeReport,
    tokens: Sequence[int],
) -> TraceBundleManifest:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"recurrent": {}, "full_attention": {}}
    for layer, cap in capture.recurrent.items():
        payload["recurrent"][str(int(layer))] = {
            "keys": cap.keys.detach().cpu(),
            "values": cap.values.detach().cpu(),
            "beta": cap.beta.detach().cpu(),
            "decay": cap.decay.detach().cpu(),
            "conv_inputs": None if cap.conv_inputs is None else cap.conv_inputs.detach().cpu(),
            "conv_kernel_size": int(cap.conv_kernel_size),
        }
    for layer, cap in capture.full_attention_inputs.items():
        payload["full_attention"][str(int(layer))] = {
            "hidden_states": cap.hidden_states.detach().cpu(),
            "position_ids": None if cap.position_ids is None else cap.position_ids.detach().cpu(),
            "position_embeddings": None if cap.position_embeddings is None else tuple(x.detach().cpu() for x in cap.position_embeddings),
        }
    tensor_path = root / "capture_tensors.pt"
    torch.save(payload, tensor_path)
    manifest = TraceBundleManifest(
        schema="mini-agi-v4.2-trace-bundle-v1",
        generation_digest=generation.digest,
        token_digest=_token_digest(tokens),
        token_count=len(tuple(tokens)),
        model_probe=probe.jsonable(),
        recurrent_layers=tuple(sorted(map(int, capture.recurrent))),
        full_attention_layers=tuple(sorted(map(int, capture.full_attention_inputs))),
        tensor_payload=tensor_path.name,
        tensor_payload_sha256=_sha256_file(tensor_path),
    )
    (root / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
    return manifest


def load_trace_bundle(
    directory: str | Path,
    *,
    generation: EffectiveModelGeneration | None = None,
) -> tuple[TraceBundleManifest, Mapping[int, CapturedGDNLayer], Mapping[int, CapturedFullAttentionLayer]]:
    root = Path(directory)
    data = json.loads((root / "manifest.json").read_text())
    manifest = TraceBundleManifest(
        schema=data["schema"],
        generation_digest=data["generation_digest"],
        token_digest=data["token_digest"],
        token_count=int(data["token_count"]),
        model_probe=data["model_probe"],
        recurrent_layers=tuple(map(int, data["recurrent_layers"])),
        full_attention_layers=tuple(map(int, data["full_attention_layers"])),
        tensor_payload=data["tensor_payload"],
        tensor_payload_sha256=data["tensor_payload_sha256"],
    )
    if manifest.schema != "mini-agi-v4.2-trace-bundle-v1":
        raise ValueError("unsupported trace bundle schema")
    if generation is not None:
        manifest.assert_generation(generation)
    path = root / manifest.tensor_payload
    if _sha256_file(path) != manifest.tensor_payload_sha256:
        raise ValueError("trace tensor payload hash mismatch")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older PyTorch; retained for portability
        payload = torch.load(path, map_location="cpu")
    recurrent: dict[int, CapturedGDNLayer] = {}
    for key, item in payload["recurrent"].items():
        layer = int(key)
        recurrent[layer] = CapturedGDNLayer(
            layer,
            item["keys"], item["values"], item["beta"], item["decay"],
            item["conv_inputs"], int(item["conv_kernel_size"]),
        )
    full: dict[int, CapturedFullAttentionLayer] = {}
    for key, item in payload["full_attention"].items():
        layer = int(key)
        pe = item["position_embeddings"]
        full[layer] = CapturedFullAttentionLayer(
            layer,
            item["hidden_states"],
            item["position_ids"],
            None if pe is None else tuple(pe),
        )
    if tuple(sorted(recurrent)) != manifest.recurrent_layers:
        raise ValueError("trace recurrent-layer manifest mismatch")
    if tuple(sorted(full)) != manifest.full_attention_layers:
        raise ValueError("trace attention-layer manifest mismatch")
    return manifest, recurrent, full


__all__ = ["TraceBundleManifest", "save_trace_bundle", "load_trace_bundle"]
