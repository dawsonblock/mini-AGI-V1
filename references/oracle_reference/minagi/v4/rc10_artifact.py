from __future__ import annotations

"""Portable, hash-verified persistence for compiled RC10 block artifacts.

Canonical memory never depends on this representation.  A block artifact can be
removed and recompiled from canonical tokens/traces without changing what the
system remembers.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json

import torch

from .conv_boundary import ConvBoundaryState
from .generation import CacheCompatibility, EffectiveModelGeneration
from .pic import FAPICSegment
from .rc10 import (
    AffineSegmentSummary, BoundaryPayload, RC10BlockCache, RecurrentVariant,
    TransitionOrientation,
)


SCHEMA = "mini-agi-v4.3-rc10-block-v1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fa_payload(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, FAPICSegment):
        payload = {-1: payload}
    if not isinstance(payload, Mapping):
        raise TypeError("unsupported full-attention payload")
    out = {}
    for layer, seg in payload.items():
        if not isinstance(seg, FAPICSegment):
            raise TypeError("full-attention payload values must be FAPICSegment")
        seg.validate()
        out[str(int(layer))] = {
            "key": seg.key.detach().cpu(),
            "value": seg.value.detach().cpu(),
            "source_positions": tuple(map(int, seg.source_positions)),
            "token_digest": str(seg.token_digest),
        }
    return out


def _conv_payload(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError("trailing convolution payload must be a layer mapping")
    out = {}
    for layer, state in payload.items():
        if not isinstance(state, ConvBoundaryState):
            raise TypeError("convolution payload values must be ConvBoundaryState")
        state.validate()
        out[str(int(layer))] = {
            "history": state.history.detach().cpu(),
            "kernel_size": int(state.kernel_size),
        }
    return out


def _cache_to_payload(cache: RC10BlockCache) -> dict[str, Any]:
    variants = []
    for variant in cache.recurrent_variants:
        summaries = []
        for summary in variant.summaries:
            summaries.append({
                "transition": summary.transition.detach().cpu(),
                "zero_state": summary.zero_state.detach().cpu(),
                "orientation": summary.orientation.value,
                "seam_width": int(summary.seam_width),
                "layer": summary.layer,
                "segment_id": str(summary.segment_id),
            })
        variants.append({"seam_width": int(variant.seam_width), "summaries": summaries})
    return {
        "compatibility": asdict(cache.compatibility),
        "recurrent_variants": variants,
        "full_attention": _fa_payload(cache.full_attention_payload),
        "boundary": {
            "leading_tokens": tuple(map(int, cache.boundary.leading_tokens)),
            "trailing_conv_payload": _conv_payload(cache.boundary.trailing_conv_payload),
            "reference_hidden_digest": str(cache.boundary.reference_hidden_digest),
            "max_repair_width": int(cache.boundary.max_repair_width),
        },
    }


@dataclass(frozen=True)
class RC10BlockArtifactManifest:
    schema: str
    block_id: str
    token_count: int
    token_digest: str
    generation_digest: str
    cache_abi_version: int
    tier: str
    payload_file: str
    payload_sha256: str

    def assert_generation(self, generation: EffectiveModelGeneration) -> None:
        if self.generation_digest != generation.digest:
            raise ValueError("compiled RC10 block belongs to another effective model generation")
        if self.cache_abi_version != generation.cache_abi_version:
            raise ValueError("compiled RC10 block cache ABI mismatch")


def save_rc10_block_artifact(directory: str | Path, cache: RC10BlockCache) -> RC10BlockArtifactManifest:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    payload_path = root / "payload.pt"
    torch.save(_cache_to_payload(cache), payload_path)
    manifest = RC10BlockArtifactManifest(
        SCHEMA, cache.block_id, int(cache.token_count), cache.token_digest,
        cache.compatibility.generation_digest, int(cache.compatibility.cache_abi_version),
        cache.tier, payload_path.name, _sha256_file(payload_path),
    )
    (root / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
    return manifest


def _load_tensor_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_rc10_block_artifact(
    directory: str | Path,
    *,
    generation: EffectiveModelGeneration | None = None,
) -> tuple[RC10BlockArtifactManifest, RC10BlockCache]:
    root = Path(directory)
    doc = json.loads((root / "manifest.json").read_text())
    manifest = RC10BlockArtifactManifest(**doc)
    if manifest.schema != SCHEMA:
        raise ValueError("unsupported RC10 block artifact schema")
    if generation is not None:
        manifest.assert_generation(generation)
    payload_path = root / manifest.payload_file
    if _sha256_file(payload_path) != manifest.payload_sha256:
        raise ValueError("RC10 block payload hash mismatch")
    payload = _load_tensor_payload(payload_path)

    compat = CacheCompatibility(**payload["compatibility"])
    variants = []
    for item in payload["recurrent_variants"]:
        summaries = []
        for s in item["summaries"]:
            summaries.append(AffineSegmentSummary(
                s["transition"], s["zero_state"], TransitionOrientation(s["orientation"]),
                int(s["seam_width"]), None if s["layer"] is None else int(s["layer"]),
                str(s["segment_id"]),
            ))
        variants.append(RecurrentVariant(int(item["seam_width"]), tuple(summaries)))

    fa = {}
    for layer, item in payload["full_attention"].items():
        fa[int(layer)] = FAPICSegment(
            item["key"], item["value"], tuple(map(int, item["source_positions"])),
            str(item["token_digest"]),
        )

    conv = {}
    for layer, item in payload["boundary"]["trailing_conv_payload"].items():
        conv[int(layer)] = ConvBoundaryState(item["history"], int(item["kernel_size"]))

    boundary = BoundaryPayload(
        tuple(map(int, payload["boundary"]["leading_tokens"])),
        conv or None,
        str(payload["boundary"]["reference_hidden_digest"]),
        int(payload["boundary"]["max_repair_width"]),
    )
    cache = RC10BlockCache(
        manifest.block_id, int(manifest.token_count), manifest.token_digest, compat,
        tuple(variants), fa or None, boundary, manifest.tier,
    )
    if cache.compatibility.generation_digest != manifest.generation_digest:
        raise ValueError("RC10 block manifest/payload generation mismatch")
    if cache.compatibility.cache_abi_version != manifest.cache_abi_version:
        raise ValueError("RC10 block manifest/payload ABI mismatch")
    return manifest, cache


__all__ = [
    "RC10BlockArtifactManifest", "save_rc10_block_artifact",
    "load_rc10_block_artifact", "SCHEMA",
]
