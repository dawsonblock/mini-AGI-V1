from __future__ import annotations

"""Compilation helpers from canonical blocks/captured model signals to RC10 artifacts."""

from dataclasses import dataclass
from typing import Mapping, Sequence
import hashlib
import torch

from .generation import CacheCompatibility, EffectiveModelGeneration
from .pic import FAPICSegment
from .qwen35_rc10 import CapturedGDNLayer, compile_layer, compile_recurrent_variant
from .rc10 import BoundaryPayload, RC10BlockCache, RecurrentVariant, TransitionOrientation


@dataclass(frozen=True)
class CompilePolicy:
    tier: str = "warm"
    warm_seams: tuple[int, ...] = (8,)
    hot_seams: tuple[int, ...] = (8, 32)
    critical_seams: tuple[int, ...] = (8, 32, 128)

    def seams_for_tier(self) -> tuple[int, ...]:
        if self.tier == "cold":
            return ()
        if self.tier == "warm":
            return self.warm_seams
        if self.tier == "hot":
            return self.hot_seams
        if self.tier == "critical":
            return self.critical_seams
        raise ValueError("unknown cache tier")


def token_digest(tokens: Sequence[int]) -> str:
    h = hashlib.sha256()
    for token in tokens:
        h.update(int(token).to_bytes(8, "little", signed=True))
    return h.hexdigest()


def compile_qwen35_block(
    *,
    block_id: str,
    tokens: Sequence[int],
    generation: EffectiveModelGeneration,
    captures: Sequence[CapturedGDNLayer],
    full_attention: FAPICSegment | Mapping[int, FAPICSegment] | None = None,
    policy: CompilePolicy = CompilePolicy(),
    orientation: TransitionOrientation = TransitionOrientation.LEFT,
    storage_dtype: str = "fp16",
) -> RC10BlockCache:
    """Compile a canonical block into a disposable model-versioned RC10 cache.

    The block remains canonical as tokens/provenance outside this function.  If
    ``tier='cold'`` no neural summaries are produced.
    """
    seams = policy.seams_for_tier()
    recurrent: list[RecurrentVariant] = []
    for seam in seams:
        # A short block may not support a wider seam.  Skip unsupported variants
        # instead of creating a malformed transition.
        if captures and seam < captures[0].keys.shape[captures[0].time_dim() % captures[0].keys.ndim]:
            recurrent.append(compile_recurrent_variant(
                captures,
                seam_width=seam,
                orientation=orientation,
                segment_id=str(block_id),
            ))

    conv_payload = {}
    max_width = 0
    for cap in captures:
        compiled = compile_layer(
            cap,
            seams=(min(seams) if seams else 0,) if seams else (0,),
            orientation=orientation,
            segment_id=str(block_id),
        ) if seams else None
        if compiled is not None and compiled.conv_boundary is not None:
            conv_payload[int(cap.layer)] = compiled.conv_boundary
        if seams:
            max_width = max(max_width, max(w for w in seams if w < len(tokens)) if any(w < len(tokens) for w in seams) else 0)

    compatibility = CacheCompatibility(
        generation_digest=generation.digest,
        cache_abi_version=generation.cache_abi_version,
        storage_dtype=str(storage_dtype),
        transition_orientation=orientation.value,
    )
    leading = tuple(map(int, tokens[:max_width])) if max_width else ()
    return RC10BlockCache(
        block_id=str(block_id),
        token_count=len(tokens),
        token_digest=token_digest(tokens),
        compatibility=compatibility,
        recurrent_variants=tuple(recurrent),
        full_attention_payload=full_attention,
        boundary=BoundaryPayload(
            leading_tokens=leading,
            trailing_conv_payload=conv_payload or None,
            max_repair_width=max_width,
        ),
        tier=policy.tier,
    )
