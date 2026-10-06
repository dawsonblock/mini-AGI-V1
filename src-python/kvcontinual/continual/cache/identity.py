from __future__ import annotations

from kvcontinual.continual.types import CacheIdentity, Compatibility


def compare_cache_identity(old: CacheIdentity, new: CacheIdentity) -> Compatibility:
    """Fail-closed execution-cache compatibility.

    Exact equality is reusable. An adapter-only change requires source replay.
    Structural changes (base weights, tokenizer, layer layout, position scheme,
    recurrence implementation) invalidate the execution cache. APPROX_COMPATIBLE
    is reserved for a future explicitly-qualified compatibility certificate and
    is never inferred heuristically here.
    """
    if old == new:
        return Compatibility.EXACT_COMPATIBLE

    structural_equal = (
        old.base_model_digest == new.base_model_digest
        and old.tokenizer_digest == new.tokenizer_digest
        and old.layer_layout_digest == new.layer_layout_digest
        and old.position_scheme == new.position_scheme
        and old.recurrence_impl == new.recurrence_impl
    )
    if structural_equal and old.adapter_set_digest != new.adapter_set_digest:
        return Compatibility.REPLAY_REQUIRED
    return Compatibility.INVALID
