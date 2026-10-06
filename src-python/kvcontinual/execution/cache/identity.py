from __future__ import annotations

from kvcontinual.execution.types import CacheIdentity, Compatibility, ExecutionIdentity


def compare_execution_identity(old: ExecutionIdentity, new: ExecutionIdentity) -> Compatibility:
    if old == new:
        return Compatibility.EXACT_COMPATIBLE
    if old.model.base_weights != new.model.base_weights or old.model.tokenizer != new.model.tokenizer:
        return Compatibility.INVALID
    # Any weight-bearing model change is replay-required by default.
    if old.model != new.model or old.runtime_learning_state != new.runtime_learning_state:
        return Compatibility.REPLAY_REQUIRED
    # Same model semantics but changed numerical/runtime implementation.
    numerical_fields_equal = (
        old.recurrence_impl == new.recurrence_impl
        and old.tensor_layout == new.tensor_layout
        and old.activation_dtype == new.activation_dtype
        and old.cache_dtype == new.cache_dtype
    )
    if numerical_fields_equal and old.kernel_abi != new.kernel_abi:
        return Compatibility.NUMERICALLY_COMPATIBLE
    return Compatibility.REPLAY_REQUIRED


def compare_cache_identity(old: CacheIdentity, new: CacheIdentity) -> Compatibility:
    if old == new:
        return Compatibility.EXACT_COMPATIBLE
    structural = (
        old.base_model_digest == new.base_model_digest
        and old.tokenizer_digest == new.tokenizer_digest
        and old.layer_layout_digest == new.layer_layout_digest
        and old.position_scheme == new.position_scheme
        and old.recurrence_impl == new.recurrence_impl
    )
    if structural and old.adapter_set_digest != new.adapter_set_digest:
        return Compatibility.REPLAY_REQUIRED
    if old.base_model_digest == new.base_model_digest and old.tokenizer_digest == new.tokenizer_digest:
        return Compatibility.REPLAY_REQUIRED
    return Compatibility.INVALID
