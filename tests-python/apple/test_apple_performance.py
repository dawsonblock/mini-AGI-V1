from kvcontinual.execution.platforms.apple_performance import (
    help_supports,
    llama_optional_args,
    plan_apple_performance,
)


def test_memory_planner_reduces_unsafe_context():
    p = plan_apple_performance(
        unified_memory_gib=16,
        model_weight_gib=9,
        requested_context_tokens=131072,
        kv_bytes_per_token=128 * 1024,
    )
    assert p.recommended_context_tokens < 131072
    assert p.recommended_quantization_bits == 4
    assert p.warnings


def test_memory_planner_preserves_reasonable_context():
    p = plan_apple_performance(
        unified_memory_gib=64,
        model_weight_gib=8,
        requested_context_tokens=8192,
        kv_bytes_per_token=32 * 1024,
    )
    assert p.recommended_context_tokens == 8192
    assert p.estimated_headroom_gib > 0


def test_llama_args_only_use_advertised_flags():
    help_text = """
      --model-draft FNAME
      --cache-type-k TYPE
      --batch-size N
      --flash-attn [on|off|auto]
    """
    args = llama_optional_args(
        help_text,
        draft_model="draft.gguf",
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        batch_size=512,
        ubatch_size=128,
        threads=8,
        flash_attention=True,
    )
    assert args == [
        "--model-draft", "draft.gguf",
        "--cache-type-k", "q8_0",
        "--batch-size", "512",
        "--flash-attn", "on",
    ]
    assert help_supports(help_text, "--model-draft")
    assert not help_supports(help_text, "--threads")
