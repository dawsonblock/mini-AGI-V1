from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any

_GIB = 1024 ** 3


@dataclass(frozen=True)
class ApplePerformanceProfile:
    unified_memory_gib: float
    model_weight_gib: float
    requested_context_tokens: int
    recommended_context_tokens: int
    parallel_requests: int
    kv_bytes_per_token: int
    reserved_system_gib: float
    estimated_workspace_gib: float
    estimated_kv_gib: float
    estimated_headroom_gib: float
    memory_pressure_ratio: float
    recommended_backend: str
    recommended_quantization_bits: int
    use_speculative_decoding: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _round_context(tokens: int) -> int:
    if tokens <= 0:
        return 1024
    # Keep recommendations reproducible and backend-friendly.
    step = 1024
    return max(step, (tokens // step) * step)


def plan_apple_performance(
    *,
    unified_memory_gib: float,
    model_weight_gib: float,
    requested_context_tokens: int,
    parallel_requests: int = 1,
    kv_bytes_per_token: int = 64 * 1024,
    backend: str = "auto",
    system_reserve_gib: float | None = None,
    workspace_fraction: float = 0.18,
    min_workspace_gib: float = 1.5,
) -> ApplePerformanceProfile:
    """Conservative unified-memory planner.

    This is deliberately model-agnostic. `kv_bytes_per_token` must be measured or
    supplied for the exact model/runtime before the recommendation is treated as a
    qualification result. The planner exists to avoid obviously unsafe profiles,
    not to claim exact memory consumption.
    """
    if unified_memory_gib <= 0 or model_weight_gib <= 0:
        raise ValueError("memory and model size must be positive")
    if requested_context_tokens <= 0 or parallel_requests <= 0 or kv_bytes_per_token <= 0:
        raise ValueError("context, parallelism and kv_bytes_per_token must be positive")

    reserve = system_reserve_gib
    if reserve is None:
        reserve = max(4.0, unified_memory_gib * 0.18)
    workspace = max(min_workspace_gib, model_weight_gib * workspace_fraction)
    budget = max(0.0, unified_memory_gib - reserve - model_weight_gib - workspace)
    bytes_per_context_token = kv_bytes_per_token * parallel_requests
    max_ctx = int((budget * _GIB) // bytes_per_context_token) if bytes_per_context_token else 0
    recommended_ctx = _round_context(min(requested_context_tokens, max_ctx)) if max_ctx >= 1024 else 1024
    kv_gib = (recommended_ctx * bytes_per_context_token) / _GIB
    headroom = unified_memory_gib - reserve - model_weight_gib - workspace - kv_gib
    used = unified_memory_gib - max(0.0, headroom)
    pressure = min(1.5, used / unified_memory_gib)

    backend_norm = backend.lower().strip()
    if backend_norm == "auto":
        # MLX is preferred for native Apple execution and Python integration;
        # llama.cpp remains a strong GGUF/quantized alternative.
        backend_norm = "mlx" if unified_memory_gib >= 16 else "llama.cpp"
    if backend_norm not in {"mlx", "llama.cpp"}:
        raise ValueError("backend must be auto, mlx, or llama.cpp")

    # Recommendation only; actual format support is runtime/model dependent.
    if model_weight_gib > unified_memory_gib * 0.55:
        qbits = 4
    elif model_weight_gib > unified_memory_gib * 0.35:
        qbits = 6
    else:
        qbits = 8

    warnings: list[str] = []
    if max_ctx < 1024:
        warnings.append("Estimated model/workspace footprint leaves <1024 tokens of KV budget; use a smaller/quantized model.")
    if recommended_ctx < requested_context_tokens:
        warnings.append("Requested context was reduced to preserve unified-memory headroom.")
    if headroom < 2.0:
        warnings.append("Estimated headroom is below 2 GiB; memory-pressure stalls or OOM are plausible.")
    if parallel_requests > 1 and headroom < 4.0:
        warnings.append("Parallel decoding is memory-constrained under the supplied KV estimate.")

    speculative = model_weight_gib >= 6.0 and headroom >= 2.0
    return ApplePerformanceProfile(
        unified_memory_gib=round(unified_memory_gib, 3),
        model_weight_gib=round(model_weight_gib, 3),
        requested_context_tokens=int(requested_context_tokens),
        recommended_context_tokens=int(recommended_ctx),
        parallel_requests=int(parallel_requests),
        kv_bytes_per_token=int(kv_bytes_per_token),
        reserved_system_gib=round(reserve, 3),
        estimated_workspace_gib=round(workspace, 3),
        estimated_kv_gib=round(kv_gib, 3),
        estimated_headroom_gib=round(headroom, 3),
        memory_pressure_ratio=round(pressure, 4),
        recommended_backend=backend_norm,
        recommended_quantization_bits=qbits,
        use_speculative_decoding=speculative,
        warnings=tuple(warnings),
    )


def help_supports(help_text: str, flag: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_-]){re.escape(flag)}(?![A-Za-z0-9_-])", help_text))


def _help_line(help_text: str, flag: str) -> str:
    for line in help_text.splitlines():
        if flag in line:
            return line.strip()
    return ""


def llama_optional_args(
    help_text: str,
    *,
    draft_model: str | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    threads: int | None = None,
    flash_attention: bool = True,
) -> list[str]:
    """Build optional llama-server args only when the installed binary advertises them.

    The function intentionally avoids assuming a particular llama.cpp version.
    Unsupported requested features are simply omitted by this low-level helper;
    callers that require a feature should check with `help_supports` and fail closed.
    """
    args: list[str] = []
    value_options = [
        ("--model-draft", draft_model),
        ("--cache-type-k", cache_type_k),
        ("--cache-type-v", cache_type_v),
        ("--batch-size", batch_size),
        ("--ubatch-size", ubatch_size),
        ("--threads", threads),
    ]
    for flag, value in value_options:
        if value is not None and help_supports(help_text, flag):
            args.extend([flag, str(value)])

    if flash_attention and help_supports(help_text, "--flash-attn"):
        line = _help_line(help_text, "--flash-attn").lower()
        # Newer builds often accept on/off/auto; older builds use a boolean flag.
        if any(token in line for token in ("on|off", "on/off", "auto", "[on", "{on")):
            args.extend(["--flash-attn", "on"])
        else:
            args.append("--flash-attn")
    return args
