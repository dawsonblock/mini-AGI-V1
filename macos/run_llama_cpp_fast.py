#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SRC = ROOT / "src-python"
if str(RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SRC))

from kvcontinual.execution.platforms.apple_performance import help_supports, llama_optional_args


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _help(server: str) -> str:
    proc = subprocess.run([server, "--help"], text=True, capture_output=True, check=False)
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def main() -> int:
    server = os.getenv("LLAMA_SERVER") or shutil.which("llama-server")
    if not server:
        print("llama-server not found; install llama.cpp or set LLAMA_SERVER", file=sys.stderr)
        return 2
    help_text = _help(server)
    model = os.getenv("QW3_MODEL")
    hf_model = os.getenv("QW3_HF_MODEL")
    if not model and not hf_model:
        print("Set QW3_MODEL=/path/model.gguf or QW3_HF_MODEL=<repo-or-repo:file>", file=sys.stderr)
        return 2

    ctx = _env_int("QW3_MAC_CONTEXT", 32768)
    parallel = _env_int("QW3_MAC_PARALLEL", 1)
    port = _env_int("QW3_MAC_BACKEND_PORT", 8080)
    gpu_layers = _env_int("QW3_MAC_GPU_LAYERS", 999)

    cmd = [server]
    if model:
        cmd += ["-m", model]
    else:
        cmd += ["-hf", hf_model]
    cmd += ["-ngl", str(gpu_layers), "-c", str(ctx), "-np", str(parallel)]

    lora = os.getenv("QW3_LLAMA_LORA")
    if lora:
        cmd += ["--lora", lora]
    if os.getenv("QW3_LLAMA_LORA_HOT_SWITCH", "0") == "1" and help_supports(help_text, "--lora-init-without-apply"):
        cmd.append("--lora-init-without-apply")

    draft = os.getenv("QW3_DRAFT_MODEL")
    if draft and not help_supports(help_text, "--model-draft"):
        print("Requested speculative decoding, but this llama-server does not advertise --model-draft; refusing silent fallback.", file=sys.stderr)
        return 3

    optional = llama_optional_args(
        help_text,
        draft_model=draft,
        cache_type_k=os.getenv("QW3_MAC_CACHE_K") or None,
        cache_type_v=os.getenv("QW3_MAC_CACHE_V") or None,
        batch_size=_env_int("QW3_MAC_BATCH", 512) if os.getenv("QW3_MAC_BATCH") else None,
        ubatch_size=_env_int("QW3_MAC_UBATCH", 128) if os.getenv("QW3_MAC_UBATCH") else None,
        threads=_env_int("QW3_MAC_THREADS", 0) or None,
        flash_attention=os.getenv("QW3_MAC_FLASH_ATTN", "1") == "1",
    )
    cmd += optional
    cmd += ["--host", "127.0.0.1", "--port", str(port)]

    if os.getenv("QW3_PRINT_COMMAND", "0") == "1":
        print(" ".join(shlex.quote(x) for x in cmd))
        return 0
    os.execv(server, cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
