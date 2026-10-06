#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SRC = ROOT / "src-python"
if str(RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SRC))

from kvcontinual.execution.platforms.apple_performance import plan_apple_performance


def _sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:
        return ""


def _memory_gib() -> float:
    raw = _sysctl("hw.memsize")
    if not raw:
        raise RuntimeError("could not read hw.memsize; this command targets macOS")
    return int(raw) / (1024 ** 3)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a conservative Apple Silicon inference profile.")
    ap.add_argument("--model-weight-gib", type=float, required=True)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--kv-bytes-per-token", type=int, default=64 * 1024,
                    help="Measured/estimated KV+recurrent bytes per token per request")
    ap.add_argument("--backend", choices=["auto", "mlx", "llama.cpp"], default="auto")
    ap.add_argument("--json-out")
    ap.add_argument("--env-out")
    args = ap.parse_args()

    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        print("warning: profile generation is intended for Apple Silicon macOS", file=sys.stderr)

    p = plan_apple_performance(
        unified_memory_gib=_memory_gib(),
        model_weight_gib=args.model_weight_gib,
        requested_context_tokens=args.context,
        parallel_requests=args.parallel,
        kv_bytes_per_token=args.kv_bytes_per_token,
        backend=args.backend,
    )
    payload = p.to_dict()
    payload["hardware"] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "brand": _sysctl("machdep.cpu.brand_string") or _sysctl("hw.model"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.env_out:
        lines = [
            f"export QW3_MAC_CONTEXT={p.recommended_context_tokens}",
            f"export QW3_MAC_PARALLEL={p.parallel_requests}",
            f"export QW3_APPLE_BACKEND={p.recommended_backend}",
            f"export QW3_APPLE_RECOMMENDED_QUANT_BITS={p.recommended_quantization_bits}",
            "export QW3_MAC_FLASH_ATTN=1",
        ]
        Path(args.env_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
