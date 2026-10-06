#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except Exception:
        return ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit Apple performance qualification evidence. This does not self-authorize promotion.")
    ap.add_argument("--benchmark-json", required=True)
    ap.add_argument("--model-file")
    ap.add_argument("--max-ttft", type=float)
    ap.add_argument("--min-decode-tps", type=float)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bench = json.loads(Path(args.benchmark_json).read_text())
    s = bench["summary"]
    gates = {
        "ttft_ok": args.max_ttft is None or s["ttft_median_s"] <= args.max_ttft,
        "decode_tps_ok": args.min_decode_tps is None or s["decode_tps_median"] >= args.min_decode_tps,
    }
    model_digest = None
    if args.model_file:
        p = Path(args.model_file)
        model_digest = _sha256(p) if p.is_file() else None
    payload = {
        "schema": "egai.apple.performance-qualification-evidence.v1",
        "authoritative_promotion": False,
        "note": "Evidence artifact only. Independent qualification/promotion authority must bind and sign it before deployment.",
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "macos": _cmd(["sw_vers", "-productVersion"]),
            "hardware_model": _cmd(["sysctl", "-n", "hw.model"]),
            "chip": _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "memory_bytes": _cmd(["sysctl", "-n", "hw.memsize"]),
        },
        "software": {
            "python": sys.version.split()[0],
            "mlx_version": _cmd([sys.executable, "-c", "import mlx; print(getattr(mlx,'__version__','unknown'))"]),
            "mlx_lm_version": _cmd([sys.executable, "-c", "import importlib.metadata as m; print(m.version('mlx-lm'))"]),
            "llama_server_version": _cmd(["llama-server", "--version"]),
        },
        "model_digest": model_digest,
        "benchmark_summary": s,
        "gates": gates,
        "passed_local_performance_gates": all(gates.values()),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["evidence_digest"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 4


if __name__ == "__main__":
    raise SystemExit(main())
