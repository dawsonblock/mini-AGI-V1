#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main() -> int:
    results = []
    results.append(check("OS", platform.system() == "Darwin", f"{platform.system()} {platform.release()}"))
    results.append(check("CPU", platform.machine() == "arm64", platform.machine()))
    mem = None
    if platform.system() == "Darwin":
        try:
            mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        except Exception:
            pass
    if mem:
        gib = mem / (1024 ** 3)
        results.append(check("Unified memory", gib >= 16, f"{gib:.1f} GiB"))
    else:
        print("[WARN] Unified memory: could not query hw.memsize")
    results.append(check("Python", sys.version_info >= (3, 11), sys.version.split()[0]))
    llama = shutil.which("llama-server")
    mlx = shutil.which("mlx_lm.server") or shutil.which("mlx_lm")
    results.append(check("Inference backend", bool(llama or mlx), f"llama-server={llama or '-'} mlx={mlx or '-'}"))

    endpoint = os.getenv("KVCONTINUAL_MODEL_ENDPOINT", "http://127.0.0.1:8080/v1")
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=2) as response:
            body = json.loads(response.read().decode("utf-8"))
            results.append(check("Backend API", response.status < 400, f"{endpoint} models={len(body.get('data', []))}"))
    except Exception as exc:
        print(f"[WARN] Backend API: not running at {endpoint}: {exc}")

    print("\nMac mode intentionally delegates model execution to llama.cpp/Metal or MLX.")
    print("The custom CUDA KVMem kernels are not available on Apple GPUs; the persistent continual-memory plane remains available.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
