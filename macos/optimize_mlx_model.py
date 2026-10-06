#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _help() -> str:
    p = subprocess.run([sys.executable, "-m", "mlx_lm.convert", "--help"], text=True, capture_output=True, check=False)
    return (p.stdout or "") + "\n" + (p.stderr or "")


def _has(h: str, flag: str) -> bool:
    return flag in h


def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big")); h.update(rel)
        data = p.read_bytes(); h.update(len(data).to_bytes(8, "big")); h.update(hashlib.sha256(data).digest())
    return "sha256:" + h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a local quantized MLX model only if the installed MLX-LM advertises the required converter flags.")
    ap.add_argument("model")
    ap.add_argument("output")
    ap.add_argument("--bits", type=int, choices=[4, 8], default=4)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    h = _help()
    required = ["--hf-path", "--mlx-path", "--quantize", "--q-bits"]
    missing = [x for x in required if not _has(h, x)]
    if missing:
        print(json.dumps({"supported": False, "missing_flags": missing}, indent=2))
        return 3
    out = Path(args.output).expanduser().resolve()
    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", args.model,
        "--mlx-path", str(out),
        "--quantize",
        "--q-bits", str(args.bits),
    ]
    print("COMMAND:", " ".join(cmd))
    if not args.execute:
        print("Dry run only. Add --execute after reviewing the command.")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    manifest = {
        "source_model": args.model,
        "quantization_bits": args.bits,
        "output": str(out),
        "tree_digest": _tree_digest(out),
    }
    (out / "EGAI_APPLE_MODEL_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
