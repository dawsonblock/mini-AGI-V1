#!/usr/bin/env python3
"""Compute the canonical RC11 ExecutionIdentity digest from explicit JSON."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
from kvcontinual.execution.types import ExecutionIdentity, ModelIdentity


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("identity_json", help="JSON containing model + execution identity fields")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    raw = json.loads(Path(a.identity_json).read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise SystemExit("identity JSON must contain a model object")
    model = ModelIdentity(**raw["model"])
    execution_fields = {k: v for k, v in raw.items() if k not in {"model", "digest"}}
    execution = ExecutionIdentity(model=model, **execution_fields)
    payload = execution.to_dict()
    if raw.get("digest") is not None and raw.get("digest") != payload["digest"]:
        raise SystemExit("provided identity digest does not match canonical ExecutionIdentity")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
