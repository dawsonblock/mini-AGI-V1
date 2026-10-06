#!/usr/bin/env python3
"""v16.2 Phase 2 — import probe.

Proves the packaged modules import in the deployed environment, not just in
the build environment. Covers the module groups the campaign depends on:

  minagi, minagi.v161.*, RC14.7 evidence modules, grounded replay,
  Dream-RSI, plasticity controller, Colab runtime, CUDA backend, PEFT trainer.

Exits 0 only if every probe imports cleanly. Writes IMPORT_PROBE.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import detect_root, ensure_path

PROBES = {
    "minagi": ["minagi"],
    "minagi.v161": [
        "minagi.v161",
        "minagi.v161.campaign_plan",
        "minagi.v161.converged_evidence",
        "minagi.v161.dataset_manifest",
        "minagi.v161.evaluator_registry",
        "minagi.v161.evaluators",
        "minagi.v161.executed_run",
        "minagi.v161.runtime_closure3",
    ],
    "rc14.7_evidence": [
        "minagi.rc14.authority_artifacts",
        "minagi.rc14.executed_runs",
        "minagi.rc14.experiment_protocol",
        "minagi.rc14.falsification",
        "minagi.rc14.independent_reproduction",
        "minagi.rc14.qualification",
    ],
    "grounded_replay": [
        "minagi.rc14.replay_policy",
        "minagi.egai.replay",
        "egai.replay.policy",
        "egai.replay.vault",
        "egai.replay.world",
        "egai.authority.replay_policy",
    ],
    "dream_rsi": [
        "dream_rsi_governed",
        "dream_rsi_governed.experiment",
        "dream_rsi_governed.ledger",
        "dream_rsi_governed.promotion",
        "dream_rsi_governed.qualification",
        "dream_rsi_governed.replay",
        "dream_rsi_governed.canary",
    ],
    "plasticity_controller": [
        "minagi.v16.control_plane",
        "minagi.v16.plasticity_execution",
    ],
    "colab_runtime": [
        "minagi.platforms.colab.doctor",
        "minagi.platforms.colab.environment",
        "minagi.platforms.colab.storage",
    ],
    "cuda_backend": [
        "minagi.platforms.cuda.evaluate",
        "minagi.platforms.cuda.hf_runtime",
    ],
    "peft_trainer": [
        "minagi.platforms.cuda.peft_trainer",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="release root (auto-detected when run inside the tree)")
    ap.add_argument("--output", default="IMPORT_PROBE.json")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else detect_root()
    ensure_path(root)

    results = []
    ok_all = True
    for group, modules in PROBES.items():
        for mod in modules:
            try:
                importlib.import_module(mod)
                results.append({"group": group, "module": mod, "status": "OK"})
            except Exception as exc:  # noqa: BLE001 - report every failure verbatim
                ok_all = False
                results.append({
                    "group": group,
                    "module": mod,
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().splitlines()[-6:],
                })

    report = {
        "schema": "mini-agi-v16.2-import-probe-v1",
        "root": str(root),
        "python": sys.version.split()[0],
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "OK"),
        "failed": [r for r in results if r["status"] != "OK"],
        "results": results,
        "status": "PASS" if ok_all else "FAIL",
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: report[k] for k in ("schema", "total", "passed", "status")}, indent=2))
    for r in report["failed"]:
        print(f"FAIL {r['module']}: {r['error']}", file=sys.stderr)
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
