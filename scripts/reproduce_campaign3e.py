#!/usr/bin/env python3
"""Reproduce a committed Campaign 3E run under its recorded conditions.

The recorded result (v4.0.1 evidence, independently VERIFIED by the
v4.0.2 reconciler):

    v401-3e-8fam-e8-s0  — 640 steps, 8 families, exposure 8, seed 0
    fit gate 0.9875 PASS; within-family validation 0.208 -> 0.542
    (n=24 word-disjoint unseen inputs); hidden flat; compositional weak.

A reproduction is only a reproduction if the CLI configuration matches
the run's committed CONFIG.json — corpus, runner config, family set,
steps/exposure, seed, rank, learning rate and every preservation flag.
NOT the command floating around in review notes: that one names
``--config configs/campaign3e.yaml``, which binds a DIFFERENT
generation contract (max_new_tokens 32 vs the committed 16) and would
produce a new experiment, not a reproduction.

    # reconcile + print the exact command (no model loaded)
    python scripts/reproduce_campaign3e.py \
        --from-run results/campaign3e/runs/v401-3e-8fam-e8-s0

    # execute it, then independently re-verify the fresh run dir
    python scripts/reproduce_campaign3e.py \
        --from-run results/campaign3e/runs/v401-3e-8fam-e8-s0 \
        --execute --run-id campaign3e-reproduce-s0 --requalify
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

RUNNER = ROOT / "scripts" / "positive_control_fit.py"


def load_run_config(run_dir: Path) -> dict:
    p = Path(run_dir) / "CONFIG.json"
    if not p.is_file():
        raise SystemExit(f"no CONFIG.json in {run_dir}")
    return json.loads(p.read_text())


def build_command(cfg: dict, run_root: str, run_id: str) -> tuple[list, list]:
    """Derive the runner CLI from the committed config; report which
    fields were bound and which could not be recovered (a reproduction
    with unrecoverable settings is a NEW replication attempt)."""
    unbound = []
    cmd = [sys.executable, str(RUNNER),
           "--corpus", str(cfg["corpus"]),
           "--config", str(cfg["config"]),
           "--families", ",".join(cfg["families"]),
           "--seed", str(cfg["seed"]),
           "--rank", str(cfg["rank"]),
           "--learning-rate", str(cfg["learning_rate"]),
           "--run-root", run_root, "--run-id", run_id]
    # exposure is the recorded allocation; steps must equal
    # exposure * training_examples, matching the original invocation.
    contrast = cfg.get("contrast") or {}
    if contrast.get("exposure_target") is not None:
        cmd += ["--exposure", str(contrast["exposure_target"])]
        if contrast.get("steps") and contrast["exposure_target"] * \
                contrast.get("training_examples", 0) != \
                contrast["steps"]:
            unbound.append("exposure*examples != recorded steps")
    else:
        cmd += ["--steps", str(cfg["steps"])]
    if cfg.get("shuffle_labels"):
        cmd.append("--shuffle-labels")
    reh = cfg.get("rehearsal") or {}
    kl = cfg.get("kl_preservation") or {}
    if reh.get("fraction"):
        cmd += ["--rehearsal-fraction", str(reh["fraction"])]
        # corpus is a real path only when a custom rehearsal corpus was
        # recorded; the default value is a description, not a file
        rc = reh.get("corpus")
        if rc and str(rc).endswith(".jsonl"):
            cmd += ["--rehearsal-corpus", str(rc)]
    if kl.get("weight"):
        cmd += ["--kl-weight", str(kl["weight"])]
        if kl.get("every_microbatches"):
            cmd += ["--kl-every", str(kl["every_microbatches"])]
        if kl.get("prompts"):
            cmd += ["--kl-prompts", str(kl["prompts"])]
        if kl.get("corpus"):
            cmd += ["--kl-corpus", str(kl["corpus"])]
    for field in ("model_id", "model_revision", "score_oracle"):
        if not cfg.get(field):
            unbound.append(field)
    return cmd, unbound


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", required=True,
                    help="committed run directory carrying CONFIG.json")
    ap.add_argument("--run-root", default="results/campaign3e/reproduction")
    ap.add_argument("--run-id", default="campaign3e-reproduce-s0")
    ap.add_argument("--execute", action="store_true",
                    help="run the reconciled command (loads the model)")
    ap.add_argument("--requalify", action="store_true",
                    help="after execution, verify the fresh run dir with "
                         "the independent reconciler")
    args = ap.parse_args()

    cfg = load_run_config(args.from_run)
    cmd, unbound = build_command(cfg, args.run_root, args.run_id)

    print("reproduction of", args.from_run)
    print(json.dumps({
        "bound_fields": {
            "corpus": cfg["corpus"], "config": cfg["config"],
            "families": cfg["families"],
            "steps": cfg.get("steps"), "seed": cfg.get("seed"),
            "rank": cfg.get("rank"),
            "learning_rate": cfg.get("learning_rate"),
            "model_id": cfg.get("model_id"),
            "model_revision": cfg.get("model_revision"),
            "max_new_tokens": cfg.get("max_new_tokens"),
            "prompt_contract_digest":
                cfg.get("prompt_contract_digest"),
            "score_oracle": cfg.get("score_oracle"),
        },
        "unbound": unbound,
        "label": "new replication attempt — unrecoverable settings"
                 if unbound else "exact-condition reproduction",
        "command": cmd}, indent=1))
    if unbound:
        print("NOTE: settings above could not be recovered from the "
              "committed CONFIG — the result is a new replication "
              "attempt, not an exact reproduction", file=sys.stderr)

    if not args.execute:
        print("\n(dry run — pass --execute to run)")
        return 0
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if args.requalify:
        from minagi.v161.evidence_reconciliation import verify_run
        rep = verify_run(ROOT / args.run_root / args.run_id,
                         repo_root=ROOT)
        print(json.dumps({"fresh_run": rep["status"],
                          "mismatches": rep["mismatches"]}, indent=1))
        hist = json.loads((Path(args.from_run) / "METRICS.json"
                           ).read_text())
        fresh_m = (ROOT / args.run_root / args.run_id / "METRICS.json")
        if fresh_m.is_file():
            fm = json.loads(fresh_m.read_text())
            print(json.dumps({
                "historical": {"frozen_val": hist["frozen"]["validation"],
                               "adapted_val":
                               hist["adapted"]["validation"]},
                "reproduction": {"frozen_val": fm["frozen"]["validation"],
                                 "adapted_val":
                                 fm["adapted"]["validation"]}},
                indent=1))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
