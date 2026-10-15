#!/usr/bin/env python3
"""Campaign 3B.3 P4 — matched multi-seed comparison.

Four principal training conditions against the same corpus, contract,
rank/alpha/scaling, optimizer, steps and budget — varying ONLY the
preservation mechanism:

  fixed      plain fixed-rank LoRA (the Campaign 3B.1 condition)
  rehearsal  LoRA + verified rehearsal interleaved into the GRADIENT
             stream (P2) — NOT inference-time replay (L7)
  kl         LoRA + frozen-reference KL regularization (P3)
  both       rehearsal + KL

Seeds 0,1,2 — matched per condition so pairings are seed-aligned.

Every run is an immutable positive_control_fit run; the aggregator
pairs per-row hidden outcomes against frozen under the STRICT oracle
(exact-match-v2) and reports gained/destroyed/net, family-level
effects, a sign-test p, and train_seconds cost.

Usage:
  # run all cells sequentially (each run is ~2-5 min on CPU)
  python3 scripts/analysis/campaign3b3_multiseed.py run

  # aggregate after runs exist
  python3 scripts/analysis/campaign3b3_multiseed.py analyze
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results/campaign3b3/runs"
REPORT = ROOT / "results/campaign3b3/MULTISEED_COMPARISON.json"

FAMILIES = "pc-upper,pc-rev"
STEPS = 300
RANK = 16
SEEDS = [0, 1, 2]
CONDITIONS = {
    "fixed": [],
    "rehearsal": ["--rehearsal-fraction", "0.25"],
    # KL evaluated every 4th microbatch — a bounded preservation
    # cadence matching the rehearsal schedule but on offset indices
    # (rehearsal fires micro % 4 == 3, KL fires micro % 4 == 0)
    "kl": ["--kl-weight", "1.0", "--kl-every", "4"],
    "both": ["--rehearsal-fraction", "0.25", "--kl-weight", "1.0",
             "--kl-every", "4"],
}


def run_cell(cond: str, seed: int, extra: list[str]) -> Path:
    run_id = f"c3b3-{cond}-s{seed}"
    final = RUNS / run_id
    if final.is_dir():
        print(f"  {run_id}: exists, skipping")
        return final
    cmd = [sys.executable, str(ROOT / "scripts/positive_control_fit.py"),
           "--families", FAMILIES, "--steps", str(STEPS),
           "--rank", str(RANK), "--seed", str(seed),
           "--run-root", str(RUNS.relative_to(ROOT)),
           "--run-id", run_id] + CONDITIONS[cond] + extra
    print(f"  {run_id}: {' '.join(cmd[2:])}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        print(f"  {run_id}: FAILED rc={r.returncode}")
    else:
        for ln in r.stdout.splitlines():
            if ln.startswith(("fit gate", "published")):
                print(f"    {ln.strip()}")
    return final


def cmd_run(args) -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    for cond, seed in product(CONDITIONS, SEEDS):
        if only and f"{cond}-s{seed}" not in only and cond not in only:
            continue
        run_cell(cond, seed, args.extra or [])
    return 0


def _sign_test_p(gained: int, destroyed: int) -> float:
    """Exact two-sided sign test on net wins/losses."""
    n = gained + destroyed
    if n == 0:
        return 1.0
    k = min(gained, destroyed)
    tail = sum(_c(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _c(n: int, k: int) -> int:
    from math import comb
    return comb(n, k)


def _load_cell(run_id: str) -> dict | None:
    d = RUNS / run_id
    if not (d / "METRICS.json").is_file():
        return None
    m = json.loads((d / "METRICS.json").read_text())
    rc = (d / "RUN_COMMIT.json")
    status = json.loads(rc.read_text())["status"] if rc.is_file() \
        else "?"
    receipt = {}
    tr = d / "TRAINING_RECEIPT.json"
    if tr.is_file():
        receipt = json.loads(tr.read_text())
    return {
        "run_id": run_id, "status": status,
        "fit_accuracy": m["fit_gate"]["adapted_subset_accuracy"],
        "fit_passed": m["fit_gate"]["passed"],
        "frozen_hidden": m["frozen"]["hidden"],
        "adapted_hidden": m["adapted"]["hidden"],
        "destroyed_ids": m["capability_destruction"]["destroyed_ids"],
        "gained_ids": m["capability_destruction"]["gained_ids"],
        "n_frozen_correct":
            m["capability_destruction"]["n_frozen_correct_hidden"],
        "train_seconds": m["training"]["train_seconds"],
        "rehearsal_presentations":
            receipt.get("rehearsal_presentations", 0),
        "kl_term_mean": (receipt.get("kl_preservation") or {})
            .get("kl_term_mean"),
        "adapter_delta_l2": receipt.get("adapter_delta_l2"),
    }


def cmd_analyze(_args) -> int:
    cells = {}
    for cond in CONDITIONS:
        for seed in SEEDS:
            c = _load_cell(f"c3b3-{cond}-s{seed}")
            if c:
                cells[(cond, seed)] = c

    if not cells:
        raise SystemExit(f"no completed runs under {RUNS}")

    frozen_hidden = None
    for c in cells.values():
        frozen_hidden = c["frozen_hidden"]
    out = {"schema": "campaign3b3-multiseed-v1", "oracle": "exact-match-v2",
           "families": FAMILIES, "steps": STEPS, "rank": RANK,
           "seeds": SEEDS,
           "conditions": {c: {"flags": CONDITIONS[c]}
                          for c in CONDITIONS},
           "cells": {f"{c}-s{s}": v for (c, s), v in cells.items()}}

    summary = {}
    for cond in CONDITIONS:
        cs = [cells[(cond, s)] for s in SEEDS if (cond, s) in cells]
        if not cs:
            continue
        nets = [len(c["gained_ids"]) - len(c["destroyed_ids"])
                for c in cs]
        gained = sum(len(c["gained_ids"]) for c in cs)
        destroyed = sum(len(c["destroyed_ids"]) for c in cs)
        # family-level adapted outcome per row id prefix
        per_row = {}
        for c in cs:
            mets = json.loads(
                (RUNS / c["run_id"] / "METRICS.json").read_text())
            for rid, ok in mets["hidden_correct"].items():
                per_row.setdefault(rid, []).append(int(ok))
        fam_acc = {}
        for rid, oks in per_row.items():
            fam = rid.rsplit("-", 1)[0]
            fam_acc.setdefault(fam, []).extend(oks)
        summary[cond] = {
            "n_seeds": len(cs),
            "adapted_hidden_mean": statistics.fmean(
                c["adapted_hidden"] for c in cs),
            "adapted_hidden_per_seed": [c["adapted_hidden"] for c in cs],
            "fit_mean": statistics.fmean(c["fit_accuracy"] for c in cs),
            "fit_passed_seeds": sum(1 for c in cs if c["fit_passed"]),
            "gained_total": gained, "destroyed_total": destroyed,
            "net_per_seed": nets,
            "net_mean": statistics.fmean(nets),
            "sign_test_p_vs_frozen": _sign_test_p(gained, destroyed),
            "train_seconds_mean": statistics.fmean(
                c["train_seconds"] for c in cs),
            "family_accuracy": {k: statistics.fmean(v)
                                for k, v in sorted(fam_acc.items())},
        }
    out["frozen_hidden_baseline"] = frozen_hidden
    out["condition_summary"] = summary

    # reference arms under the strict oracle: the campaign rescore
    # summary carries frozen (L1) and verified-replay (L5b) hidden
    # rows directly
    ref = {}
    summ = ROOT / "results/campaign3b2/RESCORE_SUMMARY.json"
    if summ.is_file():
        arms = (json.loads(summ.read_text())
                .get("campaign3b", {}).get("arms", {}))
        for arm in ("L1", "L5b", "L6", "L7", "NC"):
            if arm in arms:
                a = arms[arm]
                ref[arm] = {"v1_hidden": a.get("hidden_v1"),
                            "v2_hidden": a.get("hidden_v2"),
                            "n_rows": a.get("n_hidden")}
        if "L1" in ref and "L5b" in ref:
            ref["replay_minus_frozen_v2"] = (
                ref["L5b"]["v2_hidden"] - ref["L1"]["v2_hidden"])
            ref["replay_advantage_note"] = (
                "single-seed campaign arm, 12 hidden rows; the strict "
                "net advantage is +1 row — a paired sign test on the "
                "flips gives p=1.0, so the replay edge is NOT "
                "statistically established "
                "(see CAMPAIGN3B_RESCORE_REPORT.md)")
    out["reference_arms_v2"] = ref

    # honest statistical-resolution note: hidden is 12 rows x 3 seeds =
    # 36 paired observations per condition — sign tests are powered
    # only for large effects at this sample size
    out["resolution_note"] = (
        "hidden split = 12 rows; per condition the paired test pools "
        "seeds (12 x n_seeds cells) but rows are shared, so effective "
        "resolution is rows x seeds with row-level dependence — small "
        "effects are underpowered by construction; report magnitudes, "
        "not just p-values")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"{'condition':10} {'acc':>6} {'fit':>5} {'+/-':>6} "
          f"{'net/seed':>14} {'p':>6} {'sec':>6}")
    for cond, s in sorted(summary.items()):
        print(f"{cond:10} {s['adapted_hidden_mean']:>6.3f} "
              f"{s['fit_mean']:>5.2f} "
              f"{s['gained_total']:>3}/{s['destroyed_total']:<2} "
              f"{str(s['net_per_seed']):>14} "
              f"{s['sign_test_p_vs_frozen']:>6.3f} "
              f"{s['train_seconds_mean']:>6.0f}")
    print(f"frozen hidden baseline: {frozen_hidden}")
    print(f"wrote {REPORT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run")
    rp.add_argument("--only", default=None,
                    help="comma subset e.g. 'rehearsal-s0,both'")
    rp.add_argument("--extra", nargs="*", default=None,
                    help="extra args forwarded to the fit runner")
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    return cmd_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
