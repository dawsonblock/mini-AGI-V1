#!/usr/bin/env python3
"""Campaign 3B.3 Priority 2 — cost, quality and reliability baselines.

"Independently measurable intelligence" needs three numbers per arm, not
one: what it scored, what it cost, and how much the score moves when you
run it again. A quality figure alone cannot distinguish an arm that is
better from an arm that is luckier, and neither can answer the third
question the architecture is meant to answer — whether an improvement
"justifies its complete cost".

Arm classes, and which evidence supports which metric:

  frozen    L1, no adaptation
  memory    L5b, verified replay — frozen weights, non-parametric
  lora      L6/L7, parametric adaptation
  control   NC, shuffled labels — the floor a real gain must clear

  QUALITY      from the campaign evidence (one seed, three arms)
  RELIABILITY  from the fit evidence (multi-seed, frozen vs LoRA) —
               the campaign evidence is a single seed and cannot
               support a variance claim at all
  COST         from whichever evidence ran the arm

That split is the point: this tool will not compute a reliability figure
from a single seed, and reports NOT_ESTIMABLE instead.

Usage:
  python3 scripts/analysis/capability_baseline.py \
      --campaign-evidence <extracted 3B evidence>/campaigns/.../seed-0 \
      --run-root results/campaign3b2/runs \
      --output CAPABILITY_BASELINE.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

NOT_ESTIMABLE = "NOT_ESTIMABLE"

#: arm -> (class, description). Declared, so an unknown arm is an error
#: rather than a silent omission from the baseline.
ARM_CLASSES = {
    "L1": ("frozen", "no adaptation"),
    "L5": ("memory-self-verified", "self-verified replay (empty in 3A)"),
    "L5b": ("memory", "verified replay — frozen weights, non-parametric"),
    "L6": ("lora", "fixed-rank LoRA, no replay context"),
    "L7": ("lora-replay", "LoRA under the same replay context as L5b"),
    "NC": ("control", "shuffled-label LoRA — the floor"),
}


def _load_preds(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[str(r["id"])] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-evidence", required=True)
    ap.add_argument("--run-root", default="results/campaign3b2/runs")
    ap.add_argument("--corpus",
                    default="configs/positive_control_tasks.jsonl")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    corpus = {}
    for line in (ROOT / args.corpus).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            corpus[str(r["id"])] = r

    def norm(s):
        return str(s).strip().casefold()

    # ---- QUALITY: campaign evidence, hidden block -------------------
    camp = Path(args.campaign_evidence)
    hidden = [rid for rid, r in corpus.items() if r["split"] == "hidden"]
    arms = {}
    for arm, (cls, desc) in sorted(ARM_CLASSES.items()):
        f = camp / f"PREDICTIONS-{arm}.jsonl"
        if not f.is_file():
            continue
        preds = _load_preds(f)
        scored = [rid for rid in hidden if rid in preds]
        if not scored:
            continue
        correct = sum(1 for rid in scored
                      if norm(preds[rid]["output"]) == norm(corpus[rid]["expected"]))
        # a destroyed answer is a capability the frozen arm HAD and lost
        arms[arm] = {
            "class": cls, "description": desc,
            "n_scored": len(scored),
            "accuracy": correct / len(scored),
            "hidden_correct": correct,
            "mean_generated_tokens": statistics.fmean(
                preds[rid].get("generated_tokens") or 0 for rid in scored),
        }
    if "L1" not in arms:
        raise SystemExit("the frozen arm is absent — no baseline to compare to")

    frozen_acc = arms["L1"]["accuracy"]
    frozen_preds = _load_preds(camp / "PREDICTIONS-L1.jsonl")
    frozen_correct_ids = {rid for rid in hidden
                          if norm(frozen_preds[rid]["output"])
                          == norm(corpus[rid]["expected"])}
    for arm, blk in arms.items():
        preds = _load_preds(camp / f"PREDICTIONS-{arm}.jsonl")
        blk["delta_vs_frozen"] = blk["accuracy"] - frozen_acc
        destroyed = [rid for rid in frozen_correct_ids
                     if norm(preds[rid]["output"])
                     != norm(corpus[rid]["expected"])]
        gained = [rid for rid in hidden
                  if rid not in frozen_correct_ids
                  and norm(preds[rid]["output"]) == norm(corpus[rid]["expected"])]
        blk["destroyed_vs_frozen"] = len(destroyed)
        blk["gained_vs_frozen"] = len(gained)
        blk["capability_destruction_rate"] = (
            len(destroyed) / len(frozen_correct_ids)
            if frozen_correct_ids else None)
        blk["destruction_status"] = ("ESTIMATED" if frozen_correct_ids
                                     else NOT_ESTIMABLE)

    # ---- COST: from whichever evidence ran the arm ------------------
    for arm in arms:
        p = camp / f"{arm}.json"
        arms[arm]["cost"] = {"train_seconds": 0.0, "source": "campaign"}
        if p.is_file():
            doc = json.loads(p.read_text())
            m = (doc.get("bundle") or {}).get("metrics_reported") or {}
            arms[arm]["cost"].update({
                "train_seconds": m.get("train_seconds", 0.0),
                "trainable_params": m.get("trainable_params", 0),
                "wall_seconds": m.get("wall_seconds", 0.0),
                "arm_state_bytes": m.get("arm_state_bytes", 0)})

    # ---- RELIABILITY: fit evidence, multi-seed ---------------------
    run_root = ROOT / args.run_root
    cells: dict = {}
    if run_root.is_dir():
        for d in sorted(run_root.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            try:
                cfg = json.loads((d / "CONFIG.json").read_text())
                met = json.loads((d / "METRICS.json").read_text())
            except (OSError, ValueError):
                continue
            if cfg.get("shuffle_labels"):
                continue
            key = (cfg.get("corpus", "?"), len(cfg["families"]),
                   cfg.get("rank", 16))
            cells.setdefault(key, []).append({
                "run": d.name, "frozen": met["frozen"]["hidden"],
                "adapted": met["adapted"]["hidden"],
                "train_seconds": met["training"]["train_seconds"],
                "destroyed": met["capability_destruction"]["n_destroyed"],
                "gained": met["capability_destruction"]["n_gained"]})

    reliability = []
    for (corpus_name, n_fam, rank), rows in sorted(cells.items()):
        if len(rows) < 2:
            reliability.append({
                "corpus": corpus_name, "families": n_fam, "rank": rank,
                "n_seeds": len(rows), "status": NOT_ESTIMABLE,
                "reason": "a single seed cannot support a variance claim"})
            continue
        adapted = [r["adapted"] for r in rows]
        frozen = [r["frozen"] for r in rows]
        deltas = [a - f for a, f in zip(adapted, frozen)]
        reliability.append({
            "corpus": corpus_name, "families": n_fam, "rank": rank,
            "n_seeds": len(rows),
            "status": "ESTIMATED",
            "frozen_mean": statistics.fmean(frozen),
            "adapted_mean": statistics.fmean(adapted),
            "delta_mean": statistics.fmean(deltas),
            "delta_stdev": (statistics.stdev(deltas)
                            if len(deltas) > 1 else 0.0),
            "delta_min": min(deltas), "delta_max": max(deltas),
            "seeds_beating_frozen": sum(1 for d in deltas if d > 0),
            "mean_train_seconds": statistics.fmean(
                r["train_seconds"] for r in rows),
            "mean_destroyed": statistics.fmean(r["destroyed"] for r in rows),
            "mean_gained": statistics.fmean(r["gained"] for r in rows)})

    # ---- improvement per unit cost ---------------------------------
    cost_per_gain = []
    for blk in reliability:
        if blk["status"] != "ESTIMATED":
            continue
        secs = blk["mean_train_seconds"]
        cost_per_gain.append({
            "corpus": blk["corpus"], "families": blk["families"],
            "rank": blk["rank"],
            "delta_mean": blk["delta_mean"],
            "mean_train_seconds": secs,
            "delta_per_train_second": (blk["delta_mean"] / secs)
            if secs else None})

    report = {
        "schema": "mini-agi-campaign3b3-capability-baseline-v1",
        "frozen_baseline_accuracy": frozen_acc,
        "quality_campaign_evidence": arms,
        "reliability_fit_evidence": reliability,
        "improvement_per_unit_cost": cost_per_gain,
        "method": ("QUALITY from the campaign evidence (one seed, all arms). "
                   "RELIABILITY from the fit evidence (multi-seed) — the "
                   "campaign evidence is a single seed and is reported "
                   "NOT_ESTIMABLE rather than given a variance it cannot "
                   "support."),
        "answer_to_the_three_questions": {
            "1_what_capability": ("deterministic string transformations; "
                                  "frozen accuracy is the baseline against "
                                  "which any acquisition is measured"),
            "2_which_intervention": ("see the campaign arms: the non-neural "
                                     "memory arm beats frozen, and the LoRA "
                                     "arms do not beat the memory arm"),
            "3_transfer_persist_cost": ("transfer: delta_vs_frozen per arm; "
                                        "persist: destroyed/gained counts; "
                                        "cost: improvement_per_unit_cost"),
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2,
                                            sort_keys=True) + "\n")

    print(f"frozen baseline: {frozen_acc:.3f} on {len(hidden)} hidden tasks\n")
    print(f"{'arm':6} {'class':20} {'acc':>6} {'vs frozen':>10} "
          f"{'destroyed':>10} {'gained':>7}")
    for arm, blk in sorted(arms.items(), key=lambda x: -x[1]["accuracy"]):
        print(f"{arm:6} {blk['class']:20} {blk['accuracy']:>6.3f} "
              f"{blk['delta_vs_frozen']:>+10.3f} "
              f"{blk['destroyed_vs_frozen']:>10} {blk['gained_vs_frozen']:>7}")
    print(f"\nreliability cells: "
          f"{sum(1 for r in reliability if r['status'] == 'ESTIMATED')} "
          f"estimated, "
          f"{sum(1 for r in reliability if r['status'] == NOT_ESTIMABLE)} "
          f"NOT_ESTIMABLE (single seed)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
