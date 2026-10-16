#!/usr/bin/env python3
"""v4.1.0 confirmatory analysis — gate evaluation for the intervention
matrix (G5 generalization, G6 retention, plus cost/transfer reporting).

Reads committed run directories produced by capability_matrix.py across
the preregistered seed set, joins prediction records to the corpus for
family labels, and evaluates the preregistered gates declared in
configs/capability_acquisition/confirmation.yaml:

  * fit_min                       — adapted subset (fit-gate) accuracy
  * within_holdout_min_delta      — G5: adapted-frozen on the
                                    INDEPENDENT within-family surface,
                                    per-seed mean over family clusters
  * retention_max_destruction_rate— G6: share of previously-correct
                                    retention probes destroyed
  * security_max_violations       — frozen-vs-adapted security
                                    regressions

Uncertainty is cluster-aware: families (not rows) are the bootstrap
unit, per the preregistered `analysis_unit: family` — rows within a
family are correlated by construction and must not be treated as
independent.

Usage:
    python scripts/analyze_v410_matrix.py \
        --matrix-id v410-4a --run-root results/campaign4a/matrix/runs \
        --corpus configs/campaign4a_tasks.jsonl \
        --gates configs/capability_acquisition/confirmation.yaml \
        --seeds 0,1,2,3,4 \
        --arms fixed,rehearsal,kl,both,shuffled
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines()
            if ln.strip()]


def _family_map(corpus: Path) -> dict[str, str]:
    return {str(r["id"]): str(r["family"])
            for r in _load_jsonl(corpus)}


def _correct_map(run: Path, fname: str) -> dict[str, int]:
    """id -> 0/1 correctness, from the committed prediction file."""
    p = run / fname
    if not p.is_file():
        return {}
    out = {}
    for r in _load_jsonl(p):
        out[str(r["id"])] = int(r.get("score", 0))
    return out


def _by_family(scores: dict[str, int],
               fams: dict[str, str]) -> dict[str, list[int]]:
    per: dict[str, list[int]] = {}
    for rid, s in scores.items():
        per.setdefault(fams.get(rid, "?"), []).append(s)
    return per


def cluster_bootstrap_delta(
        frozen: dict[str, list[int]],
        adapted: dict[str, list[int]],
        *, resamples: int, alpha: float,
        rng: random.Random) -> dict:
    """Bootstrap the adapted-minus-frozen within-family accuracy by
    resampling FAMILIES (clusters) with replacement."""
    fams = sorted(set(frozen) | set(adapted))
    if not fams:
        return {"delta": None, "ci_low": None, "ci_high": None}

    def mean_acc(m: dict[str, list[int]], keys) -> float:
        vals = [s for k in keys for s in m.get(k, [])]
        return sum(vals) / len(vals) if vals else 0.0

    point = mean_acc(adapted, fams) - mean_acc(frozen, fams)
    boots = []
    for _ in range(resamples):
        keys = [rng.choice(fams) for _ in fams]
        boots.append(mean_acc(adapted, keys) - mean_acc(frozen, keys))
    boots.sort()
    lo = boots[int((alpha / 2) * len(boots))]
    hi = boots[min(len(boots) - 1,
                   int((1 - alpha / 2) * len(boots)) - 1)]
    return {"delta": round(point, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "resamples": resamples, "clusters": len(fams)}


def run_summary(run: Path, fams: dict[str, str]) -> dict | None:
    mpath = run / "METRICS.json"
    if not mpath.is_file():
        return None
    m = json.loads(mpath.read_text())
    who = m.get("within_holdout") or {}
    ret = m.get("retention") or {}
    val = m.get("validation") or {}
    sec = m.get("security") or {}
    fr_who = _correct_map(run, "PREDICTIONS-FROZEN-WITHIN_HOLDOUT.jsonl")
    ad_who = _correct_map(run, "PREDICTIONS-ADAPTED-WITHIN_HOLDOUT.jsonl")
    return {
        "run_id": run.name,
        "fit_gate_passed": bool((m.get("fit_gate") or {}).get("passed")),
        "fit_subset": (m.get("fit_gate") or {}).get(
            "adapted_subset_accuracy"),
        "validation": {"frozen": val.get("frozen"),
                       "adapted": val.get("adapted")},
        "within_holdout": {"frozen": who.get("frozen"),
                           "adapted": who.get("adapted")},
        "hidden": {"frozen": (m.get("hidden") or {}).get("frozen")
                   if isinstance(m.get("hidden"), dict) else
                   (m.get("frozen") or {}).get("hidden"),
                   "adapted": (m.get("adapted") or {}).get("hidden")},
        "compositional": m.get("compositional"),
        "retention": {"frozen": ret.get("frozen"),
                      "adapted": ret.get("adapted"),
                      "destruction_rate": ret.get("destruction_rate"),
                      "acquisition_rate": ret.get("acquisition_rate"),
                      "n_frozen_correct": ret.get("n_frozen_correct"),
                      "n_destroyed": ret.get("n_destroyed")},
        "security": sec if isinstance(sec, dict) else {},
        "train_seconds": (m.get("training") or {}).get("train_seconds"),
        "_families": {"frozen_who": _by_family(fr_who, fams),
                      "adapted_who": _by_family(ad_who, fams)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-id", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gates",
                    default="configs/capability_acquisition/"
                            "confirmation.yaml")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--arms", default="fixed,rehearsal,kl,both,shuffled")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import yaml
    gcfg = yaml.safe_load((ROOT / args.gates).read_text())
    gates = gcfg.get("gates") or {}
    fams = _family_map(ROOT / args.corpus
                       if not Path(args.corpus).is_absolute()
                       else Path(args.corpus))
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rng = random.Random(20241010)

    matrix: dict[str, dict] = {}
    missing: list[str] = []
    for arm in arms:
        cells = []
        for seed in seeds:
            rid = f"{args.matrix_id}-{arm}-s{seed}"
            s = run_summary(run_root / rid, fams)
            if s is None:
                missing.append(rid)
            else:
                cells.append(s)
        matrix[arm] = {"seeds": cells, "n_seeds": len(cells)}

    # ---- per-arm aggregate + gates ------------------------------------
    report: dict[str, dict] = {}
    for arm, blk in matrix.items():
        cells = blk["seeds"]
        if not cells:
            report[arm] = {"status": "MISSING"}
            continue
        fit_ok = all(c["fit_gate_passed"] for c in cells)
        who_deltas = [(c["within_holdout"]["adapted"] or 0)
                      - (c["within_holdout"]["frozen"] or 0)
                      for c in cells]
        ret_dest = [c["retention"]["destruction_rate"] or 0
                    for c in cells]
        # pooled family-cluster bootstrap over all seeds' within_holdout
        fr: dict[str, list[int]] = {}
        ad: dict[str, list[int]] = {}
        for c in cells:
            for k, v in c["_families"]["frozen_who"].items():
                fr.setdefault(k, []).extend(v)
            for k, v in c["_families"]["adapted_who"].items():
                ad.setdefault(k, []).extend(v)
        boot = cluster_bootstrap_delta(
            fr, ad,
            resamples=int(gates.get("bootstrap_resamples", 2000)),
            alpha=float(gates.get("ci_alpha", 0.05)), rng=rng)
        # Security is evaluated by the GOVERNED lane
        # (safety_regression.declarative_score over adapted-model
        # generations on the security split), not by the fit runner —
        # a missing block must fail closed, never silently pass.
        sec_blocks = [c["security"] for c in cells
                      if isinstance(c["security"], dict)
                      and c["security"]]
        sec_viol = (sum(int(sv.get("regressions",
                                 sv.get("violations", 0)) or 0)
                        for sv in sec_blocks)
                    if len(sec_blocks) == len(cells) else None)
        mean_delta = (sum(who_deltas) / len(who_deltas)
                      if who_deltas else 0.0)
        verdict = {
            "n_seeds": len(cells),
            "within_holdout_delta_per_seed":
                [round(d, 4) for d in who_deltas],
            "within_holdout_delta_mean": round(mean_delta, 4),
            "within_holdout_delta_clustered_ci": boot,
            "retention_destruction_per_seed":
                [round(d, 4) for d in ret_dest],
            "retention_destruction_max": max(ret_dest, default=0.0),
            "security_regressions_total": sec_viol,
            "security_evaluated": sec_viol is not None,
            "train_seconds_mean": (
                round(sum(c["train_seconds"] or 0 for c in cells)
                      / len(cells), 1)),
            "gates": {
                "fit_all_pass": fit_ok,
                "G5_within_holdout": mean_delta >= float(
                    gates.get("within_holdout_min_delta", 0.0)),
                "G6_retention": max(ret_dest, default=1.0) <= float(
                    gates.get("retention_max_destruction_rate", 0.0)),
                # fail-closed: unevaluated security is NOT a pass
                "security": (sec_viol <= int(
                    gates.get("security_max_violations", 0)))
                if sec_viol is not None else None,
            }}
        verdict["all_gates_pass"] = all(
            v is True for v in verdict["gates"].values())
        report[arm] = verdict

    doc = {"schema": "mini-agi-v4.1-confirmation-analysis-v1",
           "matrix_id": args.matrix_id, "corpus": args.corpus,
           "seeds": seeds, "arms": arms,
           "gates_declared": gates,
           "missing_cells": missing,
           "complete": not missing,
           "arms_report": report}
    text = json.dumps(doc, indent=2, sort_keys=True)
    out = args.output or str(
        run_root.parent / f"CONFIRMATION_ANALYSIS-{args.matrix_id}.json")
    Path(out).write_text(text + "\n")
    print(text)
    print(f"\nwrote {out}")
    if missing:
        print(f"MISSING cells: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
