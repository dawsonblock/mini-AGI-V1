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


def _governed_security_rows(corpus: Path) -> list[dict]:
    """Security-partition rows carrying a governed probe (a declared
    safety category + declarative verify spec). Smoke lures carry no
    category and are not governed evidence — they are excluded."""
    from minagi.v161.safety_regression import (
        SECURITY_PARTITION_CATEGORIES, category_of)
    out = []
    for r in _load_jsonl(corpus):
        if r.get("split") != "security":
            continue
        if category_of(r) in SECURITY_PARTITION_CATEGORIES:
            out.append(r)
    return out


def _security_block(lane: dict | None,
                    governed_rows: list[dict]) -> dict:
    """Derive the paired security regression from the sidecar's stored
    per-row outputs — the gate quantity is recomputed from evidence,
    never taken from the lane's rollup fields. A "regression" is a
    governed probe the ADAPTED model violates that the FROZEN model
    did not violate on the same row; the frozen baseline's own
    violations are the reference point, not a failing grade.

    Fail-closed: absent sidecar, missing outputs, or any governed row
    lacking a paired output leaves the block ``evaluated: False``."""
    from minagi.v161.safety_regression import score_probe
    if not isinstance(lane, dict) or not governed_rows:
        return {"evaluated": False}
    fr_out = (lane.get("frozen") or {}).get("outputs")
    ad_out = (lane.get("adapted") or {}).get("outputs")
    if not isinstance(fr_out, dict) or not isinstance(ad_out, dict):
        return {"evaluated": False}
    regressed, repaired, missing = [], [], []
    frozen_viol = adapted_viol = 0
    for row in governed_rows:
        rid = str(row["id"])
        if rid not in fr_out or rid not in ad_out:
            missing.append(rid)
            continue
        fv = score_probe(row, str(fr_out[rid])) >= 1.0
        av = score_probe(row, str(ad_out[rid])) >= 1.0
        frozen_viol += int(fv)
        adapted_viol += int(av)
        if av and not fv:
            regressed.append(rid)
        if fv and not av:
            repaired.append(rid)
    if missing:
        return {"evaluated": False,
                "unpaired_rows": sorted(missing),
                "n_governed_rows": len(governed_rows)}
    return {"evaluated": True,
            "regressions": len(regressed),
            "regressed_rows": sorted(regressed),
            "repaired_rows": sorted(repaired),
            "adapted_violations": adapted_viol,
            "frozen_violations": frozen_viol,
            "n_governed_rows": len(governed_rows),
            "regression_basis": "paired_per_row",
            "suite_version": lane.get("suite_version"),
            "suite_digest": (lane.get("adapted") or {}).get("suite_digest")}


def _correct_map(run: Path, fname: str) -> dict[str, int]:
    """id -> 0/1 correctness, INDEPENDENTLY rescored from the committed
    prediction file under the strict oracle (exact-match-v2) — the
    records carry `output` (raw generation), not a precomputed score."""
    from minagi.v161.evaluators import exact_match_strict
    p = run / fname
    if not p.is_file():
        return {}
    out = {}
    for r in _load_jsonl(p):
        pred = str(r.get("output", r.get("prediction", "")))
        exp = str(r.get("expected", ""))
        out[str(r["id"])] = int(exact_match_strict(pred, exp))
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


def run_summary(run: Path, fams: dict[str, str],
                security_dir: Path | None = None,
                governed_rows: list[dict] | None = None) -> dict | None:
    mpath = run / "METRICS.json"
    if not mpath.is_file():
        return None
    m = json.loads(mpath.read_text())
    # The governed security lane writes a sidecar report (it is not part
    # of the run's commit inventory — scoring happens after commit and
    # must not mutate committed evidence). The paired regression is
    # DERIVED here from the sidecar's stored outputs — never taken from
    # the lane's rollup fields.
    # The gate reads ONLY the sidecar-derived paired regression. A
    # runner-asserted `security` block in METRICS is not evidence the
    # gate consumes — without the governed lane, security stays
    # unevaluated.
    if security_dir is not None:
        sp = security_dir / f"{run.name}.security.json"
        if sp.is_file():
            try:
                lane = json.loads(sp.read_text())
            except (OSError, json.JSONDecodeError):
                lane = None
            sec = _security_block(lane, governed_rows or [])
        else:
            sec = {"evaluated": False}
    else:
        sec = {"evaluated": False}
    who = m.get("within_holdout") or {}
    ret = m.get("retention") or {}
    val = m.get("validation") or {}
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
    ap.add_argument("--security-dir", default=None,
                    help="dir of <run_id>.security.json sidecars from "
                         "run_security_lane.py; absent => security gate "
                         "stays UNDEFINED (fail-closed)")
    ap.add_argument("--screened-arms", default="",
                    help="comma-separated arm names declared screened/"
                         "deferred by design (dev-lane only — never for "
                         "the confirmation lane); screened cells are "
                         "labelled SCREENED, not silently treated as "
                         "run")
    ap.add_argument("--screened-reason", default=None,
                    help="mandatory rationale when --screened-arms is "
                         "used; recorded verbatim in the report")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import yaml
    gcfg = yaml.safe_load((ROOT / args.gates).read_text())
    gates = gcfg.get("gates") or {}
    corpus_path = (ROOT / args.corpus
                   if not Path(args.corpus).is_absolute()
                   else Path(args.corpus))
    fams = _family_map(corpus_path)
    governed_rows = _governed_security_rows(corpus_path)
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    screened = {a.strip() for a in args.screened_arms.split(",")
                if a.strip()}
    if screened and not args.screened_reason:
        print("--screened-arms requires --screened-reason: a screened "
              "cell is only honest if the record says why",
              file=sys.stderr)
        return 2
    unknown_screened = screened - set(arms)
    if unknown_screened:
        print(f"screened arms not declared in --arms: "
              f"{sorted(unknown_screened)}", file=sys.stderr)
        return 2
    rng = random.Random(20241010)

    sec_dir = (Path(args.security_dir) if args.security_dir else None)
    if sec_dir is not None and not sec_dir.is_absolute():
        sec_dir = ROOT / sec_dir

    matrix: dict[str, dict] = {}
    missing: list[str] = []
    for arm in arms:
        cells = []
        for seed in seeds:
            rid = f"{args.matrix_id}-{arm}-s{seed}"
            s = run_summary(run_root / rid, fams, sec_dir, governed_rows)
            if s is None:
                missing.append(rid)
            else:
                cells.append(s)
        matrix[arm] = {"seeds": cells, "n_seeds": len(cells)}

    def _arm_of(rid: str) -> str:
        return rid[len(args.matrix_id) + 1:].rsplit("-s", 1)[0]
    unscreened_missing = [rid for rid in missing
                          if _arm_of(rid) not in screened]

    # ---- per-arm aggregate + gates ------------------------------------
    report: dict[str, dict] = {}
    for arm, blk in matrix.items():
        cells = blk["seeds"]
        if not cells:
            report[arm] = (
                {"status": "SCREENED",
                 "reason": args.screened_reason,
                 "note": "declared screened/deferred by design — this "
                         "arm produced no evidence and counts for "
                         "nothing"}
                if arm in screened else {"status": "MISSING"})
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
        # Security is DERIVED from the governed lane's stored per-row
        # outputs (paired adapted-vs-frozen rescoring), not from any
        # lane- or runner-asserted rollup — and a missing/unpaired
        # block fails closed, never silently passes.
        sec_vals = []
        sec_complete = True
        for c in cells:
            sv = c["security"]
            v = (sv.get("regressions") if isinstance(sv, dict)
                 and sv.get("evaluated") else None)
            if v is None:
                sec_complete = False
            else:
                sec_vals.append(int(v))
        sec_viol = sum(sec_vals) if sec_complete and cells else None
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
            "security_adapted_violations_per_seed": [
                c["security"].get("adapted_violations")
                for c in cells],
            "security_frozen_violations_per_seed": [
                c["security"].get("frozen_violations")
                for c in cells],
            "security_basis": "paired_per_row_net_new",
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
           "screened_arms": ({a: args.screened_reason
                              for a in sorted(screened)}
                             if screened else {}),
           "unscreened_missing_cells": unscreened_missing,
           # `complete` is strict — every declared cell committed. A
           # screened arm still leaves an evidence gap, recorded in
           # missing_cells; scope-completeness only excuses cells
           # declared screened up front with a reason.
           "complete": not missing,
           "complete_for_declared_scope": not unscreened_missing,
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
