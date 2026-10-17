#!/usr/bin/env python3
"""v4.1.0 confirmatory analysis — gate evaluation for the intervention
matrix (G5 generalization, G6 retention, security), v4.0.3-r1 evidence
closure.

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
  * required_evidence_level       — the evidence level every cell must
                                    reach (integrity|local|independent|
                                    qualified)

EVIDENCE CLOSURE (v4.0.3-r1) — before any gate evaluates a cell, the
cell's evidence is closed:

  * a valid committed run inventory (RUN_COMMIT.json + hashes);
  * exact model/adapter identity bindings, scorer identity, dataset
    and protocol digests;
  * every expected prediction file present with complete, unique ids;
  * every gate input INDEPENDENTLY RESCORED from committed predictions
    (runner-asserted METRICS fields are never consumed);
  * paired per-row security evidence derived from the governed lane
    sidecar;
  * the configured evidence level: ``independent`` requires a signed
    independent-evaluation-v1 receipt from an evaluation-role
    authority, ``qualified`` additionally requires a signed
    qualification decision.

A cell that fails closure is INVALID — its gates are NOT_EVALUABLE,
never PASS. Each metric has three states: measured, explicitly
optional/not-applicable, or missing required evidence; a gate that
needs a missing value is NOT_EVALUABLE. Missing retention evidence is
NOT a zero regression, and an undefined confidence interval is not
passing statistical evidence.

MATRIX COMPLETENESS — --manifest names the experiment manifest
(seeds x arms) when present; otherwise --seeds/--arms declare it.
Screened arms stay honest: a NOT_EXECUTED/screened arm leaves the full
program INCOMPLETE and the campaign qualification non-passing.

Usage:
    python scripts/analyze_v410_matrix.py \
        --matrix-id v410-4a --run-root results/campaign4a/matrix/runs \
        --corpus configs/campaign4a_tasks.jsonl \
        --gates configs/capability_acquisition/confirmation.yaml \
        --manifest configs/capability_acquisition/matrix_manifest.yaml \
        --trust-root path/to/TRUST_ROOT.json \
        --security-dir results/campaign4a/security
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.qualified_outcome import (  # noqa: E402
    STATE_INDEPENDENT, STATE_LOCAL, STATE_QUALIFIED, STATE_RANK,
    derive_qualified_outcome)

#: Evidence levels a matrix cell can be required to reach.
LEVEL_RANK = {"integrity": 1, "local": 2, "independent": 3,
              "qualified": 4}
_LEVEL_STATE = {1: "INTEGRITY_VERIFIED", 2: STATE_LOCAL,
                3: STATE_INDEPENDENT, 4: STATE_QUALIFIED}
_LEVEL_ALIASES = {
    "integrity_verified": "integrity",
    "locally_recomputed": "local",
    "independently_verified": "independent",
    "evaluation_verified": "independent",   # deprecated v4.0.3 name
}


def _level_of(name: str) -> str | None:
    n = str(name).strip().lower()
    return n if n in LEVEL_RANK else _LEVEL_ALIASES.get(n)


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


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _destruction_rate(frozen: dict[str, int],
                      adapted: dict[str, int]) -> float | None:
    """Share of previously-correct answers the adapted model destroyed,
    rescored per id. None when there is nothing to measure — a missing
    denominator is not a zero regression."""
    if not frozen or not adapted:
        return None
    n_fc = sum(1 for v in frozen.values() if v == 1)
    if not n_fc:
        return None
    destroyed = sum(1 for rid, v in frozen.items()
                    if v == 1 and adapted.get(rid) == 0)
    return destroyed / n_fc


def cluster_bootstrap_delta(
        frozen: dict[str, list[int]],
        adapted: dict[str, list[int]], *, resamples: int, alpha: float,
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
                governed_rows: list[dict] | None = None, *,
                repo_root=None, corpus_rows=None,
                verifier=None, authority=None) -> dict | None:
    """Evidence-closed per-cell summary.

    Returns None only when the run directory does not exist at all
    (a MISSING cell). Everything else — committed inventory, rescore
    reconciliation, receipt/qualification authority — comes back in
    ``evidence``. Gate inputs are rescore-derived; a cell that did not
    reach LOCALLY_RECOMPUTED carries no evaluable gate value."""
    run = Path(run)
    if not run.is_dir():
        return None
    m = {}
    mpath = run / "METRICS.json"
    if mpath.is_file():
        try:
            m = json.loads(mpath.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            m = {}
    # The governed security lane writes a sidecar report (it is not part
    # of the run's commit inventory — scoring happens after commit and
    # must not mutate committed evidence). The paired regression is
    # DERIVED here from the sidecar's stored outputs — never taken from
    # the lane's rollup fields, and a missing/unpaired block fails
    # closed, never silently passes. When the evaluator bound the same
    # sidecar into the receipt's security_metrics_digest, the document
    # is presented to the binding check as well.
    lane = None
    if security_dir is not None:
        sp = security_dir / f"{run.name}.security.json"
        if sp.is_file():
            try:
                lane = json.loads(sp.read_text())
            except (OSError, json.JSONDecodeError):
                lane = None
    sec = _security_block(lane, governed_rows or []) \
        if security_dir is not None else {"evaluated": False}
    oc = derive_qualified_outcome(run, verifier=verifier,
                                  authority=authority,
                                  repo_root=repo_root,
                                  corpus_rows=corpus_rows,
                                  security_doc=lane)
    fr_who = _correct_map(run, "PREDICTIONS-FROZEN-WITHIN_HOLDOUT.jsonl")
    ad_who = _correct_map(run, "PREDICTIONS-ADAPTED-WITHIN_HOLDOUT.jsonl")
    fr_ret = _correct_map(run, "PREDICTIONS-FROZEN-RETENTION.jsonl")
    ad_ret = _correct_map(run, "PREDICTIONS-ADAPTED-RETENTION.jsonl")
    subset = _correct_map(run, "PREDICTIONS-SUBSET.jsonl")
    who_delta = None
    if fr_who and ad_who:
        who_delta = (sum(ad_who.values()) / len(ad_who)
                     - sum(fr_who.values()) / len(fr_who))
    val = m.get("validation") or {}
    who = m.get("within_holdout") or {}
    ret = m.get("retention") or {}
    return {
        "run_id": run.name,
        "verification_state": oc.verification_state,
        "evaluation_status": oc.evaluation_status,
        "qualification_status": oc.qualification_status,
        "evidence_reasons": list(oc.reasons[:8]),
        # ---- rescore-derived gate inputs (never METRICS-asserted) ---
        "fit_subset_rescore": _mean(subset.values()),
        "within_holdout_delta_rescore": who_delta,
        "retention_destruction_rescore":
            _destruction_rate(fr_ret, ad_ret),
        "retention_n_frozen_correct":
            sum(1 for v in fr_ret.values() if v == 1) if fr_ret else None,
        "security": sec if isinstance(sec, dict) else {},
        # ---- reported metrics, audit only — gates never read these ---
        "reported": {
            "fit_gate_passed":
                bool((m.get("fit_gate") or {}).get("passed")),
            "within_holdout": {"frozen": who.get("frozen"),
                               "adapted": who.get("adapted")},
            "retention": {"destruction_rate":
                          ret.get("destruction_rate")},
            "validation": {"frozen": val.get("frozen"),
                           "adapted": val.get("adapted")},
        },
        "train_seconds": (m.get("training") or {}).get("train_seconds"),
        "_families": {"frozen_who": _by_family(fr_who, fams),
                      "adapted_who": _by_family(ad_who, fams)}}


def _gate(value, op: str, bound, *, evaluable: bool) -> dict:
    """Three-state gate verdict. ``value`` None means the measurement
    was never taken — that is missing evidence, NOT a zero and NOT a
    pass."""
    if not evaluable:
        return {"status": "NOT_EVALUABLE", "value": None}
    if value is None:
        return {"status": "NOT_EVALUABLE", "value": None}
    ok = value >= bound if op == "min" else value <= bound
    return {"status": "PASS" if ok else "FAIL", "value": value}


def _bool_of(status: str):
    return {"PASS": True, "FAIL": False}.get(status)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-id", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gates",
                    default="configs/capability_acquisition/"
                            "confirmation.yaml")
    ap.add_argument("--manifest", default=None,
                    help="experiment manifest (YAML or JSON) declaring "
                         "the expected seeds x arms; when given it "
                         "overrides --seeds/--arms")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--arms", default="fixed,rehearsal,kl,both,shuffled")
    ap.add_argument("--security-dir", default=None,
                    help="dir of <run_id>.security.json sidecars from "
                         "run_security_lane.py; absent => security gate "
                         "stays NOT_EVALUABLE (fail-closed)")
    ap.add_argument("--evidence-level",
                    choices=sorted(LEVEL_RANK), default=None,
                    help="minimum evidence level every cell must reach "
                         "(default: gates.required_evidence_level, "
                         "else 'local')")
    ap.add_argument("--trust-root", default=None,
                    help="TRUST_ROOT.json — required for evidence "
                         "levels independent/qualified (receipt and "
                         "qualification signatures verify against it)")
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
    corpus_rows = _load_jsonl(corpus_path)
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = ROOT / run_root

    # ---- expected matrix shape: manifest wins over CLI declarations --
    manifest_doc = None
    if args.manifest:
        mpath = Path(args.manifest)
        if not mpath.is_absolute():
            mpath = ROOT / mpath
        manifest_doc = yaml.safe_load(mpath.read_text())
        if not isinstance(manifest_doc, dict):
            print("experiment manifest is not a mapping",
                  file=sys.stderr)
            return 2
        if manifest_doc.get("matrix_id") \
                and str(manifest_doc["matrix_id"]) != args.matrix_id:
            print(f"manifest matrix_id "
                  f"{manifest_doc['matrix_id']!r} != --matrix-id "
                  f"{args.matrix_id!r}", file=sys.stderr)
            return 2
        seeds = [int(s) for s in manifest_doc["seeds"]]
        arms = [str(a) for a in manifest_doc["arms"]]
    else:
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

    required_level = _level_of(
        args.evidence_level
        or gates.get("required_evidence_level") or "local")
    if required_level is None:
        print(f"unknown evidence level "
              f"{args.evidence_level or gates.get('required_evidence_level')!r}",
              file=sys.stderr)
        return 2
    required_rank = LEVEL_RANK[required_level]
    evaluable_rank = max(required_rank, LEVEL_RANK["local"])

    verifier = None
    authority = None
    if args.trust_root:
        from minagi.v161.evaluator_authority import load_trust_root
        authority, verifier = load_trust_root(
            (ROOT / args.trust_root) if not Path(
                args.trust_root).is_absolute()
            else args.trust_root)
    elif required_rank >= LEVEL_RANK["independent"]:
        print("note: --trust-root absent — receipts cannot verify; "
              f"cells cannot reach {required_level!r} and will be "
              "INSUFFICIENT_EVIDENCE", file=sys.stderr)

    rng = random.Random(20241010)
    sec_dir = (Path(args.security_dir) if args.security_dir else None)
    if sec_dir is not None and not sec_dir.is_absolute():
        sec_dir = ROOT / sec_dir

    matrix: dict[str, dict] = {}
    missing: list[str] = []
    invalid: list[str] = []
    for arm in arms:
        cells = []
        for seed in seeds:
            rid = f"{args.matrix_id}-{arm}-s{seed}"
            s = run_summary(run_root / rid, fams, sec_dir,
                            governed_rows, repo_root=ROOT,
                            corpus_rows=corpus_rows,
                            verifier=verifier, authority=authority)
            if s is None:
                missing.append(rid)
            else:
                rank = STATE_RANK.get(s["verification_state"], 0)
                s["evidence_rank"] = rank
                s["evidence_level_met"] = rank >= required_rank
                s["evaluable"] = rank >= evaluable_rank
                cells.append(s)
                if rank == 0 or not s["evidence_level_met"]:
                    invalid.append(rid)
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
        n_expected = len(seeds)
        n_missing = n_expected - len(cells)
        evaluable = all(c["evaluable"] for c in cells) \
            and n_missing == 0
        level_met = all(c["evidence_level_met"] for c in cells) \
            and n_missing == 0

        # ---- rescore-derived per-seed gate inputs --------------------
        fit_vals = [c["fit_subset_rescore"] for c in cells]
        who_deltas = [c["within_holdout_delta_rescore"] for c in cells]
        ret_dest = [c["retention_destruction_rescore"] for c in cells]
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
        sec_complete = evaluable
        for c in cells:
            sv = c["security"]
            v = (sv.get("regressions") if isinstance(sv, dict)
                 and sv.get("evaluated") else None)
            if v is None:
                sec_complete = False
            else:
                sec_vals.append(int(v))
        sec_viol = sum(sec_vals) if sec_complete and cells else None
        mean_delta = _mean(who_deltas)
        fit_min = float(gates.get("fit_min", 0.0))
        fit_ok = all(v is not None and v >= fit_min for v in fit_vals)

        gate_detail = {
            "fit_all_pass": _gate(
                min((v for v in fit_vals if v is not None), default=None),
                "min", fit_min, evaluable=evaluable),
            "G5_within_holdout": _gate(
                mean_delta, "min",
                float(gates.get("within_holdout_min_delta", 0.0)),
                evaluable=evaluable),
            "G6_retention": _gate(
                max((v for v in ret_dest if v is not None), default=None),
                "max",
                float(gates.get("retention_max_destruction_rate", 0.0)),
                evaluable=evaluable
                and all(v is not None for v in ret_dest)),
            "security": _gate(
                sec_viol, "max",
                int(gates.get("security_max_violations", 0)),
                evaluable=sec_viol is not None),
        }
        gates_bool = {k: _bool_of(g["status"])
                      for k, g in gate_detail.items()}
        all_pass = all(g["status"] == "PASS"
                       for g in gate_detail.values())

        if n_missing:
            cell_state = "INCOMPLETE"
        elif level_met:
            cell_state = "VALID"
        elif all(c["evaluable"] for c in cells):
            cell_state = "INSUFFICIENT_EVIDENCE"
        else:
            cell_state = "INVALID"

        verdict = {
            "experiment_status": cell_state,
            "n_seeds": len(cells),
            "n_missing_cells": n_missing,
            "evidence_level_met": level_met,
            "within_holdout_delta_per_seed":
                [round(d, 4) if d is not None else None
                 for d in who_deltas],
            "within_holdout_delta_mean":
                round(mean_delta, 4) if mean_delta is not None else None,
            "within_holdout_delta_clustered_ci": boot,
            "retention_destruction_per_seed":
                [round(d, 4) if d is not None else None
                 for d in ret_dest],
            "retention_destruction_max":
                max((v for v in ret_dest if v is not None),
                    default=None),
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
            "cells": [{"run_id": c["run_id"],
                       "verification_state": c["verification_state"],
                       "evidence_level_met": c["evidence_level_met"],
                       "reasons": c["evidence_reasons"][:4]}
                      for c in cells],
            "gates": gates_bool,
            "gate_detail": gate_detail,
            "all_gates_pass": all_pass,
            "fit_ok": fit_ok,
        }
        report[arm] = verdict

    # ---- matrix status -------------------------------------------------
    # A cell is INVALID when it exists but its evidence cannot verify at
    # the required level; INCOMPLETE when expected cells never executed
    # (including screened arms — a deferred arm is still an evidence
    # gap for the full program). The campaign qualification passes only
    # on COMPLETE.
    matrix_status = "COMPLETE"
    if invalid:
        matrix_status = "INVALID"
    elif missing:
        matrix_status = "INCOMPLETE"
    matrix_qualified = matrix_status == "COMPLETE"

    doc = {"schema": "mini-agi-v4.1-confirmation-analysis-v2",
           "matrix_id": args.matrix_id, "corpus": args.corpus,
           "seeds": seeds, "arms": arms,
           "manifest": (str(args.manifest) if args.manifest else None),
           "evidence_level_required": required_level,
           "gates_declared": gates,
           "missing_cells": missing,
           "invalid_cells": invalid,
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
           "matrix_status": matrix_status,
           "matrix_qualified": matrix_qualified,
           "arms_report": report}
    text = json.dumps(doc, indent=2, sort_keys=True)
    out = args.output or str(
        run_root.parent / f"CONFIRMATION_ANALYSIS-{args.matrix_id}.json")
    Path(out).write_text(text + "\n")
    print(text)
    print(f"\nwrote {out}")
    if missing or invalid or not matrix_qualified:
        if missing:
            print(f"MISSING cells: {missing}", file=sys.stderr)
        if invalid:
            print(f"INVALID/insufficient cells: {invalid}",
                  file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
