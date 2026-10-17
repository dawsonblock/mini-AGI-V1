#!/usr/bin/env python3
"""Campaign 3B.3 P0 — rescore every retained prediction under the
corrected oracle (RD-18).

The original scorer folded case (``exact-match-v1`` —
``strip().casefold()``), so a wrong-case output was scored correct on
tasks whose answer is literally a case transformation (``pc-upper``,
``pc-vowels``). This tool recomputes every retained prediction under
``exact-match-v2`` (``strip() == strip()``) WITHOUT modifying the
original records — each run keeps its v1 metrics for provenance and
gains a ``RESCORE-STRICT.json`` overlay naming both oracles.

Scope:
  * ``--runs-root`` : positive-control fit/ladder runs
    (PREDICTIONS-{FROZEN,ADAPTED,SUBSET}.jsonl carry ``expected``
    inline — pure recompute).
  * ``--campaign-evidence`` : the published Campaign 3B evidence
    extraction (PREDICTIONS-{L1,L5,L5a,L5b,L6,L7,NC}.jsonl join the
    committed corpus by task id — expected answers never lived in the
    evidence records).

Outputs ``RESCORE_SUMMARY.json`` and prints the corrected headline
table. The report writer lives in RESCORE_REPORT.md generation here so
the numbers cannot drift from the data.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def strict(out: str, exp: str) -> int:
    return int(str(out).strip() == str(exp).strip())


def folded(out: str, exp: str) -> int:
    return int(str(out).strip().casefold() == str(exp).strip().casefold())


def _sign_test_pvalue(gains: int, losses: int) -> float:
    """Exact two-sided sign test on discordant pairs (McNemar exact)."""
    n = gains + losses
    if n == 0:
        return 1.0
    k = min(gains, losses)
    # two-sided binomial tail: 2 * P(X <= k), clipped at 1
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * p)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def rescore_run_dir(run_dir: Path) -> dict | None:
    """Rescore one fit/ladder run. Returns the overlay document or None
    when no prediction files exist."""
    files = sorted(run_dir.glob("PREDICTIONS-*.jsonl"))
    if not files:
        return None
    overlay = {"schema": "mini-agi-campaign3b3-rescore-v1",
               "run_id": run_dir.name,
               "oracles": {"v1": "strip().casefold() == strip().casefold()",
                           "v2": "strip() == strip()"},
               "files": {}}
    for pf in files:
        rows = _read_jsonl(pf)
        v1_scores, v2_scores, flips = {}, {}, []
        per_family = defaultdict(lambda: {"v1": 0, "v2": 0, "n": 0})
        for r in rows:
            rid = str(r["id"])
            out, exp = str(r["output"]), str(r["expected"])
            s1, s2 = folded(out, exp), strict(out, exp)
            v1_scores[rid], v2_scores[rid] = s1, s2
            fam = str(r.get("family", "?"))
            per_family[fam]["v1"] += s1
            per_family[fam]["v2"] += s2
            per_family[fam]["n"] += 1
            if s1 != s2:
                flips.append({"id": rid, "family": fam,
                              "output": out[:80], "expected": exp[:80],
                              "v1": s1, "v2": s2})
        n = len(rows) or 1
        overlay["files"][pf.name] = {
            "n_rows": len(rows),
            "v1_accuracy": sum(v1_scores.values()) / n,
            "v2_accuracy": sum(v2_scores.values()) / n,
            "n_flipped": len(flips),
            "flipped_rows": flips,
            "per_family": {f: {"v1": d["v1"] / d["n"],
                               "v2": d["v2"] / d["n"], "n": d["n"]}
                           for f, d in sorted(per_family.items())},
            "v2_scores": v2_scores}
    # paired destruction/gains under each oracle, hidden rows only
    fh = overlay["files"].get("PREDICTIONS-FROZEN.jsonl", {}).get("v2_scores")
    ah = overlay["files"].get("PREDICTIONS-ADAPTED.jsonl", {}).get("v2_scores")
    if fh and ah:
        destroyed = [k for k in fh if fh[k] == 1 and ah.get(k, 0) == 0]
        gained = [k for k in fh if fh[k] == 0 and ah.get(k, 0) == 1]
        overlay["hidden_pairwise_v2"] = {
            "n_frozen_correct": sum(fh.values()),
            "n_destroyed": len(destroyed), "destroyed_ids": sorted(destroyed),
            "n_gained": len(gained), "gained_ids": sorted(gained),
            "net": len(gained) - len(destroyed),
            "sign_test_p": _sign_test_pvalue(len(gained), len(destroyed))}
    return overlay


def rescore_campaign(evidence_root: Path, corpus: Path) -> dict:
    """Rescore the published Campaign 3B arm predictions by joining them
    to the committed corpus (expected values are corpus-side)."""
    rows = {str(r["id"]): r for r in _read_jsonl(corpus)}
    by_split = defaultdict(dict)
    for r in rows.values():
        by_split[str(r["split"])][str(r["id"])] = r
    out = {"schema": "mini-agi-campaign3b3-campaign-rescore-v1",
           "oracles": {"v1": "casefold", "v2": "strict"},
           "arms": {}}
    for pf in sorted(evidence_root.rglob("PREDICTIONS-*.jsonl")):
        arm = pf.stem.replace("PREDICTIONS-", "")
        preds = _read_jsonl(pf)
        v1_h, v2_h, flips = {}, {}, []
        fam_stats = defaultdict(lambda: {"v1": 0, "v2": 0, "n": 0})
        for p in preds:
            rid = str(p["id"])
            gold = rows.get(rid)
            if gold is None or str(gold.get("split")) != "hidden":
                continue
            s1, s2 = folded(str(p["output"]), str(gold["expected"])), \
                strict(str(p["output"]), str(gold["expected"]))
            v1_h[rid], v2_h[rid] = s1, s2
            fam = str(gold["family"])
            fam_stats[fam]["v1"] += s1
            fam_stats[fam]["v2"] += s2
            fam_stats[fam]["n"] += 1
            if s1 != s2:
                flips.append({"id": rid, "family": fam,
                              "output": str(p["output"])[:80],
                              "expected": str(gold["expected"])[:80]})
        n = len(v1_h) or 1
        out["arms"][arm] = {
            "file": str(pf.relative_to(evidence_root)),
            "hidden_v1": sum(v1_h.values()) / n,
            "hidden_v2": sum(v2_h.values()) / n,
            "n_hidden": len(v1_h), "n_flipped": len(flips),
            "flipped_rows": flips,
            "per_family": {f: {"v1": d["v1"] / d["n"],
                               "v2": d["v2"] / d["n"], "n": d["n"]}
                           for f, d in sorted(fam_stats.items())},
            "v2_scores": v2_h}
    # paired primary comparisons under v2
    arms = out["arms"]
    if "L1" in arms:
        l1 = arms["L1"]["v2_scores"]
        for arm in ("L5b", "L6", "L7", "NC"):
            if arm not in arms:
                continue
            a = arms[arm]["v2_scores"]
            keys = set(l1) & set(a)
            gained = sum(1 for k in keys if l1[k] == 0 and a[k] == 1)
            destroyed = sum(1 for k in keys if l1[k] == 1 and a[k] == 0)
            arms[arm]["paired_vs_L1_v2"] = {
                "n_pairs": len(keys), "gained": gained,
                "destroyed": destroyed, "net": gained - destroyed,
                "mean_delta": (sum(a[k] for k in keys)
                               - sum(l1[k] for k in keys)) / (len(keys) or 1),
                "sign_test_p": _sign_test_pvalue(gained, destroyed)}
    if "L5b" in arms and "L7" in arms:
        a, b = arms["L5b"]["v2_scores"], arms["L7"]["v2_scores"]
        keys = set(a) & set(b)
        gained = sum(1 for k in keys if a[k] == 0 and b[k] == 1)
        destroyed = sum(1 for k in keys if a[k] == 1 and b[k] == 0)
        out["l7_minus_l5b_v2"] = {
            "n_pairs": len(keys), "l7_gained": gained,
            "l7_destroyed": destroyed, "delta": gained - destroyed,
            "mean_delta": (sum(b[k] for k in keys)
                           - sum(a[k] for k in keys)) / (len(keys) or 1)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="results/campaign3b2/runs")
    ap.add_argument("--campaign-evidence", default=None,
                    help="extracted Campaign 3B evidence directory")
    ap.add_argument("--corpus",
                    default="configs/positive_control_tasks.jsonl")
    ap.add_argument("--out", default="results/campaign3b2/RESCORE_SUMMARY.json")
    args = ap.parse_args()

    summary = {"schema": "mini-agi-campaign3b3-rescore-summary-v1",
               "oracles": {"v1": "casefold (historical)",
                           "v2": "strict case-sensitive (corrected)"},
               "runs": {}}
    runs_root = Path(args.runs_root)
    if runs_root.is_dir():
        for run_dir in sorted(runs_root.iterdir()):
            if not run_dir.is_dir():
                continue
            overlay = rescore_run_dir(run_dir)
            if overlay is None:
                continue
            (run_dir / "RESCORE-STRICT.json").write_text(
                json.dumps(overlay, indent=1, sort_keys=True) + "\n")
            f_ad = overlay["files"].get("PREDICTIONS-ADAPTED.jsonl", {})
            f_fr = overlay["files"].get("PREDICTIONS-FROZEN.jsonl", {})
            f_sub = overlay["files"].get("PREDICTIONS-SUBSET.jsonl", {})
            summary["runs"][run_dir.name] = {
                "hidden_v1": f_ad.get("v1_accuracy"),
                "hidden_v2": f_ad.get("v2_accuracy"),
                "frozen_v1": f_fr.get("v1_accuracy"),
                "frozen_v2": f_fr.get("v2_accuracy"),
                "subset_v1": f_sub.get("v1_accuracy"),
                "subset_v2": f_sub.get("v2_accuracy"),
                "n_flipped": sum(f["n_flipped"]
                                 for f in overlay["files"].values()),
                "pairwise_v2": overlay.get("hidden_pairwise_v2")}

    if args.campaign_evidence:
        camp = rescore_campaign(Path(args.campaign_evidence),
                                ROOT / args.corpus)
        summary["campaign3b"] = camp

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")

    # headline table
    camp = summary.get("campaign3b", {}).get("arms", {})
    if camp:
        print("Campaign 3B rescore (hidden, v1 casefold -> v2 strict):")
        for arm in ("L1", "L5", "L5b", "L6", "L7", "NC"):
            if arm in camp:
                a = camp[arm]
                print(f"  {arm:4s} {a['hidden_v1']:.3f} -> {a['hidden_v2']:.3f}"
                      f"   ({a['n_flipped']} rows flipped)")
        if "l7_minus_l5b_v2" in summary["campaign3b"]:
            d = summary["campaign3b"]["l7_minus_l5b_v2"]
            print(f"  L7-L5b strict: {d['mean_delta']:+.3f} "
                  f"(destroyed {d['l7_destroyed']}, gained {d['l7_gained']})")
    runs = summary["runs"]
    if runs:
        print(f"\nfit/ladder runs rescored: {len(runs)}")
        deltas = [r["hidden_v2"] - r["hidden_v1"] for r in runs.values()
                  if r.get("hidden_v2") is not None]
        if deltas:
            print(f"  hidden accuracy mean shift: {sum(deltas)/len(deltas):+.4f}"
                  f"  (min {min(deltas):+.3f}, max {max(deltas):+.3f})")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
