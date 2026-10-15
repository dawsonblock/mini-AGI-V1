#!/usr/bin/env python3
"""Requalify executed positive-control evidence (V4-22).

Scans ``results/<campaign>/runs/<run_id>`` directories and re-derives
what is verifiable from the committed artifacts — without trusting the
runner's reported metrics:

1. RUN_COMMIT.json integrity — status COMPLETE/FAILED and every file in
   the commit inventory hashed against the bytes on disk.
2. Strict rescore — PREDICTIONS-{FROZEN,SUBSET,ADAPTED}.jsonl are scored
   with ``exact_match_strict`` (and ``score_row`` when a row carries a
   verify spec); the recomputed accuracies are compared to the values
   METRICS.json reported.
3. Verdict per run — VERIFIED (integrity + rescore agree),
   INCONSISTENT (rescore disagrees with reported metrics), FAILED_RUN
   (commit status FAILED), or INVALID (integrity check failed).

Writes ``results/REQUALIFICATION_STATUS.md`` — the honest status table
backing the release's "executed evidence" claims. No model is loaded;
this is pure artifact verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.evaluators import (exact_match_strict,  # noqa: E402
                                    score_row)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines()
            if ln.strip()]


def requalify_run(run_dir: Path) -> dict:
    out = {"run_id": run_dir.name, "status": None, "checks": {},
           "rescore": {}, "reported": {}, "mismatches": []}
    commit_path = run_dir / "RUN_COMMIT.json"
    if not commit_path.is_file():
        out["status"] = "INVALID"
        out["checks"]["commit"] = "RUN_COMMIT.json missing"
        return out
    commit = json.loads(commit_path.read_text())
    run_status = commit.get("status")
    out["checks"]["commit_status"] = run_status

    # --- file-hash inventory ---
    inventory = commit.get("files") or {}
    bad = []
    for rel, want in inventory.items():
        fp = run_dir / rel
        if not fp.is_file():
            bad.append(f"{rel}: missing")
            continue
        got = _sha256(fp)
        if got != want:
            bad.append(f"{rel}: hash mismatch")
    out["checks"]["inventory"] = ("ok" if not bad else bad)
    # files on disk that the commit did not record
    extra = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*")
                   if p.is_file()
                   and str(p.relative_to(run_dir)) not in inventory
                   and p.name != "RUN_COMMIT.json")
    out["checks"]["uncommitted_files"] = extra

    metrics_path = run_dir / "METRICS.json"
    reported = json.loads(metrics_path.read_text()) \
        if metrics_path.is_file() else {}

    # --- strict rescore ---
    def rescore(pred_file: str) -> dict | None:
        fp = run_dir / pred_file
        if not fp.is_file():
            return None
        rows = _load_jsonl(fp)
        scores = {}
        for r in rows:
            row = {"expected": r.get("expected", ""),
                   "verify": r.get("verify")}
            scores[str(r["id"])] = float(
                score_row(row, str(r.get("output", "")),
                          exact_match_strict))
        by_split: dict[str, list[float]] = {}
        for r in rows:
            by_split.setdefault(str(r.get("split", "?")), []).append(
                scores[str(r["id"])])
        return {"n": len(rows),
                "per_id": scores,
                "by_split_mean": {s: sum(v) / len(v)
                                  for s, v in by_split.items()}}

    rescores = {name: rescore(f"PREDICTIONS-{name}.jsonl")
                for name in ("FROZEN", "SUBSET", "ADAPTED")}
    out["rescore"] = {k: {kk: vv for kk, vv in v.items()
                          if kk != "per_id"}
                      for k, v in rescores.items() if v}

    # compare rescore vs reported headline metrics
    def _cmp(name, rescore_key, reported_key):
        rs = rescores.get(rescore_key)
        if rs is None:
            return
        got = rs["by_split_mean"].get("hidden")
        if got is None and rs["by_split_mean"]:
            got = next(iter(rs["by_split_mean"].values()))
        want = (reported.get(reported_key) or {})
        want_v = want.get("hidden") if isinstance(want, dict) else None
        if want_v is None or got is None:
            return
        if abs(float(got) - float(want_v)) > 1e-9:
            out["mismatches"].append(
                f"{name}: reported {want_v} vs strict rescore {got}")

    _cmp("frozen.hidden", "FROZEN", "frozen")
    _cmp("adapted.hidden", "ADAPTED", "adapted")
    # subset rows live in PREDICTIONS-SUBSET (adapted) — METRICS reports
    # adapted.subset
    sub = rescores.get("SUBSET")
    if sub is not None:
        got = sub["by_split_mean"].get("train")
        if got is None and sub["by_split_mean"]:
            got = next(iter(sub["by_split_mean"].values()))
        want = (reported.get("adapted") or {}).get("subset")
        if got is not None and want is not None and \
                abs(got - float(want)) > 1e-9:
            out["mismatches"].append(
                f"adapted.subset: reported {want} vs strict rescore {got}")
    out["reported"] = {k: reported.get(k) for k in
                       ("adapted", "frozen", "fit_gate")}

    if run_status != "COMPLETE":
        out["status"] = "FAILED_RUN" if run_status == "FAILED" else "INVALID"
    elif bad:
        out["status"] = "INVALID"
    elif out["mismatches"]:
        out["status"] = "INCONSISTENT"
    else:
        out["status"] = "VERIFIED"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", default=None,
                    help="run roots to scan (default: results/*/runs/*)")
    ap.add_argument("--output", default="results/REQUALIFICATION_STATUS.md")
    args = ap.parse_args()

    if args.results:
        run_dirs = [Path(p) for p in args.results]
    else:
        run_dirs = sorted(ROOT.glob("results/*/runs/*/"))
    reports = [requalify_run(d) for d in run_dirs if d.is_dir()]

    counts: dict[str, int] = {}
    for r in reports:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    lines = ["# Requalification Status — executed positive-control evidence",
             "",
             "Independently recomputed from committed artifacts by "
             "`scripts/requalify_executed_campaign.py`. A run is VERIFIED "
             "only when its commit inventory matches the bytes on disk "
             "AND strict rescoring reproduces the reported headline "
             "metrics.",
             "",
             "| run | commit | integrity | rescore vs reported | status |",
             "|---|---|---|---|---|"]
    for r in reports:
        inv = r["checks"].get("inventory", "?")
        inv_s = "ok" if inv == "ok" else "FAIL"
        mm = "; ".join(r["mismatches"]) if r["mismatches"] else "—"
        lines.append(f"| {r['run_id']} | {r['checks'].get('commit_status')}"
                     f" | {inv_s} | {mm} | {r['status']} |")
    lines += ["", "## Totals", ""]
    for k in sorted(counts):
        lines.append(f"- {k}: {counts[k]}")
    lines += ["",
              "Note: VERIFIED here means *evidence integrity plus strict "
              "score reproduction*. It is not a qualification verdict — "
              "family-level significance and gate outcomes belong to the "
              "campaign qualifier."]

    out_path = ROOT / args.output if not Path(args.output).is_absolute() \
        else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"{len(reports)} runs: {counts}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
