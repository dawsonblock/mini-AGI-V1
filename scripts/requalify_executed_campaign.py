#!/usr/bin/env python3
"""Requalify executed positive-control evidence (V4-22, V4-27).

Scans ``results/<campaign>/runs/<run_id>`` directories and re-derives
what is verifiable from the committed artifacts — without trusting the
runner's reported metrics:

1. RUN_COMMIT.json integrity — status COMPLETE/FAILED and every file in
   the commit inventory hashed against the bytes on disk.
2. Prediction coverage — the exact expected file set (reconstructed
   from CONFIG.json's declared split counts), unique task ids, no
   missing/duplicated/unexpected predictions, and — when the declared
   corpus resolves — every prediction joined back to its corpus row
   (split membership, prompt_sha256, expected answer).
3. Strict rescore of EVERY prediction file — hidden, train-subset,
   validation, and compositional — compared against the reported
   metrics including the within-family decomposition and the
   capability-destruction id lists.
4. Identity bindings — commit/config digest agreement, adapter
   manifest ↔ adapter bytes ↔ training receipt/spec digests, model
   identity, contract digests, scorer identity.

Verdicts: VERIFIED / INCONSISTENT / FAILED_RUN / INVALID.

Writes ``results/REQUALIFICATION_STATUS.md`` — the honest status table
backing the release's "executed evidence" claims. No model is loaded;
this is pure artifact verification.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.evidence_reconciliation import verify_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", default=None,
                    help="run roots to scan (default: results/*/runs/*)")
    ap.add_argument("--output", default="results/REQUALIFICATION_STATUS.md")
    args = ap.parse_args()

    if args.results:
        run_dirs = [Path(p) for p in args.results]
    else:
        # dot-named entries are the v4.0.2 reservation internals
        # (.staging/, .claims/) — not committed runs.
        run_dirs = sorted(d for d in ROOT.glob("results/*/runs/*/")
                          if not d.name.startswith("."))
    corpus_cache: dict[str, list[dict]] = {}

    def _corpus_for(d: Path):
        import json
        cfg = d / "CONFIG.json"
        if not cfg.is_file():
            return None
        c = json.loads(cfg.read_text()).get("corpus")
        if not c:
            return None
        if c not in corpus_cache:
            p = ROOT / str(c)
            corpus_cache[c] = ([json.loads(ln) for ln in
                                p.read_text().splitlines() if ln.strip()]
                               if p.is_file() else [])
        return corpus_cache[c] or None

    reports = [verify_run(d, repo_root=ROOT,
                          corpus_rows=_corpus_for(d))
               for d in run_dirs if d.is_dir()]

    counts: dict[str, int] = {}
    for r in reports:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    lines = ["# Requalification Status — executed positive-control evidence",
             "",
             "Independently recomputed from committed artifacts by "
             "`scripts/requalify_executed_campaign.py`. A run is VERIFIED "
             "only when its commit inventory matches the bytes on disk, "
             "every committed prediction file rescored under the strict "
             "oracle reproduces the reported metrics, and the "
             "commit/config/adapter/receipt identity bindings all agree.",
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
    n_pred = sum(len(r["rescore"]) for r in reports)
    lines += ["",
              f"Prediction files independently rescored: {n_pred}",
              "",
              "Note: VERIFIED here means *evidence integrity plus strict "
              "score reproduction across every committed split plus "
              "identity binding*. It is not a qualification verdict — "
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
