#!/usr/bin/env python3
"""Scorer calibration — run BEFORE campaign preregistration.

Evaluates containment_match against the frozen, manually labeled
calibration set (configs/scorer_calibration.jsonl). Emits
SCORER_CALIBRATION.json with per-case agreement and exits non-zero
below the preregistered agreement floor.

The calibration artifact digest is bound into the retention evaluator
artifact config, so the signed campaign plan transitively binds the
exact scorer + calibration evidence used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import detect_root, ensure_path

MIN_AGREEMENT = 0.90


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--output", default="SCORER_CALIBRATION.json")
    args = ap.parse_args()
    root = ensure_path(Path(args.root).resolve() if args.root else detect_root())

    from minagi.v161.evaluators import containment_match
    from egai.common.canonical import digest

    rows = [json.loads(line) for line in
            (root / "configs" / "scorer_calibration.jsonl").read_text().splitlines()
            if line.strip()]
    cases = []
    agree = 0
    for r in rows:
        got = int(containment_match(r["prediction"], r["expected"]))
        ok = got == int(r["label"])
        agree += ok
        cases.append({"id": r["id"], "expected_label": r["label"],
                      "scored": got, "agree": ok,
                      "note": r.get("note", "")})
    total = len(rows)
    agreement = agree / total
    record = {
        "schema": "mini-agi-v16.2-scorer-calibration-v1",
        "scorer": "containment_match",
        "calibration_set": "configs/scorer_calibration.jsonl",
        "calibration_set_digest": digest(rows),
        "n_cases": total,
        "agreements": agree,
        "agreement_rate": agreement,
        "min_agreement_required": MIN_AGREEMENT,
        "mismatches": [c for c in cases if not c["agree"]],
        "cases": cases,
        "status": "PASS" if agreement >= MIN_AGREEMENT else "FAIL",
    }
    Path(args.output).write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({k: record[k] for k in
                      ("schema", "n_cases", "agreements", "agreement_rate",
                       "status")}, indent=2))
    for m in record["mismatches"]:
        print(f"[MISMATCH] {m['id']}: label={m['expected_label']} "
              f"scored={m['scored']} {m['note']}", file=sys.stderr)
    return 0 if record["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
