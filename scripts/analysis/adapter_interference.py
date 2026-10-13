#!/usr/bin/env python3
"""Campaign 3B.2 Step 7 — reproduce adapter interference from preserved
predictions.

The Campaign 3B.1 report claimed L7 loses six rows and wins none against
L5b across twelve held-out tasks. That claim must be reproducible from
the preserved per-row predictions rather than from a summary table, and
every task must receive an explicit outcome:

    preserved-correct   both baseline and treatment correct
    newly-correct       baseline wrong, treatment correct
    destroyed-correct   baseline correct, treatment wrong
    still-incorrect     both wrong

It also reports the capability-destruction rate

    D = #(baseline correct and treatment incorrect) / #(baseline correct)

with D = NOT_ESTIMABLE when the denominator is zero — never 0.0, which
would claim demonstrated preservation from evidence that cannot measure
it (the SCI-305 rule).

Step 9 diagnostics carried alongside, because a bare score cannot say
what the adapter did:

  family-confusion   does the treatment output equal a TRAINED family's
                     transformation of the same input? (one learned
                     transformation applied to an unrelated task)
  format-drift       did the output contract break (prose, multi-line,
                     over-length) where the baseline held it?
  length-change      generated length before/after

Usage:
  python3 scripts/analysis/adapter_interference.py \
      --predictions-dir <dir with PREDICTIONS-<ARM>.jsonl> \
      --corpus configs/positive_control_tasks.jsonl \
      --treatment L7 --baseline L5b \
      --output INTERFERENCE.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "scripts"))

NOT_ESTIMABLE = "NOT_ESTIMABLE"

#: The answer-shaped single line the frozen contract requires. Prose,
#: multi-line and over-length output are format drift.
import re  # noqa: E402
ANSWER_SHAPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ,\-]{0,31}")


def _trained_transforms():
    """The corpus's own train-family transformations.

    Imported from the generator rather than reimplemented, so the
    confusion check cannot drift from the corpus it analyses.
    """
    from generate_positive_control import TRAIN_FAMILIES
    return {fam: fn for fam, (_line, fn) in TRAIN_FAMILIES.items()
            if fn is not None}


def load_predictions(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[str(rec["id"])] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-dir", required=True)
    ap.add_argument("--corpus",
                    default="configs/positive_control_tasks.jsonl")
    ap.add_argument("--treatment", default="L7")
    ap.add_argument("--baseline", default="L5b")
    ap.add_argument("--split", default="hidden")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pred_dir = Path(args.predictions_dir)
    corpus = {}
    for line in (ROOT / args.corpus).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            corpus[str(row["id"])] = row

    base_preds = load_predictions(pred_dir / f"PREDICTIONS-{args.baseline}.jsonl")
    treat_preds = load_predictions(pred_dir / f"PREDICTIONS-{args.treatment}.jsonl")
    transforms = _trained_transforms()

    ids = sorted(rid for rid, row in corpus.items()
                 if row["split"] == args.split)
    if not ids:
        raise SystemExit(f"no {args.split!r} rows in {args.corpus}")

    def norm(s: str) -> str:
        return str(s).strip().casefold()

    rows, outcomes = [], {"preserved-correct": 0, "newly-correct": 0,
                          "destroyed-correct": 0, "still-incorrect": 0}
    missing = []
    for rid in ids:
        row = corpus[rid]
        b = base_preds.get(rid)
        t = treat_preds.get(rid)
        if b is None or t is None:
            missing.append(rid)
            continue
        expected = str(row["expected"])
        bo, to = str(b.get("output", "")), str(t.get("output", ""))
        bc = int(norm(bo) == norm(expected))
        tc = int(norm(to) == norm(expected))
        outcome = ("preserved-correct" if bc and tc else
                   "newly-correct" if tc else
                   "destroyed-correct" if bc else "still-incorrect")
        outcomes[outcome] += 1

        # Step 9: is the treatment emitting a TRAINED transformation?
        payload = str(row["prompt"]).split("Input: ", 1)[-1] \
            .rsplit("\nAnswer:", 1)[0]
        # Case-SENSITIVE first: scoring is casefolded (so "ICE" scores
        # for "IcE"), but for diagnosis a casefolded comparison conflates
        # distinct transformations — pc-sort("egg") is "egg", which
        # casefolds equal to pc-upper("egg")="EGG". Reporting both would
        # over-count the confusion signal, so exact matches are separated
        # from casefold-only matches.
        stripped = to.strip()
        confused = sorted(fam for fam, fn in transforms.items()
                          if str(fn(payload)).strip() == stripped)
        confused_casefold_only = sorted(
            fam for fam, fn in transforms.items()
            if fam not in confused
            and norm(fn(payload)) == norm(stripped))
        b_fmt = bool(ANSWER_SHAPE.fullmatch(bo.strip()))
        t_fmt = bool(ANSWER_SHAPE.fullmatch(to.strip()))

        rows.append({
            "task_id": rid, "family_id": row["family"],
            "expected_answer": expected,
            "baseline_arm": args.baseline, "baseline_output": bo,
            "adapted_arm": args.treatment, "adapted_output": to,
            "baseline_correct": bc, "adapted_correct": tc,
            "outcome": outcome,
            "input_sha256_baseline": b.get("input_sha256"),
            "input_sha256_treatment": t.get("input_sha256"),
            "input_identical": b.get("input_sha256") == t.get("input_sha256"),
            "truncation_status": {
                "baseline": b.get("max_new_tokens") is not None
                and b.get("generated_tokens") == b.get("max_new_tokens"),
                "treatment": t.get("max_new_tokens") is not None
                and t.get("generated_tokens") == t.get("max_new_tokens")},
            "generation_finish_reason": {
                "baseline": b.get("status", "ok"),
                "treatment": t.get("status", "ok")},
            "generated_tokens": {"baseline": b.get("generated_tokens"),
                                 "treatment": t.get("generated_tokens")},
            "family_confusion": confused,
            "family_confusion_casefold_only": confused_casefold_only,
            "format_drift": {"baseline_valid": b_fmt, "treatment_valid": t_fmt,
                             "broke": b_fmt and not t_fmt},
        })

    n_base_correct = sum(r["baseline_correct"] for r in rows)
    n_destroyed = sum(1 for r in rows if r["outcome"] == "destroyed-correct")
    n_gained = sum(1 for r in rows if r["outcome"] == "newly-correct")

    by_family: dict = {}
    for r in rows:
        f = by_family.setdefault(r["family_id"], {
            "n": 0, "baseline_correct": 0, "adapted_correct": 0,
            "destroyed": 0, "gained": 0})
        f["n"] += 1
        f["baseline_correct"] += r["baseline_correct"]
        f["adapted_correct"] += r["adapted_correct"]
        f["destroyed"] += r["outcome"] == "destroyed-correct"
        f["gained"] += r["outcome"] == "newly-correct"

    report = {
        "schema": "mini-agi-campaign3b2-interference-report-v1",
        "predictions_dir": str(pred_dir),
        "corpus": args.corpus, "split": args.split,
        "baseline_arm": args.baseline, "treatment_arm": args.treatment,
        "n_tasks": len(rows), "n_missing_predictions": len(missing),
        "missing_task_ids": missing,
        "scores": {
            "baseline_correct": n_base_correct,
            "treatment_correct": sum(r["adapted_correct"] for r in rows),
            "baseline_accuracy": n_base_correct / len(rows) if rows else None,
            "treatment_accuracy": (sum(r["adapted_correct"] for r in rows)
                                   / len(rows)) if rows else None},
        "outcomes": outcomes,
        "paired": {"wins": n_gained, "losses": n_destroyed,
                   "ties": len(rows) - n_gained - n_destroyed,
                   "delta": ((n_gained - n_destroyed) / len(rows))
                   if rows else None},
        "capability_destruction_rate": {
            "numerator": n_destroyed, "denominator": n_base_correct,
            "D": (n_destroyed / n_base_correct) if n_base_correct
            else None,
            "status": "ESTIMATED" if n_base_correct else NOT_ESTIMABLE},
        "input_identity": {
            "n_identical": sum(r["input_identical"] for r in rows),
            "n": len(rows),
            "all_identical": all(r["input_identical"] for r in rows)},
        "format_drift": {
            "n_broke": sum(r["format_drift"]["broke"] for r in rows)},
        "family_confusion": {
            "n_confused": sum(1 for r in rows if r["family_confusion"]),
            "by_family": {fam: sum(1 for r in rows
                                   if fam in r["family_confusion"])
                          for fam in sorted(transforms)},
            "n_casefold_only": sum(1 for r in rows
                                   if r["family_confusion_casefold_only"]),
            "by_family_casefold_only": {
                fam: sum(1 for r in rows
                         if fam in r["family_confusion_casefold_only"])
                for fam in sorted(transforms)}},
        "per_family": {k: by_family[k] for k in sorted(by_family)},
        "tasks": rows,
    }
    Path(args.output).write_text(json.dumps(report, indent=2,
                                            sort_keys=True) + "\n")

    print(f"{args.treatment} vs {args.baseline} on {len(rows)} {args.split} tasks")
    print(f"  baseline {report['scores']['baseline_accuracy']:.3f} -> "
          f"treatment {report['scores']['treatment_accuracy']:.3f}")
    print(f"  outcomes: {outcomes}")
    print(f"  D (destroyed/baseline-correct): "
          f"{report['capability_destruction_rate']['D']} "
          f"[{report['capability_destruction_rate']['status']}]")
    print(f"  inputs identical: "
          f"{report['input_identity']['n_identical']}/"
          f"{report['input_identity']['n']}")
    print(f"  family confusion: {report['family_confusion']['n_confused']} "
          f"tasks emit a trained transformation")
    if missing:
        print(f"  WARNING: {len(missing)} tasks had no prediction")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
