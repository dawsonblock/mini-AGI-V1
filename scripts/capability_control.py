#!/usr/bin/env python3
"""Capability acquisition controller CLI (V4-13).

Reads measured evidence (a JSON mapping of FailureEvidence fields),
produces a diagnosis and an intervention decision, and appends verified
outcomes to the controller's intervention history.

    # diagnose + select from a metrics file
    python3 scripts/capability_control.py decide \
        --evidence evidence.json --history history.jsonl

    # record a verified experimental outcome (closes the loop)
    python3 scripts/capability_control.py record \
        --history history.jsonl --intervention fixed_lora \
        --delta 0.12 --retention-drop 0.02 --verified --cost 900

The history file is the controller's memory: prior verified failures
block the same intervention under unchanged conditions unless
--changed-condition documents what is different.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.capability_controller import (  # noqa: E402
    FailureEvidence, InterventionRecord, diagnose,
    intervention_history_append, intervention_history_load,
    select_intervention, _signature)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decide")
    d.add_argument("--evidence", required=True,
                   help="JSON file with FailureEvidence fields")
    d.add_argument("--history", default="history.jsonl")
    d.add_argument("--target", default="")
    d.add_argument("--conditions", default="")
    d.add_argument("--changed-condition", default=None)
    d.add_argument("--output", default=None)

    r = sub.add_parser("record")
    r.add_argument("--history", default="history.jsonl")
    r.add_argument("--intervention", required=True)
    r.add_argument("--delta", type=float, required=True)
    r.add_argument("--retention-drop", type=float, default=0.0)
    r.add_argument("--cost", type=float, default=0.0)
    r.add_argument("--verified", action="store_true")
    r.add_argument("--target", default="")
    r.add_argument("--conditions", default="")

    args = ap.parse_args()

    if args.cmd == "decide":
        ev = FailureEvidence(**json.loads(
            Path(args.evidence).read_text()))
        hyps = diagnose(ev)
        hist = intervention_history_load(Path(args.history))
        dec = select_intervention(
            hyps, hist, target=args.target, conditions=args.conditions,
            changed_condition=args.changed_condition)
        doc = {"hypotheses": [h.__dict__ for h in hyps],
               "decision": dec.__dict__}
        text = json.dumps(doc, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text + "\n")
        print(text)
        return 0

    rec = InterventionRecord(
        signature=_signature(args.intervention, args.target,
                             args.conditions),
        intervention=args.intervention,
        outcome_delta=args.delta,
        retention_drop=args.retention_drop,
        verified=bool(args.verified),
        cost=args.cost,
        conditions=args.conditions)
    intervention_history_append(Path(args.history), rec)
    print(json.dumps(rec.__dict__, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
