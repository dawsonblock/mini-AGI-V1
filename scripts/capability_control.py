#!/usr/bin/env python3
"""Capability acquisition controller CLI (V4-13, V4-31).

Reads measured evidence (a JSON mapping of FailureEvidence fields),
produces a diagnosis and an intervention decision, and appends verified
outcomes to the controller's intervention history.

    # diagnose + select from a metrics file
    python3 scripts/capability_control.py decide \
        --evidence evidence.json --history history.jsonl

    # record a verified experimental outcome (closes the loop)
    python3 scripts/capability_control.py record \
        --history history.jsonl --intervention fixed_lora \
        --delta 0.12 --retention-drop 0.02 --cost 900 \
        --verified --receipt results/.../runs/<run_id>

V4-31: ``--verified`` alone does NOT establish verification. It requires
``--receipt`` naming independently verifiable evidence — a committed
run directory (RUN_COMMIT inventory + rescorable metrics) or a signed
receipt document checked against ``--trust-root``. The recorded entry
carries the receipt digest so the verification claim can be re-audited.

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

from dataclasses import asdict  # noqa: E402

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
    d.add_argument("--policy", choices=["hypothesis", "adaptive"],
                   default="hypothesis",
                   help="adaptive = v4.2.0 measured-outcome selection "
                        "over --outcomes (receipt-gated); hypothesis = "
                        "the v4.0 prior policy")
    d.add_argument("--outcomes", default=None,
                   help="jsonl of OutcomeRecord rows (adaptive policy)")
    d.add_argument("--receipt-root", default=None,
                   help="directory whose per-run subdirs resolve each "
                        "record's run_id for receipt verification")
    d.add_argument("--trust-root", default=None,
                   help="TRUST_ROOT.json for signed receipt documents")

    r = sub.add_parser("record")
    r.add_argument("--history", default="history.jsonl")
    r.add_argument("--intervention", required=True)
    r.add_argument("--delta", type=float, required=True)
    r.add_argument("--retention-drop", type=float, default=0.0)
    r.add_argument("--cost", type=float, default=0.0)
    r.add_argument("--verified", action="store_true")
    r.add_argument("--receipt", default=None,
                   help="path to independently verifiable outcome "
                        "evidence (a committed run dir, RUN_COMMIT.json, "
                        "or a signed receipt document). Required for "
                        "--verified.")
    r.add_argument("--trust-root", default=None,
                   help="TRUST_ROOT.json for checking a signed receipt "
                        "document's signature")
    r.add_argument("--target", default="")
    r.add_argument("--conditions", default="")

    o = sub.add_parser("record-outcome",
                       help="v4.2.0 outcome model — multi-surface "
                            "deltas + measured resources; receipt "
                            "required")
    o.add_argument("--outcomes", default="outcomes.jsonl")
    o.add_argument("--intervention", required=True)
    o.add_argument("--target", default="string_transformation")
    o.add_argument("--within-delta", type=float, required=True)
    o.add_argument("--cross-delta", type=float, default=0.0)
    o.add_argument("--retention-regression", type=float, default=0.0)
    o.add_argument("--qualified", action="store_true",
                   help="the independent qualifier passed this outcome")
    o.add_argument("--train-seconds", type=float, default=0.0)
    o.add_argument("--gpu-seconds", type=float, default=0.0)
    o.add_argument("--eval-seconds", type=float, default=0.0)
    o.add_argument("--run-id", default="")
    o.add_argument("--seed", type=int, default=None)
    o.add_argument("--receipt", required=True,
                   help="committed run dir or signed receipt — the "
                        "record's digest binds to it")
    o.add_argument("--trust-root", default=None)

    args = ap.parse_args()

    if args.cmd == "decide":
        ev = FailureEvidence(**json.loads(
            Path(args.evidence).read_text()))
        hyps = diagnose(ev)
        if args.policy == "adaptive":
            from minagi.v161.adaptive_intervention import (
                load_outcomes, verify_outcomes)
            from minagi.v161.capability_controller import (
                select_intervention_adaptive)
            from minagi.v161.capability_outcomes import (
                verify_outcome_receipt)

            verifier = None
            if args.trust_root:
                from minagi.v161.authority import AuthorityRegistry
                verifier = AuthorityRegistry.load(
                    args.trust_root).verifier()

            def receipt_ok(rec) -> bool:
                if not rec.run_id or not args.receipt_root:
                    return False
                res = verify_outcome_receipt(
                    Path(args.receipt_root) / rec.run_id, verifier)
                return bool(res.get("verified")) and \
                    res.get("receipt_digest") == rec.evidence_receipt_digest

            outcomes = load_outcomes(args.outcomes) if args.outcomes else []
            split = verify_outcomes(outcomes, receipt_ok)
            dec = select_intervention_adaptive(
                hyps, split["verified"], target=args.target,
                conditions=args.conditions)
            dec["unverified_records_ignored"] = len(split["unverified"])
            text = json.dumps({"hypotheses": [h.__dict__ for h in hyps],
                               "decision": dec}, indent=2,
                              sort_keys=True)
            if args.output:
                Path(args.output).write_text(text + "\n")
            print(text)
            return 0
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

    if args.cmd == "record-outcome":
        from minagi.v161.adaptive_intervention import (
            OutcomeRecord, append_outcome)
        from minagi.v161.capability_outcomes import (
            verify_outcome_receipt)

        verifier = None
        if args.trust_root:
            from minagi.v161.authority import AuthorityRegistry
            verifier = AuthorityRegistry.load(
                args.trust_root).verifier()
        res = verify_outcome_receipt(args.receipt, verifier)
        if not res.get("verified"):
            print(f"error: receipt failed verification — "
                  f"{res.get('reason')} (a v4.2 outcome record must "
                  f"bind verified evidence)", file=sys.stderr)
            return 2
        import time as _t
        rec = OutcomeRecord(
            intervention=args.intervention,
            target_capability=args.target,
            evaluation={
                "within_family_delta": args.within_delta,
                "cross_family_delta": args.cross_delta,
                "retention_regression": args.retention_regression,
                "qualification_passed": bool(args.qualified)},
            resources={"training_seconds": args.train_seconds,
                       "gpu_seconds": args.gpu_seconds,
                       "evaluation_seconds": args.eval_seconds},
            evidence_receipt_digest=res["receipt_digest"],
            run_id=args.run_id or str(res.get("run_id") or ""),
            seed=args.seed,
            recorded_utc=_t.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     _t.gmtime()))
        append_outcome(args.outcomes, rec)
        print(json.dumps(asdict(rec), indent=1, sort_keys=True))
        return 0

    # ---- record -------------------------------------------------------
    from minagi.v161.capability_outcomes import verify_outcome_receipt

    verified = bool(args.verified)
    receipt_digest = ""
    if args.verified and not args.receipt:
        print("error: --verified requires --receipt naming independently "
              "verifiable evidence — a bare flag cannot establish "
              "verification (V4-31)", file=sys.stderr)
        return 2
    if args.receipt:
        verifier = None
        if args.trust_root:
            from minagi.v161.authority import AuthorityRegistry
            verifier = AuthorityRegistry.load(args.trust_root).verifier()
        res = verify_outcome_receipt(args.receipt, verifier)
        if not res.get("verified"):
            if args.verified:
                print(f"error: receipt failed verification — "
                      f"{res.get('reason')} (refusing to record a "
                      f"verified outcome)", file=sys.stderr)
                return 2
            print(f"warning: receipt not verifiable "
                  f"({res.get('reason')}) — recorded unverified",
                  file=sys.stderr)
            verified = False
        else:
            receipt_digest = res["receipt_digest"]

    rec = InterventionRecord(
        signature=_signature(args.intervention, args.target,
                             args.conditions),
        intervention=args.intervention,
        outcome_delta=args.delta,
        retention_drop=args.retention_drop,
        verified=verified,
        cost=args.cost,
        conditions=args.conditions,
        receipt_digest=receipt_digest)
    intervention_history_append(Path(args.history), rec)
    print(json.dumps(rec.__dict__, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
