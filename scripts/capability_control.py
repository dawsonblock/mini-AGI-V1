#!/usr/bin/env python3
"""Capability acquisition controller CLI (V4-13, V4-31, v4.0.3).

Reads measured evidence (a JSON mapping of FailureEvidence fields),
produces a diagnosis and an intervention decision, and appends verified
outcomes to the controller's intervention history.

    # diagnose + select from a metrics file
    python3 scripts/capability_control.py decide \
        --evidence evidence.json --history history.jsonl

    # record a verified experimental outcome — DERIVED FROM EVIDENCE
    python3 scripts/capability_control.py record \
        --history history.jsonl \
        --receipt results/.../runs/<run_id> \
        --trust-root config/trust/authority.json

v4.0.3 contract (fabricated-outcome closure):

* ``--receipt`` names independently verifiable evidence: a committed
  run directory (RUN_COMMIT inventory + strict prediction rescore +
  identity bindings via evidence_reconciliation.verify_run) or a
  signed receipt document verified against ``--trust-root``.
* Verified records DERIVE their numbers — delta, retention regression,
  cost, intervention — from the rescored evidence via
  qualified_outcome.derive_qualified_outcome. ``--delta``/``--cost``/
  ``--retention-drop`` are REJECTED alongside --receipt: the controller
  may never substitute operator-supplied numbers for measured ones.
* Manual numbers remain available ONLY as an explicitly unverified
  diagnostic record (``--diagnostic``). Diagnostic records carry
  verified=false and can never promote into verified history.
* A run reaches verified history only in state EVALUATION_VERIFIED
  (independent rescore) or QUALIFIED (signed decision from a
  qualification-authorized signer). A hash-consistent self-authored
  commit is INTEGRITY_VERIFIED at best and is refused.

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


def _load_authority(trust_root):
    """Trust root -> (AuthorityRegistry, verifier) — both, so role
    authorization is enforced, not just key presence."""
    from minagi.v161.authority import AuthorityRegistry
    reg = AuthorityRegistry.load(trust_root)
    return reg, reg.verifier()


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
    d.add_argument("--repo-root", default=str(ROOT),
                   help="repository root for corpus-join checks during "
                        "run-dir receipt verification")

    r = sub.add_parser("record")
    r.add_argument("--history", default="history.jsonl")
    r.add_argument("--receipt", default=None,
                   help="path to independently verifiable outcome "
                        "evidence (a committed run dir, RUN_COMMIT.json, "
                        "or a signed receipt document). Verified "
                        "records are DERIVED from this evidence.")
    r.add_argument("--trust-root", default=None,
                   help="TRUST_ROOT.json — required for signed receipt "
                        "documents and for qualification decisions")
    r.add_argument("--repo-root", default=str(ROOT))
    r.add_argument("--diagnostic", action="store_true",
                   help="explicitly UNVERIFIED diagnostic record: manual "
                        "--delta/--cost/--retention-drop permitted; the "
                        "record never counts as verified")
    r.add_argument("--intervention", default=None,
                   help="with --receipt: must equal the intervention "
                        "derived from evidence; with --diagnostic: "
                        "required manual label")
    r.add_argument("--delta", type=float, default=None)
    r.add_argument("--retention-drop", type=float, default=None)
    r.add_argument("--cost", type=float, default=None)
    r.add_argument("--verified", action="store_true",
                   help="DEPRECATED no-op — verification is established "
                        "by --receipt evidence, never by assertion")
    r.add_argument("--target", default="")
    r.add_argument("--conditions", default="")

    o = sub.add_parser("record-outcome",
                       help="v4.2.0 outcome model — multi-surface "
                            "deltas + measured resources; derived from "
                            "--receipt evidence (or --diagnostic)")
    o.add_argument("--outcomes", default="outcomes.jsonl")
    o.add_argument("--receipt", default=None,
                   help="committed run dir or signed receipt — the "
                        "record's fields are derived from it")
    o.add_argument("--trust-root", default=None)
    o.add_argument("--repo-root", default=str(ROOT))
    o.add_argument("--diagnostic", action="store_true",
                   help="explicitly UNVERIFIED manual record")
    o.add_argument("--intervention", default=None)
    o.add_argument("--target", default="string_transformation")
    o.add_argument("--within-delta", type=float, default=None)
    o.add_argument("--cross-delta", type=float, default=None)
    o.add_argument("--retention-regression", type=float, default=None)
    o.add_argument("--qualified", action="store_true",
                   help="asserts the evidence carries a signed "
                        "qualification decision — fails if it does not")
    o.add_argument("--train-seconds", type=float, default=None)
    o.add_argument("--gpu-seconds", type=float, default=None)
    o.add_argument("--eval-seconds", type=float, default=None)
    o.add_argument("--run-id", default="")
    o.add_argument("--seed", type=int, default=None)

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
            authority = None
            if args.trust_root:
                authority, verifier = _load_authority(args.trust_root)

            def receipt_ok(rec) -> bool:
                if not rec.run_id or not args.receipt_root:
                    return False
                res = verify_outcome_receipt(
                    Path(args.receipt_root) / rec.run_id, verifier,
                    authority=authority, repo_root=args.repo_root)
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
        from minagi.v161.qualified_outcome import QUALIFIED

        manual = any(v is not None for v in (
            args.within_delta, args.cross_delta, args.retention_regression,
            args.train_seconds, args.gpu_seconds, args.eval_seconds))
        if args.diagnostic:
            if args.within_delta is None or not args.intervention:
                print("error: --diagnostic records require --intervention "
                      "and --within-delta (manual numbers permitted only "
                      "here; the record is unverified)", file=sys.stderr)
                return 2
            import time as _t
            rec = OutcomeRecord(
                intervention=args.intervention,
                target_capability=args.target,
                evaluation={
                    "within_family_delta": args.within_delta,
                    "cross_family_delta": args.cross_delta or 0.0,
                    "retention_regression": args.retention_regression
                    or 0.0,
                    "qualification_passed": False},
                resources={"training_seconds": args.train_seconds or 0.0,
                           "gpu_seconds": args.gpu_seconds or 0.0,
                           "evaluation_seconds": args.eval_seconds or 0.0},
                evidence_receipt_digest="",
                run_id=args.run_id,
                seed=args.seed,
                recorded_utc=_t.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         _t.gmtime()))
            append_outcome(args.outcomes, rec)
            print(json.dumps({**asdict(rec), "provenance": "diagnostic"},
                             indent=1, sort_keys=True))
            return 0

        if not args.receipt:
            print("error: record-outcome requires --receipt (evidence "
                  "the record is derived from) or --diagnostic "
                  "(explicitly unverified)", file=sys.stderr)
            return 2
        if manual:
            print("error: verified records DERIVE their numbers from "
                  "evidence — --within-delta/--cross-delta/"
                  "--retention-regression/--*-seconds are rejected with "
                  "--receipt (use --diagnostic for manual records)",
                  file=sys.stderr)
            return 2
        if not args.intervention:
            print("error: --intervention required (the recorded "
                  "intervention name; evidence-derived numbers bind to "
                  "it)", file=sys.stderr)
            return 2

        from minagi.v161.capability_outcomes import verify_outcome_receipt
        verifier = None
        authority = None
        if args.trust_root:
            authority, verifier = _load_authority(args.trust_root)
        res = verify_outcome_receipt(args.receipt, verifier,
                                     authority=authority,
                                     repo_root=args.repo_root)
        if not res.get("verified"):
            print(f"error: receipt failed verification — "
                  f"{res.get('reason')} (a v4.2 outcome record must "
                  f"bind verified evidence)", file=sys.stderr)
            return 2
        oc = (res.get("outcome") or {})
        qualified = res.get("qualification_status") == QUALIFIED
        if args.qualified and not qualified:
            print("error: --qualified asserted but the evidence carries "
                  "no valid signed qualification decision — a caller "
                  "cannot declare qualification", file=sys.stderr)
            return 2
        run_id = args.run_id or str(oc.get("run_id") or "")
        if oc.get("run_id") and args.run_id \
                and str(oc["run_id"]) != args.run_id:
            print("error: --run-id contradicts the evidence's run_id",
                  file=sys.stderr)
            return 2
        import time as _t
        rec = OutcomeRecord(
            intervention=args.intervention,
            target_capability=args.target,
            evaluation={
                "within_family_delta": oc.get("outcome_delta")
                if oc.get("outcome_delta") is not None else 0.0,
                "cross_family_delta": oc.get("cross_family_delta")
                if oc.get("cross_family_delta") is not None else 0.0,
                "retention_regression": oc.get("retention_regression")
                if oc.get("retention_regression") is not None else 0.0,
                "qualification_passed": bool(qualified)},
            resources={
                "training_seconds": oc.get("training_seconds", 0.0),
                "gpu_seconds": 0.0,
                "evaluation_seconds": oc.get("evaluation_seconds", 0.0)},
            evidence_receipt_digest=res["receipt_digest"],
            run_id=run_id,
            seed=args.seed if args.seed is not None else oc.get("seed"),
            recorded_utc=_t.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     _t.gmtime()))
        append_outcome(args.outcomes, rec)
        print(json.dumps({**asdict(rec),
                          "provenance": "evidence",
                          "verification_state":
                          res.get("verification_state")},
                         indent=1, sort_keys=True))
        return 0

    # ---- record -------------------------------------------------------
    from minagi.v161.capability_outcomes import verify_outcome_receipt

    if args.verified and not args.receipt:
        print("error: --verified cannot establish verification — it is "
              "a deprecated no-op; verification requires --receipt "
              "naming independently verifiable evidence (V4-31/v4.0.3)",
              file=sys.stderr)
        return 2
    if not args.receipt and not args.diagnostic:
        print("error: record requires --receipt (verified, "
              "evidence-derived) or --diagnostic (explicitly "
              "unverified manual record)", file=sys.stderr)
        return 2
    if args.receipt and any(v is not None for v in (
            args.delta, args.retention_drop, args.cost)):
        print("error: verified records DERIVE numbers from evidence — "
              "--delta/--retention-drop/--cost are rejected with "
              "--receipt (use --diagnostic for manual records)",
              file=sys.stderr)
        return 2

    if args.diagnostic:
        if args.receipt:
            print("error: --diagnostic and --receipt are mutually "
                  "exclusive", file=sys.stderr)
            return 2
        if not args.intervention or args.delta is None:
            print("error: --diagnostic records require --intervention "
                  "and --delta", file=sys.stderr)
            return 2
        rec = InterventionRecord(
            signature=_signature(args.intervention, args.target,
                                 args.conditions),
            intervention=args.intervention,
            outcome_delta=args.delta,
            retention_drop=args.retention_drop or 0.0,
            verified=False,
            cost=args.cost or 0.0,
            conditions=args.conditions,
            receipt_digest="")
        intervention_history_append(Path(args.history), rec)
        print(json.dumps({**rec.__dict__, "provenance": "diagnostic"},
                         indent=1, sort_keys=True))
        return 0

    verifier = None
    authority = None
    if args.trust_root:
        authority, verifier = _load_authority(args.trust_root)
    res = verify_outcome_receipt(args.receipt, verifier,
                                 authority=authority,
                                 repo_root=args.repo_root)
    if not res.get("verified"):
        print(f"error: evidence did not verify — "
              f"{res.get('reason')} (refusing to record a verified "
              f"outcome)", file=sys.stderr)
        return 2
    oc = res.get("outcome") or {}
    intervention = oc.get("intervention") or args.intervention
    if not intervention:
        print("error: evidence does not determine an intervention and "
              "none was supplied", file=sys.stderr)
        return 2
    if args.intervention and oc.get("intervention") \
            and args.intervention != oc["intervention"]:
        print(f"error: --intervention {args.intervention!r} contradicts "
              f"the evidence-derived intervention "
              f"{oc['intervention']!r}", file=sys.stderr)
        return 2

    rec = InterventionRecord(
        signature=_signature(intervention, args.target,
                             args.conditions),
        intervention=intervention,
        outcome_delta=oc.get("outcome_delta")
        if oc.get("outcome_delta") is not None else 0.0,
        retention_drop=oc.get("retention_regression")
        if oc.get("retention_regression") is not None else 0.0,
        verified=True,
        cost=(oc.get("training_seconds") or 0.0)
        + (oc.get("evaluation_seconds") or 0.0),
        conditions=args.conditions,
        receipt_digest=(res.get("outcome_digest")
                        or res["receipt_digest"]))
    intervention_history_append(Path(args.history), rec)
    print(json.dumps({**rec.__dict__,
                      "provenance": "evidence",
                      "verification_state":
                      res.get("verification_state"),
                      "run_id": oc.get("run_id")},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
