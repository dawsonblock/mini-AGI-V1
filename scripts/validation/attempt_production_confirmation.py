#!/usr/bin/env python3
"""Campaign 3B.3 P5 — production-mode sealed confirmation attempt.

The honest version of "independently qualify the winner": run the real
production confirmation path against real evidence, on this host, and
record EXACTLY where it stops. On a same-UID host with no independently
held evaluator and no qualified research record, the correct outcome is
a refusal — and this script records which gate refused, with the
machine-checked custody report alongside.

Emits P5_CONFIRMATION_ATTEMPT.json — never a QUALIFIED claim. If this
ever emits production_valid=true, something is wrong: this host cannot
hold independent evaluator custody.

Usage:
  python3 scripts/validation/attempt_production_confirmation.py \
      --storage <storage-root> --campaign-id <id> \
      --sealed <sealed.jsonl> --submission <submission.json> \
      --output P5_CONFIRMATION_ATTEMPT.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "scripts/validation"))

import run_sealed_confirmation as rsc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--sealed", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    storage = Path(args.storage).resolve()
    sealed_path = Path(args.sealed).resolve()
    record: dict = {
        "schema": "mini-agi-campaign3b3-p5-attempt-v1",
        "host_facts": {
            "euid": os.geteuid() if hasattr(os, "geteuid") else None,
            "storage_owner": (storage.stat().st_uid
                              if storage.exists() else None),
            "sealed_owner": (sealed_path.stat().st_uid
                             if sealed_path.exists() else None),
            "note": ("same-UID host: the sealed set and the research "
                     "tree are owned by one OS identity, so independent "
                     "evaluator custody cannot be established here")},
    }

    # 1. machine-checked custody report — the load-bearing fact
    try:
        custody = rsc.sealed_custody_report(sealed_path, storage=storage)
    except Exception as exc:
        custody = {"error": f"{type(exc).__name__}: {exc}",
                   "independent_isolation_established": False}
    record["custody_report"] = custody
    record["custody_established"] = bool(
        custody.get("independent_isolation_established"))

    # 2. attempt the real production confirmation; record where it
    #    stops. ConfirmationRefused = a gate refused (expected here).
    try:
        campaign_dir = storage / "campaigns" / args.campaign_id
        plan = rsc._load_plan(campaign_dir)
        signers = rsc._load_signers(storage)
        submission = rsc.load_submission(Path(args.submission))
        sealed_rows = rsc.load_sealed(sealed_path, storage=storage)
        protocol = None
        proto_path = campaign_dir / "EXPERIMENT_PROTOCOL.json"
        if proto_path.is_file():
            from minagi.v161.experiment_protocol import \
                ExperimentProtocolV1
            protocol = ExperimentProtocolV1(**json.loads(
                proto_path.read_text())["value"])
        execute_arm = rsc._model_executor(storage, args.campaign_id,
                                          plan, submission, protocol)
        conf = rsc.run_confirmation(
            storage=storage, plan=plan, submission=submission,
            sealed_rows=sealed_rows, execute_arm=execute_arm,
            signers=signers, custody=custody, mode="production",
            retention_scorer=str(getattr(protocol, "retention_scorer",
                                         "retention_score")),
            verify_artifacts=lambda: rsc.verify_candidate_artifacts(
                storage, args.campaign_id, plan))
        record["confirmation"] = {
            "reached": True,
            "decision": conf["value"]["decision"],
            "production_valid": conf["value"]["production_valid"],
            "reasons": conf["value"]["reasons"]}
    except rsc.ConfirmationRefused as exc:
        record["confirmation"] = {"reached": False,
                                  "refused_at": str(exc)}
    except Exception as exc:
        record["confirmation"] = {
            "reached": False,
            "error": f"{type(exc).__name__}: {exc}"}

    # 3. verdict — this attempt can never produce a qualification
    record["verdict"] = "NOT_QUALIFIED"
    record["statement"] = (
        "A production sealed confirmation was attempted against real "
        "campaign evidence on a same-UID host. It could not produce a "
        "qualified result: no independently held evaluator custody "
        "exists here, and the campaign's own research record is "
        "REFUSED — there is no qualified candidate to confirm. The "
        "refusal is the correct, fail-closed output. P5 remains OPEN "
        "until a candidate survives matched multi-seed evaluation AND "
        "an evaluator on a distinct OS identity verifies it against a "
        "fresh sealed benchmark.")

    Path(args.output).write_text(json.dumps(record, indent=2,
                                            sort_keys=True) + "\n")
    print(json.dumps({"verdict": record["verdict"],
                      "custody_established": record["custody_established"],
                      "confirmation": record["confirmation"]},
                     indent=2))
    return 2  # never 0 — this attempt does not qualify anything


if __name__ == "__main__":
    raise SystemExit(main())
