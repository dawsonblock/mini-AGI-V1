#!/usr/bin/env python3
"""v16.2 Phases 16-17 — independent artifact-only campaign qualification.

The qualifier consumes ONLY persisted artifacts (campaign plan, dataset proof,
execution public key, A0/A1 receipts, adapter files). It never touches live
trainer objects. It independently:

  * verifies the sealed campaign plan digest
  * verifies every receipt signature against the published execution key
  * binds each receipt to the plan (campaign, evaluator, dataset digests)
  * re-hashes adapter artifacts and compares to the digest bound in receipts
  * refuses incomplete seed matrices
  * reconstructs acquisition / retention / security outcomes and re-derives
    the campaign decision from the preregistered thresholds

Writes QUALIFICATION_RECORD.json inside the campaign dir by default.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import detect_root, ensure_path

ZERO = "sha256:" + "0" * 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--storage", required=True, help="campaign storage root (contains campaigns/ and adapters/)")
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--output", default=None, help="default: <campaign-dir>/QUALIFICATION_RECORD.json")
    args = ap.parse_args()

    root = ensure_path(Path(args.root).resolve() if args.root else detect_root())

    from egai.common.crypto import Ed25519Verifier, SignedEnvelope
    from minagi.v161.campaign_plan import ColabCampaignPlanV161
    from minagi.v161.executed_run import ExecutedRunReceiptV161
    from minagi.v161.runtime_closure3 import sha256_path

    storage = Path(args.storage)
    campaign_dir = storage / "campaigns" / args.campaign_id
    output = Path(args.output) if args.output else campaign_dir / "QUALIFICATION_RECORD.json"

    failures: list[dict] = []

    def fail(where, detail):
        failures.append({"where": where, "detail": detail})

    # --- Plan ---
    plan_doc = json.loads((campaign_dir / "CAMPAIGN_PLAN.json").read_text())
    plan = ColabCampaignPlanV161(**plan_doc["value"])
    if plan.digest != plan_doc["digest"]:
        fail("plan", "CAMPAIGN_PLAN.json digest field does not match recomputed plan digest")

    proof = json.loads((campaign_dir / "DATASET_PROOF.json").read_text())
    hidden_digest = proof["hidden_digest"]

    verifier = Ed25519Verifier()
    key_id = (campaign_dir / "EXECUTION_KEY_ID.txt").read_text().strip()
    verifier.register(key_id, (campaign_dir / "EXECUTION_PUBLIC_KEY.bin").read_bytes())

    # --- Cell matrix ---
    matrix: dict[str, dict[str, str]] = {}
    receipts: dict[tuple[int, str], ExecutedRunReceiptV161] = {}
    for seed in plan.seeds:
        run_dir = campaign_dir / f"seed-{seed}"
        matrix[f"seed-{seed}"] = {}
        for arm in ("A0", "A1"):
            cell_path = run_dir / f"{arm}.json"
            state = "missing"
            if (run_dir / "COMPLETE").is_file() and (run_dir / "SEED_RESULT.json").is_file():
                if cell_path.is_file():
                    state = "present"
                else:
                    state = "incomplete"
            matrix[f"seed-{seed}"][arm] = state
            if state != "present":
                fail(f"seed-{seed}/{arm}", f"cell {state}")
                continue
            doc = json.loads(cell_path.read_text())
            try:
                receipt = ExecutedRunReceiptV161(**doc["receipt"])
            except Exception as exc:  # noqa: BLE001
                fail(f"seed-{seed}/{arm}", f"receipt parse: {type(exc).__name__}: {exc}")
                continue
            if receipt.arm != arm:
                fail(f"seed-{seed}/{arm}", f"arm field {receipt.arm} != {arm}")
            if int(receipt.seed) != seed:
                fail(f"seed-{seed}/{arm}", f"seed field {receipt.seed} != {seed}")
            if receipt.campaign_digest != plan.digest:
                fail(f"seed-{seed}/{arm}", "receipt campaign_digest does not match plan")
            if receipt.evaluator_digest != plan.scorer_artifact_digest:
                fail(f"seed-{seed}/{arm}", "receipt evaluator_digest != preregistered scorer")
            if receipt.dataset_digest != hidden_digest:
                fail(f"seed-{seed}/{arm}", "receipt dataset_digest != sealed hidden partition")
            if receipt.signer_key_id != key_id:
                fail(f"seed-{seed}/{arm}", "receipt signed by unexpected key_id")
            if not receipt.verify(verifier):
                fail(f"seed-{seed}/{arm}", "signature verification failed")
            receipts[(seed, arm)] = receipt

    # --- Adapter physical closure ---
    for (seed, arm), receipt in receipts.items():
        if arm == "A0":
            if receipt.adapter_digest != ZERO:
                fail(f"seed-{seed}/A0", "frozen arm carries a non-zero adapter digest")
            continue
        adapter_dir = storage / "adapters" / args.campaign_id / f"seed-{seed}"
        if not adapter_dir.is_dir():
            fail(f"seed-{seed}/A1", f"adapter dir missing: {adapter_dir}")
            continue
        actual = sha256_path(adapter_dir)
        if actual != receipt.adapter_digest:
            fail(f"seed-{seed}/A1",
                 f"adapter digest mismatch disk={actual} receipt={receipt.adapter_digest}")

    # --- Environment consistency ---
    envs = {r.environment_digest for r in receipts.values()}
    if len(envs) != 1:
        fail("environment", f"receipts span {len(envs)} environment digests: {sorted(envs)}")

    # --- Reconstruct outcomes ---
    per_seed = []
    for seed in plan.seeds:
        a0 = receipts.get((seed, "A0"))
        a1 = receipts.get((seed, "A1"))
        if a0 is None or a1 is None:
            continue
        acq0 = float(a0.metrics.get("hidden_exact_match", 0.0))
        acq1 = float(a1.metrics.get("hidden_exact_match", 0.0))
        per_seed.append({
            "seed": seed,
            "acquisition_a0": acq0,
            "acquisition_a1": acq1,
            "forward_transfer_delta": acq1 - acq0,
            "retention": float(a1.metrics.get("retention", 0.0)),
            "security_regressions": int(a1.metrics.get("security_regressions", 0)),
        })

    complete = not failures and len(per_seed) == len(plan.seeds)
    decision = "REFUSE"
    reasons = []
    if not complete:
        reasons.append("incomplete or invalid evidence matrix")
    else:
        deltas = [x["forward_transfer_delta"] for x in per_seed]
        rets = [x["retention"] for x in per_seed]
        sec = sum(x["security_regressions"] for x in per_seed)
        if sum(deltas) / len(deltas) < plan.minimum_forward_transfer:
            reasons.append("mean forward-transfer delta below preregistered minimum")
        if sum(rets) / len(rets) < plan.minimum_retention:
            reasons.append("mean retention below preregistered minimum")
        if plan.require_zero_security_regressions and sec > 0:
            reasons.append(f"{sec} security regressions")
        decision = "QUALIFIED" if not reasons else "REFUSE"

    stats = {}
    if per_seed:
        ds = [x["forward_transfer_delta"] for x in per_seed]
        stats = {
            "n": len(ds),
            "mean": statistics.fmean(ds),
            "median": statistics.median(ds),
            "stdev": statistics.stdev(ds) if len(ds) > 1 else 0.0,
            "worst": min(ds),
            "best": max(ds),
        }

    recorded_result = None
    result_path = campaign_dir / "RESULT.json"
    if result_path.is_file():
        recorded_result = json.loads(result_path.read_text())
    agreement = None
    if recorded_result is not None:
        agreement = (recorded_result.get("decision") == "PASS") == (decision == "QUALIFIED")

    record = {
        "schema": "mini-agi-v16.2-qualification-record-v1",
        "campaign_id": args.campaign_id,
        "campaign_plan_digest": plan.digest,
        "campaign_dir": str(campaign_dir),
        "matrix": matrix,
        "per_seed": per_seed,
        "forward_transfer_delta_stats": stats,
        "decision": decision,
        "reasons": reasons,
        "failures": failures,
        "runner_decision_agreement": agreement,
        "note": "QUALIFIED is experimental qualification only; not production promotion authority.",
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({k: record[k] for k in ("schema", "campaign_id", "decision", "runner_decision_agreement")}, indent=2))
    for f in failures:
        print(f"[FAIL] {f['where']}: {f['detail']}", file=sys.stderr)
    return 0 if decision == "QUALIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
