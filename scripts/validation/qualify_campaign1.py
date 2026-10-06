#!/usr/bin/env python3
"""v16.2 Phases 16-17 — independent artifact-only Campaign 1 qualification.

Consumes ONLY persisted artifacts: the signed campaign plan, dataset
proof, execution public key, per-(seed, arm) receipt cells, arm state
dirs, and adapter dirs. Never touches live trainer objects.

Checks:
  * plan digest AND Ed25519 signature (plan signed before execution)
  * every receipt signature against the published execution key
  * receipt binding: campaign, evaluator, hidden-dataset digests
  * physical closure: adapter digests (L6/NC) and non-parametric arm
    state digests (L2-L5) recomputed from disk
  * complete arm x seed matrix; L1 adapter_digest must be ZERO
  * environment consistency across receipts
  * negative control: NC forward transfer must stay within the
    preregistered bound
  * reconstructs mean(L6 - L5) and re-derives the decision from the
    preregistered thresholds
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
    ap.add_argument("--storage", required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    root = ensure_path(Path(args.root).resolve() if args.root else detect_root())

    from egai.common.crypto import Ed25519Verifier, SignedEnvelope
    from minagi.v161.campaign_plan import ColabCampaignPlanV162
    from minagi.v161.executed_run import ExecutedRunReceiptV162
    from minagi.v161.runtime_closure3 import sha256_path

    storage = Path(args.storage)
    campaign_dir = storage / "campaigns" / args.campaign_id
    output = Path(args.output) if args.output else campaign_dir / "QUALIFICATION_RECORD.json"

    failures: list[dict] = []

    def fail(where, detail):
        failures.append({"where": where, "detail": detail})

    # --- Plan: digest AND signature ---
    plan_doc = json.loads((campaign_dir / "CAMPAIGN_PLAN.json").read_text())
    plan = ColabCampaignPlanV162(**plan_doc["value"])
    if plan.digest != plan_doc["digest"]:
        fail("plan", "CAMPAIGN_PLAN.json digest field does not match recomputed plan digest")

    verifier = Ed25519Verifier()
    key_id = (campaign_dir / "EXECUTION_KEY_ID.txt").read_text().strip()
    verifier.register(key_id, (campaign_dir / "EXECUTION_PUBLIC_KEY.bin").read_bytes())
    if plan_doc.get("signer_key_id") != key_id:
        fail("plan", "plan signed by unexpected key")
    if not verifier.verify(plan_doc["value"],
                           SignedEnvelope(plan_doc["signer_key_id"],
                                          plan_doc["signature_b64"])):
        fail("plan", "plan signature verification failed")

    proof = json.loads((campaign_dir / "DATASET_PROOF.json").read_text())
    hidden_digest = proof["hidden_digest"]

    # --- Arm x seed matrix ---
    matrix: dict[str, dict[str, str]] = {}
    receipts: dict[tuple[int, str], ExecutedRunReceiptV162] = {}
    for seed in plan.seeds:
        run_dir = campaign_dir / f"seed-{seed}"
        matrix[f"seed-{seed}"] = {}
        seed_ok = (run_dir / "COMPLETE").is_file() and (run_dir / "SEED_RESULT.json").is_file()
        for arm in plan.arms:
            cell_path = run_dir / f"{arm}.json"
            state = "present" if seed_ok and cell_path.is_file() else \
                ("incomplete" if cell_path.is_file() else "missing")
            matrix[f"seed-{seed}"][arm] = state
            if state != "present":
                fail(f"seed-{seed}/{arm}", f"cell {state}")
                continue
            doc = json.loads(cell_path.read_text())
            try:
                receipt = ExecutedRunReceiptV162(**doc["receipt"])
            except Exception as exc:  # noqa: BLE001
                fail(f"seed-{seed}/{arm}", f"receipt parse: {type(exc).__name__}: {exc}")
                continue
            if receipt.arm != arm or int(receipt.seed) != seed:
                fail(f"seed-{seed}/{arm}", "arm/seed field mismatch")
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

    # --- Physical closure: adapters (L6/NC) and arm states (L2-L5) ---
    for (seed, arm), receipt in receipts.items():
        if receipt.adapter_digest == ZERO:
            if arm in ("L6", "NC"):
                fail(f"seed-{seed}/{arm}", "parametric arm missing adapter digest")
        else:
            adir = storage / "adapters" / args.campaign_id / arm / f"seed-{seed}"
            if not adir.is_dir():
                fail(f"seed-{seed}/{arm}", f"adapter dir missing: {adir}")
            elif sha256_path(adir) != receipt.adapter_digest:
                fail(f"seed-{seed}/{arm}", "adapter digest mismatch vs disk")
        if receipt.state_digest != ZERO:
            sdir = storage / "arms" / args.campaign_id / (
                f"seed-{seed}/{arm}" if arm == "L5" else arm)
            if not sdir.is_dir():
                fail(f"seed-{seed}/{arm}", f"arm state dir missing: {sdir}")
            elif sha256_path(sdir) != receipt.state_digest:
                fail(f"seed-{seed}/{arm}", "arm state digest mismatch vs disk")
        if arm == "L1" and (receipt.adapter_digest != ZERO or receipt.state_digest != ZERO):
            fail(f"seed-{seed}/L1", "frozen arm carries non-zero digests")

    envs = {r.environment_digest for r in receipts.values()}
    if len(envs) != 1:
        fail("environment", f"receipts span {len(envs)} environment digests")

    # --- Reconstruct outcomes ---
    per_seed = []
    for seed in plan.seeds:
        cell = {a: receipts.get((seed, a)) for a in plan.arms}
        if any(v is None for v in cell.values()):
            continue
        h = {a: float(cell[a].metrics.get("hidden_exact_match", 0.0)) for a in plan.arms}
        per_seed.append({
            "seed": seed,
            "hidden_exact_match": h,
            "ft_vs_L1": {a: h[a] - h["L1"] for a in plan.arms if a != "L1"},
            "delta_ft_neural": h["L6"] - h["L5"],
            "l6_retention": float(cell["L6"].metrics.get("retention", 0.0)),
            "security_regressions": int(sum(cell[a].metrics.get("security_regressions", 0)
                                          for a in plan.arms)),
        })

    complete = not failures and len(per_seed) == len(plan.seeds)
    reasons: list[str] = []
    stats = {}
    if per_seed:
        deltas = [x["delta_ft_neural"] for x in per_seed]
        stats = {"n": len(deltas), "mean": statistics.fmean(deltas),
                 "median": statistics.median(deltas),
                 "stdev": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                 "worst": min(deltas), "best": max(deltas)}
        nc_fts = [x["ft_vs_L1"].get("NC", 0.0) for x in per_seed]
        nc_mean = statistics.fmean(nc_fts)
        ret_mean = statistics.fmean([x["l6_retention"] for x in per_seed])
        sec_total = sum(x["security_regressions"] for x in per_seed)
        if statistics.fmean(deltas) < plan.minimum_neural_incremental_ft:
            reasons.append("mean neural incremental FT (L6-L5) below preregistered minimum")
        if ret_mean < plan.minimum_retention:
            reasons.append("mean L6 retention below preregistered minimum")
        if plan.require_zero_security_regressions and sec_total > 0:
            reasons.append(f"{sec_total} security regressions")
        if nc_mean > plan.negative_control_max_ft:
            reasons.append(f"negative control FT {nc_mean:.4f} exceeded bound "
                           f"{plan.negative_control_max_ft} — pipeline suspect")
    if not complete:
        reasons.insert(0, "incomplete or invalid evidence matrix")
    decision = "QUALIFIED" if complete and not reasons else "REFUSE"

    recorded_result = None
    result_path = campaign_dir / "RESULT.json"
    if result_path.is_file():
        recorded_result = json.loads(result_path.read_text())
    agreement = None
    if recorded_result is not None:
        agreement = (recorded_result.get("decision") == "PASS") == (decision == "QUALIFIED")

    record = {
        "schema": "mini-agi-v16.2-campaign1-qualification-v1",
        "campaign_id": args.campaign_id,
        "campaign_plan_digest": plan.digest,
        "matrix": matrix,
        "per_seed": per_seed,
        "delta_ft_neural_stats": stats,
        "decision": decision,
        "reasons": reasons,
        "failures": failures,
        "runner_decision_agreement": agreement,
        "reproduction_policy": {
            "semantics": plan.reproduction_semantics,
            "adapter_bitwise_required": plan.adapter_bitwise_required,
            "metric_tolerance": plan.metric_tolerance,
        },
        "note": "QUALIFIED is experimental qualification only; not "
                "production promotion authority.",
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({k: record[k] for k in
                      ("schema", "campaign_id", "decision", "runner_decision_agreement")},
                     indent=2))
    for f in failures:
        print(f"[FAIL] {f['where']}: {f['detail']}", file=sys.stderr)
    return 0 if decision == "QUALIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
