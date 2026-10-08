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
    from minagi.v161.campaign_plan import (ColabCampaignPlanV162,
                                           ColabCampaignPlanV163,
                                           ColabCampaignPlanV164)
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
    plan_cls = {"mini-agi-v16.3-colab-campaign-plan-v1": ColabCampaignPlanV163,
                "mini-agi-v16.4-colab-campaign-plan-v1": ColabCampaignPlanV164,
                }.get(plan_doc["value"].get("schema"), ColabCampaignPlanV162)
    plan = plan_cls(**plan_doc["value"])
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

    # v164: the signed experiment protocol must be present and digest-bound
    if isinstance(plan, ColabCampaignPlanV164):
        from minagi.v161.experiment_protocol import ExperimentProtocolV1
        proto_path = campaign_dir / "EXPERIMENT_PROTOCOL.json"
        if not proto_path.is_file():
            fail("protocol", "EXPERIMENT_PROTOCOL.json missing for v164 plan")
        else:
            try:
                pdoc = json.loads(proto_path.read_text())
                proto = ExperimentProtocolV1(**pdoc["value"])
                if pdoc.get("digest") != proto.digest:
                    fail("protocol", "EXPERIMENT_PROTOCOL.json digest field "
                                     "does not match recomputed digest")
                if proto.digest != plan.experiment_protocol_digest:
                    fail("protocol", "experiment protocol digest != "
                                     "preregistered plan binding")
            except Exception as exc:  # noqa: BLE001
                fail("protocol", f"protocol parse: {type(exc).__name__}: {exc}")

    proof = json.loads((campaign_dir / "DATASET_PROOF.json").read_text())
    hidden_digest = proof["hidden_digest"]

    # --- Rebuild the partition set from the committed corpus ---
    # The proof document is runner output; do not trust it. The task
    # corpus is part of the source tree, so the qualifier can
    # independently recompute partition digests and compare them to
    # both the proof and the signed plan.
    dataset_cfg = root / "configs" / "campaign1_tasks.jsonl"
    import yaml
    campaign_cfg = yaml.safe_load((root / "configs" / "campaign1.yaml").read_text())
    fam_disjoint = bool(campaign_cfg.get("require_family_disjoint_hidden", True))
    from minagi.v161.dataset_manifest import (DatasetMember,
                                              DatasetMembershipManifest,
                                              DatasetPartitionSet)
    from egai.common.canonical import sha256_bytes
    rows = [json.loads(l) for l in dataset_cfg.read_text().splitlines()
            if l.strip()]

    def _member(row):
        payload = json.dumps(row, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode()
        return DatasetMember(str(row["id"]), str(row["family"]),
                             sha256_bytes(payload),
                             str(row.get("source", "local")),
                             str(row.get("generator", "manual")))

    by_split = {s: [r for r in rows if r["split"] == s]
                for s in ("train", "validation", "hidden", "retention", "security")}
    parts = DatasetPartitionSet(
        DatasetMembershipManifest("train", tuple(_member(r) for r in by_split["train"])),
        DatasetMembershipManifest("validation", tuple(_member(r) for r in by_split["validation"])),
        DatasetMembershipManifest("hidden", tuple(_member(r) for r in by_split["hidden"])),
        DatasetMembershipManifest("retention", tuple(_member(r) for r in by_split["retention"])),
        DatasetMembershipManifest("security", tuple(_member(r) for r in by_split["security"])),
        require_family_disjoint_hidden=fam_disjoint)
    if parts.digest != plan.dataset_partition_digest:
        fail("dataset", "recomputed partition digest != plan digest")
    if parts.hidden.digest != hidden_digest:
        fail("dataset", "recomputed hidden digest != DATASET_PROOF.json")

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
    models = {r.model_digest for r in receipts.values()}
    if len(models) != 1:
        fail("model", f"receipts span {len(models)} model digests")
    toks = {r.tokenizer_digest for r in receipts.values()}
    if len(toks) != 1:
        fail("tokenizer", f"receipts span {len(toks)} tokenizer digests")
    if isinstance(plan, ColabCampaignPlanV163):
        # symmetric binding: every arm's receipts must carry the exact
        # preregistered identities
        if models and models != {plan.model_digest}:
            fail("model", "receipt model_digest != plan model_digest")
        if toks and toks != {plan.tokenizer_digest}:
            fail("tokenizer", "receipt tokenizer_digest != plan tokenizer_digest")

    # --- Reconstruct outcomes ---
    per_seed = []
    for seed in plan.seeds:
        cell = {a: receipts.get((seed, a)) for a in plan.arms}
        if any(v is None for v in cell.values()):
            continue
        h = {a: float(cell[a].metrics.get("hidden_exact_match", 0.0)) for a in plan.arms}
        n_sec = len(parts.security.members) if parts.security is not None else 1
        sec_rate = {a: 1.0 - float(cell[a].metrics.get("security_regressions", 0)) / max(1, n_sec)
                    for a in plan.arms}
        per_seed.append({
            "seed": seed,
            "hidden_exact_match": h,
            "ft_vs_L1": {a: h[a] - h["L1"] for a in plan.arms if a != "L1"},
            "delta_ft_neural": h["L6"] - h["L5"],
            "l6_retention": float(cell["L6"].metrics.get("retention", 0.0)),
            "l1_retention": float(cell["L1"].metrics.get("retention", 0.0)),
            "security_pass_rate": sec_rate,
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
        if statistics.fmean(deltas) < plan.minimum_neural_incremental_ft:
            reasons.append("mean neural incremental FT (L6-L5) below preregistered minimum")
        if isinstance(plan, ColabCampaignPlanV163):
            n_pos = sum(1 for x in per_seed if x["delta_ft_neural"] > 0)
            if n_pos < plan.min_seeds_positive_ft:
                reasons.append(f"only {n_pos}/{len(per_seed)} seeds positive "
                               f"< {plan.min_seeds_positive_ft}")
            l1_ret = statistics.fmean([x["l1_retention"] for x in per_seed])
            l6_ret = statistics.fmean([x["l6_retention"] for x in per_seed])
            if l1_ret - l6_ret > plan.retention_max_drop:
                reasons.append(f"L6 retention drop {l1_ret - l6_ret:.4f} > "
                               f"{plan.retention_max_drop}")
            l1_sec = statistics.fmean([x["security_pass_rate"]["L1"] for x in per_seed])
            l6_sec = statistics.fmean([x["security_pass_rate"]["L6"] for x in per_seed])
            if l6_sec < plan.security_min_pass_rate:
                reasons.append(f"L6 security pass rate {l6_sec:.3f} < "
                               f"{plan.security_min_pass_rate}")
            if l6_sec - l1_sec < -plan.security_max_drop_vs_L1:
                reasons.append(f"L6 security drop {l6_sec - l1_sec:.4f} below "
                               f"-{plan.security_max_drop_vs_L1} vs L1")
        else:
            ret_mean = statistics.fmean([x["l6_retention"] for x in per_seed])
            sec_total = sum(x["security_regressions"] for x in per_seed)
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
