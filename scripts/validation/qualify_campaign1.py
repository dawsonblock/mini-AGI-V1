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
    ap.add_argument("--holdout", default=None,
                    help="(v166) path to the evaluator-sealed final "
                         "holdout JSONL; its manifest digest must equal "
                         "plan.final_holdout_digest")
    args = ap.parse_args()

    root = ensure_path(Path(args.root).resolve() if args.root else detect_root())

    from egai.common.crypto import Ed25519Verifier, SignedEnvelope
    from minagi.v161.campaign_plan import (ColabCampaignPlanV162,
                                           ColabCampaignPlanV163,
                                           ColabCampaignPlanV164,
                                           ColabCampaignPlanV165,
                                           ColabCampaignPlanV166)
    from minagi.v161.executed_run import ExecutedRunReceiptV162
    from minagi.v161.runtime_closure3 import sha256_path
    from minagi.v161.stats import bootstrap_ci

    storage = Path(args.storage)
    campaign_dir = storage / "campaigns" / args.campaign_id
    output = Path(args.output) if args.output else campaign_dir / "QUALIFICATION_RECORD.json"

    failures: list[dict] = []

    def fail(where, detail):
        failures.append({"where": where, "detail": detail})

    # --- Plan: digest AND signature ---
    plan_doc = json.loads((campaign_dir / "CAMPAIGN_PLAN.json").read_text())
    schema = plan_doc["value"].get("schema")
    plan_cls = {"mini-agi-v16.6-colab-campaign-plan-v1": ColabCampaignPlanV166,
                "mini-agi-v16.5-colab-campaign-plan-v1": ColabCampaignPlanV165,
                "mini-agi-v16.4-colab-campaign-plan-v1": ColabCampaignPlanV164,
                "mini-agi-v16.3-colab-campaign-plan-v1": ColabCampaignPlanV163,
                }.get(schema, ColabCampaignPlanV162)
    plan = plan_cls(**plan_doc["value"])
    if plan.digest != plan_doc["digest"]:
        fail("plan", "CAMPAIGN_PLAN.json digest field does not match recomputed plan digest")

    verifier = Ed25519Verifier()
    key_id = (campaign_dir / "EXECUTION_KEY_ID.txt").read_text().strip()
    verifier.register(key_id, (campaign_dir / "EXECUTION_PUBLIC_KEY.bin").read_bytes())

    registry = None
    authority_verifier = None
    if isinstance(plan, ColabCampaignPlanV165):
        # v165: signatures mean nothing without authorized signers. The
        # plan must be signed by the registered plan authority; receipts
        # by the execution-witness role; both checked against the
        # out-of-band trust root, never the evidence directory alone.
        from minagi.v161.authority import (AuthorityLedger,
                                           AuthorityRegistry)
        trust_path = storage / "trust_root.json"
        if not trust_path.is_file():
            fail("authority", "trust_root.json missing for v165 campaign")
        else:
            try:
                registry = AuthorityRegistry.load(trust_path)
                authority_verifier = registry.verifier()
            except Exception as exc:  # noqa: BLE001
                fail("authority",
                     f"trust root load: {type(exc).__name__}: {exc}")
        if registry is not None:
            if not registry.is_authorized(
                    "plan", plan_doc.get("signer_key_id", "")):
                fail("plan", "plan not signed by registered plan authority")
            elif not authority_verifier.verify(
                    plan_doc["value"],
                    SignedEnvelope(plan_doc["signer_key_id"],
                                   plan_doc["signature_b64"])):
                fail("plan", "plan signature verification failed")
    else:
        if plan_doc.get("signer_key_id") != key_id:
            fail("plan", "plan signed by unexpected key")
        if not verifier.verify(plan_doc["value"],
                               SignedEnvelope(plan_doc["signer_key_id"],
                                              plan_doc["signature_b64"])):
            fail("plan", "plan signature verification failed")

    # v165: the signed experiment protocol must be present and digest-bound
    proto = None
    if isinstance(plan, ColabCampaignPlanV165):
        from minagi.v161.experiment_protocol import ExperimentProtocolV1
        proto_path = campaign_dir / "EXPERIMENT_PROTOCOL.json"
        if not proto_path.is_file():
            fail("protocol", "EXPERIMENT_PROTOCOL.json missing for v165 plan")
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
    # v164 plans bind the corpus path and disjointness flag into the
    # signed plan; older plans fall back to the campaign1 conventions.
    if isinstance(plan, ColabCampaignPlanV164):
        dataset_cfg = root / plan.dataset_path
        fam_disjoint = bool(plan.require_family_disjoint_hidden)
    else:
        dataset_cfg = root / "configs" / "campaign1_tasks.jsonl"
        import yaml
        campaign_cfg = yaml.safe_load(
            (root / "configs" / "campaign1.yaml").read_text())
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

    # v166: the sealed final holdout is bound by manifest digest; the
    # evaluator supplies the file and we re-derive — never trust the
    # plan doc's claim alone. Holdout content is confirmation-set only:
    # it is never joined to scoring cells here.
    if isinstance(plan, ColabCampaignPlanV166) and plan.final_holdout_digest:
        if not args.holdout:
            fail("holdout", "v166 plan binds final_holdout_digest but "
                            "--holdout was not supplied for verification")
        else:
            try:
                hrows = [json.loads(l) for l in
                         Path(args.holdout).read_text().splitlines()
                         if l.strip()]
                hman = DatasetMembershipManifest(
                    "final_holdout",
                    tuple(_member(r) for r in hrows))
                if hman.digest != plan.final_holdout_digest:
                    fail("holdout", "sealed holdout manifest digest != "
                                    "plan.final_holdout_digest")
            except Exception as exc:  # noqa: BLE001
                fail("holdout",
                     f"holdout rebuild failed: {type(exc).__name__}: {exc}")

    class _V3Cell:
        """Uniform view: an EvidenceReceiptV3 whose reported metrics were
        replaced by the INDEPENDENTLY recomputed values — downstream
        reconstruction consumes verified scores, never worker claims."""
        __slots__ = ("_receipt", "metrics")

        def __init__(self, receipt, metrics):
            self._receipt = receipt
            self.metrics = metrics

        def __getattr__(self, name):
            return getattr(self._receipt, name)

    # --- Arm x seed matrix ---
    matrix: dict[str, dict[str, str]] = {}
    receipts: dict[tuple[int, str], ExecutedRunReceiptV162] = {}
    recomputed_all: dict[tuple[int, str], dict] = {}
    hidden_outputs: dict = {}
    per_id_hidden: dict = {}

    if isinstance(plan, ColabCampaignPlanV165):
        # Independent rescore plumbing (v3): prediction records join the
        # committed corpus by task id; expected answers never enter the
        # evidence artifacts.
        from minagi.v161.evaluators import (containment_match, exact_match,
                                            retention_score, score_row,
                                            security_regression)
        from minagi.v161.evidence_receipt_v3 import (
            EvidenceReceiptV3, evaluation_bundle,
            input_manifest_digest, predictions_digest_of,
            read_predictions)
        from minagi.v161.stats import (bootstrap_ci, cluster_bootstrap_ci,
                                       false_activation_rate)
        from egai.common.canonical import digest as _digest, sha256_bytes as _sha
        ret_impl = {"containment_match": containment_match,
                    "retention_score": retention_score}
        ret_fn = ret_impl[getattr(proto, "retention_scorer",
                                  "retention_score")]
        sec_rows_by_id = {str(r["id"]): r for r in by_split["security"]}
        hid_rows_by_id = {str(r["id"]): r for r in by_split["hidden"]}
        ret_probe = [r for r in by_split["retention"]
                     if r.get("probe") != "delayed"]
        ret_delayed = [r for r in by_split["retention"]
                       if r.get("probe") == "delayed"]
        ret_rows_by_id = {str(r["id"]): r for r in ret_probe}
        delayed_rows_by_id = {str(r["id"]): r for r in ret_delayed}
        eval_set_expected = _digest(
            {"hidden": parts.hidden.digest,
             "retention": parts.retention.digest,
             "security": parts.security.digest})
        evaluator_set_expected = _digest(
            [plan.scorer_artifact_digest, plan.retention_artifact_digest,
             plan.security_artifact_digest])

        def _rescore_cell(seed, arm, cell_name, r3, doc):
            """Verify evidence chain + recompute metrics from canonical
            predictions. Returns recomputed metrics dict or None."""
            where = f"seed-{seed}/{cell_name}"
            preds_path = campaign_dir / f"seed-{seed}" / \
                f"PREDICTIONS-{cell_name}.jsonl"
            if not preds_path.is_file():
                fail(where, "canonical predictions file missing")
                return None
            preds = read_predictions(preds_path)
            if predictions_digest_of(preds) != r3.predictions_digest:
                fail(where, "predictions digest mismatch vs receipt")
                return None
            if input_manifest_digest(preds) != r3.input_manifest_digest:
                fail(where, "input manifest digest mismatch vs receipt")
                return None
            if _digest(doc.get("bundle") or {}) != \
                    r3.evaluation_bundle_digest:
                fail(where, "evaluation bundle digest mismatch vs receipt")
                return None
            block_index = {"hidden": (hid_rows_by_id, exact_match,
                                      "hidden_exact_match", "mean"),
                           "retention": (ret_rows_by_id, ret_fn,
                                         "retention", "mean"),
                           "retention_delayed": (delayed_rows_by_id, ret_fn,
                                                 "retention_delayed", "mean"),
                           "security": (sec_rows_by_id, security_regression,
                                        "security_regressions", "count")}
            metrics = {}
            for rec in preds:
                blk = rec.get("block")
                if blk not in block_index:
                    fail(where, f"unknown prediction block {blk!r}")
                    return None
                if _sha(str(rec.get("input", "")).encode()) != \
                        rec.get("input_sha256"):
                    fail(where, f"input digest mismatch in record {rec.get('id')}")
                    return None
            by_block = {}
            for rec in preds:
                by_block.setdefault(rec["block"], []).append(rec)
            for blk, recs in by_block.items():
                index, fn, metric, how = block_index[blk]
                vals = []
                for rec in recs:
                    row = index.get(str(rec["id"]))
                    if row is None:
                        fail(where, f"prediction id {rec['id']} not in "
                                    f"{blk} partition — off-corpus input")
                        return None
                    score = score_row(row, str(rec["output"]), fn)
                    vals.append(float(score))
                    if blk == "hidden":
                        hidden_outputs.setdefault((seed, arm), {})[
                            str(rec["id"])] = str(rec["output"])
                        per_id_hidden.setdefault((seed, arm), {})[
                            str(rec["id"])] = float(score)
                metrics[metric] = (sum(vals) if how == "count"
                                   else (sum(vals) / len(vals) if vals else 0.0))
            # worker-reported metrics become checkable claims
            reported = dict((doc.get("bundle") or {})
                            .get("metrics_reported") or {})
            for k, v in metrics.items():
                if abs(float(reported.get(k, float("nan"))) - float(v)) > 1e-9:
                    fail(where, f"reported metric {k}={reported.get(k)} "
                                f"!= recomputed {v} — fabricated score")
                    return None
            for k in ("eval_seconds", "generated_tokens",
                      "arm_state_bytes", "wall_seconds",
                      "train_seconds", "trainable_params"):
                if k in reported:
                    metrics[k] = reported[k]
            return metrics

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
                # v165 distinguishes absent evidence (INCOMPLETE — e.g.
                # another lane's unfinished seed) from invalid evidence
                # (a cell file present without a verified atomic commit
                # — grafted/torn material). Only the latter is a failure.
                if state == "incomplete" \
                        or not isinstance(plan, ColabCampaignPlanV165):
                    fail(f"seed-{seed}/{arm}", f"cell {state}")
                continue
            doc = json.loads(cell_path.read_text())
            if isinstance(plan, ColabCampaignPlanV165):
                try:
                    r3 = EvidenceReceiptV3(**doc["receipt"])
                except Exception as exc:  # noqa: BLE001
                    fail(f"seed-{seed}/{arm}",
                         f"receipt parse: {type(exc).__name__}: {exc}")
                    continue
                if r3.arm != arm or int(r3.seed) != seed:
                    fail(f"seed-{seed}/{arm}", "arm/seed field mismatch")
                if r3.campaign_digest != plan.digest:
                    fail(f"seed-{seed}/{arm}",
                         "receipt campaign_digest does not match plan")
                if r3.protocol_digest != plan.experiment_protocol_digest:
                    fail(f"seed-{seed}/{arm}",
                         "receipt protocol_digest != preregistered protocol")
                if r3.evaluation_dataset_digest != eval_set_expected:
                    fail(f"seed-{seed}/{arm}",
                         "receipt evaluation_dataset_digest != partitions")
                if r3.evaluator_digest != evaluator_set_expected:
                    fail(f"seed-{seed}/{arm}",
                         "receipt evaluator_digest != preregistered evaluator set")
                if registry is not None and not registry.is_authorized(
                        "execution_witness", r3.signer_key_id):
                    fail(f"seed-{seed}/{arm}",
                         "receipt signer not authorized as execution_witness")
                elif authority_verifier is not None and \
                        not r3.verify(authority_verifier):
                    fail(f"seed-{seed}/{arm}", "signature verification failed")
                metrics = _rescore_cell(seed, arm, arm, r3, doc)
                if metrics is None:
                    continue
                recomputed_all[(seed, arm)] = metrics
                receipts[(seed, arm)] = _V3Cell(r3, metrics)
                continue
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

    # --- v164: delayed-retention persistence cells ---
    delayed: dict[tuple[int, str], float] = {}
    if isinstance(plan, ColabCampaignPlanV164):
        for seed in plan.seeds:
            run_dir = campaign_dir / f"seed-{seed}"
            for cell_name, arm_id in (("L1_delayed", "L1"), ("L6_delayed", "L6")):
                cell_path = run_dir / f"{cell_name}.json"
                if not cell_path.is_file():
                    fail(f"seed-{seed}/{cell_name}", "delayed cell missing")
                    continue
                doc = json.loads(cell_path.read_text())
                if isinstance(plan, ColabCampaignPlanV165):
                    try:
                        r3 = EvidenceReceiptV3(**doc["receipt"])
                    except Exception as exc:  # noqa: BLE001
                        fail(f"seed-{seed}/{cell_name}",
                             f"receipt parse: {type(exc).__name__}: {exc}")
                        continue
                    if r3.arm != arm_id or int(r3.seed) != seed:
                        fail(f"seed-{seed}/{cell_name}",
                             "arm/seed field mismatch")
                    if r3.campaign_digest != plan.digest:
                        fail(f"seed-{seed}/{cell_name}",
                             "receipt campaign_digest != plan")
                    if registry is not None and not registry.is_authorized(
                            "execution_witness", r3.signer_key_id):
                        fail(f"seed-{seed}/{cell_name}",
                             "delayed receipt signer not authorized")
                    elif authority_verifier is not None and \
                            not r3.verify(authority_verifier):
                        fail(f"seed-{seed}/{cell_name}",
                             "signature verification failed")
                    if (doc.get("bundle") or {}).get("cell") != cell_name:
                        fail(f"seed-{seed}/{cell_name}",
                             "bundle cell marker mismatch")
                    metrics = _rescore_cell(seed, arm_id, cell_name, r3, doc)
                    if metrics is None:
                        continue
                    if "retention_delayed" not in metrics:
                        fail(f"seed-{seed}/{cell_name}",
                             "retention_delayed metric missing")
                        continue
                    delayed[(seed, arm_id)] = float(
                        metrics["retention_delayed"])
                else:
                    try:
                        rec = ExecutedRunReceiptV162(**doc["receipt"])
                    except Exception as exc:  # noqa: BLE001
                        fail(f"seed-{seed}/{cell_name}",
                             f"receipt parse: {type(exc).__name__}: {exc}")
                        continue
                    if rec.arm != arm_id or int(rec.seed) != seed:
                        fail(f"seed-{seed}/{cell_name}", "arm/seed field mismatch")
                    if rec.campaign_digest != plan.digest:
                        fail(f"seed-{seed}/{cell_name}", "receipt campaign_digest != plan")
                    if rec.evaluator_digest != plan.scorer_artifact_digest:
                        fail(f"seed-{seed}/{cell_name}", "receipt evaluator_digest != scorer")
                    if rec.dataset_digest != hidden_digest:
                        fail(f"seed-{seed}/{cell_name}", "receipt dataset_digest != hidden")
                    if rec.signer_key_id != key_id or not rec.verify(verifier):
                        fail(f"seed-{seed}/{cell_name}", "signature verification failed")
                    if doc.get("probe_kind") != "delayed":
                        fail(f"seed-{seed}/{cell_name}", "probe_kind marker missing")
                    if "retention_delayed" not in rec.metrics:
                        fail(f"seed-{seed}/{cell_name}", "retention_delayed metric missing")
                    delayed[(seed, arm_id)] = float(rec.metrics.get("retention_delayed", 0.0))
                # L6 delayed cell re-verifies the persisted adapter bytes
                if arm_id == "L6":
                    _rec = (r3 if isinstance(plan, ColabCampaignPlanV165)
                            else rec)
                    adir = storage / "adapters" / args.campaign_id / "L6" / f"seed-{seed}"
                    if not adir.is_dir():
                        fail(f"seed-{seed}/{cell_name}", "L6 adapter dir missing for delayed probe")
                    elif sha256_path(adir) != _rec.adapter_digest:
                        fail(f"seed-{seed}/{cell_name}", "L6 delayed adapter digest mismatch")

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

    if isinstance(plan, ColabCampaignPlanV164):
        # Per-seed atomicity (resume-tolerant): every cell of a seed —
        # arms + delayed probes — must share ONE environment digest, so
        # within-seed comparisons are single-environment. Different
        # seeds may come from different runtimes (the runner's resume
        # semantics redo a seed entirely in one environment); this is
        # the multi-site-trial analogue, and env provenance is still
        # bound into every receipt.
        delayed_receipts = {}
        for seed in plan.seeds:
            cell_path = campaign_dir / f"seed-{seed}"
            for nm in ("L1_delayed", "L6_delayed"):
                p = cell_path / f"{nm}.json"
                if p.is_file():
                    try:
                        rdoc = json.loads(p.read_text())["receipt"]
                        delayed_receipts[(seed, nm)] = (
                            EvidenceReceiptV3(**rdoc)
                            if isinstance(plan, ColabCampaignPlanV165)
                            else ExecutedRunReceiptV162(**rdoc))
                    except Exception:
                        pass
        for seed in plan.seeds:
            seed_envs = {receipts[(seed, a)].environment_digest
                         for a in plan.arms if (seed, a) in receipts}
            seed_envs |= {r.environment_digest for (s, _), r in
                          delayed_receipts.items() if s == seed}
            if len(seed_envs) != 1:
                fail("environment", f"seed-{seed} spans "
                                    f"{len(seed_envs)} environment digests")
    else:
        envs = {r.environment_digest for r in receipts.values()}
        if len(envs) != 1:
            fail("environment",
                 f"receipts span {len(envs)} environment digests")
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
        if isinstance(plan, ColabCampaignPlanV164):
            n_pos = sum(1 for x in per_seed if x["delta_ft_neural"] > 0)
            if n_pos < plan.min_seeds_positive_ft:
                reasons.append(f"only {n_pos}/{len(per_seed)} seeds positive "
                               f"< {plan.min_seeds_positive_ft}")
            ci = bootstrap_ci(deltas, plan.bootstrap_resamples, plan.ci_alpha)
            stats["bootstrap_ci"] = ci
            if ci["lower"] <= plan.min_delta_ft_ci_lower:
                reasons.append(f"bootstrap CI lower bound {ci['lower']:.4f} <= "
                               f"{plan.min_delta_ft_ci_lower}")
            if isinstance(plan, ColabCampaignPlanV166):
                # Phase-6 (REPAIR-033): the uncertainty unit is the
                # cluster, not the row — paired per-row L6-L5 deltas
                # pooled across seeds are resampled by task FAMILY.
                hidx = {str(r["id"]): r for r in by_split["hidden"]}
                pairs, fams = [], []
                for seed in plan.seeds:
                    l5m = per_id_hidden.get((seed, "L5"), {})
                    l6m = per_id_hidden.get((seed, "L6"), {})
                    for rid in sorted(set(l5m) & set(l6m)):
                        pairs.append(l6m[rid] - l5m[rid])
                        fams.append(str(hidx.get(rid, {})
                                        .get("family", "unknown")))
                if pairs and plan.cluster_unit == "family":
                    cci = cluster_bootstrap_ci(
                        pairs, fams, plan.bootstrap_resamples,
                        plan.ci_alpha)
                    stats["cluster_bootstrap_ci_row_delta"] = cci
                    if cci["lower"] <= plan.min_delta_ft_ci_lower:
                        reasons.append(
                            f"family-clustered CI lower bound "
                            f"{cci['lower']:.4f} <= "
                            f"{plan.min_delta_ft_ci_lower}")
                # false activation: L6 changed a correct L1 answer to a
                # wrong one — learned behavior firing off-task. The rate
                # is conditioned on previously-correct L1 cases; the
                # broader conditional regression is reported alongside.
                fa_rates = []
                cr_rates = []
                for seed in plan.seeds:
                    base = hidden_outputs.get((seed, "L1"), {})
                    armed = hidden_outputs.get((seed, "L6"), {})
                    golds = {rid: str(hidx[rid]["expected"])
                             for rid in set(base) & set(armed)
                             if rid in hidx}
                    fa = false_activation_rate(base, armed, golds,
                                               exact_match)
                    fa_rates.append(fa["false_activation_rate"])
                    cr_rates.append(fa["conditional_regression_rate"])
                if fa_rates:
                    stats["false_activation_rate_mean"] = \
                        statistics.fmean(fa_rates)
                    stats["conditional_regression_rate_mean"] = \
                        statistics.fmean(cr_rates)
                    if statistics.fmean(fa_rates) > \
                            plan.max_false_activation_rate:
                        reasons.append(
                            f"false-activation rate "
                            f"{statistics.fmean(fa_rates):.3f} > "
                            f"{plan.max_false_activation_rate}")
            l1_ret = statistics.fmean([x["l1_retention"] for x in per_seed])
            l6_ret = statistics.fmean([x["l6_retention"] for x in per_seed])
            if l1_ret - l6_ret > plan.retention_max_drop:
                reasons.append(f"L6 retention drop {l1_ret - l6_ret:.4f} > "
                               f"{plan.retention_max_drop}")
            d_l1 = [delayed.get((s, "L1")) for s in plan.seeds]
            d_l6 = [delayed.get((s, "L6")) for s in plan.seeds]
            if any(v is None for v in d_l1 + d_l6):
                reasons.append("delayed-retention probes missing")
            else:
                d_drop = statistics.fmean(d_l1) - statistics.fmean(d_l6)
                stats["retention_delayed"] = {"L1": statistics.fmean(d_l1),
                                              "L6": statistics.fmean(d_l6),
                                              "drop": d_drop}
                if d_drop > plan.delayed_retention_max_drop:
                    reasons.append(f"L6 delayed-retention drop {d_drop:.4f} > "
                                   f"{plan.delayed_retention_max_drop}")
            l1_sec = statistics.fmean([x["security_pass_rate"]["L1"] for x in per_seed])
            l6_sec = statistics.fmean([x["security_pass_rate"]["L6"] for x in per_seed])
            if l6_sec < plan.security_min_pass_rate:
                reasons.append(f"L6 security pass rate {l6_sec:.3f} < "
                               f"{plan.security_min_pass_rate}")
            if l6_sec - l1_sec < -plan.security_max_drop_vs_L1:
                reasons.append(f"L6 security drop {l6_sec - l1_sec:.4f} below "
                               f"-{plan.security_max_drop_vs_L1} vs L1")
        elif isinstance(plan, ColabCampaignPlanV163):
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
    # Explicit failure states (REPAIR-018): v165 distinguishes
    # INCOMPLETE (absent cells) / INVALID_EVIDENCE (integrity failures)
    # / REFUSED (gates unmet) / QUALIFIED. Older plans keep their
    # historical vocabulary for evidence compatibility.
    if isinstance(plan, ColabCampaignPlanV165):
        if failures:
            decision = "INVALID_EVIDENCE"
        elif len(per_seed) < len(plan.seeds):
            decision = "INCOMPLETE"
        elif reasons:
            decision = "REFUSED"
        else:
            decision = "QUALIFIED"
    else:
        decision = "QUALIFIED" if complete and not reasons else "REFUSE"

    recorded_result = None
    result_path = campaign_dir / "RESULT.json"
    if result_path.is_file():
        recorded_result = json.loads(result_path.read_text())
    agreement = None
    if recorded_result is not None:
        if isinstance(plan, ColabCampaignPlanV165):
            agreement = (recorded_result.get("decision") == decision)
        else:
            agreement = (recorded_result.get("decision") == "PASS") == \
                        (decision == "QUALIFIED")

    record = {
        "schema": "mini-agi-v16.2-campaign1-qualification-v1",
        "campaign_id": args.campaign_id,
        "campaign_plan_digest": plan.digest,
        "environment_digests": sorted({r.environment_digest
                                       for r in receipts.values()}),
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

    if isinstance(plan, ColabCampaignPlanV165) and registry is not None:
        # Independent authority output: the evaluation authority signs
        # the recomputed metrics + evidence root; the qualification
        # authority signs the gate decision; both land in the ledger.
        from egai.common.crypto import Ed25519Signer
        from egai.common.canonical import digest as _d

        def _role_signer(role):
            p = storage / ".keys" / f"{role}.pem"
            if not p.is_file():
                return None
            s = Ed25519Signer.from_private_bytes(p.read_bytes())
            return s if registry.is_authorized(role, s.key_id) else None

        eval_signer = _role_signer("evaluation")
        qual_signer = _role_signer("qualification")
        evidence_root = sha256_path(campaign_dir)
        bundle_body = {
            "schema": "mini-agi-v16.5-evaluation-record-v1",
            "campaign_id": args.campaign_id,
            "campaign_plan_digest": plan.digest,
            "protocol_digest": plan.experiment_protocol_digest,
            "evidence_root_digest": evidence_root,
            "evaluator_set_digest": evaluator_set_expected,
            "recomputed_per_cell": {f"{s}-{a}": m
                                   for (s, a), m in
                                   sorted(recomputed_all.items())},
            "per_seed": per_seed,
        }
        if eval_signer is not None:
            env = eval_signer.sign(bundle_body)
            bundle_doc = {"value": bundle_body,
                          "digest": _d(bundle_body),
                          "signer_key_id": env.key_id,
                          "signature_b64": env.signature_b64}
            (campaign_dir / "EVALUATION_BUNDLE.json").write_text(
                json.dumps(bundle_doc, indent=2, sort_keys=True))
            AuthorityLedger(storage / "AUTHORITY_LEDGER.jsonl") \
                .append(eval_signer, "evaluation_bundle", bundle_body)
            record["evaluation_bundle_digest"] = _d(bundle_body)
            record["evidence_root_digest"] = evidence_root
            record["scoring"] = "independently-recomputed-from-predictions"
        else:
            fail("authority", "evaluation authority key unavailable — "
                              "record left unsigned")
        if qual_signer is not None:
            qenv = qual_signer.sign(record)
            record_doc = {"value": record, "digest": _d(record),
                          "signer_key_id": qenv.key_id,
                          "signature_b64": qenv.signature_b64}
            output.write_text(json.dumps(record_doc, indent=2,
                                         sort_keys=True))
            AuthorityLedger(storage / "AUTHORITY_LEDGER.jsonl") \
                .append(qual_signer, "qualification_record", record)
        else:
            fail("authority", "qualification authority key unavailable — "
                              "record left unsigned")
            output.write_text(json.dumps(record, indent=2, sort_keys=True))
    else:
        output.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({k: record[k] for k in
                      ("schema", "campaign_id", "decision", "runner_decision_agreement")},
                     indent=2))
    for f in failures:
        print(f"[FAIL] {f['where']}: {f['detail']}", file=sys.stderr)
    return 0 if decision == "QUALIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
