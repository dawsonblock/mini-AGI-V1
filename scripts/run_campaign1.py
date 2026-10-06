#!/usr/bin/env python3
"""v16.2 Campaign 1 — six-arm x multi-seed experimental campaign.

Arm ladder: L1 frozen -> L2 retrieval -> L3 semantic-memory -> L4 skills
-> L5 grounded-replay -> L6 neural-adapter, plus NC negative control
(label-shuffled LoRA).

Flow per run:
  1. load dataset partitions (train/val/hidden/retention/security)
  2. register evaluator artifacts
  3. resolve+pins the immutable model revision
  4. build + persist non-parametric arm states
  5. create AND SIGN the campaign plan BEFORE any model evaluation
  6. per seed: execute every arm, emit a signed ExecutedRunReceiptV162
     per (seed, arm), with adapter/state digests bound to disk
  7. emit RESULT.json with per-arm metrics and the primary comparison
     mean(L6 - L5)

Resume semantics: a seed directory counts as evidence only when it
contains COMPLETE + SEED_RESULT.json; partial seeds are re-executed.
"""
from __future__ import annotations

import argparse, gc, json, random, sys, time
from dataclasses import asdict
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest, sha256_bytes
from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from minagi.platforms.colab.environment import probe_environment
from minagi.platforms.colab.storage import ColabStorage
from minagi.platforms.cuda.hf_runtime import HFLoadSpec, load_tokenizer, load_causal_lm
from minagi.platforms.cuda.peft_trainer import LoraTrainSpec, train_lora
from minagi.v161.arms import (ARMS, ARM_IDS, ZERO_DIGEST, dir_size_bytes,
                              shuffled_label_texts)
from minagi.v161.dataset_manifest import (DatasetMember,
                                          DatasetMembershipManifest,
                                          DatasetPartitionSet)
from minagi.v161.evaluator_registry import EvaluatorArtifact, EvaluatorRegistry
from minagi.v161.evaluators import exact_match, retention_score, security_regression
from minagi.v161.campaign_plan import ColabCampaignPlanV162
from minagi.v161.executed_run import ExecutedRunReceiptV162
from minagi.v161.runtime_closure3 import sha256_path


def load_rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def member(row):
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return DatasetMember(str(row["id"]), str(row["family"]), sha256_bytes(payload),
                         str(row.get("source", "local")), str(row.get("generator", "manual")))


def generate(model, tokenizer, prompt, max_new_tokens):
    import torch
    with torch.inference_mode():
        device = next(model.parameters()).device
        batch = tokenizer(prompt, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.generate(**batch, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
        new_tokens = int(out.shape[1] - batch["input_ids"].shape[1])
        text = tokenizer.decode(out[0, batch["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        return text, new_tokens


def evaluate(model, tokenizer, rows, max_new_tokens, scorer, arm=None):
    scores, outputs, tokens = [], [], 0
    t0 = time.time()
    for row in rows:
        prompt = str(row["prompt"])
        if arm is not None:
            prompt = arm.augment(prompt, str(row.get("family", "")))
        pred, nt = generate(model, tokenizer, prompt, max_new_tokens)
        tokens += nt
        score = float(scorer(pred, str(row["expected"])))
        scores.append(score)
        outputs.append({"id": row["id"], "prediction": pred,
                        "expected": row["expected"], "score": score})
    mean = sum(scores) / len(scores) if scores else 0.0
    return mean, outputs, tokens, time.time() - t0


def model_identity(model, spec):
    commit = str(getattr(getattr(model, "config", None), "_commit_hash", "") or spec.revision)
    cfg = getattr(model, "config", None)
    cfgdoc = cfg.to_dict() if cfg is not None and hasattr(cfg, "to_dict") else {}
    return digest({"model_id": spec.model_id, "resolved_revision": commit, "config": cfgdoc})


def tokenizer_identity(tokenizer, spec):
    return digest({"model_id": spec.model_id,
                   "revision": str(getattr(tokenizer, "_commit_hash", "") or spec.revision),
                   "class": tokenizer.__class__.__name__, "vocab_size": len(tokenizer)})


def resolve_revision(model_id: str, configured: str) -> str:
    """Pin 'auto' to the immutable upstream commit; a concrete sha
    passes through unchanged."""
    if configured and configured != "auto":
        return configured
    from huggingface_hub import HfApi
    return str(HfApi().model_info(model_id).sha)


def security_eval(model, tokenizer, rows, max_new_tokens, security_fn, arm=None):
    outputs, count, tokens = [], 0, 0
    for row in rows:
        prompt = str(row["prompt"])
        if arm is not None:
            prompt = arm.augment(prompt, str(row.get("family", "")))
        pred, nt = generate(model, tokenizer, prompt, max_new_tokens)
        tokens += nt
        flag = int(security_fn(pred, str(row["expected"])))
        count += flag
        outputs.append({"id": row["id"], "prediction": pred, "flagged": flag})
    return count, outputs, tokens


def free_model(model):
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/campaign1.yaml")
    ap.add_argument("--storage", default="/content/minagi_campaign1")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())

    storage = ColabStorage(args.storage)
    env = probe_environment(str(storage.root))
    if cfg.get("require_cuda", True) and not env.cuda_available:
        raise SystemExit("CUDA GPU required by this campaign config")

    rows = load_rows(ROOT / cfg["dataset"])
    by_split = {s: [r for r in rows if r["split"] == s]
                for s in ("train", "validation", "hidden", "retention", "security")}
    train, val, hidden = by_split["train"], by_split["validation"], by_split["hidden"]
    retention_rows, security_rows = by_split["retention"], by_split["security"]
    parts = DatasetPartitionSet(
        DatasetMembershipManifest("train", tuple(member(r) for r in train)),
        DatasetMembershipManifest("validation", tuple(member(r) for r in val)),
        DatasetMembershipManifest("hidden", tuple(member(r) for r in hidden)),
        DatasetMembershipManifest("retention", tuple(member(r) for r in retention_rows)),
        DatasetMembershipManifest("security", tuple(member(r) for r in security_rows)),
        require_family_disjoint_hidden=bool(cfg.get("require_family_disjoint_hidden", False)))

    registry = EvaluatorRegistry(storage.root / "evidence" / "evaluators")
    arts = [EvaluatorArtifact.from_callable("exact-match-v1", exact_match),
            EvaluatorArtifact.from_callable("retention-v1", retention_score),
            EvaluatorArtifact.from_callable("security-v1", security_regression)]
    for a in arts:
        registry.register(a)
    score_fn = registry.resolve(arts[0].digest)
    retention_fn = registry.resolve(arts[1].digest)
    security_fn = registry.resolve(arts[2].digest)

    model_cfg = cfg["model"]
    revision = resolve_revision(model_cfg["id"], str(model_cfg.get("revision", "auto")))
    spec = HFLoadSpec(model_cfg["id"], revision, str(model_cfg.get("dtype", "auto")),
                      str(model_cfg.get("quantization", "none")),
                      bool(model_cfg.get("trust_remote_code", False)))
    seeds = tuple(int(x) for x in cfg.get("seeds", [0]))
    arms_in_plan = tuple(str(a) for a in cfg.get("arms", list(ARM_IDS)))
    max_new = int(cfg.get("max_new_tokens", 32))

    # --- Build + persist non-parametric arm states (before signing) ---
    arms_state_root = storage.root / "arms" / cfg["campaign_id"]
    arm_instances, arm_state_dirs = {}, {}
    for arm_id in ("L2", "L3", "L4"):
        if arm_id not in arms_in_plan:
            continue
        arm = ARMS[arm_id]()
        if arm_id == "L2":
            arm.k = int(cfg.get("retrieval_k", 3))
        if arm_id == "L3":
            arm.k = int(cfg.get("memory_k", 2))
        sdir = arms_state_root / arm_id
        arm.build_state(sdir, train)
        arm_instances[arm_id] = arm
        arm_state_dirs[arm_id] = sdir

    # --- Plan: created and SIGNED before any model evaluation ---
    plan = ColabCampaignPlanV162(
        str(cfg["campaign_id"]), spec.model_id, spec.revision,
        parts.digest, arts[0].digest, arts[1].digest, arts[2].digest,
        seeds, arms_in_plan, "NC",
        float(cfg.get("negative_control_max_ft", 0.02)),
        float(cfg.get("minimum_neural_incremental_ft", 0.02)),
        float(cfg.get("minimum_retention", 0.95)), True,
        "statistical-equivalence", False, float(cfg.get("metric_tolerance", 0.05)))

    campaign_dir = storage.root / "campaigns" / cfg["campaign_id"]
    campaign_dir.mkdir(parents=True, exist_ok=True)
    signer = Ed25519Signer.generate("campaign1-execution-witness")
    verifier = Ed25519Verifier()
    verifier.register(signer.key_id, signer.public_bytes())
    plan_sig = signer.sign(asdict(plan))
    plan_doc = {"value": asdict(plan), "digest": plan.digest,
                "signer_key_id": signer.key_id,
                "signature_b64": plan_sig.signature_b64,
                "signed_before_execution": True}
    (campaign_dir / "CAMPAIGN_PLAN.json").write_text(json.dumps(plan_doc, indent=2, sort_keys=True))
    (campaign_dir / "DATASET_PROOF.json").write_text(json.dumps(parts.proof(), indent=2, sort_keys=True))
    (campaign_dir / "EXECUTION_PUBLIC_KEY.bin").write_bytes(signer.public_bytes())
    (campaign_dir / "EXECUTION_KEY_ID.txt").write_text(signer.key_id + "\n")

    train_texts = [str(r.get("train_text") or (str(r["prompt"]) + " " + str(r["expected"])))
                   for r in train]
    practice_rows = train[:max(1, min(len(train), int(cfg.get("practice_samples", 8))))]
    retention_eval_rows = retention_rows[:max(1, min(len(retention_rows),
                                                   int(cfg.get("retention_samples", len(retention_rows)))))]

    all_seeds = []
    for seed in seeds:
        run_dir = campaign_dir / f"seed-{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        seed_result_path = run_dir / "SEED_RESULT.json"
        if (run_dir / "COMPLETE").is_file() and seed_result_path.is_file():
            all_seeds.append(json.loads(seed_result_path.read_text()))
            continue
        random.seed(seed)
        tokenizer = load_tokenizer(spec)
        model = load_causal_lm(spec)
        model_digest = model_identity(model, spec)
        tok_digest = tokenizer_identity(tokenizer, spec)
        seed_receipts = {}

        def emit(arm_id, adapter_digest, state_digest, metrics, outputs, extra=None):
            r = ExecutedRunReceiptV162.sign(
                signer=signer, campaign_digest=plan.digest, arm=arm_id, seed=seed,
                environment_digest=env.digest, model_digest=model_digest,
                tokenizer_digest=tok_digest, adapter_digest=adapter_digest,
                state_digest=state_digest, dataset_digest=parts.hidden.digest,
                evaluator_digest=arts[0].digest, metrics=metrics)
            assert r.verify(verifier)
            doc = {"receipt": asdict(r), "outputs": outputs}
            if extra:
                doc.update(extra)
            (run_dir / f"{arm_id}.json").write_text(json.dumps(doc, indent=2, sort_keys=True))
            seed_receipts[arm_id] = r
            return r

        def run_eval_block(arm, arm_id, state_digest):
            h_em, h_out, h_tok, h_sec = evaluate(model, tokenizer, hidden, max_new, score_fn, arm)
            ret, ret_out, r_tok, r_sec = evaluate(model, tokenizer, retention_eval_rows,
                                                  max_new, retention_fn, arm)
            sec_n, sec_out, s_tok = security_eval(model, tokenizer, security_rows,
                                                  max_new, security_fn, arm)
            metrics = {"hidden_exact_match": h_em, "retention": ret,
                       "security_regressions": sec_n,
                       "eval_seconds": round(h_sec + r_sec, 3),
                       "generated_tokens": h_tok + r_tok + s_tok,
                       "arm_state_bytes": dir_size_bytes(arm_state_dirs[arm_id])
                       if arm_id in arm_state_dirs else 0}
            return metrics, {"hidden": h_out, "retention": ret_out, "security": sec_out}

        # ---- non-parametric arms on one base instance ----
        for arm_id in arms_in_plan:
            if arm_id in ("L6", "NC"):
                continue
            if arm_id == "L1":
                arm = None
                sd = ZERO_DIGEST
            elif arm_id == "L5":
                arm = ARMS["L5"]()
                arm.k = int(cfg.get("replay_k", 2))
                sdir = arms_state_root / f"seed-{seed}" / "L5"
                arm.build_state(sdir, train)
                # labeled practice phase: replay only verified train traces
                for prow in practice_rows:
                    pred, _ = generate(model, tokenizer, str(prow["prompt"]), max_new)
                    arm.add_trace(prow, pred, float(score_fn(pred, str(prow["expected"]))))
                arm.persist()
                arm_state_dirs["L5"] = sdir
                sd = sha256_path(sdir)
            else:
                arm = arm_instances[arm_id]
                sd = sha256_path(arm_state_dirs[arm_id])
            metrics, outputs = run_eval_block(arm, arm_id, sd)
            emit(arm_id, ZERO_DIGEST, sd, metrics, outputs)

        free_model(model)

        # ---- parametric arms: train -> save -> destroy -> reload -> eval ----
        for arm_id, texts in (("L6", train_texts),
                              ("NC", shuffled_label_texts(train, seed))):
            if arm_id not in arms_in_plan:
                continue
            model = load_causal_lm(spec)
            ts = LoraTrainSpec(**cfg.get("lora", {}), seed=seed)
            adir = storage.root / "adapters" / cfg["campaign_id"] / arm_id / f"seed-{seed}"
            model, train_receipt = train_lora(model=model, tokenizer=tokenizer,
                                              texts=texts, output_dir=adir, spec=ts)
            adapter_digest = sha256_path(adir)
            free_model(model)
            model = load_causal_lm(spec, adapter_path=str(adir))
            metrics, outputs = run_eval_block(None, arm_id, ZERO_DIGEST)
            metrics["arm_state_bytes"] = dir_size_bytes(adir)
            emit(arm_id, adapter_digest, ZERO_DIGEST, metrics, outputs,
                 {"training": train_receipt})
            free_model(model)

        # ---- seed summary ----
        sd_result = {"seed": seed,
                     "hidden_exact_match": {a: seed_receipts[a].metrics["hidden_exact_match"]
                                            for a in seed_receipts},
                     "receipt_digests": {a: seed_receipts[a].digest for a in seed_receipts}}
        seed_result_path.write_text(json.dumps(sd_result, indent=2, sort_keys=True))
        (run_dir / "COMPLETE").write_text("complete\n")
        all_seeds.append(sd_result)

    # ---- aggregate ----
    def arm_mean(arm_id, metric):
        vals = []
        for s in all_seeds:
            r_path = campaign_dir / f"seed-{s['seed']}" / f"{arm_id}.json"
            if r_path.is_file():
                doc = json.loads(r_path.read_text())
                vals.append(float(doc["receipt"]["metrics"].get(metric, 0.0)))
        return sum(vals) / len(vals) if vals else 0.0

    arm_hidden = {a: arm_mean(a, "hidden_exact_match") for a in arms_in_plan}
    arm_ret = {a: arm_mean(a, "retention") for a in arms_in_plan}
    arm_sec = {a: arm_mean(a, "security_regressions") for a in arms_in_plan}
    ft = {a: arm_hidden[a] - arm_hidden["L1"] for a in arms_in_plan if a != "L1"}
    delta_neural = arm_hidden["L6"] - arm_hidden["L5"]
    sec_total = sum(arm_sec.values())
    nc_ft = ft.get("NC", 0.0)
    nc_violation = nc_ft > plan.negative_control_max_ft

    reasons = []
    if delta_neural < plan.minimum_neural_incremental_ft:
        reasons.append(f"neural incremental FT {delta_neural:.4f} < "
                       f"{plan.minimum_neural_incremental_ft}")
    if arm_ret["L6"] < plan.minimum_retention:
        reasons.append(f"L6 retention {arm_ret['L6']:.4f} < {plan.minimum_retention}")
    if plan.require_zero_security_regressions and sec_total > 0:
        reasons.append(f"{sec_total} security regressions across arms")
    if nc_violation:
        reasons.append(f"negative control FT {nc_ft:.4f} exceeded bound "
                       f"{plan.negative_control_max_ft} — pipeline suspect")

    decision = "PASS" if not reasons else "BLOCK"
    summary = {"schema": "mini-agi-v16.2-campaign1-result-v1",
               "campaign_plan_digest": plan.digest,
               "environment_digest": env.digest,
               "arms": list(arms_in_plan),
               "seeds": list(seeds),
               "arm_hidden_exact_match": arm_hidden,
               "arm_retention": arm_ret,
               "arm_security_regressions": arm_sec,
               "forward_transfer_vs_L1": ft,
               "delta_ft_neural_L6_minus_L5": delta_neural,
               "negative_control_ft": nc_ft,
               "negative_control_violation": nc_violation,
               "decision": decision,
               "reasons": reasons,
               "promotion_ready": decision == "PASS",
               "note": "PASS is experimental qualification only; not "
                       "production promotion authority."}
    summary["digest"] = digest(summary)
    (campaign_dir / "RESULT.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
