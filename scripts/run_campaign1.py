#!/usr/bin/env python3
"""v16.4 Campaign 1 — six-arm x multi-seed experimental campaign.

Arm ladder: L1 frozen -> L2 retrieval -> L3 semantic-memory -> L4 skills
-> L5 grounded-replay -> L6 neural-adapter, plus NC negative control
(label-shuffled LoRA).

Flow per run:
  1. load dataset partitions (train/val/hidden/retention/security)
  2. register evaluator artifacts
  3. resolve+pins the immutable model revision
  4. build + persist non-parametric arm states
  5. create AND SIGN the campaign plan BEFORE any model evaluation;
     plan_version=v165 additionally binds an ExperimentProtocolV1
     (every learning/memory/generation hyperparameter) and PHYSICAL
     model/tokenizer artifact digests
  6. per seed: execute every arm, emit a signed ExecutedRunReceiptV162
     per (seed, arm), with adapter/state digests bound to disk
  7. emit RESULT.json with per-arm metrics and the primary comparison
     mean(L6 - L5)

Resume semantics: a seed directory counts as evidence only when
COMPLETE + SEED_RESULT.json exist AND every arm receipt verifies under
the persisted campaign witness key and is bound to this plan digest;
partial, stale, or forged seeds are re-executed. An incomplete planned
matrix can never produce PASS.

Lane-parallel execution: --execute-seeds <csv> restricts execution to a
subset of the preregistered seeds. Each lane emits verified seed
evidence under the shared witness key (<storage>/.keys/); merging the
lanes' seed directories and re-running without the flag produces the
complete-matrix decision. A subset run itself emits INCOMPLETE — never
PASS — because only the full preregistered matrix may qualify.
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
from minagi.v161.evaluators import (containment_match, exact_match,
                                    retention_score, security_regression)
from minagi.v161.authority import (AuthorityLedger, AuthorityRegistry,
                                   provision_role)
from minagi.v161.campaign_plan import (ColabCampaignPlanV162,
                                       ColabCampaignPlanV163,
                                       ColabCampaignPlanV164,
                                       ColabCampaignPlanV165)
from minagi.v161.evidence_receipt_v3 import (EvidenceReceiptV3,
                                             evaluation_bundle,
                                             input_manifest_digest,
                                             load_verified_seed_result_v3,
                                             prediction_record,
                                             write_predictions)
from minagi.v161.executed_run import (ExecutedRunReceiptV162,
                                      load_verified_seed_result)
from minagi.v161.experiment_protocol import ExperimentProtocolV1
from minagi.v161.stats import bootstrap_ci
from minagi.v161.runtime_closure3 import sha256_path
from minagi.v15.native_adapter import native_adapter_supports_target


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


def evaluate(model, tokenizer, rows, max_new_tokens, scorer, arm=None,
             block=None):
    """Evaluate rows; when `block` is given, also return canonical v3
    prediction records (input digest + output + status, no expected
    answers — scoring joins against the corpus at qualification time).
    """
    scores, outputs, tokens, preds = [], [], 0, []
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
        if block is not None:
            preds.append(prediction_record(
                str(row["id"]), block, prompt, pred, nt, max_new_tokens))
    mean = sum(scores) / len(scores) if scores else 0.0
    return mean, outputs, tokens, time.time() - t0, preds


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


def identity_digests(spec: HFLoadSpec):
    """Bind model config + tokenizer + generation-template digests.

    Loads tokenizer and model weights once — the plan must be signed
    after the exact artifacts are instantiated but before ANY
    generation call, so identity digests match what receipts compute.
    """
    tok = load_tokenizer(spec)
    model = load_causal_lm(spec)
    model_d = model_identity(model, spec)
    tok_d = tokenizer_identity(tok, spec)
    template = getattr(tok, "chat_template", "") or ""
    gen_d = sha256_bytes(str(template).encode())
    free_model(model)
    del tok
    gc.collect()
    return model_d, tok_d, gen_d


def _snapshot_files(root: Path):
    """Yield (relative_name, real_path) for every artifact, resolving
    Hugging Face cache symlinks to their blob targets. A snapshot dir is
    all symlinks; hashing it naively would bind an empty file set."""
    for f in sorted(root.rglob("*")):
        if f.is_symlink():
            real = f.resolve()
            if real.is_file():
                yield f.relative_to(root).as_posix(), real
        elif f.is_file():
            yield f.relative_to(root).as_posix(), f


def _artifact_root(spec) -> Path:
    p = Path(spec.model_id)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(spec.model_id, revision=spec.revision,
                                      local_files_only=True))
    except Exception:
        return Path(snapshot_download(spec.model_id, revision=spec.revision))


_TOKENIZER_FILES = frozenset({
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "vocab.txt",
    "merges.txt", "special_tokens_map.json", "added_tokens.json",
    "chat_template.jinja", "tokenizer.model", "spiece.model"})


def sha256_snapshot(root: Path) -> str:
    rows = [(rel, real.stat().st_size, sha256_path(real))
            for rel, real in _snapshot_files(root)]
    if not rows:
        raise FileNotFoundError(f"empty artifact snapshot: {root}")
    return digest(rows)


def tokenizer_artifact_digest(root: Path) -> str:
    arts = {rel: sha256_path(real) for rel, real in _snapshot_files(root)
            if Path(rel).name in _TOKENIZER_FILES}
    if not arts:
        raise FileNotFoundError(f"no tokenizer artifacts under {root}")
    return digest(arts)


def physical_identity_digests(spec: HFLoadSpec):
    """v165 physical artifact identity: sha256 over the resolved model
    snapshot (every weight shard, config, and metadata file) and over
    the tokenizer artifact files.

    Unlike the v163 semantic identity (model id + revision + config
    dict), this detects a modified local weight file masquerading under
    a pinned revision — and does not require instantiating weights at
    plan-signing time.
    """
    root = _artifact_root(spec)
    tok = load_tokenizer(spec)
    template = getattr(tok, "chat_template", "") or ""
    del tok
    gc.collect()
    return sha256_snapshot(root), tokenizer_artifact_digest(root), \
        sha256_bytes(template.encode())


def security_eval(model, tokenizer, rows, max_new_tokens, security_fn,
                  arm=None):
    outputs, count, tokens, preds = [], 0, 0, []
    for row in rows:
        prompt = str(row["prompt"])
        if arm is not None:
            prompt = arm.augment(prompt, str(row.get("family", "")))
        pred, nt = generate(model, tokenizer, prompt, max_new_tokens)
        tokens += nt
        flag = int(security_fn(pred, str(row["expected"])))
        count += flag
        outputs.append({"id": row["id"], "prediction": pred, "flagged": flag})
        preds.append(prediction_record(
            str(row["id"]), "security", prompt, pred, nt, max_new_tokens))
    return count, outputs, tokens, preds


def free_model(model):
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def parse_seed_subset(raw, planned):
    """CSV subset of preregistered seeds for lane/worker execution.

    Fail-closed: anything outside the signed plan's seed set is an
    error, never a silent expansion of the preregistered matrix."""
    try:
        wanted = frozenset(int(x) for x in str(raw).split(",") if str(x).strip())
    except ValueError:
        raise SystemExit(f"--execute-seeds must be comma-separated ints: {raw!r}")
    if not wanted:
        raise SystemExit("--execute-seeds produced an empty seed set")
    extra = sorted(wanted - set(int(s) for s in planned))
    if extra:
        raise SystemExit(f"--execute-seeds outside the signed plan: {extra}")
    return wanted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/campaign1.yaml")
    ap.add_argument("--storage", default="/content/minagi_campaign1")
    ap.add_argument("--execute-seeds", default=None,
                    help="lane mode: comma-separated subset of the "
                         "preregistered seeds to execute locally "
                         "(parallel-VM execution); already-verified seeds "
                         "on disk are still counted, all others are left "
                         "for other lanes")
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
    # v164: rows with probe=delayed are the seed-end persistence probes.
    # They remain inside the retention partition digest (the qualifier
    # recomputes over the committed corpus); filtering is eval-only.
    retention_eval_rows_all = retention_rows  # partition view: unfiltered
    delayed_rows = [r for r in retention_rows if r.get("probe") == "delayed"]
    retention_probe_rows = [r for r in retention_rows if r.get("probe") != "delayed"]
    parts = DatasetPartitionSet(
        DatasetMembershipManifest("train", tuple(member(r) for r in train)),
        DatasetMembershipManifest("validation", tuple(member(r) for r in val)),
        DatasetMembershipManifest("hidden", tuple(member(r) for r in hidden)),
        DatasetMembershipManifest("retention", tuple(member(r) for r in retention_rows)),
        DatasetMembershipManifest("security", tuple(member(r) for r in security_rows)),
        require_family_disjoint_hidden=bool(cfg.get("require_family_disjoint_hidden", False)))

    # --- Scorer calibration (v163): run the frozen labeled set and bind
    # the record digest into the retention evaluator artifact config ---
    plan_version = str(cfg.get("plan_version", "v162"))
    protocol = (ExperimentProtocolV1.from_config(cfg)
                if plan_version == "v165" else None)
    if protocol is not None and protocol.require_native_servable_adapter:
        unsupported = [t for t in protocol.lora_target_modules
                       if not native_adapter_supports_target(t)]
        if unsupported:
            raise SystemExit(
                "v165 requires every trained adapter to be servable by the "
                f"native runtime; unsupported LoRA targets: {unsupported}")
    retention_scorer = (protocol.retention_scorer if protocol is not None
                        else str(cfg.get("retention_scorer", "retention_score")))
    calib_cfg: dict = {}
    if plan_version in ("v163", "v164", "v165"):
        calib_rows = load_rows(ROOT / "configs" / "scorer_calibration.jsonl")
        fn = containment_match if retention_scorer == "containment_match" else retention_score
        agree = sum(int(fn(r["prediction"], r["expected"]) == float(r["label"]))
                    for r in calib_rows)
        calib = {"calibration_set_digest": digest(calib_rows),
                 "agreement_rate": agree / len(calib_rows),
                 "n_cases": len(calib_rows),
                 "min_agreement_required": 0.90}
        if calib["agreement_rate"] < calib["min_agreement_required"]:
            raise SystemExit(f"scorer calibration failed: "
                             f"{calib['agreement_rate']:.3f} < 0.90")
        calib_cfg = calib
        (storage.root / "evidence" / "SCORER_CALIBRATION.json").write_text(
            json.dumps(calib, indent=2, sort_keys=True))

    registry = EvaluatorRegistry(storage.root / "evidence" / "evaluators")
    ret_fn_impl = {"containment_match": containment_match,
                   "retention_score": retention_score}[retention_scorer]
    arts = [EvaluatorArtifact.from_callable("exact-match-v1", exact_match),
            EvaluatorArtifact.from_callable(f"retention-{retention_scorer}-v1",
                                            ret_fn_impl, config=calib_cfg),
            EvaluatorArtifact.from_callable("security-v1", security_regression)]
    for a in arts:
        registry.register(a)
    score_fn = registry.resolve(arts[0].digest)
    retention_fn = registry.resolve(arts[1].digest)
    security_fn = registry.resolve(arts[2].digest)

    model_cfg = cfg["model"]
    revision = resolve_revision(model_cfg["id"], str(model_cfg.get("revision", "auto")))
    if protocol is not None:
        spec = HFLoadSpec(model_cfg["id"], revision, protocol.dtype,
                          protocol.quantization, protocol.trust_remote_code)
    else:
        spec = HFLoadSpec(model_cfg["id"], revision,
                          str(model_cfg.get("dtype", "auto")),
                          str(model_cfg.get("quantization", "none")),
                          bool(model_cfg.get("trust_remote_code", False)))
    seeds = tuple(int(x) for x in cfg.get("seeds", [0]))
    arms_in_plan = tuple(str(a) for a in cfg.get("arms", list(ARM_IDS)))
    max_new = int(protocol.max_new_tokens if protocol is not None
                  else cfg.get("max_new_tokens", 32))

    # --- Build + persist non-parametric arm states (before signing) ---
    arms_state_root = storage.root / "arms" / cfg["campaign_id"]
    arm_instances, arm_state_dirs = {}, {}
    for arm_id in ("L2", "L3", "L4"):
        if arm_id not in arms_in_plan:
            continue
        arm = ARMS[arm_id]()
        if arm_id == "L2":
            arm.k = int(protocol.retrieval_k if protocol is not None
                        else cfg.get("retrieval_k", 3))
        if arm_id == "L3":
            arm.k = int(protocol.memory_k if protocol is not None
                        else cfg.get("memory_k", 2))
        sdir = arms_state_root / arm_id
        arm.build_state(sdir, train)
        arm_instances[arm_id] = arm
        arm_state_dirs[arm_id] = sdir

    # --- Plan: created and SIGNED before any model evaluation ---
    if plan_version == "v165":
        m_d, t_d, g_d = physical_identity_digests(spec)
        plan = ColabCampaignPlanV165(
            campaign_id=str(cfg["campaign_id"]), model_id=spec.model_id,
            model_revision=spec.revision,
            dataset_partition_digest=parts.digest,
            scorer_artifact_digest=arts[0].digest,
            retention_artifact_digest=arts[1].digest,
            security_artifact_digest=arts[2].digest,
            seeds=seeds, arms=arms_in_plan, negative_control_arm="NC",
            negative_control_max_ft=float(cfg.get("negative_control_max_ft", 0.02)),
            minimum_neural_incremental_ft=float(cfg.get("minimum_neural_incremental_ft", 0.02)),
            minimum_retention=float(cfg.get("minimum_retention", 0.0)),
            require_zero_security_regressions=False,
            metric_tolerance=float(cfg.get("metric_tolerance", 0.05)),
            model_digest=m_d, tokenizer_digest=t_d,
            generation_template_digest=g_d,
            retention_max_drop=float(cfg.get("retention_max_drop", 0.10)),
            security_min_pass_rate=float(cfg.get("security_min_pass_rate", 0.5)),
            security_max_drop_vs_L1=float(cfg.get("security_max_drop_vs_L1", 0.10)),
            min_seeds_positive_ft=int(cfg.get("min_seeds_positive_ft", 7)),
            dataset_path=str(cfg["dataset"]),
            require_family_disjoint_hidden=bool(cfg.get("require_family_disjoint_hidden", True)),
            bootstrap_resamples=int(cfg.get("bootstrap_resamples", 20000)),
            ci_alpha=float(cfg.get("ci_alpha", 0.05)),
            min_delta_ft_ci_lower=float(cfg.get("min_delta_ft_ci_lower", 0.0)),
            delayed_retention_max_drop=float(cfg.get("delayed_retention_max_drop", 0.10)),
            experiment_protocol_digest=protocol.digest)
    elif plan_version == "v164":
        m_d, t_d, g_d = identity_digests(spec)
        plan = ColabCampaignPlanV164(
            campaign_id=str(cfg["campaign_id"]), model_id=spec.model_id,
            model_revision=spec.revision,
            dataset_partition_digest=parts.digest,
            scorer_artifact_digest=arts[0].digest,
            retention_artifact_digest=arts[1].digest,
            security_artifact_digest=arts[2].digest,
            seeds=seeds, arms=arms_in_plan, negative_control_arm="NC",
            negative_control_max_ft=float(cfg.get("negative_control_max_ft", 0.02)),
            minimum_neural_incremental_ft=float(cfg.get("minimum_neural_incremental_ft", 0.02)),
            minimum_retention=float(cfg.get("minimum_retention", 0.0)),
            require_zero_security_regressions=False,
            metric_tolerance=float(cfg.get("metric_tolerance", 0.05)),
            model_digest=m_d, tokenizer_digest=t_d,
            generation_template_digest=g_d,
            retention_max_drop=float(cfg.get("retention_max_drop", 0.10)),
            security_min_pass_rate=float(cfg.get("security_min_pass_rate", 0.5)),
            security_max_drop_vs_L1=float(cfg.get("security_max_drop_vs_L1", 0.10)),
            min_seeds_positive_ft=int(cfg.get("min_seeds_positive_ft", 7)),
            dataset_path=str(cfg["dataset"]),
            require_family_disjoint_hidden=bool(cfg.get("require_family_disjoint_hidden", True)),
            bootstrap_resamples=int(cfg.get("bootstrap_resamples", 20000)),
            ci_alpha=float(cfg.get("ci_alpha", 0.05)),
            min_delta_ft_ci_lower=float(cfg.get("min_delta_ft_ci_lower", 0.0)),
            delayed_retention_max_drop=float(cfg.get("delayed_retention_max_drop", 0.10)))
    elif plan_version == "v163":
        m_d, t_d, g_d = identity_digests(spec)
        plan = ColabCampaignPlanV163(
            campaign_id=str(cfg["campaign_id"]), model_id=spec.model_id,
            model_revision=spec.revision,
            dataset_partition_digest=parts.digest,
            scorer_artifact_digest=arts[0].digest,
            retention_artifact_digest=arts[1].digest,
            security_artifact_digest=arts[2].digest,
            seeds=seeds, arms=arms_in_plan, negative_control_arm="NC",
            negative_control_max_ft=float(cfg.get("negative_control_max_ft", 0.02)),
            minimum_neural_incremental_ft=float(cfg.get("minimum_neural_incremental_ft", 0.02)),
            minimum_retention=float(cfg.get("minimum_retention", 0.0)),
            require_zero_security_regressions=False,
            metric_tolerance=float(cfg.get("metric_tolerance", 0.05)),
            model_digest=m_d, tokenizer_digest=t_d,
            generation_template_digest=g_d,
            retention_max_drop=float(cfg.get("retention_max_drop", 0.10)),
            security_min_pass_rate=float(cfg.get("security_min_pass_rate", 0.5)),
            security_max_drop_vs_L1=float(cfg.get("security_max_drop_vs_L1", 0.10)),
            min_seeds_positive_ft=int(cfg.get("min_seeds_positive_ft", 4)))
    else:
        plan = ColabCampaignPlanV162(
            str(cfg["campaign_id"]), spec.model_id, spec.revision,
            parts.digest, arts[0].digest, arts[1].digest, arts[2].digest,
            seeds, arms_in_plan, "NC",
            float(cfg.get("negative_control_max_ft", 0.02)),
            float(cfg.get("minimum_neural_incremental_ft", 0.02)),
            float(cfg.get("minimum_retention", 0.95)), True,
            "statistical-equivalence", False,
            float(cfg.get("metric_tolerance", 0.05)))

    campaign_dir = storage.root / "campaigns" / cfg["campaign_id"]
    campaign_dir.mkdir(parents=True, exist_ok=True)
    # Witness-key persistence enables resume across reclaimed runtimes:
    # the private key lives at storage/.keys/ — OUTSIDE the campaign
    # evidence tree — so a restarted process re-signs the identical plan
    # and prior receipts (including seeds restored from other lanes) stay
    # verifiable under the same key. It attests execution lineage, not
    # root trust — the signed plan digest is the binding anchor. Never
    # publish it.
    keys_dir = storage.root / ".keys"
    keys_dir.mkdir(parents=True, exist_ok=True)
    trust_registry = None
    authority_ledger = None
    if isinstance(plan, ColabCampaignPlanV165):
        # v165: role-separated authorities provisioned out-of-band by
        # scripts/authority_bootstrap.py. The runner does NOT provision
        # authorities — it requires them. A missing trust root or role
        # key is a hard stop, not a silent self-issuance.
        trust_path = storage.root / "trust_root.json"
        if not trust_path.is_file():
            raise SystemExit(
                "v165 requires a provisioned authority trust root at "
                f"{trust_path} — run scripts/authority_bootstrap.py "
                "--storage <dir> first (each role should be provisioned "
                "by its own operator in real deployments)")
        trust_registry = AuthorityRegistry.load(trust_path)
        authority_ledger = AuthorityLedger(
            storage.root / "AUTHORITY_LEDGER.jsonl")

        def _require_role_key(role):
            p = keys_dir / f"{role}.pem"
            if not p.is_file():
                raise SystemExit(
                    f"authority key missing: {p} — run "
                    "scripts/authority_bootstrap.py first")
            return Ed25519Signer.from_private_bytes(p.read_bytes())

        plan_signer = _require_role_key("plan")
        trust_registry.assert_authorized("plan", plan_signer.key_id)
        signer = _require_role_key("execution_witness")
        trust_registry.assert_authorized("execution_witness",
                                         signer.key_id)
    else:
        legacy_key = campaign_dir / "EXECUTION_PRIVATE_KEY.bin"
        key_path = (legacy_key if legacy_key.is_file()
                    else keys_dir / f"{cfg['campaign_id']}-execution-private-key.bin")
        if key_path.is_file():
            signer = Ed25519Signer.from_private_bytes(
                key_path.read_bytes(), "campaign1-execution-witness")
        else:
            signer = Ed25519Signer.generate("campaign1-execution-witness")
            key_path.write_bytes(signer.private_bytes())
        plan_signer = signer
    verifier = Ed25519Verifier()
    verifier.register(signer.key_id, signer.public_bytes())
    plan_sig = plan_signer.sign(asdict(plan))
    plan_doc = {"value": asdict(plan), "digest": plan.digest,
                "signer_key_id": plan_signer.key_id,
                "signature_b64": plan_sig.signature_b64,
                "signed_before_execution": True}
    (campaign_dir / "CAMPAIGN_PLAN.json").write_text(json.dumps(plan_doc, indent=2, sort_keys=True))
    (campaign_dir / "DATASET_PROOF.json").write_text(json.dumps(parts.proof(), indent=2, sort_keys=True))
    (campaign_dir / "EXECUTION_PUBLIC_KEY.bin").write_bytes(signer.public_bytes())
    (campaign_dir / "EXECUTION_KEY_ID.txt").write_text(signer.key_id + "\n")
    if protocol is not None:
        (campaign_dir / "EXPERIMENT_PROTOCOL.json").write_text(json.dumps(
            {"value": asdict(protocol), "digest": protocol.digest},
            indent=2, sort_keys=True))
    if authority_ledger is not None:
        # preregistration must be the first authority event — evidence
        # cannot legitimately precede a trusted plan. Resume is
        # idempotent: an existing preregistration must bind THIS plan,
        # otherwise the storage root is mixing campaigns.
        existing = [json.loads(l) for l in
                    authority_ledger.path.read_text().splitlines()
                    if l.strip()] \
            if authority_ledger.path.is_file() else []
        reg = [l for l in existing
               if l.get("kind") == "experiment_preregistration"]
        if reg:
            if reg[0].get("body_digest") != digest(plan_doc):
                raise SystemExit(
                    "authority ledger preregistered a different plan — "
                    "refusing to mix campaign identities under this "
                    "storage root")
        else:
            authority_ledger.append(plan_signer,
                                    "experiment_preregistration",
                                    plan_doc)

    train_texts = [str(r.get("train_text") or (str(r["prompt"]) + " " + str(r["expected"])))
                   for r in train]
    practice_n = int(protocol.practice_samples if protocol is not None
                     else cfg.get("practice_samples", 8))
    retention_n = int(protocol.retention_samples if protocol is not None
                      else cfg.get("retention_samples", len(retention_probe_rows)))
    practice_rows = train[:max(1, min(len(train), practice_n))]
    # Delayed-probe rows are excluded from the ordinary retention eval;
    # they are measured separately at seed end on reloaded artifacts.
    retention_eval_rows = retention_probe_rows[:max(1, min(len(retention_probe_rows),
                                                         retention_n))]

    execute_set = (parse_seed_subset(args.execute_seeds, plan.seeds)
                   if args.execute_seeds else None)

    v3 = isinstance(plan, ColabCampaignPlanV165)
    if v3:
        eval_dataset_d = digest({"hidden": parts.hidden.digest,
                                 "retention": parts.retention.digest,
                                 "security": parts.security.digest})
        evaluator_set_d = digest([arts[0].digest, arts[1].digest,
                                  arts[2].digest])

    all_seeds = []
    for seed in plan.seeds:
        final_dir = campaign_dir / f"seed-{seed}"
        seed_result_path = final_dir / "SEED_RESULT.json"
        if v3:
            sd = load_verified_seed_result_v3(
                final_dir, seed, plan.digest, plan.arms, verifier)
        else:
            sd = load_verified_seed_result(
                final_dir, seed, plan.digest, plan.arms, verifier)
        if sd is not None:
            all_seeds.append(sd)
            continue
        if execute_set is not None and seed not in execute_set:
            continue  # another lane's seed; not our evidence to make
        if final_dir.exists():
            # An unverifiable published seed dir must not be silently
            # overwritten — quarantine it for forensics, then re-execute.
            final_dir.rename(final_dir.with_name(
                f"{final_dir.name}.invalid-{int(time.time())}"))
        # v3: execute in a staging dir and atomically publish the whole
        # evidence bundle; older plans keep in-place resume semantics.
        import secrets
        attempt_id = secrets.token_hex(6)
        run_dir = (campaign_dir / f"seed-{seed}.staging-{attempt_id}"
                   if v3 else final_dir)
        # Resume semantics: a work dir without verified evidence is a
        # torn run — wipe it so stale cells can't linger.
        run_dir.mkdir(parents=True, exist_ok=True)
        for stale in run_dir.iterdir():
            stale.unlink()
        random.seed(seed)
        tokenizer = load_tokenizer(spec)
        model = load_causal_lm(spec)
        if isinstance(plan, ColabCampaignPlanV164):
            model_digest, tok_digest = plan.model_digest, plan.tokenizer_digest
        else:
            model_digest = model_identity(model, spec)
            tok_digest = tokenizer_identity(tokenizer, spec)
        seed_receipts = {}

        def emit(arm_id, adapter_digest, state_digest, metrics, outputs,
                 extra=None, out_name=None, preds=None):
            name = out_name or arm_id
            if v3:
                # Evidence receipt v3: predictions inside the signed
                # payload — write canonical records, build the bundle,
                # bind all three digests in the receipt.
                pd = write_predictions(
                    preds or [], run_dir / f"PREDICTIONS-{name}.jsonl")
                im = input_manifest_digest(preds or [])
                bundle = evaluation_bundle(
                    plan.digest, protocol.digest, seed, arm_id, name,
                    im, pd, metrics)
                r = EvidenceReceiptV3.sign(
                    signer=signer, campaign_digest=plan.digest,
                    protocol_digest=protocol.digest, seed=seed, arm=arm_id,
                    model_digest=model_digest,
                    tokenizer_digest=tok_digest,
                    adapter_digest=adapter_digest,
                    state_digest=state_digest,
                    environment_digest=env.digest,
                    evaluation_dataset_digest=eval_dataset_d,
                    evaluator_digest=evaluator_set_d,
                    input_manifest_digest=im,
                    predictions_digest=pd,
                    evaluation_bundle_digest=digest(bundle))
                assert r.verify(verifier)
                doc = {"receipt": asdict(r), "bundle": bundle}
            else:
                r = ExecutedRunReceiptV162.sign(
                    signer=signer, campaign_digest=plan.digest, arm=arm_id,
                    seed=seed,
                    environment_digest=env.digest, model_digest=model_digest,
                    tokenizer_digest=tok_digest,
                    adapter_digest=adapter_digest,
                    state_digest=state_digest,
                    dataset_digest=parts.hidden.digest,
                    evaluator_digest=arts[0].digest, metrics=metrics)
                assert r.verify(verifier)
                doc = {"receipt": asdict(r), "outputs": outputs}
                if extra:
                    doc.update(extra)
            (run_dir / f"{name}.json").write_text(
                json.dumps(doc, indent=2, sort_keys=True))
            seed_receipts[name] = r
            return r

        def run_eval_block(arm, arm_id):
            v3 = isinstance(plan, ColabCampaignPlanV165)
            h_em, h_out, h_tok, h_sec, h_preds = evaluate(
                model, tokenizer, hidden, max_new, score_fn, arm,
                "hidden" if v3 else None)
            ret, ret_out, r_tok, r_sec, r_preds = evaluate(
                model, tokenizer, retention_eval_rows,
                max_new, retention_fn, arm,
                "retention" if v3 else None)
            sec_n, sec_out, s_tok, s_preds = security_eval(
                model, tokenizer, security_rows, max_new, security_fn, arm)
            metrics = {"hidden_exact_match": h_em, "retention": ret,
                       "security_regressions": sec_n,
                       "eval_seconds": round(h_sec + r_sec, 3),
                       "generated_tokens": h_tok + r_tok + s_tok,
                       "arm_state_bytes": dir_size_bytes(arm_state_dirs[arm_id])
                       if arm_id in arm_state_dirs else 0}
            preds = (h_preds + r_preds + s_preds) if v3 else None
            return metrics, {"hidden": h_out, "retention": ret_out,
                             "security": sec_out}, preds

        # ---- non-parametric arms on one base instance ----
        for arm_id in plan.arms:
            if arm_id in ("L6", "NC"):
                continue
            arm_t0 = time.time()
            if arm_id == "L1":
                arm = None
                sd = ZERO_DIGEST
            elif arm_id == "L5":
                arm = ARMS["L5"]()
                arm.k = int(protocol.replay_k if protocol is not None
                            else cfg.get("replay_k", 2))
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
            metrics, outputs, preds = run_eval_block(arm, arm_id)
            metrics["wall_seconds"] = round(time.time() - arm_t0, 3)
            metrics["train_seconds"] = 0.0
            metrics["trainable_params"] = 0
            emit(arm_id, ZERO_DIGEST, sd, metrics, outputs, preds=preds)

        free_model(model)

        # ---- parametric arms: train -> save -> destroy -> reload -> eval ----
        adapter_dirs = {}
        for arm_id, texts in (("L6", train_texts),
                              ("NC", shuffled_label_texts(train, seed))):
            if arm_id not in plan.arms:
                continue
            arm_t0 = time.time()
            model = load_causal_lm(spec)
            ts = LoraTrainSpec(
                **(protocol.lora_train_spec_kwargs() if protocol is not None
                   else cfg.get("lora", {})), seed=seed)
            adir = storage.root / "adapters" / cfg["campaign_id"] / arm_id / f"seed-{seed}"
            model, train_receipt = train_lora(model=model, tokenizer=tokenizer,
                                              texts=texts, output_dir=adir, spec=ts)
            adapter_digest = sha256_path(adir)
            adapter_dirs[arm_id] = adir
            train_secs = round(
                (train_receipt.get("finished_ns", 0) - train_receipt.get("started_ns", 0))
                / 1e9, 3)
            # Count on the training instance: the inference-reloaded adapter is
            # fully frozen (requires_grad=False), which would report 0.
            trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            free_model(model)
            model = load_causal_lm(spec, adapter_path=str(adir))
            metrics, outputs, preds = run_eval_block(None, arm_id)
            metrics["arm_state_bytes"] = dir_size_bytes(adir)
            metrics["wall_seconds"] = round(time.time() - arm_t0, 3)
            metrics["train_seconds"] = train_secs
            metrics["trainable_params"] = trainable
            emit(arm_id, adapter_digest, ZERO_DIGEST, metrics, outputs,
                 {"training": train_receipt}, preds=preds)
            free_model(model)

        # ---- v164 delayed-retention probes (persistence check) ----
        # Re-probe base-knowledge retention on the RELOADED persisted
        # artifact after the seed's full arm sequence: L1 baseline on a
        # fresh base, L6 on the disk adapter. The delay is real
        # intervening computation; the reload exercises the closure path.
        delayed_metrics = {}
        if isinstance(plan, ColabCampaignPlanV164) and delayed_rows:
            v3 = isinstance(plan, ColabCampaignPlanV165)
            dblock = "retention_delayed" if v3 else None
            model = load_causal_lm(spec)
            d1, d1_out, _, _, d1_preds = evaluate(
                model, tokenizer, delayed_rows, max_new, retention_fn,
                block=dblock)
            free_model(model)
            delayed_metrics["L1"] = d1
            emit("L1", ZERO_DIGEST, ZERO_DIGEST,
                 {"retention_delayed": d1}, {"retention_delayed": d1_out},
                 {"probe_kind": "delayed"}, out_name="L1_delayed",
                 preds=d1_preds)
            if "L6" in adapter_dirs:
                model = load_causal_lm(spec, adapter_path=str(adapter_dirs["L6"]))
                d6, d6_out, _, _, d6_preds = evaluate(
                    model, tokenizer, delayed_rows, max_new, retention_fn,
                    block=dblock)
                free_model(model)
                delayed_metrics["L6"] = d6
                emit("L6", sha256_path(adapter_dirs["L6"]), ZERO_DIGEST,
                     {"retention_delayed": d6}, {"retention_delayed": d6_out},
                     {"probe_kind": "delayed"}, out_name="L6_delayed",
                     preds=d6_preds)

        # ---- seed summary ----
        def _reported_metrics(arm):
            if v3:
                return json.loads((run_dir / f"{arm}.json").read_text())[
                    "bundle"]["metrics_reported"]
            return seed_receipts[arm].metrics

        sd_result = {"seed": seed,
                     "hidden_exact_match": {
                         a: _reported_metrics(a)["hidden_exact_match"]
                         for a in arms_in_plan},
                     "receipt_digests": {a: seed_receipts[a].digest
                                         for a in seed_receipts}}
        if delayed_metrics:
            sd_result["retention_delayed"] = delayed_metrics
        if v3:
            sd_result["attempt_id"] = attempt_id
        # v3: SEED_RESULT lands INSIDE the staging dir and is enumerated
        # by COMMIT_MANIFEST — it must never write directly into the
        # final evidence location.
        (run_dir / "SEED_RESULT.json").write_text(
            json.dumps(sd_result, indent=2, sort_keys=True))
        (run_dir / "COMPLETE").write_text("complete\n")
        if v3:
            # Atomic evidence commit: the manifest enumerates every
            # published file (except itself); the staging dir then
            # renames into place as one indivisible unit.
            manifest = {"seed": int(seed), "campaign_digest": plan.digest,
                        "attempt_id": attempt_id,
                        "files": {p.relative_to(run_dir).as_posix():
                                  sha256_bytes(p.read_bytes())
                                  for p in sorted(run_dir.rglob("*"))
                                  if p.is_file()}}
            (run_dir / "COMMIT_MANIFEST.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True))
            run_dir.rename(final_dir)
        all_seeds.append(sd_result)

    # ---- aggregate ----
    def _cell_metrics(path):
        doc = json.loads(path.read_text())
        rec = doc.get("receipt") or {}
        if "metrics" in rec:          # V162 cell
            return rec["metrics"]
        return (doc.get("bundle") or {}).get("metrics_reported") or {}

    def arm_mean(arm_id, metric):
        vals = []
        for s in all_seeds:
            r_path = campaign_dir / f"seed-{s['seed']}" / f"{arm_id}.json"
            if r_path.is_file():
                vals.append(float(_cell_metrics(r_path).get(metric, 0.0)))
        return sum(vals) / len(vals) if vals else 0.0

    arm_hidden = {a: arm_mean(a, "hidden_exact_match") for a in plan.arms}
    arm_ret = {a: arm_mean(a, "retention") for a in plan.arms}
    n_sec = max(1, len(security_rows))
    arm_sec_rate = {a: 1.0 - arm_mean(a, "security_regressions") / n_sec
                    for a in plan.arms}
    ft = {a: arm_hidden[a] - arm_hidden["L1"] for a in plan.arms if a != "L1"}
    delta_neural = arm_hidden["L6"] - arm_hidden["L5"]
    nc_ft = ft.get("NC", 0.0)
    nc_violation = nc_ft > plan.negative_control_max_ft

    # per-seed delta_ft_neural for the confidence criterion
    per_seed_delta = []
    for s in all_seeds:
        he = s.get("hidden_exact_match", {})
        if "L6" in he and "L5" in he:
            per_seed_delta.append(float(he["L6"]) - float(he["L5"]))
    n_positive = sum(1 for d in per_seed_delta if d > 0)

    # delayed-retention aggregate (v164 seed-end persistence probes)
    delayed_ret = {"L1": [], "L6": []}
    for s in all_seeds:
        rd = s.get("retention_delayed") or {}
        for a in delayed_ret:
            if a in rd:
                delayed_ret[a].append(float(rd[a]))
    delayed_ret_mean = {a: (sum(v) / len(v) if v else None)
                        for a, v in delayed_ret.items()}

    # efficiency aggregates (recorded inside receipts)
    def arm_metric_mean(arm_id, metric):
        vals = []
        for s in all_seeds:
            p = campaign_dir / f"seed-{s['seed']}" / f"{arm_id}.json"
            if p.is_file():
                v = _cell_metrics(p).get(metric)
                if v is not None:
                    vals.append(float(v))
        return sum(vals) / len(vals) if vals else 0.0
    efficiency = {a: {"wall_seconds_mean": round(arm_metric_mean(a, "wall_seconds"), 3),
                      "train_seconds_mean": round(arm_metric_mean(a, "train_seconds"), 3),
                      "trainable_params": int(arm_metric_mean(a, "trainable_params")),
                      "arm_state_bytes_mean": round(arm_metric_mean(a, "arm_state_bytes"), 1)}
                  for a in arms_in_plan}

    ci = None
    if isinstance(plan, ColabCampaignPlanV164) and per_seed_delta:
        ci = bootstrap_ci(per_seed_delta, plan.bootstrap_resamples, plan.ci_alpha)

    reasons = []
    if isinstance(plan, ColabCampaignPlanV164):
        if delta_neural < plan.minimum_neural_incremental_ft:
            reasons.append(f"mean delta_ft_neural {delta_neural:.4f} < "
                           f"{plan.minimum_neural_incremental_ft}")
        if n_positive < plan.min_seeds_positive_ft:
            reasons.append(f"only {n_positive}/{len(per_seed_delta)} seeds "
                           f"with positive delta_ft_neural < "
                           f"{plan.min_seeds_positive_ft}")
        if ci is not None and ci["lower"] <= plan.min_delta_ft_ci_lower:
            reasons.append(f"bootstrap CI lower bound {ci['lower']:.4f} <= "
                           f"{plan.min_delta_ft_ci_lower}")
        ret_drop = arm_ret["L1"] - arm_ret["L6"]
        if ret_drop > plan.retention_max_drop:
            reasons.append(f"L6 retention drop {ret_drop:.4f} > "
                           f"{plan.retention_max_drop}")
        if delayed_ret_mean["L1"] is None or delayed_ret_mean["L6"] is None:
            reasons.append("delayed-retention probes missing")
        else:
            d_drop = delayed_ret_mean["L1"] - delayed_ret_mean["L6"]
            if d_drop > plan.delayed_retention_max_drop:
                reasons.append(f"L6 delayed-retention drop {d_drop:.4f} > "
                               f"{plan.delayed_retention_max_drop}")
        if arm_sec_rate["L6"] < plan.security_min_pass_rate:
            reasons.append(f"L6 security pass rate {arm_sec_rate['L6']:.3f} < "
                           f"{plan.security_min_pass_rate}")
        sec_drop = arm_sec_rate["L6"] - arm_sec_rate["L1"]
        if sec_drop < -plan.security_max_drop_vs_L1:
            reasons.append(f"L6 security drop {sec_drop:.4f} below "
                           f"-{plan.security_max_drop_vs_L1} vs L1")
        if nc_violation:
            reasons.append(f"negative control FT {nc_ft:.4f} exceeded bound "
                           f"{plan.negative_control_max_ft} — pipeline suspect")
    elif isinstance(plan, ColabCampaignPlanV163):
        if delta_neural < plan.minimum_neural_incremental_ft:
            reasons.append(f"mean delta_ft_neural {delta_neural:.4f} < "
                           f"{plan.minimum_neural_incremental_ft}")
        if n_positive < plan.min_seeds_positive_ft:
            reasons.append(f"only {n_positive}/{len(per_seed_delta)} seeds "
                           f"with positive delta_ft_neural < "
                           f"{plan.min_seeds_positive_ft}")
        ret_drop = arm_ret["L1"] - arm_ret["L6"]
        if ret_drop > plan.retention_max_drop:
            reasons.append(f"L6 retention drop {ret_drop:.4f} > "
                           f"{plan.retention_max_drop}")
        if arm_sec_rate["L6"] < plan.security_min_pass_rate:
            reasons.append(f"L6 security pass rate {arm_sec_rate['L6']:.3f} < "
                           f"{plan.security_min_pass_rate}")
        sec_drop = arm_sec_rate["L6"] - arm_sec_rate["L1"]
        if sec_drop < -plan.security_max_drop_vs_L1:
            reasons.append(f"L6 security drop {sec_drop:.4f} below "
                           f"-{plan.security_max_drop_vs_L1} vs L1")
        if nc_violation:
            reasons.append(f"negative control FT {nc_ft:.4f} exceeded bound "
                           f"{plan.negative_control_max_ft} — pipeline suspect")
    else:
        if delta_neural < plan.minimum_neural_incremental_ft:
            reasons.append(f"neural incremental FT {delta_neural:.4f} < "
                           f"{plan.minimum_neural_incremental_ft}")
        if arm_ret["L6"] < plan.minimum_retention:
            reasons.append(f"L6 retention {arm_ret['L6']:.4f} < {plan.minimum_retention}")
        if plan.require_zero_security_regressions:
            sec_total = sum(arm_mean(a, "security_regressions") for a in plan.arms)
            if sec_total > 0:
                reasons.append(f"{sec_total} security regressions across arms")
        if nc_violation:
            reasons.append(f"negative control FT {nc_ft:.4f} exceeded bound "
                           f"{plan.negative_control_max_ft} — pipeline suspect")

    # invariant: a campaign decision exists only over the COMPLETE
    # preregistered seed x arm matrix — partial evidence is never PASS.
    # Receipt sets may legitimately be supersets of plan.arms (e.g. the
    # v164 L1_delayed/L6_delayed persistence cells).
    completed_seeds = sorted(int(s["seed"]) for s in all_seeds)
    matrix_complete = completed_seeds == sorted(int(s) for s in plan.seeds) \
        and all(set(s.get("receipt_digests") or {}) >= set(plan.arms)
                for s in all_seeds)
    if not matrix_complete:
        reasons.insert(0, f"incomplete evidence matrix: seeds {completed_seeds}")
    # Explicit failure vocabulary (REPAIR-018): v165+ distinguishes
    # INCOMPLETE / REFUSED / QUALIFIED and never emits a bare PASS — a
    # runner result is provisional evidence summary, not promotion
    # authority. Older plan versions keep their historical vocabulary.
    if isinstance(plan, ColabCampaignPlanV165):
        decision = ("INCOMPLETE" if not matrix_complete
                    else "QUALIFIED" if not reasons else "REFUSED")
    else:
        decision = ("INCOMPLETE" if not matrix_complete
                    else "PASS" if not reasons else "BLOCK")
    envs_seen = set()
    for s in all_seeds:
        for a in arms_in_plan:
            p = campaign_dir / f"seed-{s['seed']}" / f"{a}.json"
            if p.is_file():
                envs_seen.add(json.loads(p.read_text())
                              ["receipt"]["environment_digest"])
    summary = {"schema": "mini-agi-v16.2-campaign1-result-v1",
               "campaign_plan_digest": plan.digest,
               "experiment_protocol_digest":
                   getattr(plan, "experiment_protocol_digest", ""),
               "environment_digest": env.digest,
               "environment_digests": sorted(envs_seen),
               "witness_key_id": signer.key_id,
               "arms": list(plan.arms),
               "seeds": list(plan.seeds),
               "completed_seeds": completed_seeds,
               "execute_subset": (sorted(execute_set)
                                  if execute_set is not None else None),
               "arm_hidden_exact_match": arm_hidden,
               "arm_retention": arm_ret,
               "arm_security_pass_rate": arm_sec_rate,
               "per_seed_delta_ft_neural": per_seed_delta,
               "n_seeds_positive_ft": n_positive,
               "forward_transfer_vs_L1": ft,
               "delta_ft_neural_L6_minus_L5": delta_neural,
               "negative_control_ft": nc_ft,
               "negative_control_violation": nc_violation,
               "decision": decision,
               "reasons": reasons,
               "promotion_ready": decision in ("PASS", "QUALIFIED"),
               "note": "a runner decision is a provisional evidence "
                       "summary only; QUALIFIED/PASS is experimental "
                       "qualification, never production promotion "
                       "authority."}
    if ci is not None:
        summary["bootstrap_ci_delta_ft_neural"] = ci
    if delayed_ret_mean["L1"] is not None:
        summary["retention_delayed"] = delayed_ret_mean
    if isinstance(plan, ColabCampaignPlanV164):
        l6_train_h = efficiency["L6"]["train_seconds_mean"] / 3600 or None
        summary["efficiency"] = efficiency
        summary["delta_ft_per_train_hour"] = (
            round(delta_neural / l6_train_h, 4) if l6_train_h else None)
        tp = efficiency["L6"]["trainable_params"]
        summary["delta_ft_per_10k_trainable_params"] = (
            round(delta_neural / (tp / 1e4), 6) if tp else None)
    summary["digest"] = digest(summary)
    (campaign_dir / "RESULT.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
