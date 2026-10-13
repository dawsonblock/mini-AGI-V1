#!/usr/bin/env python3
"""Campaign 3B.1 Step 4 — can fixed-rank LoRA fit a small verified subset?

The Campaign 3B diagnostic never cleared the 95% fit gate (0.767 train
accuracy on 30 examples across 6 families), so its cross-family
"generalization failure" may partly be an under-fitting failure. Step 4
separates those: train on a SMALLER, carefully verified subset until the
adapter demonstrably fits it, then ask whether the interference
signature survives.

SCI-402 — immutable run artifacts. An earlier revision wrote every
adapter to <output>.parent/adapter, so runs sharing an output directory
destroyed each other's adapters: five of six adapters from the first
experiment were lost. Each run now owns a unique directory, is built in
staging, hashed, verified by reload, and atomically published. A failed
run is recorded as FAILED rather than vanishing.

SCI-403 — the generation contract is the only source of the token
budget. The earlier revision accepted --max-new-tokens separately from
the frozen contract, so a run could generate with a budget the contract
did not authenticate. The budget now comes from
contract.output.max_new_tokens and the effective decoding configuration
is recorded in the run.

Usage:
  python3 scripts/positive_control_fit.py \
      --families pc-upper,pc-rev --steps 300 \
      --run-root results/campaign3b2/runs
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_inventory(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): sha256_file(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="configs/positive_control_tasks.jsonl")
    ap.add_argument("--config",
                    default="configs/campaign3b_positive_control.yaml")
    ap.add_argument("--families", default="pc-upper,pc-rev",
                    help="comma-separated train families to fit")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="negative control: same prompts, deranged "
                         "responses (same permutation semantics as the "
                         "campaign NC arm). A fit gain under shuffled "
                         "labels is not learning.")
    ap.add_argument("--run-root", default="results/campaign3b2/runs",
                    help="parent directory for immutable per-run outputs")
    ap.add_argument("--run-id", default=None,
                    help="explicit run id; refused if it already exists")
    args = ap.parse_args()

    import yaml
    import torch
    from egai.common.canonical import digest
    from minagi.platforms.cuda.hf_runtime import (HFLoadSpec, load_causal_lm,
                                                  load_tokenizer)
    from minagi.platforms.cuda.peft_trainer import LoraTrainSpec, train_lora
    from minagi.v161.prompt_contract import (build_prompt_contract,
                                             encode_generation,
                                             output_contract_from_config)

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    if not families:
        raise SystemExit("--families produced an empty set")

    rows = [json.loads(line) for line in
            (ROOT / args.corpus).read_text().splitlines() if line.strip()]
    train_all = [r for r in rows if r["split"] == "train"]
    # Validate the family names before anything else, so a typo reports the
    # typo rather than an empty-subset symptom.
    known = sorted({r["family"] for r in train_all})
    unknown = [f for f in families if f not in known]
    if unknown:
        raise SystemExit(f"unknown train families: {unknown} (have {known})")
    subset = [r for r in train_all if r["family"] in families]
    if not subset:
        raise SystemExit(f"no train rows in families {families}")
    hidden = [r for r in rows if r["split"] == "hidden"]

    output_contract = output_contract_from_config(
        cfg, int(cfg.get("max_new_tokens", 16)))
    spec = HFLoadSpec(cfg["model"]["id"], cfg["model"]["revision"],
                      "float32" if not torch.cuda.is_available()
                      else str(cfg["model"].get("dtype", "auto")), "none",
                      False)
    tokenizer = load_tokenizer(spec)
    contract = build_prompt_contract(tokenizer, output_contract)
    max_new = int(contract.output.max_new_tokens)

    lora = cfg["lora"]
    train_spec = LoraTrainSpec(
        rank=args.rank, alpha=int(lora.get("alpha", 32)),
        dropout=float(lora.get("dropout", 0.05)),
        target_modules=tuple(lora.get("target_modules", ["q_proj"])),
        learning_rate=args.learning_rate, steps=args.steps,
        max_length=int(lora.get("max_length", 128)), seed=args.seed,
        optimizer=str(lora.get("optimizer", "adamw")),
        response_only_loss=bool(lora.get("response_only_loss", True)),
        gradient_accumulation_steps=int(
            lora.get("gradient_accumulation_steps", 1)),
        microbatch_size=int(lora.get("microbatch_size", 1)),
        malformed_policy=str(lora.get("malformed_policy", "fail")))

    run_config = {
        "families": families, "steps": args.steps, "rank": args.rank,
        "learning_rate": args.learning_rate, "seed": args.seed,
        "shuffle_labels": bool(args.shuffle_labels),
        "model_id": cfg["model"]["id"],
        "model_revision": cfg["model"]["revision"],
        "corpus": args.corpus, "config": args.config,
        "subset_rows": len(subset), "hidden_rows": len(hidden),
        "prompt_contract_digest": contract.digest,
        "output_contract_digest": contract.output_contract_digest,
        "max_new_tokens": max_new,
        "lora_spec_digest": train_spec.digest,
    }
    config_digest = digest(run_config)
    attempt_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    slug = "-".join(f.split("-")[-1] for f in families)[:24] or "fit"
    run_id = args.run_id or f"{slug}-{config_digest[7:19]}-{attempt_id[:8]}"

    run_root = ROOT / args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    final_dir = run_root / run_id
    if final_dir.exists():
        raise SystemExit(
            f"run destination already exists, refusing to overwrite: "
            f"{final_dir} — a committed run is never modified")
    staging = run_root / f".staging-{run_id}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    print(f"run id   : {run_id}")
    print(f"staging  : {staging}")

    def fail(reason: str) -> int:
        """A failed run is recorded, never silently dropped."""
        (staging / "RUN_COMMIT.json").write_text(json.dumps({
            "run_id": run_id, "status": "FAILED", "reason": reason,
            "config_digest": config_digest, "attempt_id": attempt_id,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime()),
        }, indent=2, sort_keys=True) + "\n")
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(staging, final_dir)
        print(f"FAILED: {reason}\n  recorded at {final_dir}")
        return 2

    (staging / "CONFIG.json").write_text(
        json.dumps({**run_config, "config_digest": config_digest,
                    "attempt_id": attempt_id, "run_id": run_id},
                   indent=2, sort_keys=True) + "\n")

    def generate(model, prompt):
        with torch.inference_mode():
            device = next(model.parameters()).device
            ids = encode_generation(tokenizer, prompt, contract)
            batch = {"input_ids": torch.tensor([ids], device=device),
                     "attention_mask": torch.ones((1, len(ids)),
                                                  dtype=torch.long,
                                                  device=device)}
            out = model.generate(**batch, max_new_tokens=max_new,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(out[0, len(ids):],
                                    skip_special_tokens=True).strip()

    def per_row(model, group):
        return {r["id"]: {
            "family": r["family"],
            "prompt_sha256": hashlib.sha256(
                str(r["prompt"]).encode()).hexdigest(),
            "expected": str(r["expected"]),
            "output": generate(model, str(r["prompt"])),
        } for r in group}

    def score(preds, group):
        by = {r["id"]: r for r in group}
        return {rid: int(v["output"].strip().casefold()
                         == by[rid]["expected"].strip().casefold())
                for rid, v in preds.items()}

    print(f"fit subset: {families} -> {len(subset)} examples "
          f"(of {len(train_all)} train rows)")

    frozen = load_causal_lm(spec, purpose="research")
    frozen_subset = per_row(frozen, subset)
    frozen_hidden = per_row(frozen, hidden)
    del frozen
    gc.collect()

    if args.shuffle_labels:
        from minagi.v161.arms import shuffled_label_examples
        examples = shuffled_label_examples(subset, args.seed)
    else:
        examples = [{"prompt": str(r["prompt"]),
                     "response": str(r["expected"])} for r in subset]

    model = load_causal_lm(spec, purpose="research")
    adapter_dir = staging / "adapter"
    t0 = time.time()
    try:
        model, receipt = train_lora(
            model=model, tokenizer=tokenizer, examples=examples,
            output_dir=adapter_dir, spec=train_spec,
            prompt_contract=contract)
    except Exception as exc:
        return fail(f"training raised {type(exc).__name__}: {exc}")
    train_secs = time.time() - t0
    (staging / "TRAINING_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    model.eval()
    adapted_subset = per_row(model, subset)
    adapted_hidden = per_row(model, hidden)
    in_memory_sample = {rid: adapted_hidden[rid]["output"]
                        for rid in sorted(adapted_hidden)[:3]}

    del model
    gc.collect()
    reloaded = load_causal_lm(spec, adapter_path=str(adapter_dir),
                              purpose="research")
    reload_sample = {r["id"]: generate(reloaded, str(r["prompt"]))
                     for r in sorted(hidden, key=lambda x: x["id"])[:3]}
    del reloaded
    gc.collect()
    if reload_sample != in_memory_sample:
        return fail("adapter reload did not reproduce in-memory outputs")

    f_sub = score(frozen_subset, subset)
    a_sub = score(adapted_subset, subset)
    f_hid = score(frozen_hidden, hidden)
    a_hid = score(adapted_hidden, hidden)
    fit_acc = sum(a_sub.values()) / len(subset)
    destroyed = sorted(rid for rid in f_hid
                       if f_hid[rid] == 1 and a_hid.get(rid) == 0)
    gained = sorted(rid for rid in f_hid
                    if f_hid[rid] == 0 and a_hid.get(rid) == 1)

    for name, preds in (("FROZEN", frozen_hidden),
                        ("ADAPTED", adapted_hidden)):
        (staging / f"PREDICTIONS-{name}.jsonl").write_text(
            "".join(json.dumps({"split": "hidden", "id": rid, **v},
                               sort_keys=True) + "\n"
                    for rid, v in sorted(preds.items())))
    (staging / "PREDICTIONS-SUBSET.jsonl").write_text(
        "".join(json.dumps({"split": "train-subset", "id": rid, **v},
                           sort_keys=True) + "\n"
                for rid, v in sorted(adapted_subset.items())))

    metrics = {
        "fit_gate": {"threshold": 0.95, "adapted_subset_accuracy": fit_acc,
                     "passed": fit_acc >= 0.95},
        "frozen": {"subset": sum(f_sub.values()) / len(subset),
                   "hidden": sum(f_hid.values()) / len(hidden)},
        "adapted": {"subset": fit_acc,
                    "hidden": sum(a_hid.values()) / len(hidden)},
        "subset_correct": a_sub, "hidden_correct": a_hid,
        "frozen_hidden_correct": f_hid,
        "capability_destruction": {
            "n_frozen_correct_hidden": sum(f_hid.values()),
            "destroyed_ids": destroyed, "n_destroyed": len(destroyed),
            "gained_ids": gained, "n_gained": len(gained)},
        "training": {"loss_first": receipt["loss_first"],
                     "loss_last": receipt["loss_last"],
                     "grad_norm_mean": receipt["grad_norm_mean"],
                     "supervised_tokens_total":
                         receipt.get("supervised_tokens_total"),
                     "adapter_delta_l2": receipt.get("adapter_delta_l2"),
                     "adapter_delta_relative":
                         receipt.get("adapter_delta_relative"),
                     "trainable_params": receipt["trainable_params"],
                     "rejected_examples": receipt["rejected_examples"],
                     "train_seconds": round(train_secs, 3)},
        "effective_decoding": {
            "max_new_tokens": max_new, "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "stop_on_eos": bool(contract.output.stop_on_eos),
            "source": "contract.output.max_new_tokens"},
    }
    (staging / "METRICS.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (staging / "ADAPTER_MANIFEST.json").write_text(json.dumps({
        "run_id": run_id, "adapter_dir": "adapter",
        "files": dir_inventory(adapter_dir),
        "reload_verified": True, "reload_sample": reload_sample,
        "prompt_contract_digest": contract.digest,
        "output_contract_digest": contract.output_contract_digest,
    }, indent=2, sort_keys=True) + "\n")

    inventory = dir_inventory(staging)
    (staging / "RUN_COMMIT.json").write_text(json.dumps({
        "run_id": run_id, "status": "COMPLETE",
        "config_digest": config_digest, "attempt_id": attempt_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": inventory,
    }, indent=2, sort_keys=True) + "\n")

    os.replace(staging, final_dir)
    print(f"published: {final_dir}")
    print(f"fit gate : {fit_acc:.3f} "
          f"({'PASS' if fit_acc >= 0.95 else 'FAIL'} at 0.95)")
    print(f"hidden   : frozen {metrics['frozen']['hidden']:.3f} -> "
          f"adapted {metrics['adapted']['hidden']:.3f}")
    print(f"destruction: {len(destroyed)}/{sum(f_hid.values())} "
          f"frozen-correct broken; {len(gained)} gained")
    return 0 if metrics["fit_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
