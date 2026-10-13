#!/usr/bin/env python3
"""Campaign 3B.1 Step 4 — can fixed-rank LoRA fit a small verified subset?

The Campaign 3B diagnostic never cleared the 95% fit gate (0.767 train
accuracy on 30 examples across 6 families), so its cross-family
"generalization failure" may partly be an under-fitting failure. Step 4
separates those: train on a SMALLER, carefully verified subset until the
adapter demonstrably fits it, then ask whether the interference
signature survives.

The signature to watch is capability destruction: rows the frozen model
answers correctly that the adapted model breaks. In the 3B run the
adapter destroyed 4 of 5 such rows, and `pc-len` collapsed from 3/3 to
0/3 while the shuffled-label control retained 3/3.

Two questions, one run each:

  fit        can the adapter reach >=95% on the subset it was trained on?
  transfer   given a FITTED adapter, does capability destruction persist
             on the held-out families?

Both are answered against the frozen model as baseline, under the same
prompt and output contract the campaign uses.

Usage:
  python3 scripts/positive_control_fit.py \
      --families pc-upper,pc-rev --steps 300 \
      --output POSITIVE_CONTROL_FIT.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))


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
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import yaml
    import torch
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
    subset = [r for r in train_all if r["family"] in families]
    if not subset:
        raise SystemExit(f"no train rows in families {families}")
    hidden = [r for r in rows if r["split"] == "hidden"]
    known = sorted({r["family"] for r in train_all})
    unknown = [f for f in families if f not in known]
    if unknown:
        raise SystemExit(f"unknown train families: {unknown} (have {known})")

    output_contract = output_contract_from_config(
        cfg, int(cfg.get("max_new_tokens", args.max_new_tokens)))
    spec = HFLoadSpec(cfg["model"]["id"], cfg["model"]["revision"],
                      "float32" if not torch.cuda.is_available()
                      else str(cfg["model"].get("dtype", "auto")), "none",
                      False)
    tokenizer = load_tokenizer(spec)
    contract = build_prompt_contract(tokenizer, output_contract)

    def generate(model, prompt):
        with torch.inference_mode():
            device = next(model.parameters()).device
            ids = encode_generation(tokenizer, prompt, contract)
            batch = {"input_ids": torch.tensor([ids], device=device),
                     "attention_mask": torch.ones((1, len(ids)),
                                                  dtype=torch.long,
                                                  device=device)}
            out = model.generate(**batch,
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(out[0, len(ids):],
                                    skip_special_tokens=True).strip()

    def hits(model, group):
        return {r["id"]: int(generate(model, str(r["prompt"])).casefold()
                             == str(r["expected"]).casefold())
                for r in group}

    def summarize(model, label):
        return {
            "label": label,
            "subset": sum(hits(model, subset).values()) / len(subset),
            "hidden": sum(hits(model, hidden).values()) / len(hidden),
            "hidden_by_id": hits(model, hidden),
            "subset_by_id": hits(model, subset),
        }

    print(f"fit subset: {families} -> {len(subset)} examples "
          f"(of {len(train_all)} train rows)")
    frozen = load_causal_lm(spec, purpose="research")
    frozen_res = summarize(frozen, "frozen")
    del frozen
    import gc
    gc.collect()

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
    model = load_causal_lm(spec, purpose="research")
    if args.shuffle_labels:
        from minagi.v161.arms import shuffled_label_examples
        examples = shuffled_label_examples(subset, args.seed)
    else:
        examples = [{"prompt": str(r["prompt"]),
                     "response": str(r["expected"])} for r in subset]
    t0 = time.time()
    model, receipt = train_lora(
        model=model, tokenizer=tokenizer, examples=examples,
        output_dir=Path(args.output).parent / "adapter", spec=train_spec,
        prompt_contract=contract)
    train_secs = time.time() - t0
    model.eval()
    adapted = summarize(model, "adapted")

    destroyed = [rid for rid, v in frozen_res["hidden_by_id"].items()
                 if v == 1 and adapted["hidden_by_id"].get(rid) == 0]
    gained = [rid for rid, v in frozen_res["hidden_by_id"].items()
              if v == 0 and adapted["hidden_by_id"].get(rid) == 1]
    n_frozen_correct = sum(frozen_res["hidden_by_id"].values())

    report = {
        "schema": "mini-agi-campaign3b1-fit-experiment-v1",
        "model_id": cfg["model"]["id"],
        "model_revision": cfg["model"]["revision"],
        "families": families,
        "subset_rows": len(subset),
        "hidden_rows": len(hidden),
        "train_spec": {"rank": args.rank, "steps": args.steps,
                       "learning_rate": args.learning_rate,
                       "seed": args.seed},
        "supervision": ("shuffled-label (negative control)"
                        if args.shuffle_labels else "correct labels"),
        "fit_gate": {"threshold": 0.95,
                     "adapted_subset_accuracy": adapted["subset"],
                     "passed": adapted["subset"] >= 0.95},
        "frozen": {k: v for k, v in frozen_res.items()},
        "adapted": {k: v for k, v in adapted.items()},
        "training": {
            "loss_first": receipt["loss_first"],
            "loss_last": receipt["loss_last"],
            "grad_norm_mean": receipt["grad_norm_mean"],
            "supervised_tokens_total": receipt.get("supervised_tokens_total"),
            "adapter_delta_l2": receipt.get("adapter_delta_l2"),
            "adapter_delta_relative": receipt.get("adapter_delta_relative"),
            "trainable_params": receipt["trainable_params"],
            "rejected_examples": receipt["rejected_examples"],
            "train_seconds": round(train_secs, 3),
            "prompt_contract_digest": receipt["prompt_contract_digest"],
            "output_contract_digest": receipt["output_contract_digest"],
        },
        "capability_destruction": {
            "n_frozen_correct_hidden": n_frozen_correct,
            "destroyed_ids": sorted(destroyed),
            "n_destroyed": len(destroyed),
            "gained_ids": sorted(gained),
            "n_gained": len(gained),
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2,
                                            sort_keys=True) + "\n")

    fit = report["fit_gate"]
    print(f"fit gate  : {adapted['subset']:.3f} "
          f"({'PASS' if fit['passed'] else 'FAIL'} at {fit['threshold']})")
    print(f"hidden    : frozen {frozen_res['hidden']:.3f} -> "
          f"adapted {adapted['hidden']:.3f}")
    print(f"destruction: {len(destroyed)}/{n_frozen_correct} frozen-correct "
          f"hidden rows broken; {len(gained)} gained")
    print(f"wrote {args.output}")
    return 0 if fit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
