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

SCI-402 — immutable run artifacts. An earlier revision wrote every
adapter to ``<output>.parent/adapter``, so runs sharing an output
directory destroyed each other's adapters: five of six adapters from the
first experiment were lost. Each run now owns a unique directory, is
built in staging, hashed, verified by reload, and atomically published.
A failed run is recorded as FAILED rather than vanishing.

SCI-403 — the generation contract is the only source of the token
budget. The earlier revision accepted ``--max-new-tokens`` separately
from the frozen contract, so a run could generate with a budget the
contract did not authenticate. The budget now comes from
``contract.output.max_new_tokens`` and the effective decoding
configuration is recorded in the run.

Usage:
  python3 scripts/positive_control_fit.py \
      --families pc-upper,pc-rev --steps 300 \
      --run-root results/campaign3b2/runs
"""
from __future__ import annotations

import argparse
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
    # Step 11 contrasts. Each isolates one causal factor; run only one
    # of --examples-per-family / --total-examples per run.
    ap.add_argument("--examples-per-family", type=int, default=None,
                    help="Contrast B: hold examples PER FAMILY constant "
                         "while breadth grows. Examples are repeated "
                         "cyclically from the family's unique rows.")
    ap.add_argument("--total-examples", type=int, default=None,
                    help="Contrast A: hold the TOTAL training-example "
                         "count constant while breadth grows, distributing "
                         "the budget across families (as evenly as the "
                         "count allows; the remainder is reported).")
    ap.add_argument("--exposure", type=int, default=None,
                    help="Contrast C: hold PRESENTATIONS PER EXAMPLE "
                         "constant by setting steps = exposure x examples. "
                         "Overrides --steps.")
    # Campaign 3B.3 — P2 rehearsal / P3 frozen-reference KL. Both are
    # optimization-time interventions, NOT inference-time replay (L7).
    ap.add_argument("--rehearsal-fraction", type=float, default=0.0,
                    help="fraction of microbatches that draw verified-"
                         "correct rehearsal pairs into the gradient "
                         "stream (P2). 0 disables.")
    ap.add_argument("--rehearsal-corpus", default=None,
                    help="jsonl of verified-correct rehearsal pairs "
                         "{prompt, response}; default = train-split "
                         "rows the frozen model already answers "
                         "correctly (verified at run time)")
    ap.add_argument("--kl-weight", type=float, default=0.0,
                    help="weight on mean KL(frozen || adapted) over the "
                         "KL prompt set, every --kl-every microbatches "
                         "(P3). 0 disables.")
    ap.add_argument("--kl-every", type=int, default=1)
    ap.add_argument("--kl-prompts", default="configs/preservation_prompts.txt",
                    help="plain-text prompts anchoring the frozen-"
                         "reference KL term — the frozen model's own "
                         "greedy continuations become the teacher-forced "
                         "reference (default: the preservation prompt set)")
    ap.add_argument("--kl-corpus", default=None,
                    help="jsonl of verified {prompt, response|expected} "
                         "pairs to teacher-force through the KL term; "
                         "each row is verified against the frozen model "
                         "under the strict oracle before admission")
    ap.add_argument("--run-root", default="results/campaign3b2/runs",
                    help="parent directory for immutable per-run outputs")
    ap.add_argument("--run-id", default=None,
                    help="explicit run id; refused if it already exists")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Phase 0 — cheap validation + run-destination reservation.
    # V4-04: the run directory is claimed BEFORE torch/transformers and
    # the tokenizer load — a colliding destination must be refused
    # before any costly initialization, and the reservation itself
    # closes the check-then-create race.
    # ------------------------------------------------------------------
    import yaml
    from egai.common.canonical import digest, sha256_bytes

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
    if args.examples_per_family is not None and args.total_examples is not None:
        raise SystemExit("--examples-per-family and --total-examples are "
                         "different contrasts; pass only one")
    if args.examples_per_family is not None and args.examples_per_family < 1:
        raise SystemExit("--examples-per-family must be >= 1")
    if args.total_examples is not None and args.total_examples < 1:
        raise SystemExit("--total-examples must be >= 1")

    unique_by_family = {fam: [r for r in train_all if r["family"] == fam]
                        for fam in families}
    for fam, rs in unique_by_family.items():
        if not rs:
            raise SystemExit(f"no train rows in family {fam!r}")

    #: unique rows scored (a repeated row cannot be scored twice)
    subset = [r for fam in families for r in unique_by_family[fam]]
    #: rows actually trained on, after the contrast's allocation
    if args.total_examples is not None:
        base, extra = divmod(args.total_examples, len(families))
        per_family = {fam: base + (1 if i < extra else 0)
                      for i, fam in enumerate(families)}
        allocation = "total-examples"
    elif args.examples_per_family is not None:
        per_family = {fam: args.examples_per_family for fam in families}
        allocation = "examples-per-family"
    else:
        per_family = {fam: len(unique_by_family[fam]) for fam in families}
        allocation = "all-unique"
    train_rows = []
    for fam in families:
        uniq = unique_by_family[fam]
        n = per_family[fam]
        train_rows.extend(uniq[i % len(uniq)] for i in range(n))

    if args.exposure is not None:
        if args.exposure < 1:
            raise SystemExit("--exposure must be >= 1")
        args.steps = args.exposure * len(train_rows)

    hidden = [r for r in rows if r["split"] == "hidden"]
    # v4.0.1: corpora may carry a within-family validation split
    # (unseen inputs of TRAINED families — campaign3e) and a
    # compositional split; both are evaluated when present so the new
    # measurement surfaces are exercised by this lane too. Empty splits
    # add no predictions and no metric fields.
    validation = [r for r in rows if r["split"] == "validation"]
    compositional = [r for r in rows if r["split"] == "compositional"]

    # --- run identity + destination reservation (V4-04) ---------------
    # Everything needed to name this run is cheap: the family set, the
    # contrast allocation, the LoRA spec, the model identity, and the
    # preservation flags. The prompt-contract digests still bind inside
    # run_config/CONFIG.json below — they are just not needed to
    # decide WHERE the run lives.
    from minagi.platforms.cuda.peft_trainer import LoraTrainSpec

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
        malformed_policy=str(lora.get("malformed_policy", "fail")),
        rehearsal_fraction=float(args.rehearsal_fraction),
        kl_weight=float(args.kl_weight),
        kl_every=int(args.kl_every))

    # --- run identity + collision reservation, BEFORE any model load ---
    # run_id is derived from the cheap identity digest (families,
    # contrast, LoRA spec, model id, preservation flags). The full
    # run_config — which additionally binds the prompt/output contract
    # digests — is written once the tokenizer exists; it remains the
    # complete signed configuration record.
    run_identity = {
        "schema": "mini-agi-v4.0-fit-run-identity-v1",
        "families": families, "steps": args.steps, "rank": args.rank,
        "learning_rate": args.learning_rate, "seed": args.seed,
        "shuffle_labels": bool(args.shuffle_labels),
        "model_id": cfg["model"]["id"],
        "model_revision": cfg["model"]["revision"],
        "corpus": args.corpus, "config": args.config,
        "contrast": {
            "allocation": allocation,
            "families": len(families),
            "examples_per_family": dict(per_family),
            "training_examples": len(train_rows),
            "unique_scored_rows": len(subset),
            "steps": args.steps,
            "exposure_target": args.exposure,
            "presentations_per_example": (args.steps / len(train_rows))
            if train_rows else None,
        },
        "lora_spec_digest": train_spec.digest,
        "rehearsal_fraction": args.rehearsal_fraction,
        "rehearsal_corpus": args.rehearsal_corpus,
        "kl_weight": args.kl_weight, "kl_every": args.kl_every,
        "kl_prompts": args.kl_prompts,
        "kl_corpus": args.kl_corpus,
    }
    identity_digest = digest(run_identity)
    attempt_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    slug = "-".join(f.split("-")[-1] for f in families)[:24] or "fit"
    run_id = args.run_id or f"{slug}-{identity_digest[7:19]}-{attempt_id[:8]}"

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

    ctx = {"config_digest": identity_digest}

    def fail(reason: str) -> int:
        """A failed run is recorded, never silently dropped."""
        (staging / "RUN_COMMIT.json").write_text(json.dumps({
            "run_id": run_id, "status": "FAILED", "reason": reason,
            "config_digest": ctx["config_digest"],
            "attempt_id": attempt_id,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime()),
        }, indent=2, sort_keys=True) + "\n")
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(staging, final_dir)
        print(f"FAILED: {reason}\n  recorded at {final_dir}")
        return 2

    # ------------------------------------------------------------------
    # Phase 1 — model dependencies. Nothing above this line touches
    # torch, the tokenizer, or the model surface.
    # ------------------------------------------------------------------
    import torch
    from minagi.platforms.cuda.hf_runtime import (HFLoadSpec, load_causal_lm,
                                                  load_tokenizer)
    from minagi.platforms.cuda.peft_trainer import train_lora
    from minagi.v161.prompt_contract import (build_prompt_contract,
                                             encode_generation,
                                             output_contract_from_config)

    # SCI-403: the contract is the ONLY source of the generation budget.
    output_contract = output_contract_from_config(
        cfg, int(cfg.get("max_new_tokens", 16)))
    spec = HFLoadSpec(cfg["model"]["id"], cfg["model"]["revision"],
                      "float32" if not torch.cuda.is_available()
                      else str(cfg["model"].get("dtype", "auto")), "none",
                      False)
    tokenizer = load_tokenizer(spec)
    contract = build_prompt_contract(tokenizer, output_contract)
    max_new = int(contract.output.max_new_tokens)

    kl_prompts = None
    kl_prompts_digest = None
    if args.kl_weight > 0 and args.kl_prompts:
        kp_path = ROOT / args.kl_prompts
        if not kp_path.is_file():
            return fail(f"KL prompt set missing: {kp_path}")
        kl_prompts = [ln.strip() for ln in
                      kp_path.read_text().splitlines() if ln.strip()]
        if not kl_prompts:
            return fail(f"KL prompt set empty: {kp_path}")
        kl_prompts_digest = digest(
            [sha256_bytes(p.encode()) for p in kl_prompts])

    run_config = {
        "families": families, "steps": args.steps, "rank": args.rank,
        "learning_rate": args.learning_rate, "seed": args.seed,
        "shuffle_labels": bool(args.shuffle_labels),
        "model_id": cfg["model"]["id"],
        "model_revision": cfg["model"]["revision"],
        "corpus": args.corpus, "config": args.config,
        "subset_rows": len(subset), "hidden_rows": len(hidden),
        "validation_rows": len(validation),
        "compositional_rows": len(compositional),
        "contrast": run_identity["contrast"],
        "run_identity_digest": identity_digest,
        "prompt_contract_digest": contract.digest,
        "output_contract_digest": contract.output_contract_digest,
        "score_oracle": "exact-match-v2",
        "max_new_tokens": max_new,
        "lora_spec_digest": train_spec.digest,
        "rehearsal": {"fraction": args.rehearsal_fraction,
                      "corpus": args.rehearsal_corpus or
                                "frozen-correct rows outside the "
                                "trained families",
                      "excludes_trained_families": True,
                      "note": "optimization-time gradient interleave "
                              "(P2) — NOT inference-time replay (L7)"},
        "kl_preservation": {"weight": args.kl_weight,
                            "every_microbatches": args.kl_every,
                            "scope": "response_continuations",
                            "prompts": args.kl_prompts,
                            "corpus": args.kl_corpus,
                            "prompts_digest": kl_prompts_digest,
                            "reference": "same weights, adapter disabled"},
    }
    config_digest = digest(run_config)
    ctx["config_digest"] = config_digest

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
        """RD-18: strict case-sensitive oracle (exact-match-v2). A
        transformation task's answer is the transformed string itself,
        so letter case is part of correctness — the earlier casefold
        comparison silently accepted wrong-case outputs."""
        by = {r["id"]: r for r in group}
        return {rid: int(v["output"].strip()
                         == by[rid]["expected"].strip())
                for rid, v in preds.items()}

    print(f"contrast : {allocation} | {len(families)} families | "
          f"{len(train_rows)} training examples "
          f"({len(subset)} unique scored) | {args.steps} steps | "
          f"exposure "
          f"{args.steps / len(train_rows) if train_rows else 0:.1f}/example")

    frozen = load_causal_lm(spec, purpose="research")
    frozen_subset = per_row(frozen, subset)
    frozen_hidden = per_row(frozen, hidden)
    frozen_val = per_row(frozen, validation)
    frozen_comp = per_row(frozen, compositional)

    if args.shuffle_labels:
        from minagi.v161.arms import shuffled_label_examples
        # the derangement is over the TRAINED allocation, so the control
        # sees the same number of examples as its matched condition
        examples = shuffled_label_examples(train_rows, args.seed)
    else:
        examples = [{"prompt": str(r["prompt"]),
                     "response": str(r["expected"])} for r in train_rows]

    # P2 rehearsal pool (V4-03) — VERIFIED-correct pairs of capabilities
    # OTHER than the one being acquired. Rehearsing the trained rows
    # themselves rehearses the capability under acquisition (it does not
    # preserve anything distinct and can entrench the new behavior); the
    # default pool is therefore frozen-correct TRAIN-split rows whose
    # family is not among the trained families, plus frozen-correct
    # non-trained rows from the validation split — never hidden,
    # retention, security, or holdout rows (those are measurement
    # surfaces; training on them would contaminate the measurement).
    #
    # A custom --rehearsal-corpus undergoes the IDENTICAL frozen-correct
    # membership check: every candidate pair is regenerated under the
    # frozen model and admitted only when the frozen output equals the
    # claimed answer under the strict oracle. Unverifiable rows are
    # rejected and counted, never silently admitted. Rehearsal always
    # carries the TRUE expected answer even under --shuffle-labels.
    def frozen_correct(candidates, model, cached=None):
        """Filter to pairs the frozen model already answers correctly
        (strict oracle). `cached` maps prompt->frozen output when a
        scoring pass already produced it."""
        verified, rejected = [], 0
        cache = dict(cached or {})
        for c in candidates:
            out = cache.get(c["prompt"])
            if out is None:
                out = generate(model, c["prompt"])
                cache[c["prompt"]] = out
            if out.strip() == str(c["response"]).strip():
                verified.append({"prompt": c["prompt"],
                                 "response": str(c["response"])})
            else:
                rejected += 1
        return verified, rejected, cache

    rehearsal_examples = None
    rehearsal_rejected = 0
    trained_families = set(families)
    if args.rehearsal_fraction > 0 or args.kl_corpus:
        if args.rehearsal_corpus:
            cand = [{"prompt": str(r["prompt"]),
                     "response": str(r.get("response",
                                           r.get("expected", "")))}
                    for r in (json.loads(ln) for ln in
                              (ROOT / args.rehearsal_corpus)
                              .read_text().splitlines() if ln.strip())
                    if str(r.get("family", "")) not in trained_families]
        else:
            cand = [{"prompt": str(r["prompt"]),
                     "response": str(r["expected"])}
                    for r in rows
                    if r["split"] in ("train", "validation")
                    and r["family"] not in trained_families]
        rehearsal_examples, rehearsal_rejected, _ = \
            frozen_correct(cand, frozen)
        if args.rehearsal_fraction > 0 and not rehearsal_examples:
            return fail(
                "rehearsal requested but the verified pool is empty — "
                "cannot rehearse behavior the frozen model does not "
                "have outside the trained families")
        if args.rehearsal_fraction > 0:
            print(f"rehearsal: {len(rehearsal_examples)} verified pairs "
                  f"({rehearsal_rejected} rejected) outside trained "
                  f"families, fraction {args.rehearsal_fraction} of "
                  "microbatches")

    # P3 KL reference set (V4-02): verified {prompt,response} pairs are
    # teacher-forced so the penalty constrains response continuations.
    # --kl-corpus rows pass the same frozen-correct membership check;
    # --kl-prompts are handed to the trainer, which derives the FROZEN
    # model's own greedy continuations as the reference.
    kl_examples = None
    if args.kl_weight > 0 and args.kl_corpus:
        cand = [{"prompt": str(r["prompt"]),
                 "response": str(r.get("response",
                                       r.get("expected", "")))}
                for r in (json.loads(ln) for ln in
                          (ROOT / args.kl_corpus)
                          .read_text().splitlines() if ln.strip())]
        kl_examples, kl_rejected, _ = frozen_correct(cand, frozen)
        if not kl_examples:
            return fail("kl-corpus supplied but no pair survived "
                        "frozen-correct verification")
        print(f"kl corpus: {len(kl_examples)} verified pairs "
              f"({kl_rejected} rejected)")

    # The frozen pass is done — rehearsal and KL membership are settled.
    del frozen
    import gc
    gc.collect()

    model = load_causal_lm(spec, purpose="research")
    adapter_dir = staging / "adapter"
    t0 = time.time()
    try:
        model, receipt = train_lora(
            model=model, tokenizer=tokenizer, examples=examples,
            output_dir=adapter_dir, spec=train_spec,
            prompt_contract=contract,
            rehearsal_examples=rehearsal_examples,
            kl_prompts=kl_prompts, kl_examples=kl_examples)
    except Exception as exc:                       # noqa: BLE001
        return fail(f"training raised {type(exc).__name__}: {exc}")
    train_secs = time.time() - t0
    (staging / "TRAINING_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    model.eval()
    adapted_subset = per_row(model, subset)
    adapted_hidden = per_row(model, hidden)
    adapted_val = per_row(model, validation)
    adapted_comp = per_row(model, compositional)
    in_memory_sample = {rid: adapted_hidden[rid]["output"]
                        for rid in sorted(adapted_hidden)[:3]}

    # The adapter must reload from disk and reproduce the in-memory
    # outputs, or the published artifact does not describe the run.
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
    f_val = score(frozen_val, validation)
    a_val = score(adapted_val, validation)
    f_comp = score(frozen_comp, compositional)
    a_comp = score(adapted_comp, compositional)

    def _acc(scores, group, fams=None):
        sel = [scores[r["id"]] for r in group
               if fams is None or r["family"] in fams]
        return sum(sel) / len(sel) if sel else None
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
    # Per-split prediction files for the v4.0.1 surfaces — written only
    # when the corpus declares them, so legacy-corpora run inventories
    # are byte-identical to before.
    for split_name, pair in (("VALIDATION", (frozen_val, adapted_val)),
                             ("COMPOSITIONAL", (frozen_comp, adapted_comp))):
        if not pair[0] and not pair[1]:
            continue
        for name, preds in (("FROZEN", pair[0]), ("ADAPTED", pair[1])):
            (staging / f"PREDICTIONS-{name}-{split_name}.jsonl").write_text(
                "".join(json.dumps({"split": split_name.lower(), "id": rid,
                                    **v}, sort_keys=True) + "\n"
                        for rid, v in sorted(preds.items())))
    (staging / "PREDICTIONS-SUBSET.jsonl").write_text(
        "".join(json.dumps({"split": "train-subset", "id": rid, **v},
                           sort_keys=True) + "\n"
                for rid, v in sorted(adapted_subset.items())))

    metrics = {
        "fit_gate": {"threshold": 0.95, "adapted_subset_accuracy": fit_acc,
                     "passed": fit_acc >= 0.95},
        "frozen": {"subset": sum(f_sub.values()) / len(subset),
                   "hidden": sum(f_hid.values()) / len(hidden),
                   **({"validation": _acc(f_val, validation)}
                      if validation else {}),
                   **({"compositional": _acc(f_comp, compositional)}
                      if compositional else {})},
        "adapted": {"subset": fit_acc,
                    "hidden": sum(a_hid.values()) / len(hidden),
                    **({"validation": _acc(a_val, validation)}
                       if validation else {}),
                    **({"compositional": _acc(a_comp, compositional)}
                       if compositional else {})},
        "subset_correct": a_sub, "hidden_correct": a_hid,
        "frozen_hidden_correct": f_hid,
        **({"validation": {
                "rows": len(validation),
                "within_family_rows": sum(
                    1 for r in validation
                    if r["family"] in trained_families),
                "frozen": _acc(f_val, validation),
                "adapted": _acc(a_val, validation),
                "within_family": {
                    "frozen": _acc(
                        f_val, validation, trained_families),
                    "adapted": _acc(
                        a_val, validation, trained_families)},
                "untrained_family": {
                    "frozen": _acc(
                        f_val, validation,
                        {r["family"] for r in validation}
                        - trained_families),
                    "adapted": _acc(
                        a_val, validation,
                        {r["family"] for r in validation}
                        - trained_families)},
                "correct": a_val, "frozen_correct": f_val}}
           if validation else {}),
        **({"compositional": {
                "rows": len(compositional),
                "frozen": _acc(f_comp, compositional),
                "adapted": _acc(a_comp, compositional),
                "correct": a_comp, "frozen_correct": f_comp}}
           if compositional else {}),
        "capability_destruction": {
            "n_frozen_correct_hidden": sum(f_hid.values()),
            "destroyed_ids": destroyed, "n_destroyed": len(destroyed),
            "gained_ids": gained, "n_gained": len(gained)},
        "contrast": run_config["contrast"],
        "exposure_actual": {
            "sample_presentations": receipt.get("sample_presentations"),
            "optimizer_updates": receipt.get("optimizer_updates"),
            "effective_batch_size": receipt.get("effective_batch_size"),
            "training_examples": len(train_rows),
            "presentations_per_example":
                (receipt.get("sample_presentations") / len(train_rows))
                if receipt.get("sample_presentations") and train_rows
                else None},
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
        "effective_decoding": {"max_new_tokens": max_new, "do_sample": False,
                               "pad_token_id": tokenizer.eos_token_id,
                               "stop_on_eos": bool(contract.output.stop_on_eos),
                               "source": "contract.output.max_new_tokens"},
        "rehearsal_pool": {
            "excluded_trained_families": sorted(trained_families),
            "verified_pairs": len(rehearsal_examples or []),
            "rejected_by_frozen_check": rehearsal_rejected,
            "custom_corpus": args.rehearsal_corpus,
            "membership_check": "frozen-model strict-oracle regeneration"},
        "kl_reference": {
            "kl_examples": len(kl_examples or []),
            "kl_prompts": len(kl_prompts or []),
            "scope": "response_continuations"},
    }
    (staging / "METRICS.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (staging / "ADAPTER_MANIFEST.json").write_text(json.dumps({
        "run_id": run_id, "adapter_dir": "adapter",
        "files": dir_inventory(adapter_dir),
        "reload_verified": True,
        "reload_sample": reload_sample,
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
    if validation:
        wf = metrics["validation"]["within_family"]
        print(f"within-family validation: frozen "
              f"{wf['frozen']:.3f} -> adapted {wf['adapted']:.3f} "
              f"(n={metrics['validation']['within_family_rows']})")
    if compositional:
        print(f"compositional: frozen "
              f"{metrics['compositional']['frozen']:.3f} -> adapted "
              f"{metrics['compositional']['adapted']:.3f}")
    print(f"destruction: {len(destroyed)}/{sum(f_hid.values())} "
          f"frozen-correct broken; {len(gained)} gained")
    return 0 if metrics["fit_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
