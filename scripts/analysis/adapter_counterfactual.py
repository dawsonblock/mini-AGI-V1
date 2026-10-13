#!/usr/bin/env python3
"""Campaign 3B.2 Step 8 — adapter-on/off counterfactual on one model.

The Campaign 3B.1 comparison (L5b vs L7) varied the adapter, but the two
arms were produced by separate model loads and separate generation calls.
This test removes every remaining confound:

  * ONE frozen base-model instance for the core counterfactual;
  * ONE tokenized input per task, reused for every condition, so the
    model sees byte-identical token ids throughout (recorded as digests);
  * identical decoding settings throughout, taken from the frozen
    output contract;
  * the adapter toggled on and off on the SAME instance via peft's
    ``disable_adapter`` context manager.

Conditions on the primary instance:

  adapter-off-before   base model, no adapter
  adapter-on           adapter attached
  adapter-off-after    adapter disabled again on the same instance

If ``adapter-off-after`` does not reproduce ``adapter-off-before``
exactly, the wrap/unwrap is not faithful and the run reports FAILED
rather than drawing conclusions from it.

Additional adapters (shuffled-label control, independently trained) are
evaluated on their own fresh instances; that is recorded in the report
so the reload difference is visible rather than assumed away.

``pc-len`` (character counting) is tracked as a named diagnostic family:
it is a capability the frozen model has and the Campaign 3B adapter
destroyed, so it is the sharpest single probe of interference.

Usage:
  python3 scripts/analysis/adapter_counterfactual.py \
      --adapter L6=/path/to/L6/seed-0 \
      --adapter NC=/path/to/NC/seed-0 \
      --output COUNTERFACTUAL.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

#: A capability the frozen model demonstrably has, used as the sharpest
#: interference probe.
DIAGNOSTIC_FAMILIES = ("pc-len",)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", default=[],
                    metavar="NAME=PATH",
                    help="repeatable; the FIRST is the primary instance")
    ap.add_argument("--corpus",
                    default="configs/positive_control_tasks.jsonl")
    ap.add_argument("--config",
                    default="configs/campaign3b_positive_control.yaml")
    ap.add_argument("--split", default="hidden")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import torch
    import yaml
    from peft import PeftModel
    from minagi.platforms.cuda.hf_runtime import (HFLoadSpec, load_causal_lm,
                                                  load_tokenizer)
    from minagi.v161.prompt_contract import (build_prompt_contract,
                                             encode_generation,
                                             output_contract_from_config)

    adapters = []
    for spec in args.adapter:
        if "=" not in spec:
            raise SystemExit(f"--adapter must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        if not Path(path).is_dir():
            raise SystemExit(f"adapter directory not found: {path}")
        adapters.append((name, path))
    if not adapters:
        raise SystemExit("at least one --adapter is required")

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    output_contract = output_contract_from_config(
        cfg, int(cfg.get("max_new_tokens", 16)))
    spec = HFLoadSpec(cfg["model"]["id"], cfg["model"]["revision"],
                      "float32" if not torch.cuda.is_available()
                      else str(cfg["model"].get("dtype", "auto")), "none",
                      False)
    tokenizer = load_tokenizer(spec)
    contract = build_prompt_contract(tokenizer, output_contract)
    max_new = int(contract.output.max_new_tokens)

    corpus = {}
    for line in (ROOT / args.corpus).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            corpus[str(row["id"])] = row
    tasks = [corpus[rid] for rid in sorted(corpus)
             if corpus[rid]["split"] == args.split]
    if not tasks:
        raise SystemExit(f"no {args.split!r} rows in {args.corpus}")

    # ONE tokenization per task, reused by every condition.
    ids_by_task = {str(r["id"]): encode_generation(tokenizer, str(r["prompt"]),
                                                   contract)
                   for r in tasks}
    id_digest = {rid: hashlib.sha256(
        json.dumps(ids).encode()).hexdigest() for rid, ids in ids_by_task.items()}

    def generate(model, ids):
        with torch.inference_mode():
            device = next(model.parameters()).device
            t = torch.tensor([ids], device=device)
            out = model.generate(
                input_ids=t,
                attention_mask=torch.ones((1, len(ids)), dtype=torch.long,
                                          device=device),
                max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(out[0, len(ids):],
                                    skip_special_tokens=True).strip()

    def run_all(model):
        return {rid: generate(model, ids) for rid, ids in ids_by_task.items()}

    def norm(s):
        return str(s).strip().casefold()

    expected = {str(r["id"]): str(r["expected"]) for r in tasks}
    family = {str(r["id"]): str(r["family"]) for r in tasks}

    def correct(outputs):
        return {rid: int(norm(v) == norm(expected[rid]))
                for rid, v in outputs.items()}

    print(f"tasks: {len(tasks)}  ({args.split})  "
          f"max_new_tokens={max_new} from the frozen contract")

    # ---- primary instance: adapter off -> on -> off again ----------
    base = load_causal_lm(spec, purpose="research")
    off_before = run_all(base)
    primary_name, primary_path = adapters[0]
    peft_model = PeftModel.from_pretrained(base, primary_path)
    peft_model.eval()
    on = run_all(peft_model)
    with peft_model.disable_adapter():
        off_after = run_all(peft_model)

    off_faithful = off_after == off_before
    if not off_faithful:
        del peft_model, base
        gc.collect()
        report = {
            "schema": "mini-agi-campaign3b2-counterfactual-v1",
            "status": "FAILED",
            "reason": ("adapter-off-after did not reproduce "
                       "adapter-off-before; the wrap/unwrap is not faithful "
                       "and no interference conclusion may be drawn"),
            "mismatched_task_ids": sorted(
                rid for rid in off_before if off_before[rid] != off_after[rid]),
        }
        Path(args.output).write_text(json.dumps(report, indent=2,
                                                sort_keys=True) + "\n")
        print("FAILED: adapter toggle is not faithful")
        return 2
    del peft_model, base
    gc.collect()

    conditions = {"adapter-off-before": off_before,
                  f"adapter-on[{primary_name}]": on,
                  "adapter-off-after": off_after}

    # ---- secondary adapters: own fresh instances -------------------
    for name, path in adapters[1:]:
        m = load_causal_lm(spec, purpose="research")
        pm = PeftModel.from_pretrained(m, path)
        pm.eval()
        conditions[f"adapter-on[{name}]"] = run_all(pm)
        del pm, m
        gc.collect()

    c_off = correct(off_before)
    n_off_correct = sum(c_off.values())
    per_condition = {}
    for label, outputs in conditions.items():
        c = correct(outputs)
        destroyed = sorted(rid for rid in c_off
                           if c_off[rid] == 1 and c.get(rid) == 0)
        gained = sorted(rid for rid in c_off
                        if c_off[rid] == 0 and c.get(rid) == 1)
        per_condition[label] = {
            "accuracy": sum(c.values()) / len(tasks),
            "correct": c,
            "destroyed_ids": destroyed, "n_destroyed": len(destroyed),
            "gained_ids": gained, "n_gained": len(gained),
            "D": (len(destroyed) / n_off_correct) if n_off_correct else None,
            "D_status": "ESTIMATED" if n_off_correct else "NOT_ESTIMABLE",
            "diagnostic_families": {
                fam: {"n": sum(1 for rid in tasks if family[str(rid["id"])] == fam),
                      "adapter_off_correct": sum(
                          c_off[str(r["id"])] for r in tasks
                          if family[str(r["id"])] == fam),
                      "condition_correct": sum(
                          c[str(r["id"])] for r in tasks
                          if family[str(r["id"])] == fam)}
                for fam in DIAGNOSTIC_FAMILIES},
        }

    report = {
        "schema": "mini-agi-campaign3b2-counterfactual-v1",
        "status": "COMPLETE",
        "split": args.split, "n_tasks": len(tasks),
        "model_id": cfg["model"]["id"],
        "model_revision": cfg["model"]["revision"],
        "prompt_contract_digest": contract.digest,
        "output_contract_digest": contract.output_contract_digest,
        "effective_decoding": {"max_new_tokens": max_new, "do_sample": False,
                               "pad_token_id": tokenizer.eos_token_id,
                               "source": "contract.output.max_new_tokens"},
        "input_identity": {
            "note": ("one tokenization per task, reused by every condition; "
                     "digests below are the input_ids actually passed to "
                     "model.generate"),
            "input_ids_digest": id_digest,
            "n_tasks": len(id_digest)},
        "adapter_toggle_faithful": off_faithful,
        "primary_adapter": primary_name,
        "adapters": {name: path for name, path in adapters},
        "conditions": per_condition,
        "outputs": conditions,
        "task_metadata": {
            str(r["id"]): {"family": str(r["family"]),
                           "expected": str(r["expected"])} for r in tasks},
    }
    Path(args.output).write_text(json.dumps(report, indent=2,
                                            sort_keys=True) + "\n")

    print(f"adapter toggle faithful: {off_faithful}")
    print(f"adapter-off accuracy   : {per_condition['adapter-off-before']['accuracy']:.3f}")
    for label, res in per_condition.items():
        if label == "adapter-off-before":
            continue
        print(f"  {label:28} acc {res['accuracy']:.3f}  "
              f"destroyed {res['n_destroyed']}  gained {res['n_gained']}  "
              f"D {res['D'] if res['D'] is None else round(res['D'], 3)}")
    fam = per_condition[f"adapter-on[{primary_name}]"]["diagnostic_families"]
    for f, v in fam.items():
        print(f"  diagnostic family {f}: off {v['adapter_off_correct']}/"
              f"{v['n']} -> on {v['condition_correct']}/{v['n']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
