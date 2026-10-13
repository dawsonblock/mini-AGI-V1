#!/usr/bin/env python3
"""Campaign 3B Step 13 — positive-control readiness report.

Reads a completed positive-control campaign and measures the
learnability gates over the generated task groups:

  fit        train rows      — can the adapter fit known examples?
  unseen     validation rows — does it apply them to new inputs from
                               the trained families?
  transfer   hidden rows     — distinct held-out families (the harder
                               cross-family claim)

Arms compared: L1 (frozen), L6 (fixed-rank LoRA), NC (shuffled-label
LoRA); L7/L5b are reported when their artifacts exist. Gates (Step 13):

  1. prompt-token parity: the training prefix equals the inference
     input tokens for every train row (contract-enforced)
  2. training targets supervised correctly: every row encodes with a
     non-empty supervised suffix and a fully masked prompt
  3. fit accuracy >= 95% on the intentionally simple task
  4. unseen within-family accuracy meaningfully above frozen
  5. correct-label arm clearly above the shuffled-label control
  6. numerical stability: finite loss curve and non-zero gradient norm
  7. adapter reload reproducible: identical greedy outputs on a sample

Usage:
  python3 scripts/positive_control_report.py \
      --storage <campaign storage> --campaign-id campaign3b-positive-control \
      [--corpus configs/positive_control_tasks.jsonl] \
      --output POSITIVE_CONTROL_REPORT.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402

FIT_GATE = 0.95
UNSEEN_MARGIN = 0.0          # strictly above frozen
CONTROL_MARGIN = 0.10        # correct-label vs shuffled-label


def load_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def readiness_gates(metrics: dict) -> tuple[bool, list[dict]]:
    """Pure gate evaluation over collected metrics (unit-testable)."""
    checks = []

    def check(name, ok, detail):
        checks.append({"gate": name, "passed": bool(ok), "detail": detail})

    parity = metrics.get("parity", {})
    check("prompt-token parity", parity.get("checked", 0) > 0
          and parity.get("failures", 1) == 0,
          f"{parity.get('checked', 0)} rows checked, "
          f"{parity.get('failures', 0)} failures")
    sup = metrics.get("supervision", {})
    check("supervised targets", sup.get("rows", 0) > 0
          and sup.get("bad", 1) == 0,
          f"{sup.get('rows', 0)} rows, {sup.get('bad', 0)} malformed")
    acc = metrics.get("accuracy", {})
    fit = acc.get("L6", {}).get("train")
    check("fit accuracy >= 95%", fit is not None and fit >= FIT_GATE,
          f"L6 train accuracy {fit}")
    frozen_val = acc.get("L1", {}).get("validation")
    l6_val = acc.get("L6", {}).get("validation")
    check("unseen within-family above frozen",
          frozen_val is not None and l6_val is not None
          and l6_val > frozen_val + UNSEEN_MARGIN,
          f"L6 {l6_val} vs L1 {frozen_val}")
    nc_fit = acc.get("NC", {}).get("train")
    check("correct labels beat shuffled labels",
          fit is not None and nc_fit is not None
          and fit > nc_fit + CONTROL_MARGIN,
          f"L6 {fit} vs NC {nc_fit}")
    stab = metrics.get("stability", {})
    check("training numerical stability",
          bool(stab.get("finite_loss", False))
          and bool(stab.get("nonzero_grad", False)),
          f"finite_loss={stab.get('finite_loss')} "
          f"grad_norm_mean={stab.get('grad_norm_mean')}")
    check("adapter reload reproducible",
          metrics.get("reload", {}).get("checked", 0) > 0
          and metrics.get("reload", {}).get("mismatches", 1) == 0,
          f"{metrics.get('reload', {}).get('checked', 0)} generations "
          f"compared")
    return all(c["passed"] for c in checks), checks


def _finite(values) -> bool:
    return bool(values) and all(isinstance(v, (int, float))
                                and math.isfinite(v) for v in values)


def receipt_stability(adapter_dir: Path) -> dict:
    path = adapter_dir / "TRAINING_RECEIPT.json"
    if not path.is_file():
        return {"finite_loss": False, "nonzero_grad": False,
                "grad_norm_mean": None}
    rec = json.loads(path.read_text())
    return {"finite_loss": _finite(rec.get("loss_curve") or []),
            "nonzero_grad": float(rec.get("grad_norm_mean") or 0.0) > 0.0,
            "grad_norm_mean": rec.get("grad_norm_mean"),
            "loss_first": rec.get("loss_first"),
            "loss_last": rec.get("loss_last"),
            "prompt_contract_digest": rec.get("prompt_contract_digest")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--corpus", default="configs/positive_control_tasks.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--sample", type=int, default=4,
                    help="rows per group for the reload check")
    args = ap.parse_args()

    storage = Path(args.storage)
    campaign_dir = storage / "campaigns" / args.campaign_id
    plan_doc = json.loads((campaign_dir / "CAMPAIGN_PLAN.json").read_text())
    rows = load_rows(ROOT / args.corpus)
    groups = {name: [r for r in rows if r["split"] == name]
              for name in ("train", "validation", "hidden")}
    if not all(groups.values()):
        raise SystemExit("corpus is missing a required group")

    import yaml
    from minagi.platforms.cuda.hf_runtime import (HFLoadSpec, load_causal_lm,
                                                  load_tokenizer)
    from minagi.v161.prompt_contract import (build_prompt_contract,
                                             encode_generation,
                                             encode_supervised,
                                             output_contract_from_config)
    import torch

    cfg = yaml.safe_load((ROOT / "configs" /
                          "campaign3b_positive_control.yaml").read_text())
    output_contract = output_contract_from_config(
        cfg, int(cfg.get("max_new_tokens", 16)))
    spec = HFLoadSpec(cfg["model"]["id"], cfg["model"]["revision"],
                      "float32" if not torch.cuda.is_available()
                      else str(cfg["model"].get("dtype", "auto")),
                      "none", False)
    tokenizer = load_tokenizer(spec)
    contract = build_prompt_contract(tokenizer, output_contract)

    metrics: dict = {"accuracy": {}, "parity": {"checked": 0, "failures": 0},
                     "supervision": {"rows": 0, "bad": 0},
                     "reload": {"checked": 0, "mismatches": 0},
                     "stability": {}}

    # gates 1-2: contract parity + supervised targets over train rows
    for row in groups["train"]:
        metrics["supervision"]["rows"] += 1
        try:
            enc = encode_supervised(tokenizer, str(row["prompt"]),
                                    str(row["expected"]), contract,
                                    int(cfg["lora"]["max_length"]))
        except ValueError:
            metrics["supervision"]["bad"] += 1
            continue
        metrics["parity"]["checked"] += 1
        gen_ids = encode_generation(tokenizer, str(row["prompt"]), contract)
        if enc["input_ids"][:len(gen_ids)] != gen_ids:
            metrics["parity"]["failures"] += 1

    def generate(model, prompt):
        with torch.inference_mode():
            device = next(model.parameters()).device
            ids = encode_generation(tokenizer, prompt, contract)
            batch = {"input_ids": torch.tensor([ids], device=device),
                     "attention_mask": torch.ones((1, len(ids)),
                                                  dtype=torch.long,
                                                  device=device)}
            out = model.generate(**batch, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(out[0, len(ids):],
                                    skip_special_tokens=True).strip()

    def accuracy(model, group):
        hits = 0
        for row in groups[group]:
            pred = generate(model, str(row["prompt"]))
            hits += int(pred.strip().casefold()
                        == str(row["expected"]).strip().casefold())
        return hits / len(groups[group])

    def evaluate_adapter(arm_id, adapter_path=None):
        model = load_causal_lm(spec, adapter_path=adapter_path,
                               purpose="research")
        out = {g: accuracy(model, g) for g in ("train", "validation",
                                               "hidden")}
        return model, out

    base, base_acc = evaluate_adapter("L1")
    metrics["accuracy"]["L1"] = base_acc
    del base

    adapters = storage / "adapters" / args.campaign_id
    for arm_id in ("L6", "NC", "L7"):
        adir = adapters / arm_id / "seed-0"
        if not adir.is_dir():
            continue
        model, acc = evaluate_adapter(arm_id, str(adir))
        metrics["accuracy"][arm_id] = acc
        if arm_id == "L6":
            metrics["stability"] = receipt_stability(adir)
            # gate 7: reload the adapter and re-generate a sample; the
            # reload must reproduce the in-memory adapter's outputs
            model2, _ = evaluate_adapter(arm_id, str(adir))
            for row in groups["train"][:args.sample]:
                a = generate(model, str(row["prompt"]))
                b = generate(model2, str(row["prompt"]))
                metrics["reload"]["checked"] += 1
                metrics["reload"]["mismatches"] += int(a != b)
            del model2
        del model

    passed, checks = readiness_gates(metrics)
    report = {"schema": "mini-agi-v16.6-positive-control-report-v1",
              "campaign_id": args.campaign_id,
              "plan_digest": plan_doc["digest"],
              "corpus_digest": digest([r["id"] for r in rows]),
              "groups": {k: len(v) for k, v in groups.items()},
              "metrics": metrics, "gates": checks, "passed": passed}
    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['gate']}: "
              f"{c['detail']}")
    print(f"positive-control readiness: "
          f"{'PASS' if passed else 'NOT YET'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
