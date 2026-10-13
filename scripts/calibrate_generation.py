#!/usr/bin/env python3
"""Campaign 3B Step 4 — decoding calibration on development data only.

Compares a small, PREDETERMINED set of generation configurations against
the frozen pinned model on development rows (train/validation splits),
records the answer-contract diagnostics for each, and writes a frozen
calibration record plus the selected ``output_contract`` config block
for the next campaign.

Guards (Exit criterion A2):

  * refuses any split other than train/validation — the sealed final
    holdout and the hidden/retention/security partitions are never read
    here ("Do not search for the best output length using the sealed
    final holdout"),
  * greedy decoding only,
  * the configuration set is fixed in this file, not searched; the
    selection rule is deterministic and recorded.

Usage:
  python3 scripts/calibrate_generation.py \
      --config configs/campaign3a.yaml \
      --split validation --limit 24 \
      --output /path/to/GENERATION_CALIBRATION.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from minagi.platforms.cuda.hf_runtime import (HFLoadSpec,  # noqa: E402
                                              load_causal_lm,
                                              load_tokenizer)
from minagi.v161.prompt_contract import (  # noqa: E402
    OutputContract, answer_only_output_contract, build_prompt_contract,
    generation_diagnostics, render_generation_input)

#: Development splits this calibration may read.
DEV_SPLITS = ("train", "validation")

#: The predetermined configuration set (fixed; never searched).
CONFIGS = (
    ("historical-raw-32", True, OutputContract(instruction="",
                                               max_new_tokens=32)),
    ("contract-32", False, OutputContract(instruction="", max_new_tokens=32)),
    ("contract-64", False, OutputContract(instruction="", max_new_tokens=64)),
    ("contract-128", False, OutputContract(instruction="", max_new_tokens=128)),
    ("answer-only-64", False, answer_only_output_contract(64)),
)

SELECTION_RULE = ("among contract configurations: highest mean strict "
                  "exact match, then highest mean valid-answer-format "
                  "rate, then fewest mean generated tokens")


def load_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def resolve_revision(model_id: str, configured: str) -> str:
    if configured and configured != "auto":
        return configured
    from huggingface_hub import HfApi
    return str(HfApi().model_info(model_id).sha)


def _generate(model, tokenizer, prompt: str, output: OutputContract,
              raw: bool):
    """One greedy generation. ``raw`` reproduces the historical 3A
    surface (raw prompt tokenization) purely as a measurement baseline —
    it is not a supported path anywhere else."""
    import torch
    with torch.inference_mode():
        device = next(model.parameters()).device
        if raw:
            ids = list(tokenizer(prompt)["input_ids"])
        else:
            ids = render_generation_input(
                tokenizer, prompt, build_prompt_contract(tokenizer, output))
        batch = {"input_ids": torch.tensor([ids], device=device),
                 "attention_mask": torch.ones((1, len(ids)),
                                              dtype=torch.long,
                                              device=device)}
        out = model.generate(**batch,
                             max_new_tokens=output.max_new_tokens,
                             do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        nt = int(out.shape[1] - len(ids))
        text = tokenizer.decode(out[0, len(ids):],
                                skip_special_tokens=True).strip()
        eos_id = getattr(tokenizer, "eos_token_id", None)
        eos_hit = bool(eos_id is not None and nt > 0
                       and int(out[0, -1].item()) == int(eos_id))
        return text, nt, eos_hit


def _mean(rows, key):
    return sum(float(r.get(key, 0.0)) for r in rows) / len(rows) \
        if rows else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="campaign config (model + dataset identity)")
    ap.add_argument("--split", default="validation",
                    choices=DEV_SPLITS,
                    help="development split only (train|validation)")
    ap.add_argument("--limit", type=int, default=24,
                    help="development rows to calibrate on")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    dataset_path = ROOT / cfg["dataset"]
    rows = [r for r in load_rows(dataset_path) if r.get("split") == args.split]
    if not rows:
        raise SystemExit(f"no rows in development split {args.split!r}")
    dev = rows[:max(1, int(args.limit))]

    model_cfg = cfg["model"]
    revision = resolve_revision(model_cfg["id"],
                                str(model_cfg.get("revision", "auto")))
    import torch
    dtype = "float32" if not torch.cuda.is_available() else \
        str(model_cfg.get("dtype", "auto"))
    spec = HFLoadSpec(model_cfg["id"], revision, dtype, "none", False)
    tokenizer = load_tokenizer(spec)
    model = load_causal_lm(spec, purpose="research")

    record = {
        "schema": "mini-agi-v16.6-generation-calibration-v1",
        "model_id": model_cfg["id"], "revision": revision,
        "dtype": dtype, "decoding": "greedy",
        "dataset": str(cfg["dataset"]),
        "dataset_digest": digest([r["id"] for r in dev]),
        "split": args.split, "rows": len(dev),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_rule": SELECTION_RULE,
        "configs": [],
    }
    for name, raw, output in CONFIGS:
        per_row = []
        for row in dev:
            pred, nt, eos_hit = _generate(model, tokenizer,
                                          str(row["prompt"]), output, raw)
            per_row.append(generation_diagnostics(
                str(row.get("expected", "")), pred,
                generated_tokens=nt, max_new_tokens=output.max_new_tokens,
                eos_hit=eos_hit))
        record["configs"].append({
            "name": name, "raw_prompt": bool(raw),
            "output_contract": asdict(output),
            "output_contract_digest": output.digest,
            "strict_exact_match": _mean(per_row, "strict_exact_match"),
            "correct_first_line": _mean(per_row, "correct_first_line"),
            "correct_answer_prefix": _mean(per_row, "correct_answer_prefix"),
            "valid_answer_format": _mean(per_row, "valid_answer_format"),
            "eos_termination": _mean(per_row, "eos_termination"),
            "max_token_truncation": _mean(per_row, "max_token_truncation"),
            "mean_generated_tokens": _mean(per_row, "generated_tokens"),
        })
        print(f"[calibrate] {name}: "
              f"strict={record['configs'][-1]['strict_exact_match']:.3f} "
              f"format={record['configs'][-1]['valid_answer_format']:.3f} "
              f"eos={record['configs'][-1]['eos_termination']:.3f}")

    contract_cfgs = [c for c in record["configs"] if not c["raw_prompt"]]
    selected = max(contract_cfgs,
                   key=lambda c: (c["strict_exact_match"],
                                  c["valid_answer_format"],
                                  -c["mean_generated_tokens"]))
    record["selected"] = selected["name"]
    oc = selected["output_contract"]
    record["output_contract_yaml"] = (
        "output_contract:\n"
        f"  instruction: {json.dumps(oc['instruction'])}\n"
        f"  max_new_tokens: {oc['max_new_tokens']}\n"
        f"  stop_on_eos: {str(oc['stop_on_eos']).lower()}\n")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"[calibrate] selected {selected['name']} — wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
