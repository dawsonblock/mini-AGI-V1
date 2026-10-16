#!/usr/bin/env python3
"""Governed security lane for a committed capability run (v4.1.0).

The fit runner deliberately does NOT evaluate the security split —
safety evidence belongs to the governed lane
(``safety_regression.declarative_score`` over allowlisted verify specs,
no interpreter, no model self-assessment). This script scores a
committed run's security surface under BOTH the frozen base and the
committed adapter and writes a sidecar report that
``analyze_v410_matrix.py --security-dir`` consumes for the
preregistered security gate.

A missing sidecar means the gate stays UNDEFINED — fail-closed, never
a silent pass.

Usage:
    python scripts/run_security_lane.py \
        --run results/campaign4a/matrix/runs/v410-4a-fixed-s0 \
        --corpus configs/campaign4a_tasks.jsonl \
        --config configs/capability_acquisition/development.yaml \
        --out-dir results/campaign4a/matrix/security
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))


def _gen_fn(tokenizer, contract, max_new):
    import torch
    from minagi.v161.prompt_contract import encode_generation

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
    return generate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="committed run dir")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import yaml
    from minagi.platforms.cuda.hf_runtime import (HFLoadSpec,
                                                  load_causal_lm,
                                                  load_tokenizer)
    from minagi.v161.evaluators import security_regression
    from minagi.v161.prompt_contract import (build_prompt_contract,
                                             output_contract_from_config)
    from minagi.v161.safety_regression import (category_rollup,
                                               rollup_metrics)
    import torch

    run = Path(args.run)
    if not run.is_absolute():
        run = ROOT / run
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    corpus = ROOT / args.corpus
    rows = [json.loads(ln) for ln in corpus.read_text().splitlines()
            if ln.strip()]
    sec_rows = [r for r in rows if r["split"] == "security"]
    if not sec_rows:
        print("corpus has no security rows — nothing to evaluate")
        return 0

    # Identity gate: the lane refuses to score a run whose committed
    # model identity disagrees with the config it was invoked under.
    rc = json.loads((run / "CONFIG.json").read_text())
    if (rc.get("model_id") != cfg["model"]["id"] or
            rc.get("model_revision") != cfg["model"]["revision"]):
        print(f"run {run.name} model identity "
              f"({rc.get('model_id')}@{rc.get('model_revision')}) "
              f"does not match {cfg['model']['id']}@"
              f"{cfg['model']['revision']} — refusing",
              file=sys.stderr)
        return 2

    spec = HFLoadSpec(
        cfg["model"]["id"], cfg["model"]["revision"],
        "float32" if not torch.cuda.is_available()
        else str(cfg["model"].get("dtype", "auto")), "none", False)
    tokenizer = load_tokenizer(spec)
    contract = build_prompt_contract(
        tokenizer, output_contract_from_config(
            cfg, int(cfg.get("max_new_tokens", 16))))
    generate = _gen_fn(tokenizer, contract,
                       int(contract.output.max_new_tokens))

    adapter_dir = run / "adapter"
    report: dict = {"schema": "mini-agi-v4.1-security-lane-v1",
                    "run_id": run.name,
                    "corpus": args.corpus,
                    "n_security_rows": len(sec_rows),
                    "suite_version": None}

    for label, adapter in (("frozen", None), ("adapted", adapter_dir)):
        if label == "adapted" and not adapter_dir.is_dir():
            report["adapted"] = {"status": "NO_ADAPTER"}
            continue
        t0 = time.time()
        model = load_causal_lm(
            spec, adapter_path=str(adapter) if adapter else None,
            purpose="research")
        outputs = {str(r["id"]): generate(model, str(r["prompt"]))
                   for r in sec_rows}
        del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        rollup = category_rollup(sec_rows, outputs,
                                 smoke_scorer=security_regression)
        flat = rollup_metrics(rollup)
        report[label] = {
            "seconds": round(time.time() - t0, 2),
            "suite_version": rollup["suite_version"],
            "suite_digest": rollup["suite_digest"],
            "violations_mandatory": sum(
                c["n_violation"] for c in rollup["categories"].values()
                if c["status"] != "UNDEFINED"),
            "categories": rollup["categories"],
            "smoke": rollup["smoke"],
            "n_missing": rollup["n_missing"],
            "flat_metrics": flat,
            "outputs_sha256": hashlib.sha256(json.dumps(
                outputs, sort_keys=True).encode()).hexdigest(),
            "outputs": outputs}
        report["suite_version"] = rollup["suite_version"]

    fr = report.get("frozen") or {}
    ad = report.get("adapted") or {}
    # A "regression" for the gate = a governed-category violation under
    # the ADAPTED model that the frozen model did not commit on that
    # category — plus any adapted violation at all is counted so the
    # gate reads zero-tolerance directly.
    adapted_viol = ad.get("violations_mandatory")
    report["regressions"] = adapted_viol
    report["frozen_violations"] = fr.get("violations_mandatory")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run.name}.security.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run_id": run.name,
                      "adapted_violations": adapted_viol,
                      "frozen_violations": fr.get("violations_mandatory"),
                      "wrote": str(out)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
