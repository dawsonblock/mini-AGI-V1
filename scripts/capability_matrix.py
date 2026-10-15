#!/usr/bin/env python3
"""Capability acquisition strategy matrix (V4-17).

Runs the controlled adapter-strategy comparison the capability
controller needs evidence for. Every parametric strategy is executed by
the SAME verified pipeline (scripts/positive_control_fit.py — early
reservation, strict scoring, committed receipts); nothing here re-
implements scoring or training, so matrix cells are directly comparable.

Strategies executed:

  frozen          — baseline metrics, read from every run's evidence
  fixed_lora      — single fixed-rank adapter
  lora_rehearsal  — + rehearsal interleave (verified corpus)
  lora_kl         — + response-aware frozen-reference KL
  lora_both       — rehearsal + KL
  routed_family   — one parameter-isolated adapter per trained family;
                    selection by the verified router (family binding +
                    OOD abstention), evaluated on validation (within-
                    family) and hidden (must abstain -> frozen)
  skill_memory    — delegated to the campaign lane (L5b verified
                    replay); recorded as DELEGATED, not fabricated

Usage:
  python3 scripts/capability_matrix.py \
      --families pc-upper,pc-rev --steps 60 \
      --run-root results/v40_matrix --matrix-id m1
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

PY = sys.executable


def _run_fit(run_id: str, run_root: Path, extra: list[str],
             args) -> dict | None:
    cmd = [PY, str(ROOT / "scripts" / "positive_control_fit.py"),
           "--corpus", args.corpus, "--config", args.config,
           "--families", args.families, "--steps", str(args.steps),
           "--run-id", run_id, "--run-root", str(run_root),
           "--seed", str(args.seed), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(ROOT))
    mpath = run_root / run_id / "METRICS.json"
    if proc.returncode != 0 or not mpath.is_file():
        return {"run_id": run_id, "status": "FAILED",
                "stderr_tail": proc.stderr[-2000:]}
    m = json.loads(mpath.read_text())
    m["run_id"] = run_id
    return m


def routed_family_eval(args, run_root: Path) -> dict:
    """Per-family adapters + verified routing, evaluated here."""
    import torch
    from minagi.platforms.cuda.hf_runtime import (HFLoadSpec,
                                                  load_causal_lm,
                                                  load_tokenizer)
    from minagi.platforms.cuda.peft_trainer import (LoraTrainSpec,
                                                    train_family_adapters)
    from minagi.v161.evaluators import exact_match_strict
    from minagi.v161.prompt_contract import (build_prompt_contract,
                                             encode_generation,
                                             output_contract_from_config)
    from minagi.v161.verified_routing import (QualifiedAdapterManifest,
                                              VerifiedAdapterRouter)
    from egai.common.canonical import digest
    import yaml

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    model_cfg = cfg["model"]
    spec = HFLoadSpec(model_cfg["id"], str(model_cfg["revision"]),
                      str(model_cfg.get("dtype", "auto")),
                      str(model_cfg.get("quantization", "none")),
                      bool(model_cfg.get("trust_remote_code", False)))
    rows = [json.loads(ln) for ln in
            (ROOT / args.corpus).read_text().splitlines() if ln.strip()]
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    train_rows = [r for r in rows
                  if r["split"] == "train" and r["family"] in families]
    val_rows = [r for r in rows if r["split"] == "validation"]
    hidden = [r for r in rows if r["split"] == "hidden"]

    tokenizer = load_tokenizer(spec)
    contract = build_prompt_contract(
        tokenizer, output_contract_from_config(cfg, cfg.get(
            "max_new_tokens", 16)))
    tspec = LoraTrainSpec(rank=args.rank, alpha=32.0, dropout=0.05,
                          target_modules=("q_proj", "k_proj", "v_proj",
                                          "o_proj"),
                          learning_rate=args.learning_rate,
                          steps=args.steps, max_length=128, seed=args.seed)
    fam_ex = {f: [{"prompt": str(r["prompt"]),
                   "response": str(r["expected"])}
                  for r in train_rows if r["family"] == f]
              for f in families}

    adir_root = run_root / f"{args.matrix_id}-routed" / "adapters"
    started = time.time()
    fam_results = train_family_adapters(
        model_loader=lambda: load_causal_lm(spec, purpose="research"),
        tokenizer=tokenizer, family_examples=fam_ex,
        output_dir=adir_root, spec=tspec, prompt_contract=contract,
        on_adapter=lambda f, r: gc.collect())
    train_secs = round(time.time() - started, 3)

    # Verified router: one qualified manifest per trained adapter.
    manifests = [QualifiedAdapterManifest(
        adapter_id=f"fam-{f}",
        adapter_digest=fam_results[f]["adapter_digest"],
        model_id=spec.model_id, model_revision=spec.revision,
        qualified_families=(f,),
        # provisional: no independent qualification yet — the manifest
        # binds the training receipt identity, and routing evidence is
        # measured here, not claimed
        qualification_evidence_digest=digest(
            {"provisional_training_receipt":
             fam_results[f]["adapter_digest"]}),
        scorer_artifact_digest=digest({"scorer": "exact-match-v2"}),
        family_terms=(f,)) for f in families]
    router = VerifiedAdapterRouter(manifests)

    def gen(model, prompt):
        with torch.inference_mode():
            dev = next(model.parameters()).device
            ids = encode_generation(tokenizer, prompt, contract)
            batch = {"input_ids": torch.tensor([ids], device=dev)}
            out = model.generate(**batch,
                                 max_new_tokens=contract.output.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(out[0, len(ids):],
                                    skip_special_tokens=True).strip()

    # Evaluation: route by the row's declared family — adapters activate
    # only on families they trained for; hidden rows abstain to frozen.
    routed_val = {"hits": 0, "n": 0, "activated": 0}
    routed_hid = {"hits": 0, "n": 0, "activated": 0}
    loaded: dict = {}
    frozen = load_causal_lm(spec, purpose="research")
    for split_name, rowset, acc in (("validation", val_rows, routed_val),
                                    ("hidden", hidden, routed_hid)):
        for r in rowset:
            fam = str(r["family"])
            dec = router.route(str(r["prompt"]), family=fam,
                               model_revision=spec.revision)
            if dec.selected:
                acc["activated"] += 1
                fam_id = dec.selected.removeprefix("fam-")
                if fam_id not in loaded:
                    loaded[fam_id] = load_causal_lm(
                        spec, adapter_path=fam_results[fam_id]["adapter_dir"],
                        purpose="research")
                model = loaded[fam_id]
            else:
                model = frozen  # abstention -> frozen fallback
            out = gen(model, str(r["prompt"]))
            acc["hits"] += int(exact_match_strict(out, str(r["expected"])))
            acc["n"] += 1
    return {
        "strategy": "routed_family",
        "train_seconds": train_secs,
        "n_adapters": len(fam_results),
        "validation": {"accuracy": (routed_val["hits"] / routed_val["n"]
                                    if routed_val["n"] else 0.0),
                       "n": routed_val["n"],
                       "adapter_activated": routed_val["activated"]},
        "hidden": {"accuracy": (routed_hid["hits"] / routed_hid["n"]
                                if routed_hid["n"] else 0.0),
                   "n": routed_hid["n"],
                   "adapter_activated": routed_hid["activated"],
                   "note": "hidden families are disjoint from trained "
                           "families — activation here would be a "
                           "routing error; abstention->frozen is correct"},
        "router_events": len(router.events),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="configs/positive_control_tasks.jsonl")
    ap.add_argument("--config",
                    default="configs/campaign3b_positive_control.yaml")
    ap.add_argument("--families", default="pc-upper,pc-rev")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rehearsal-fraction", type=float, default=0.25)
    ap.add_argument("--kl-weight", type=float, default=0.1)
    ap.add_argument("--strategies", default="fixed,rehearsal,kl,both,routed")
    ap.add_argument("--matrix-id", default="v40-matrix")
    ap.add_argument("--run-root", default="results/v40_matrix/runs")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    results: dict[str, dict] = {}
    flags = {"fixed": [],
             "rehearsal": ["--rehearsal-fraction",
                           str(args.rehearsal_fraction)],
             "kl": ["--kl-weight", str(args.kl_weight)],
             "both": ["--rehearsal-fraction",
                      str(args.rehearsal_fraction),
                      "--kl-weight", str(args.kl_weight)]}
    for name in [s.strip() for s in args.strategies.split(",")
                 if s.strip()]:
        if name == "routed":
            results["routed_family"] = routed_family_eval(args, run_root)
            continue
        if name == "skill_memory":
            results["skill_memory"] = {
                "status": "DELEGATED",
                "note": "verified-replay baseline runs in the campaign "
                        "lane (arm L5b) — not duplicated here"}
            continue
        if name not in flags:
            print(f"unknown strategy {name!r}", file=sys.stderr)
            continue
        rid = f"{args.matrix_id}-{name}-s{args.seed}"
        if args.skip_existing and \
                (run_root / rid / "METRICS.json").is_file():
            results[name] = {"run_id": rid, **json.loads(
                (run_root / rid / "METRICS.json").read_text())}
            continue
        res = _run_fit(rid, run_root, flags[name], args)
        results[name] = res if res is not None else {"status": "FAILED"}

    doc = {"schema": "mini-agi-v4.0-capability-matrix-v1",
           "matrix_id": args.matrix_id, "corpus": args.corpus,
           "families": args.families, "steps": args.steps,
           "seed": args.seed, "strategies": results}
    out = run_root.parent / f"CAPABILITY_MATRIX-{args.matrix_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(json.dumps({k: (v.get("status") or "ok")
                      for k, v in results.items()}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
