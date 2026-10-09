#!/usr/bin/env python3
"""v16.2 Phase 9 — prove the qualified adapter genuinely affects inference.

Loads artifacts from a completed run_colab_campaign.py campaign dir and checks:

  1. The adapter dir on disk still hashes to the digest bound in the A1
     receipt (physical artifact closure at reload time).
  2. A fresh frozen model reproduces the recorded A0 outputs exactly
     (greedy decode; any drift = runtime nondeterminism finding).
  3. The same model + qualified adapter produces outputs different from A0
     on at least one prompt (the adapter changed the computation).
  4. After a fresh frozen reload (adapter removed), outputs return to the
     recorded A0 values (removal restores the original computational path).

Requires torch + transformers + peft (Colab GPU runtime). Writes
ADAPTER_EFFECT_REPORT.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import detect_root, ensure_path


def generate(model, tokenizer, prompt, max_new_tokens):
    import torch
    with torch.inference_mode():
        device = next(model.parameters()).device
        batch = tokenizer(prompt, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.generate(**batch, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()


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
    ap.add_argument("--root", default=None)
    ap.add_argument("--config", default="configs/qwen_smoke.yaml")
    ap.add_argument("--storage", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="ADAPTER_EFFECT_REPORT.json")
    args = ap.parse_args()

    root = ensure_path(Path(args.root).resolve() if args.root else detect_root())
    cfg = yaml.safe_load((root / args.config).read_text())
    campaign_id = str(cfg["campaign_id"])
    storage = Path(args.storage)
    campaign_dir = storage / "campaigns" / campaign_id
    run_dir = campaign_dir / f"seed-{args.seed}"
    adapter_dir = storage / "adapters" / campaign_id / f"seed-{args.seed}"

    a0_doc = json.loads((run_dir / "A0.json").read_text())
    a1_doc = json.loads((run_dir / "A1.json").read_text())
    a0_receipt = a0_doc["receipt"]
    a1_receipt = a1_doc["receipt"]
    a0_recorded = {o["id"]: o["prediction"] for o in a0_doc["outputs"]}
    a1_recorded = {o["id"]: o["prediction"] for o in a1_doc["outputs"]}

    from minagi.platforms.cuda.hf_runtime import HFLoadSpec, load_causal_lm, load_tokenizer
    from minagi.v161.runtime_closure3 import sha256_path
    from minagi.v161.executed_run import ExecutedRunReceiptV161
    from egai.common.crypto import Ed25519Verifier

    checks: list[dict] = []

    def record(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    # 0. Receipt signatures verify against the campaign's execution key.
    pub = (campaign_dir / "EXECUTION_PUBLIC_KEY.bin").read_bytes()
    key_id = (campaign_dir / "EXECUTION_KEY_ID.txt").read_text().strip()
    verifier = Ed25519Verifier()
    verifier.register(key_id, pub)
    try:
        r0 = ExecutedRunReceiptV161(**a0_receipt)
        r1 = ExecutedRunReceiptV161(**a1_receipt)
        record("receipt_signatures_verify", r0.verify(verifier) and r1.verify(verifier))
    except Exception as exc:  # noqa: BLE001
        record("receipt_signatures_verify", False, f"{type(exc).__name__}: {exc}")
        r0 = r1 = None

    # 1. Physical adapter digest == digest bound in the A1 receipt.
    try:
        actual_adapter_digest = sha256_path(adapter_dir)
        record("loaded_adapter_digest_matches_receipt",
               actual_adapter_digest == a1_receipt["adapter_digest"],
               f"disk={actual_adapter_digest} receipt={a1_receipt['adapter_digest']}")
    except Exception as exc:  # noqa: BLE001
        record("loaded_adapter_digest_matches_receipt", False, f"{type(exc).__name__}: {exc}")

    # Dataset rows keyed by id for prompt regeneration.
    rows = {}
    for line in (root / cfg["dataset"]).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[str(r["id"])] = str(r["prompt"])
    prompt_ids = [i for i in a0_recorded if i in rows]
    max_new = int(cfg.get("max_new_tokens", 8))

    m = cfg["model"]
    spec = HFLoadSpec(m["id"], str(m.get("revision", "main")), str(m.get("dtype", "auto")),
                      str(m.get("quantization", "none")), bool(m.get("trust_remote_code", False)))
    tokenizer = load_tokenizer(spec)

    def fresh_outputs(adapter=None):
        model = load_causal_lm(spec, adapter_path=str(adapter) if adapter
                               else None, purpose="research")
        outs = {pid: generate(model, tokenizer, rows[pid], max_new) for pid in prompt_ids}
        free_model(model)
        return outs

    # 2. Frozen reload reproduces recorded A0.
    a0_fresh = fresh_outputs()
    a0_diffs = [pid for pid in prompt_ids if a0_fresh[pid] != a0_recorded[pid]]
    record("frozen_reload_reproduces_A0", not a0_diffs,
           f"{len(prompt_ids) - len(a0_diffs)}/{len(prompt_ids)} exact; mismatched={a0_diffs}")

    # 3. Adapter changes the computation.
    a1_fresh = fresh_outputs(adapter_dir)
    effect_ids = [pid for pid in prompt_ids if a1_fresh[pid] != a0_fresh[pid]]
    record("adapter_changes_output", len(effect_ids) > 0,
           f"{len(effect_ids)}/{len(prompt_ids)} prompts differ from frozen A0")
    a1_diffs = [pid for pid in prompt_ids if a1_fresh[pid] != a1_recorded[pid]]
    record("adapter_reload_reproduces_A1", not a1_diffs,
           f"{len(prompt_ids) - len(a1_diffs)}/{len(prompt_ids)} exact; mismatched={a1_diffs}")

    # 4. Fresh frozen reload after adapter removal restores A0.
    a0_restored = fresh_outputs()
    rest_diffs = [pid for pid in prompt_ids if a0_restored[pid] != a0_recorded[pid]]
    record("adapter_removal_restores_A0", not rest_diffs,
           f"{len(prompt_ids) - len(rest_diffs)}/{len(prompt_ids)} exact; mismatched={rest_diffs}")

    failed = [c for c in checks if not c["pass"]]
    report = {
        "schema": "mini-agi-v16.2-adapter-effect-v1",
        "campaign_id": campaign_id,
        "seed": args.seed,
        "adapter_dir": str(adapter_dir),
        "prompts": prompt_ids,
        "a0_recorded": a0_recorded,
        "a1_recorded": a1_recorded,
        "a0_fresh": a0_fresh,
        "a1_fresh": a1_fresh,
        "a0_restored": a0_restored,
        "checks": checks,
        "status": "PASS" if not failed else "FAIL",
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: report[k] for k in ("schema", "campaign_id", "seed", "status")}, indent=2))
    for c in checks:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['check']} {c['detail']}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
