#!/usr/bin/env python3
"""Probe stale/empty-output gating (V4-18, F9).

The within-family comparison is only meaningful when the frozen
baseline produces scorable output. If the baseline answers with empty
strings on most hidden prompts, "adapted minus frozen" measures
generation plumbing, not capability — the F9 gate marks the comparison
NOT_ESTIMABLE instead of zero.

Two modes:

* default (offline) — scans results/*/runs/*/PREDICTIONS-FROZEN.jsonl
  and reports the non-empty-output rate over hidden rows per run, plus
  a pooled estimate. Uses only committed evidence.
* --live --corpus X --max-rows N — loads the configured base model,
  greedily generates on hidden rows, and reports the same rate for a
  corpus that has no executed evidence yet.

Exit code 0 when the pooled rate meets --min-rate, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))


def _rate(rows) -> tuple[float, int, int]:
    n = len(rows)
    ok = sum(1 for r in rows if str(r.get("output", "")).strip())
    return (ok / n if n else 0.0), ok, n


def offline() -> dict:
    per_run = []
    tot_ok = tot_n = 0
    for pf in sorted(ROOT.glob("results/*/runs/*/PREDICTIONS-FROZEN.jsonl")):
        rows = [json.loads(ln) for ln in pf.read_text().splitlines()
                if ln.strip()]
        hidden = [r for r in rows if str(r.get("split")) == "hidden"] or rows
        rate, ok, n = _rate(hidden)
        tot_ok += ok
        tot_n += n
        per_run.append({"run": pf.parent.name, "n": n, "nonempty": ok,
                        "rate": round(rate, 4)})
    return {"mode": "offline", "runs": per_run,
            "pooled_rate": round(tot_ok / tot_n, 4) if tot_n else 0.0,
            "pooled_nonempty": tot_ok, "pooled_n": tot_n}


def live(corpus: str, max_rows: int, max_new: int) -> dict:
    rows = [json.loads(ln) for ln in (ROOT / corpus).read_text().splitlines()
            if ln.strip()]
    hidden = [r for r in rows if r.get("split") == "hidden"][:max_rows]
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text()) \
        if (ROOT / "configs" / "model.yaml").is_file() else {}
    mid = cfg.get("model", {}).get("id", "Qwen/Qwen2.5-0.5B-Instruct")
    rev = cfg.get("model", {}).get("revision",
                                   "7ae557604adf67be50417f59c2c2f167def9a775")
    from minagi.platforms.cuda.hf_runtime import (HFLoadSpec,
                                                  load_causal_lm,
                                                  load_tokenizer)
    from minagi.v161.prompt_contract import (build_prompt_contract,
                                             encode_generation)
    import torch
    spec = HFLoadSpec(mid, rev, "auto", "none", False)
    tok = load_tokenizer(spec)
    model = load_causal_lm(spec, purpose="research")
    contract = build_prompt_contract(tok)
    outs = []
    with torch.inference_mode():
        dev = next(model.parameters()).device
        for r in hidden:
            ids = encode_generation(tok, str(r["prompt"]), contract)
            batch = {"input_ids": torch.tensor([ids], device=dev)}
            gen = model.generate(**batch, max_new_tokens=max_new,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            outs.append({"id": r["id"],
                         "output": tok.decode(gen[0, len(ids):],
                                              skip_special_tokens=True)})
    rate, ok, n = _rate(outs)
    return {"mode": "live", "corpus": corpus, "n": n, "nonempty": ok,
            "rate": round(rate, 4),
            "sample": outs[:5]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--corpus", default="configs/campaign3b_tasks.jsonl")
    ap.add_argument("--max-rows", type=int, default=48)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--min-rate", type=float, default=0.5,
                    help="F9 threshold for the pooled basemodel rate")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rep = live(args.corpus, args.max_rows, args.max_new) if args.live \
        else offline()
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        if rep["mode"] == "offline":
            for r in rep["runs"]:
                print(f"{r['run']:>24}  {r['nonempty']}/{r['n']} "
                      f"non-empty ({r['rate']:.2f})")
            print(f"pooled: {rep['pooled_nonempty']}/{rep['pooled_n']} "
                  f"({rep['pooled_rate']:.3f})")
        else:
            print(f"{rep['corpus']}: {rep['nonempty']}/{rep['n']} "
                  f"non-empty ({rep['rate']:.2f})")
    rate = rep.get("pooled_rate", rep.get("rate", 0.0))
    if rate < args.min_rate:
        print(f"basemodel evaluable rate {rate:.3f} < F9 {args.min_rate}"
              " — within-family comparison NOT_ESTIMABLE",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
