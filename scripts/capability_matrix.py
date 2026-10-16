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
                    OOD abstention + artifact-content verification),
                    evaluated TWO ways on validation and hidden:
                    "hinted" routes with the row's declared family
                    (dispatch-metadata path — an oracle the production
                    router does not have) and "inferred" routes with NO
                    family hint (the production-realistic measurement —
                    family_terms lexical overlap only). Hidden rows must
                    abstain to frozen under BOTH modes; any activation
                    there is a routing error and is reported as the
                    false-activation count.
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
    from minagi.platforms.cuda.peft_trainer import (
        LoraTrainSpec, adapter_artifact_digest, train_family_adapters)
    from minagi.v161.evaluators import exact_match_strict
    from minagi.v161.prompt_contract import (build_prompt_contract,
                                             encode_generation,
                                             output_contract_from_config)
    from minagi.v161.verified_routing import (QualifiedAdapterManifest,
                                              VerifiedAdapterRouter,
                                              lexical_terms)
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
    def _family_terms(fam: str) -> tuple[str, ...]:
        # The vocabulary the INFERRED routing path can legitimately use:
        # the surface text of the family's task instruction — a real
        # request carries no "pc-upper" label, so the family id itself is
        # never a routing term.
        task = next(str(r["prompt"]).split("\n")[0]
                    for r in train_rows if r["family"] == fam)
        return lexical_terms(task)

    manifests = [QualifiedAdapterManifest(
        adapter_id=f"fam-{f}",
        adapter_digest=fam_results[f]["adapter_digest"],
        model_id=spec.model_id, model_revision=spec.revision,
        qualified_families=(f,),
        # provisional: no independent qualification yet — the manifest
        # binds the training receipt identity, is UNSIGNED, and every
        # routing event records controls.signature_enforced=false so
        # this evidence can never be read as authorized routing
        qualification_evidence_digest=digest(
            {"provisional_training_receipt":
             fam_results[f]["adapter_digest"]}),
        scorer_artifact_digest=digest({"scorer": "exact-match-v2"}),
        family_terms=_family_terms(f)) for f in families]
    # Artifact contents ARE verified: the resolver re-hashes each
    # adapter's weights+config at registration (and on any registry
    # reload) — a directory that no longer matches its manifest digest
    # makes the adapter ineligible, not silently served.
    router = VerifiedAdapterRouter(
        manifests,
        artifact_resolver=lambda aid: fam_results[
            aid.removeprefix("fam-")]["adapter_dir"],
        artifact_digester=adapter_artifact_digest)

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

    # Evaluation — the two routing paths are different measurements:
    #   inferred : NO family hint — eligibility comes from lexical
    #              family_terms overlap and the OOD threshold. This is
    #              the production-realistic measurement (a real request
    #              carries no benchmark label) and is the result of
    #              record for "can the router find the capability".
    #   hinted   : family=r["family"] — dispatch metadata handed to the
    #              router. Reported separately so the two are never
    #              conflated again (the pre-v4.0.1 evaluation reported
    #              routed accuracy under this oracle path only).
    # correct_selection = the adapter of the row's OWN family was chosen
    # (meaningful only for rows whose family was trained); any selection
    # of a different adapter — and ANY selection on a hidden row — is a
    # false activation.
    modes = ("inferred", "hinted")
    acc = {(m, s): {"hits": 0, "n": 0, "activated": 0,
                    "correct_selection": 0, "false_activations": 0}
           for m in modes for s in ("validation", "hidden")}
    loaded: dict = {}
    frozen = load_causal_lm(spec, purpose="research")
    for split_name, rowset in (("validation", val_rows),
                               ("hidden", hidden)):
        for r in rowset:
            fam = str(r["family"])
            expected_aid = f"fam-{fam}"
            for mode in modes:
                dec = router.route(
                    str(r["prompt"]),
                    family=(fam if mode == "hinted" else None),
                    # the hinted path models authenticated dispatch
                    # metadata; inferred carries no hint at all (V4-28)
                    family_authenticated=(mode == "hinted"),
                    model_revision=spec.revision)
                cell = acc[(mode, split_name)]
                if dec.selected:
                    cell["activated"] += 1
                    if dec.selected == expected_aid:
                        cell["correct_selection"] += 1
                    else:
                        cell["false_activations"] += 1
                    fam_id = dec.selected.removeprefix("fam-")
                    if fam_id not in loaded:
                        loaded[fam_id] = load_causal_lm(
                            spec,
                            adapter_path=fam_results[fam_id]["adapter_dir"],
                            purpose="research")
                    model = loaded[fam_id]
                else:
                    model = frozen  # abstention -> frozen fallback
                out = gen(model, str(r["prompt"]))
                cell["hits"] += int(exact_match_strict(
                    out, str(r["expected"])))
                cell["n"] += 1

    def _block(split: str, mode: str) -> dict:
        cell = acc[(mode, split)]
        return {"accuracy": (cell["hits"] / cell["n"]
                             if cell["n"] else 0.0),
                "n": cell["n"],
                "adapter_activated": cell["activated"],
                "correct_selection": cell["correct_selection"],
                "false_activations": cell["false_activations"]}

    return {
        "strategy": "routed_family",
        "train_seconds": train_secs,
        "n_adapters": len(fam_results),
        "result_of_record": "inferred",
        "validation": {m: _block("validation", m) for m in modes},
        "hidden": {**{m: _block("hidden", m) for m in modes},
                   "note": "hidden families are disjoint from trained "
                           "families — activation here would be a "
                           "routing error; abstention->frozen is "
                           "correct"},
        "routing_modes": {
            "inferred": "no family hint — lexical family_terms + OOD "
                        "threshold (production-realistic)",
            "hinted": "gold family label as dispatch metadata — "
                      "diagnostic only, not a production capability"},
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
    known = set(flags) | {"routed", "skill_memory"}
    requested = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in requested if s not in known]
    if unknown:
        # V4-29: an unrecognized strategy is a hard failure — silently
        # skipping it would let a partially-executed matrix read as a
        # completed comparison.
        print(f"unknown strateg{'y' if len(unknown) == 1 else 'ies'}: "
              f"{unknown} (known: {sorted(known)})", file=sys.stderr)
        return 2
    for name in requested:
        if name == "routed":
            try:
                results["routed_family"] = routed_family_eval(args,
                                                              run_root)
            except Exception as exc:                    # noqa: BLE001
                results["routed_family"] = {
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}"}
            continue
        if name == "skill_memory":
            results["skill_memory"] = {
                "status": "DELEGATED",
                "note": "verified-replay baseline runs in the campaign "
                        "lane (arm L5b) — not duplicated here"}
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
    # V4-29: incomplete execution fails the command — a matrix with a
    # FAILED cell is not a finished comparison.
    failed = [k for k, v in results.items()
              if isinstance(v, dict) and v.get("status") == "FAILED"]
    if failed:
        print(f"FAILED cells: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
