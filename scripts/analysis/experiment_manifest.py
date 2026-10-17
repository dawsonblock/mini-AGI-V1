#!/usr/bin/env python3
"""Campaign 3B.2 Step 27 — machine-readable experiment and artifact manifests.

The review's P2 asks for reports derived from canonical result records
rather than several independently edited summaries. These two manifests
are generated, never hand-written:

  EXPERIMENT_MANIFEST.json  one record per run, read from that run's own
                            CONFIG.json / METRICS.json / RUN_COMMIT.json
  ARTIFACT_MANIFEST.json    every artifact file with its sha256, plus
                            the aggregate roll-up

If a summary table and these manifests disagree, the manifests are right:
they are read from the runs, not from prose.

Usage:
  python3 scripts/analysis/experiment_manifest.py \
      --run-root results/campaign3b2/runs \
      --experiments EXPERIMENT_MANIFEST.json \
      --artifacts ARTIFACT_MANIFEST.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="results/campaign3b2/runs")
    ap.add_argument("--corpus",
                    default="configs/positive_control_tasks.jsonl")
    ap.add_argument("--experiments", required=True)
    ap.add_argument("--artifacts", required=True)
    args = ap.parse_args()

    run_root = ROOT / args.run_root
    if not run_root.is_dir():
        raise SystemExit(f"run root not found: {run_root}")

    frozen = None
    experiments, artifacts = [], []
    for d in sorted(run_root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        commit = json.loads((d / "RUN_COMMIT.json").read_text())
        cfg = json.loads((d / "CONFIG.json").read_text())
        metrics = json.loads((d / "METRICS.json").read_text())
        c = metrics.get("contrast") or {}
        ex = metrics.get("exposure_actual") or {}
        if frozen is None:
            frozen = metrics["frozen"]["hidden"]
        experiments.append({
            "run_id": d.name,
            "status": commit["status"],
            "config_digest": commit.get("config_digest"),
            "attempt_id": commit.get("attempt_id"),
            "created_utc": commit.get("created_utc"),
            "corpus": cfg.get("corpus"),
            "model_id": cfg["model_id"],
            "model_revision": cfg["model_revision"],
            "seed": cfg["seed"],
            "families": cfg["families"],
            "n_families": len(cfg["families"]),
            "supervision": ("shuffled-label" if cfg["shuffle_labels"]
                            else "correct"),
            "allocation": c.get("allocation"),
            "training_examples": c.get("training_examples"),
            "unique_scored_rows": c.get("unique_scored_rows"),
            "steps": c.get("steps"),
            "exposure_target": c.get("exposure_target"),
            "presentations_per_example": c.get("presentations_per_example"),
            "realised_sample_presentations": ex.get("sample_presentations"),
            "realised_optimizer_updates": ex.get("optimizer_updates"),
            "fit_accuracy": metrics["fit_gate"]["adapted_subset_accuracy"],
            "fit_passed": metrics["fit_gate"]["passed"],
            "hidden_frozen": metrics["frozen"]["hidden"],
            "hidden_adapted": metrics["adapted"]["hidden"],
            "destroyed": metrics["capability_destruction"]["n_destroyed"],
            "gained": metrics["capability_destruction"]["n_gained"],
            "destroyed_ids": metrics["capability_destruction"]["destroyed_ids"],
            "gained_ids": metrics["capability_destruction"]["gained_ids"],
            "train_seconds": metrics["training"]["train_seconds"],
            "prompt_contract_digest": cfg["prompt_contract_digest"],
            "output_contract_digest": cfg["output_contract_digest"],
            "effective_decoding": metrics["effective_decoding"],
        })
        for p in sorted(d.rglob("*")):
            if p.is_file():
                artifacts.append({
                    "run_id": d.name,
                    "path": p.relative_to(run_root).as_posix(),
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size})

    if not experiments:
        raise SystemExit(f"no runs found under {run_root}")

    correct = [e for e in experiments if e["supervision"] == "correct"]
    shuffled = [e for e in experiments if e["supervision"] == "shuffled-label"]

    # Runs span more than one benchmark, and the frozen baseline is a
    # property of the BENCHMARK, not of the campaign: v1 scores 0.417 and
    # v2 scores 0.400 on their respective held-out sets. Pooling them
    # under one baseline would make `above_frozen` compare apples to
    # oranges, so every roll-up is grouped by corpus and uses that
    # corpus's own frozen baseline, measured from the runs themselves.
    def by_breadth(rows, n, base):
        hs = [r["hidden_adapted"] for r in rows if r["n_families"] == n]
        return ({"n": len(hs), "mean": statistics.fmean(hs),
                 "min": min(hs), "max": max(hs),
                 "frozen_baseline": base,
                 "above_frozen": sum(1 for h in hs if h > base)}
                if hs else None)

    corpora: dict = {}
    for e in correct:
        corpora.setdefault(str(e.get("corpus") or "unknown"), []).append(e)
    per_corpus = {}
    for name, rows in sorted(corpora.items()):
        bases = {round(r["hidden_frozen"], 6) for r in rows}
        base = min(bases) if len(bases) == 1 else None
        per_corpus[name] = {
            "n_runs": len(rows),
            "frozen_baseline": base,
            "frozen_baseline_note": (
                "consistent across these runs" if base is not None else
                f"NOT consistent across these runs: {sorted(bases)} — "
                "above_frozen is omitted because it would be ill-defined"),
            "by_breadth": {
                str(n): by_breadth(rows, n, base)
                for n in (1, 2, 4, 5, 6) if base is not None},
        }

    exp_manifest = {
        "schema": "mini-agi-campaign3b2-experiment-manifest-v2",
        "run_root": args.run_root,
        "corpus": args.corpus,
        "n_runs": len(experiments),
        "n_correct_label_runs": len(correct),
        "n_shuffled_control_runs": len(shuffled),
        "benchmarks": sorted(corpora),
        "note": ("Roll-ups are grouped BY CORPUS because the frozen "
                 "baseline belongs to the benchmark, not the campaign. A "
                 "single pooled baseline would compare runs evaluated on "
                 "different held-out sets."),
        "correct_labels_by_corpus": per_corpus,
        "runs": experiments,
    }
    Path(args.experiments).write_text(json.dumps(exp_manifest, indent=2,
                                                 sort_keys=True) + "\n")

    art_manifest = {
        "schema": "mini-agi-campaign3b2-artifact-manifest-v1",
        "run_root": args.run_root,
        "n_artifacts": len(artifacts),
        "total_bytes": sum(a["bytes"] for a in artifacts),
        "adapters": sum(1 for a in artifacts
                        if a["path"].endswith("adapter_model.safetensors")),
        "artifacts": artifacts,
    }
    Path(args.artifacts).write_text(json.dumps(art_manifest, indent=2,
                                               sort_keys=True) + "\n")

    print(f"runs: {len(experiments)} ({len(correct)} correct, "
          f"{len(shuffled)} shuffled)")
    for name, blk in per_corpus.items():
        base = blk["frozen_baseline"]
        print(f"  {name}  n={blk['n_runs']}  "
              f"frozen {base if base is None else round(base, 3)}")
        for n, st in blk["by_breadth"].items():
            if st is None:
                continue
            print(f"    {n} families: n={st['n']} mean={st['mean']:.3f} "
                  f"[{st['min']:.3f}-{st['max']:.3f}] "
                  f"above-frozen {st['above_frozen']}/{st['n']}")
    print(f"artifacts: {len(artifacts)} "
          f"({art_manifest['total_bytes'] / 1e6:.1f} MB, "
          f"{art_manifest['adapters']} adapters)")
    print(f"wrote {args.experiments} and {args.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
