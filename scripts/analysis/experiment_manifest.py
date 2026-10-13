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

    def by_breadth(rows, n):
        hs = [r["hidden_adapted"] for r in rows if r["n_families"] == n]
        return ({"n": len(hs), "mean": statistics.fmean(hs),
                 "min": min(hs), "max": max(hs),
                 "above_frozen": sum(1 for h in hs if h > frozen)}
                if hs else None)

    exp_manifest = {
        "schema": "mini-agi-campaign3b2-experiment-manifest-v1",
        "run_root": args.run_root,
        "corpus": args.corpus,
        "frozen_hidden_baseline": frozen,
        "n_runs": len(experiments),
        "n_correct_label_runs": len(correct),
        "n_shuffled_control_runs": len(shuffled),
        "breadth_summary_correct_labels": {
            str(n): by_breadth(correct, n) for n in (2, 4, 6)},
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
          f"{len(shuffled)} shuffled) | frozen baseline {frozen:.3f}")
    for n in (2, 4, 6):
        s = exp_manifest["breadth_summary_correct_labels"][str(n)]
        if s:
            print(f"  {n} families: n={s['n']} mean={s['mean']:.3f} "
                  f"[{s['min']:.3f}-{s['max']:.3f}] "
                  f"above-frozen {s['above_frozen']}/{s['n']}")
    print(f"artifacts: {len(artifacts)} "
          f"({art_manifest['total_bytes'] / 1e6:.1f} MB, "
          f"{art_manifest['adapters']} adapters)")
    print(f"wrote {args.experiments} and {args.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
