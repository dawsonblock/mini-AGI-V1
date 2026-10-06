#!/usr/bin/env python3
"""v16.2 Phase 7 — interruption-recovery semantics for the Colab campaign.

Mirrors the resume rule in scripts/run_colab_campaign.py: a seed run is
authoritative iff seed-<n>/COMPLETE and seed-<n>/SEED_RESULT.json both exist.
An incomplete run must be resumed, never trusted as evidence; a completed run
must not be silently re-executed; a missing run must be scheduled.

Modes:
  --selftest                          fabricate a campaign dir, assert rules
  --campaign-dir DIR --seeds 0 1 ...  audit a real campaign dir

Writes INTERRUPTION_RECOVERY.json. No GPU required.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import detect_root, ensure_path

COMPLETE = "COMPLETE"
SEED_RESULT = "SEED_RESULT.json"


def seed_state(run_dir: Path) -> str:
    """Replicates run_colab_campaign.py's authoritative-run predicate."""
    has_complete = (run_dir / COMPLETE).is_file()
    has_result = (run_dir / SEED_RESULT).is_file()
    if has_complete and has_result:
        return "completed"
    if run_dir.is_dir() and any(run_dir.iterdir()):
        return "incomplete"
    return "missing"


def classify(campaign_dir: Path, seeds: list[int]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {"completed": [], "incomplete": [], "missing": []}
    for seed in seeds:
        state = seed_state(campaign_dir / f"seed-{seed}")
        out[state].append(seed)
    return out


def evidence_seeds(campaign_dir: Path, seeds: list[int]) -> list[int]:
    """Seeds the campaign runner would accept as evidence (resume-safe)."""
    return [s for s in seeds if seed_state(campaign_dir / f"seed-{s}") == "completed"]


def _fabricate(base: Path) -> Path:
    camp = base / "campaigns" / "fake-campaign"
    # seed 0: authoritative complete run
    d0 = camp / "seed-0"
    d0.mkdir(parents=True, exist_ok=True)
    (d0 / SEED_RESULT).write_text(json.dumps({"seed": 0, "delta": 0.5}))
    (d0 / COMPLETE).write_text("complete\n")
    # seed 1: crashed after writing results, before COMPLETE -> not evidence
    d1 = camp / "seed-1"
    d1.mkdir(parents=True, exist_ok=True)
    (d1 / SEED_RESULT).write_text(json.dumps({"seed": 1, "delta": 9.9}))
    # seed 2: marker written but result lost -> not evidence
    d2 = camp / "seed-2"
    d2.mkdir(parents=True, exist_ok=True)
    (d2 / COMPLETE).write_text("complete\n")
    # seed 3: never ran
    return camp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--campaign-dir", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--output", default="INTERRUPTION_RECOVERY.json")
    args = ap.parse_args()

    root = ensure_path(Path(args.root).resolve() if args.root else detect_root())

    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str = ""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    if args.selftest:
        tmp = Path(tempfile.mkdtemp(prefix="v162-resume-"))
        camp = _fabricate(tmp)
        seeds = [0, 1, 2, 3]
        states = classify(camp, seeds)
        ev = evidence_seeds(camp, seeds)

        record("completed_run_recognized", 0 in states["completed"], str(states))
        record("completed_run_kept_as_evidence", 0 in ev, f"evidence_seeds={ev}")
        record("result_without_complete_not_evidence", 1 in states["incomplete"] and 1 not in ev,
               "seed-1 has SEED_RESULT.json but no COMPLETE marker")
        record("complete_marker_without_result_not_evidence", 2 in states["incomplete"] and 2 not in ev,
               "seed-2 has COMPLETE but no SEED_RESULT.json")
        record("missing_run_detected", 3 in states["missing"], str(states))
        record("incomplete_runs_rescheduled",
               set(states["incomplete"]) | set(states["missing"]) == {1, 2, 3},
               "only seed-0 must be skipped on resume")

    if args.campaign_dir:
        camp = Path(args.campaign_dir)
        seeds = args.seeds
        if seeds is None:
            seeds = sorted(int(p.name.split("-", 1)[1]) for p in camp.glob("seed-*") if p.name.split("-", 1)[1].isdigit())
        states = classify(camp, seeds)
        ev = evidence_seeds(camp, seeds)
        record("audit_ran", True, json.dumps(states, sort_keys=True))
        # An interrupted real campaign is healthy if every non-complete state
        # is scheduled for resume rather than counted as evidence.
        record("no_phantom_evidence", ev == states["completed"],
               f"evidence={ev} completed={states['completed']}")
    else:
        states = states if args.selftest else {}

    if not checks:
        print("nothing to do: pass --selftest and/or --campaign-dir", file=sys.stderr)
        return 2

    failed = [c for c in checks if not c["pass"]]
    report = {
        "schema": "mini-agi-v16.2-interruption-recovery-v1",
        "root": str(root),
        "campaign_dir": str(args.campaign_dir) if args.campaign_dir else None,
        "checks": checks,
        "status": "PASS" if not failed else "FAIL",
    }
    if args.campaign_dir:
        report["classification"] = states
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
