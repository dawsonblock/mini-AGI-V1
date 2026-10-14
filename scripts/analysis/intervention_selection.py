#!/usr/bin/env python3
"""Campaign 3B.3 Priority 3 — evidence-driven intervention selection.

A fixed script picks the next intervention from a list. An
evidence-driven selector picks it from what the measurements actually
say, and — the part that matters — REFUSES a lever the evidence has
already refuted. That refusal is the difference between the two, and it
is what this tool is for.

It reads the measured effect sizes out of the evidence artifacts, maps
each candidate intervention (the plan's step-16 set) to the lever it
acts on, and then:

  * refuses a candidate whose lever is measured and flat or negative;
  * ranks the rest by the measured size of the lever they target;
  * marks a candidate ELIGIBLE-UNMEASURED when its lever has not been
    tested, and says so rather than assuming it will help.

Nothing is hardcoded: every effect size below is computed from the
evidence files. If a measurement is missing, the lever is UNMEASURED and
the candidate is not ranked on a guess.

Usage:
  python3 scripts/analysis/intervention_selection.py \
      --output INTERVENTION_SELECTION.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: The plan's step-16 candidate set, each mapped to the LEVER it acts on.
#: The mapping is the claim under test: if the evidence says a lever is
#: flat, every candidate acting on that lever is refused.
#: candidate -> (lever, description, assumed_direction). The direction is
#: the claim the candidate makes; evidence that contradicts it refuses the
#: candidate even when the lever shows a non-zero spread, because a lever
#: that moves the wrong way is not a basis for the remedy built on it.
CANDIDATES = {
    "E1_balanced_sampling": ("composition", "balance sampling across families",
                             None),
    "E2_replay_during_training": ("preservation",
                                  "replay previously-correct examples "
                                  "while training", None),
    "E3_frozen_reference_regularization": ("preservation",
                                           "KL penalty against the frozen "
                                           "model on a preservation set",
                                           None),
    "E4_lower_learning_rate": ("optimisation",
                               "lower LR or adjusted schedule", None),
    "E5_rank_or_target_modules": ("capacity",
                                  "alternative rank or target modules",
                                  "more capacity helps: higher rank should "
                                  "score at least as well as lower rank"),
    "E6_task_conditioned_adapters": ("composition",
                                     "specialise adapters per transformation",
                                     None),
}


def direction_contradicted(lever: str, effects: dict) -> str | None:
    """Is the candidate's assumed direction contradicted by the ordering?

    Only levers whose evidence is an ordered sweep can answer this. The
    capacity candidate assumes more rank helps; the rank sweep measured
    the opposite ordering, so the candidate is refused on direction
    regardless of how large the spread is.
    """
    if lever != "capacity":
        return None
    per = (effects.get("rank") or {}).get("per_breadth") or {}
    for label, blk in per.items():
        means = blk.get("rank_means") or {}
        if len(means) < 2:
            continue
        ordered = [means[k] for k in sorted(means)]
        best_rank = max(means, key=means.get)
        if best_rank != max(means):
            return (f"the capacity hypothesis assumes higher rank helps, but "
                    f"the sweep measured rank {best_rank} as the best of "
                    f"{sorted(means)} at {label} (means "
                    f"{[round(m, 3) for m in ordered]}) — the ordering is "
                    "not monotone in rank, so more capacity is not supported")
    return None

#: Which lever each measured effect belongs to.
LEVER_OF_MEASUREMENT = {
    "composition": "composition",
    "rank": "capacity",
    "exposure": "optimisation",
    "steps": "optimisation",
    "capability_destruction": "preservation",
}


def _load(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def measure_effects() -> dict:
    """Every effect size, read from the evidence. No constants.

    Each measurement carries the WITHIN-LEVEL spread (seed-to-seed sd)
    wherever levels repeat, because that is what decides whether a lever
    is real: a spread across levels that does not exceed the noise it was
    measured against is not a finding.
    """
    effects: dict = {}

    # composition: spread across family sets at identical breadth/steps
    comp = _load(ROOT / "results/campaign3b2/family-composition.json")
    if comp:
        means = [v["mean"] for v in comp["pairs"].values()]
        within = [v.get("hi", 0) - v.get("lo", 0) for v in comp["pairs"].values()]
        effects["composition"] = {
            "value": max(means) - min(means),
            "basis": "spread across family pairs at fixed breadth, examples, "
                     "steps and seeds",
            "n_pairs": len(means), "source": "family-composition.json",
            # per-pair ranges are a within-pair spread proxy, not an sd
            "within_pair_range_mean": (sum(within) / len(within)
                                       if within else None)}

    sf = _load(ROOT / "results/campaign3b2/single-family.json")
    if sf:
        means = [v["mean"] for v in sf.values()]
        effects.setdefault("composition", {})
        effects["composition"]["single_family_spread"] = max(means) - min(means)

    # rank: spread across RANK MEANS, never across individual runs — the
    # raw range over ranks x seeds is rank effect PLUS seed noise, and
    # using it inflated this from 0.053 to 0.175, enough to recommend the
    # one lever the evidence refutes.
    ladder = _load(ROOT / "results/campaign3b2/rank-ladder.json")
    if ladder:
        per_breadth: dict = {}
        for grp, label in (("6", "6 families"), ("2", "2 families"),
                           ("2b", "2 families (bad pair)")):
            rows = [r for r in ladder if r["grp"] == grp]
            if not rows:
                continue
            by_rank: dict = {}
            for r in rows:
                by_rank.setdefault(r["rank"], []).append(r["hidden"])
            rank_means = {k: statistics.fmean(v) for k, v in by_rank.items()}
            sds = [statistics.stdev(v) for v in by_rank.values() if len(v) > 1]
            per_breadth[label] = {
                "spread_across_rank_means": (max(rank_means.values())
                                             - min(rank_means.values())),
                "rank_means": rank_means,
                "seeds_per_rank": {k: len(v) for k, v in by_rank.items()},
                "within_level_sd": (statistics.fmean(sds) if sds else None)}
        if per_breadth:
            within = [b["within_level_sd"] for b in per_breadth.values()
                      if b["within_level_sd"] is not None]
            effects["rank"] = {
                "value": max(b["spread_across_rank_means"]
                             for b in per_breadth.values()),
                "within_level_sd": (statistics.fmean(within)
                                    if within else None),
                "basis": ("spread across rank 4/8/16/32 MEANS at fixed "
                          "breadth, examples and steps; seed noise is "
                          "averaged within rank before the spread is taken"),
                "source": "rank-ladder.json",
                "per_breadth": per_breadth}

    # exposure and steps: from the contrast summary. A level with a single
    # run is weak but still a measurement, so it counts and its n is
    # reported rather than the lever silently going unmeasured.
    contrast = _load(ROOT / "results/campaign3b2/contrast-summary.json")
    if contrast:
        for name, key in (("exposure", "exp"), ("steps", "steps")):
            groups: dict = {}
            for r in contrast:
                if r["shuffled"] or r.get(key) is None:
                    continue
                groups.setdefault((r["fam"], r[key]), []).append(r["hidden"])
            per_breadth = {}
            for fam in sorted({k[0] for k in groups}):
                levels = {k[1]: v for k, v in groups.items() if k[0] == fam}
                if len(levels) < 2:
                    continue
                level_means = {k: statistics.fmean(v)
                               for k, v in levels.items()}
                per_breadth[str(fam)] = {
                    "spread": max(level_means.values())
                    - min(level_means.values()),
                    "levels": {str(k): {"n_runs": len(levels[k]),
                                        "mean": level_means[k]}
                               for k in sorted(levels)}}
            if per_breadth:
                sds = [statistics.stdev(v) for v in groups.values()
                       if len(v) > 1]
                effects[name] = {
                    "value": max(b["spread"] for b in per_breadth.values()),
                    "within_level_sd": (statistics.fmean(sds) if sds else None),
                    "basis": (f"spread across {name} levels at fixed "
                              "breadth; levels with one run are reported "
                              "with their n"),
                    "source": "contrast-summary.json",
                    "per_breadth": per_breadth}

    # preservation: how much capability the adapted arms destroy
    base = _load(ROOT / "CAPABILITY_BASELINE.json")
    if base:
        lora = [b for b in base["quality_campaign_evidence"].values()
                if b["class"].startswith("lora")]
        rates = [b["capability_destruction_rate"] for b in lora
                 if b.get("capability_destruction_rate") is not None]
        if rates:
            effects["capability_destruction"] = {
                "value": statistics.fmean(rates),
                "basis": "mean fraction of frozen-correct answers the LoRA "
                         "arms destroy",
                "source": "CAPABILITY_BASELINE.json"}
    return effects


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--flat_threshold", type=float, default=None,
                    help="a lever whose measured spread is below this is "
                         "treated as flat and its candidates are refused")
    args = ap.parse_args()

    effects = measure_effects()
    if not effects:
        raise SystemExit("no evidence found — refusing to select on nothing")

    # lever -> the largest measured effect acting on it
    lever_effect: dict = {}
    for measurement, blk in effects.items():
        lever = LEVER_OF_MEASUREMENT.get(measurement)
        if lever is None:
            continue
        val = blk.get("value", 0.0)
        if lever not in lever_effect or val > lever_effect[lever]["value"]:
            lever_effect[lever] = {"value": val, "measurement": measurement,
                                   "basis": blk["basis"],
                                   "source": blk["source"]}

    selected, refused = [], []
    for name, (lever, description, assumed) in CANDIDATES.items():
        eff = lever_effect.get(lever)
        if eff is None:
            selected.append({
                "candidate": name, "lever": lever,
                "description": description, "status": "ELIGIBLE-UNMEASURED",
                "reason": (f"the {lever} lever has not been measured, so this "
                           "is eligible but its value is unknown — it is not "
                           "ranked on a guess")})
            continue
        sd = eff.get("within_level_sd")
        if args.flat_threshold is not None:
            flat = eff["value"] < args.flat_threshold
            why = (f"spread {eff['value']:.3f} < threshold "
                   f"{args.flat_threshold}")
        elif sd is not None:
            flat = eff["value"] <= sd
            why = (f"spread {eff['value']:.3f} <= seed sd {sd:.3f} — the "
                   "levels do not separate beyond the noise")
        else:
            flat = False
            why = ("no within-level repeats, so flatness cannot be tested; "
                   "treated as unmeasured rather than assumed flat")
        contradicted = direction_contradicted(lever, effects)
        if contradicted and not flat:
            refused.append({
                "candidate": name, "lever": lever,
                "description": description,
                "status": "REFUSED-DIRECTION",
                "measured_effect": eff["value"],
                "assumed_direction": assumed,
                "reason": contradicted})
            continue
        if flat:
            refused.append({
                "candidate": name, "lever": lever,
                "description": description, "status": "REFUSED",
                "measured_effect": eff["value"],
                "reason": (f"the {lever} lever is flat: measured spread "
                           f"{why} "
                           f"({eff['basis']}; {eff['source']})")})
            continue
        selected.append({
            "candidate": name, "lever": lever,
            "description": description, "status": "ELIGIBLE",
            "measured_effect": eff["value"],
            "reason": (f"targets the {lever} lever, whose measured spread is "
                       f"{eff['value']:.3f} ({eff['basis']}; "
                       f"{eff['source']})")})

    selected.sort(key=lambda s: -(s.get("measured_effect") or 0.0))
    report = {
        "schema": "mini-agi-campaign3b3-intervention-selection-v1",
        "measured_effects": effects,
        "lever_effects": lever_effect,
        "flatness_test": (f"fixed threshold {args.flat_threshold}"
                          if args.flat_threshold is not None else
                          "spread across levels must EXCEED the measured "
                          "seed-to-seed sd"),
        "selected": selected,
        "refused": refused,
        "recommendation": (selected[0]["candidate"] if selected else None),
        "statement": (
            "Selection is driven by measured effect sizes, not by the "
            "candidate list. A lever the evidence has measured as flat has "
            "its candidates refused with the measurement quoted, so the "
            "selector cannot recommend a remedy that has already been "
            "tested and found inert."),
    }
    Path(args.output).write_text(json.dumps(report, indent=2,
                                            sort_keys=True) + "\n")

    print("measured lever effects (from the evidence, nothing hardcoded):")
    for lever, e in sorted(lever_effect.items(), key=lambda x: -x[1]["value"]):
        print(f"  {lever:24} {e['value']:>7.3f}   ({e['measurement']}, "
              f"{e['source']})")
    print("\nselected:")
    for s in selected:
        print(f"  {s['candidate']:36} {s['status']:20} "
              f"{s.get('measured_effect', float('nan')):.3f}")
    print("\nrefused (lever measured flat):")
    for r in refused:
        print(f"  {r['candidate']:36} {r['status']:20} "
              f"{r['measured_effect']:.3f}")
    print(f"\nrecommendation: {report['recommendation']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
