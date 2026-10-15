#!/usr/bin/env python3
"""Campaign 3B.3 P1 — evidence-driven intervention selection with
uncertainty-aware decision states.

A fixed script picks the next intervention from a list. An
evidence-driven selector picks it from what the measurements actually
say — and, crucially, it must NOT turn noisy, confounded, or missing
evidence into a categorical refusal. Reviewer requirement (3B.3
assessment #4): the selector must distinguish

  REFUTED               measured direction contradicts the
                        intervention's claim AND the effect exceeds
                        the measured noise — evidence *against*.
  CONFOUNDED            the measurement cannot isolate the lever from a
                        covarying factor (e.g. the rank ladder held
                        alpha fixed at 32, so alpha/r update scaling
                        moved inversely with rank — the sweep measured
                        capacity x scaling jointly).
  INCONCLUSIVE          evidence exists but the measured spread does
                        not exceed the seed-to-seed noise, so the
                        lever's effect cannot be distinguished from
                        seed draw.
  INSUFFICIENT-EVIDENCE too few observations to measure at all.
  FAILED-IMPLEMENTATION a mechanism whose only trial failed for reasons
                        that do not test the mechanism itself.
  HIGH-RISK             measured benefit exists but the intervention
                        carries unacceptable regression/destruction
                        risk on the evidence.
  ELIGIBLE-UNMEASURED   lever never measured — eligible on priors only.
  ELIGIBLE              lever measured, not contradicted, not dominant.
  RECOMMENDED           targets the largest lever measured positive
                        beyond noise, with no contradiction and no
                        unresolved confound.

Every effect size is recomputed under the corrected strict oracle
(exact-match-v2) from the per-run RESCORE-STRICT.json overlays — never
from the casefold-era aggregate files.

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

#: The candidate set, each mapped to the LEVER it acts on, with the
#: precise mechanism claim. A refusal/refutation can only bind the claim
#: actually stated here — not the broader mechanism family.
CANDIDATES = {
    "E1_balanced_sampling": {
        "lever": "composition",
        "claim": "balance sampling across families reduces the spread "
                 "between good and bad family sets",
        "direction": None},
    "E2_replay_during_training": {
        "lever": "preservation",
        "claim": ("interleave verified rehearsal examples into the "
                  "GRADIENT stream — rehearsal pairs carry supervised "
                  "loss with a budgeted presentation fraction. This is "
                  "NOT L7's mechanism: L7 injected replay as "
                  "inference-time context and scored 0.083 while "
                  "destroying 4/4 frozen-correct rows; E2 changes what "
                  "the optimizer minimizes, not what the prompt "
                  "contains"),
        "direction": None},
    "E3_frozen_reference_regularization": {
        "lever": "preservation",
        "claim": "KL penalty against the frozen model on a preservation "
                 "prompt set — directly opposes capability drift",
        "direction": None},
    "E4_lower_learning_rate": {
        "lever": "optimisation",
        "claim": "lower LR or adjusted schedule reduces destruction",
        "direction": None},
    "E5_rank_or_target_modules": {
        "lever": "capacity",
        "claim": "more capacity helps: higher rank should score at "
                 "least as well as lower rank",
        "direction": ("monotone non-decreasing in rank at fixed "
                      "breadth")},
    "E6_task_conditioned_adapters": {
        "lever": "composition",
        "claim": "specialise adapters per transformation family",
        "direction": None},
}

#: Confounds registered against a LEVER'S measurement — facts about the
#: experiment that prevent it isolating the lever.
LEVER_CONFOUNDS = {
    "capacity": ("the rank ladder held alpha fixed at 32 while varying "
                 "rank 4-32, so the effective LoRA scaling alpha/r "
                 "covaried INVERSELY with rank (8.0, 4.0, 2.0, 1.0): "
                 "rank 4 carried the largest update magnitude. The "
                 "sweep measured capacity x update-scale jointly and "
                 "cannot isolate either"),
}


def _load(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def _run_records(runs_root: Path) -> list[dict]:
    """Per-run records pairing CONFIG.json with its strict rescore."""
    out = []
    if not runs_root.is_dir():
        return out
    for d in sorted(runs_root.iterdir()):
        cfg, ov = d / "CONFIG.json", d / "RESCORE-STRICT.json"
        if not (d.is_dir() and cfg.is_file() and ov.is_file()):
            continue
        c, o = _load(cfg), _load(ov)
        ad = o["files"].get("PREDICTIONS-ADAPTED.jsonl", {})
        fr = o["files"].get("PREDICTIONS-FROZEN.jsonl", {})
        sub = o["files"].get("PREDICTIONS-SUBSET.jsonl", {})
        out.append({
            "run_id": d.name,
            "families": tuple(c.get("families", [])),
            "n_families": len(c.get("families", [])),
            "rank": c.get("rank"), "steps": c.get("steps"),
            "seed": c.get("seed"),
            "shuffled": bool(c.get("shuffle_labels")),
            "hidden_v2": ad.get("v2_accuracy"),
            "frozen_v2": fr.get("v2_accuracy"),
            "subset_v2": sub.get("v2_accuracy"),
            "pairwise": o.get("hidden_pairwise_v2")})
    return out


def _famset_key(families) -> str:
    fs = tuple(sorted(families))
    if len(fs) == 6:
        return "6 families"
    if fs == ("pc-rev", "pc-upper"):
        return "2 families (good pair)"
    if fs == ("pc-rot1", "pc-sort"):
        return "2 families (bad pair)"
    if len(fs) == 1:
        return f"single {fs[0]}"
    return f"{len(fs)} families: {','.join(fs)}"


def measure_effects() -> dict:
    """Effect sizes recomputed under the strict oracle from per-run
    RESCORE-STRICT.json overlays, plus the published campaign rescore.
    Each measurement carries its within-level seed sd where levels
    repeat — that is what separates a lever from noise. Keys starting
    with ``_`` carry metadata, not measurements."""
    effects: dict = {"_source": "per-run RESCORE-STRICT.json overlays "
                                "+ RESCORE_SUMMARY.json"}
    runs = _run_records(ROOT / "results/campaign3b2/runs")
    if runs:
        effects["_n_runs_rescored"] = len(runs)

    # ---- capacity: rank sweep under strict scoring ----------------
    rank_groups: dict = {}
    for r in runs:
        if r["shuffled"] or r["rank"] is None or r["hidden_v2"] is None:
            continue
        rank_groups.setdefault((_famset_key(r["families"]), r["rank"]),
                               []).append(r["hidden_v2"])
    per_breadth: dict = {}
    fams = sorted({k[0] for k in rank_groups})
    for fam in fams:
        by_rank = {rk: v for (f, rk), v in rank_groups.items()
                   if f == fam}
        if len(by_rank) < 2:
            continue
        rank_means = {k: statistics.fmean(v) for k, v in by_rank.items()}
        sds = [statistics.stdev(v) for v in by_rank.values() if len(v) > 1]
        per_breadth[fam] = {
            "rank_means": rank_means,
            "spread_across_rank_means": (max(rank_means.values())
                                         - min(rank_means.values())),
            "seeds_per_rank": {k: len(v) for k, v in by_rank.items()},
            "within_level_sd": (statistics.fmean(sds) if sds else None),
            "oracle": "exact-match-v2"}
    if per_breadth:
        within = [b["within_level_sd"] for b in per_breadth.values()
                  if b["within_level_sd"] is not None]
        effects["rank"] = {
            "value": max(b["spread_across_rank_means"]
                         for b in per_breadth.values()),
            "within_level_sd": (statistics.fmean(within)
                                if within else None),
            "confound": LEVER_CONFOUNDS["capacity"],
            "source": "results/campaign3b2/runs/*/RESCORE-STRICT.json",
            "basis": ("spread across rank MEANS at fixed breadth under "
                      "strict scoring; seed noise averaged within rank"),
            "per_breadth": per_breadth}

    # ---- composition: spread across family sets --------------------
    comp_groups: dict = {}
    for r in runs:
        if r["shuffled"] or r["hidden_v2"] is None:
            continue
        comp_groups.setdefault(_famset_key(r["families"]),
                               []).append(r["hidden_v2"])
    comp_means = {k: statistics.fmean(v)
                  for k, v in comp_groups.items() if v}
    if len(comp_means) >= 2:
        sds = [statistics.stdev(v) for v in comp_groups.values()
               if len(v) > 1]
        effects["composition"] = {
            "value": max(comp_means.values()) - min(comp_means.values()),
            "within_level_sd": (statistics.fmean(sds) if sds else None),
            "source": "results/campaign3b2/runs/*/RESCORE-STRICT.json",
            "basis": ("spread across family-set means under strict "
                      "scoring (train blocks byte-identical)"),
            "set_means": comp_means,
            "n_sets": len(comp_means)}

    # ---- optimisation: steps / exposure ---------------------------
    for name, key in (("steps", "steps"), ("exposure", "exposure")):
        groups: dict = {}
        for r in runs:
            if r["shuffled"] or r.get(key) is None \
                    or r["hidden_v2"] is None:
                continue
            groups.setdefault((r["n_families"], r[key]),
                              []).append(r["hidden_v2"])
        per_breadth = {}
        for fam_n in sorted({k[0] for k in groups}):
            levels = {k[1]: v for k, v in groups.items()
                      if k[0] == fam_n}
            if len(levels) < 2:
                continue
            means = {k: statistics.fmean(v) for k, v in levels.items()}
            per_breadth[str(fam_n)] = {
                "spread": max(means.values()) - min(means.values()),
                "levels": {str(k): {"n_runs": len(levels[k]),
                                    "mean": means[k]}
                           for k in sorted(levels)}}
        if per_breadth:
            effects[name] = {
                "value": max(b["spread"] for b in per_breadth.values()),
                "within_level_sd": None,
                "source":
                    "results/campaign3b2/runs/*/RESCORE-STRICT.json",
                "basis": f"spread across {key} levels at fixed breadth, "
                         "strict scoring",
                "per_breadth": per_breadth}

    # ---- preservation: strict capability destruction --------------
    pair_nets, pair_rates = [], []
    for r in runs:
        pw = r.get("pairwise")
        if not pw or r["shuffled"]:
            continue
        n_fc = pw.get("n_frozen_correct") or 0
        if n_fc:
            pair_rates.append(pw["n_destroyed"] / n_fc)
        pair_nets.append(pw["net"])
    if pair_rates:
        effects["preservation"] = {
            "value": statistics.fmean(pair_rates),
            "within_level_sd": (statistics.stdev(pair_rates)
                                if len(pair_rates) > 1 else None),
            "net_gain_mean": statistics.fmean(pair_nets),
            "source": "results/campaign3b2/runs/*/RESCORE-STRICT.json",
            "basis": ("mean fraction of frozen-correct hidden answers "
                      "destroyed, strict oracle, per-run pairwise"),
            "n_runs": len(pair_rates)}

    # ---- trial history per candidate mechanism ----------------------
    # Which measured conditions tested which candidate mechanism, and
    # whether those trials cleared their implementation gates — a
    # mechanism whose only trials failed implementation was never
    # actually tested (FAILED-IMPLEMENTATION), which is different from
    # both refutation and inconclusiveness.
    trial: dict = {}
    rank_trials = [r for r in runs
                   if not r["shuffled"] and r["rank"] is not None]
    if rank_trials:
        fits = [r["subset_v2"] for r in rank_trials
                if r["subset_v2"] is not None]
        trial["E5_rank_or_target_modules"] = {
            "n_trials": len(rank_trials),
            "n_fit": sum(1 for f in fits if f is not None and f >= 0.95),
            "fit_gate": 0.95,
            "note": "rank trials reached fit >= 0.95, so the mechanism "
                    "was exercised — the trial tested capacity (see "
                    "confound on the capacity measurement)"}
        effects["_trial_history"] = trial

    # ---- memory/replay (inference-time): the published campaign ----
    rescore = _load(ROOT / "results/campaign3b2/RESCORE_SUMMARY.json")
    if rescore and "campaign3b" in rescore:
        arms = rescore["campaign3b"].get("arms", {})
        l5b = arms.get("L5b", {}).get("paired_vs_L1_v2")
        if l5b:
            effects["memory_replay_inference"] = {
                "value": l5b["mean_delta"],
                "gained": l5b["gained"], "destroyed": l5b["destroyed"],
                "sign_test_p": l5b["sign_test_p"],
                "basis": ("L5b verified replay vs frozen under strict "
                          "scoring — direction positive but ONE net row "
                          "in 12: statistically null (exact sign test)"),
                "source": "results/campaign3b2/RESCORE_SUMMARY.json"}
    return effects


LEVER_OF_MEASUREMENT = {
    "composition": "composition",
    "rank": "capacity",
    "steps": "optimisation",
    "exposure": "optimisation",
    "preservation": "preservation",
    "memory_replay_inference": "memory_replay",
}


def _classify(candidate: str, spec: dict, eff: dict | None,
              effects: dict) -> dict:
    """Assign one decision state to a candidate from the measured lever
    evidence. The taxonomy is the deliverable: each state names a
    DIFFERENT kind of evidence situation, and every state below is
    reachable — pinned by tests-python/v161/test_intervention_states.py.
    """
    lever = spec["lever"]
    base = {"candidate": candidate, "lever": lever,
            "claim": spec["claim"], "priority": "normal"}

    # 1. lever never measured
    if eff is None:
        return {**base, "state": "ELIGIBLE-UNMEASURED",
                "reason": (f"the {lever} lever has not been measured — "
                           "eligible on priors only, not ranked on a "
                           "guess")}

    # 2. candidate's own trials all failed implementation gates — the
    #    mechanism itself was never exercised
    hist = (effects.get("_trial_history") or {}).get(candidate)
    if hist is not None and hist.get("n_trials", 0) > 0 \
            and hist.get("n_fit", 0) == 0:
        return {**base, "state": "FAILED-IMPLEMENTATION",
                "measured_effect": eff["value"],
                "reason": (f"{hist['n_trials']} trial(s) of this "
                           "mechanism exist but none cleared the fit "
                           f"gate ({hist['fit_gate']}) — a failed "
                           "implementation is not evidence against the "
                           "mechanism; fix the implementation before "
                           "judging it")}

    # 3. measurement confounded — cannot isolate the lever at all
    if eff.get("confound"):
        return {**base, "state": "CONFOUNDED", "priority": "low",
                "measured_effect": eff["value"],
                "reason": (f"the {lever} measurement cannot isolate the "
                           f"lever: {eff['confound']}. The narrow claim "
                           f"({spec.get('direction') or spec['claim']}) "
                           "is unsupported, but this does NOT exclude "
                           "the mechanism family"),
                "preserves": ["rank allocation at fixed scaling",
                              "alpha/r scaling sweep",
                              "dynamic LoRA"]}

    # 4. measured, but too little evidence to call it either way
    if eff.get("n_runs") is not None and eff["n_runs"] < 2:
        return {**base, "state": "INSUFFICIENT-EVIDENCE",
                "priority": "low", "measured_effect": eff["value"],
                "reason": (f"only {eff['n_runs']} run(s) measured this "
                           "lever — direction unknown, not refuted")}

    # 5. spread within seed noise — the evidence cannot distinguish the
    #    lever's effect from seed draw
    sd = eff.get("within_level_sd")
    if sd is not None and eff["value"] <= sd:
        return {**base, "state": "INCONCLUSIVE", "priority": "low",
                "measured_effect": eff["value"],
                "reason": (f"measured spread {eff['value']:.3f} does not "
                           f"exceed seed-to-seed sd {sd:.3f} — the lever "
                           "cannot be distinguished from seed draw; "
                           "neither supported nor refuted")}

    # 6. direction contradicted beyond noise — the only true refusal.
    #    Applies to ordered sweeps whose candidate states a direction.
    if spec.get("direction") and eff.get("per_breadth"):
        for label, blk in eff["per_breadth"].items():
            means = blk.get("rank_means") or {}
            if len(means) < 2 or blk.get("within_level_sd") is None:
                continue
            spread = blk.get("spread_across_rank_means", 0.0)
            if spread <= blk["within_level_sd"]:
                continue
            best_rank = max(means, key=means.get)
            if best_rank != max(means):
                return {**base, "state": "REFUTED",
                        "measured_effect": eff["value"],
                        "reason": (f"the claim is that more rank helps, "
                                   f"but at {label} rank {best_rank} "
                                   f"scored best of {sorted(means)} and "
                                   f"the ordering is not monotone — "
                                   f"spread {spread:.3f} exceeds noise "
                                   f"{blk['within_level_sd']:.3f}, so "
                                   "the contradiction is measured, not "
                                   "a noise artifact")}

    # 7. benefit measured but the mechanism's own record shows
    #    unacceptable destruction risk
    risk = eff.get("risk")
    if risk and risk.get("destruction_rate", 0.0) >= 0.5 \
            and eff["value"] > 0 and lever != "preservation":
        return {**base, "state": "HIGH-RISK",
                "measured_effect": eff["value"],
                "reason": (f"measured benefit {eff['value']:.3f} but the "
                           f"mechanism destroyed "
                           f"{risk['destruction_rate']:.0%} of "
                           "frozen-correct answers in prior trials — "
                           "needs a preservation co-intervention, not "
                           "rejection")}

    # 8. clean eligibility; preservation remedies get the measured harm
    #    quoted as their foundation
    if lever == "preservation":
        return {**base, "state": "ELIGIBLE",
                "measured_effect": eff["value"],
                "reason": (f"targets the largest measured effect: "
                           f"{eff['value']:.3f} mean destruction of "
                           f"frozen-correct answers ({eff['basis']}). "
                           "Remedy founded on a measured harm, not a "
                           "hope")}
    return {**base, "state": "ELIGIBLE",
            "measured_effect": eff["value"],
            "reason": (f"targets the {lever} lever (measured "
                       f"{eff['value']:.3f}; {eff['basis']})")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    effects = measure_effects()
    if not effects:
        raise SystemExit("no evidence found — refusing to select on nothing")

    lever_effect = {}
    for meas, blk in effects.items():
        if meas.startswith("_"):
            continue
        lever = LEVER_OF_MEASUREMENT.get(meas)
        if lever is None:
            continue
        val = blk.get("value", 0.0)
        if lever not in lever_effect or val > lever_effect[lever]["value"]:
            lever_effect[lever] = {**blk, "measurement": meas}

    decisions = [(_classify(name, spec, lever_effect.get(spec["lever"]),
                            effects))
                 for name, spec in CANDIDATES.items()]

    # RECOMMENDED = best ELIGIBLE by measured effect
    elig = [d for d in decisions if d["state"] == "ELIGIBLE"]
    rec = None
    if elig:
        best = max(elig, key=lambda d: d["measured_effect"] or 0.0)
        best["state"] = "RECOMMENDED"
        rec = best["candidate"]

    order = {"RECOMMENDED": 0, "ELIGIBLE": 1, "ELIGIBLE-UNMEASURED": 2,
             "HIGH-RISK": 3, "INCONCLUSIVE": 4, "CONFOUNDED": 5,
             "INSUFFICIENT-EVIDENCE": 6, "FAILED-IMPLEMENTATION": 7,
             "REFUTED": 8}
    decisions.sort(key=lambda d: (order[d["state"]], d["candidate"]))

    unmeasured = sorted({s["lever"] for s in CANDIDATES.values()}
                        - set(lever_effect))
    measured = {k: v for k, v in effects.items() if not k.startswith("_")}
    selected = [{k: d[k] for k in ("candidate", "lever", "state",
                                   "measured_effect", "reason")}
                for d in decisions
                if d["state"] in ("RECOMMENDED", "ELIGIBLE",
                                  "ELIGIBLE-UNMEASURED")]
    refused = [{k: d[k] for k in ("candidate", "lever", "state",
                                  "measured_effect", "reason")}
               for d in decisions if d["state"] == "REFUTED"]
    report = {
        "schema": "mini-agi-campaign3b3-intervention-selection-v2",
        "oracle": "exact-match-v2 (strict case-sensitive)",
        "flatness_test": ("a lever's spread must EXCEED the measured "
                          "within-level seed sd before it is treated as "
                          "an effect — noise-level spread is "
                          "INCONCLUSIVE, never a refutation"),
        "decision_states": {k: v for k, v in {
            "REFUTED": "measured direction contradicts the claim beyond noise",
            "CONFOUNDED": "measurement cannot isolate the lever",
            "INCONCLUSIVE": "spread within seed noise",
            "INSUFFICIENT-EVIDENCE": "too few observations",
            "FAILED-IMPLEMENTATION": "mechanism untested by a failed trial",
            "HIGH-RISK": "benefit measured but unacceptable regression risk",
            "ELIGIBLE-UNMEASURED": "lever never measured",
            "ELIGIBLE": "measured, not contradicted",
            "RECOMMENDED": "targets the dominant measured positive lever",
        }.items()},
        "measured_effects": measured,
        "n_runs_rescored": effects.get("_n_runs_rescored", 0),
        "trial_history": effects.get("_trial_history", {}),
        "lever_effects": lever_effect,
        "decisions": decisions,
        "selected": selected,
        "refused": refused,
        "recommendation": rec,
        "selection_complete": not unmeasured,
        "unmeasured_levers": unmeasured,
        "completeness_warning": (
            "" if not unmeasured else
            f"INCOMPLETE SELECTION — levers {unmeasured} carry no "
            "measurement, so their candidates were NOT tested for "
            "refusal; treat this selection as weaker than a complete "
            "one"),
        "statement": (
            "Selection is driven by measured effect sizes under the "
            "strict oracle. Uncertainty is a decision state, not a "
            "refusal: INCONCLUSIVE, CONFOUNDED and INSUFFICIENT-EVIDENCE "
            "all keep a direction open at low priority, and only a "
            "measured effect that contradicts the claim beyond noise "
            "counts as REFUTED."),
    }
    Path(args.output).write_text(json.dumps(report, indent=2,
                                            sort_keys=True) + "\n")

    print("measured lever effects (strict oracle):")
    for lever, e in sorted(lever_effect.items(),
                           key=lambda x: -(x[1]["value"] or 0)):
        extra = f"  sd={e['within_level_sd']:.3f}" \
            if e.get("within_level_sd") is not None else ""
        conf = "  [CONFOUNDED]" if e.get("confound") else ""
        print(f"  {lever:22} {e['value']:>7.3f}{extra}   "
              f"({e['measurement']}){conf}")
    print("\ndecisions:")
    for d in decisions:
        me = d.get("measured_effect")
        print(f"  {d['candidate']:38} {d['state']:22} "
              + (f"{me:.3f}" if me is not None else "-"))
    print(f"\nrecommendation: {rec}")
    if unmeasured:
        print(f"unmeasured levers: {unmeasured}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
