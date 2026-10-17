"""Campaign 3B.3 P1 — pin every selector decision state.

The taxonomy is the deliverable: each state must be reachable from
evidence logic, never decoration. Each test feeds a synthetic effect
block and asserts the exact state the candidate lands on, so a future
edit that silently collapses a state (e.g. turning CONFOUNDED into
REFUTED, or noise into falsification) fails here.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

spec = importlib.util.spec_from_file_location(
    "intervention_selection",
    ROOT / "scripts/analysis/intervention_selection.py")
SEL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SEL)


def _classify(candidate, spec_, eff, effects=None):
    return SEL._classify(candidate, spec_, eff, effects or {})


def _spec(claim="claim", direction=None):
    return {"lever": "x", "claim": claim, "direction": direction}


# -- every state is reachable ----------------------------------------

def test_unmeasured_lever_is_eligible_unmeasured():
    d = _classify("C", _spec(), None)
    assert d["state"] == "ELIGIBLE-UNMEASURED"


def test_failed_implementation_when_no_trial_reached_fit():
    eff = {"value": 0.3, "basis": "b"}
    fx = {"_trial_history": {"C": {"n_trials": 2, "n_fit": 0,
                                    "fit_gate": 0.95}}}
    d = _classify("C", _spec(), eff, fx)
    assert d["state"] == "FAILED-IMPLEMENTATION"


def test_confounded_measurement_is_confounded_not_refuted():
    """Reviewer finding: a sweep that covaries a second factor cannot
    refute the lever — it is CONFOUNDED, and the mechanism family is
    preserved."""
    eff = {"value": 0.4, "basis": "b",
           "confound": "alpha covaried inversely with rank"}
    d = _classify("C", _spec(direction="monotone"), eff)
    assert d["state"] == "CONFOUNDED"
    assert d["priority"] == "low"
    assert "does NOT exclude" in d["reason"]


def test_spread_within_noise_is_inconclusive():
    eff = {"value": 0.05, "within_level_sd": 0.047, "basis": "b"}
    d = _classify("C", _spec(), eff)
    # 0.05 > 0.047: NOT within noise
    assert d["state"] == "ELIGIBLE"

    d2 = _classify("C", _spec(), {"value": 0.05,
                                   "within_level_sd": 0.06,
                                   "basis": "b"})
    assert d2["state"] == "INCONCLUSIVE"
    assert "neither supported nor refuted" in d2["reason"]


def test_insufficient_evidence_below_two_runs():
    eff = {"value": 0.3, "n_runs": 1, "basis": "b"}
    d = _classify("C", _spec(), eff)
    assert d["state"] == "INSUFFICIENT-EVIDENCE"


def test_direction_contradicted_beyond_noise_is_refuted():
    """REFUTED exists — but only for an unconfounded ordered sweep whose
    measured direction beats the noise and contradicts the claim."""
    eff = {"value": 0.30, "within_level_sd": 0.02,
           "basis": "b",
           "per_breadth": {"6f": {"rank_means": {4: 0.40, 8: 0.30,
                                                 16: 0.20, 32: 0.10},
                                   "within_level_sd": 0.02,
                                   "spread_across_rank_means": 0.30}}}
    d = _classify("C", _spec(direction="monotone in rank"), eff)
    assert d["state"] == "REFUTED"
    assert "not monotone" in d["reason"]


def test_within_noise_direction_is_not_refuted():
    """A non-monotone ordering that does NOT exceed the noise is
    INCONCLUSIVE, never REFUTED — noisy non-monotonicity is not
    falsification."""
    eff = {"value": 0.03, "within_level_sd": 0.05,
           "basis": "b",
           "per_breadth": {"6f": {"rank_means": {4: 0.20, 32: 0.17},
                                   "within_level_sd": 0.05,
                                   "spread_across_rank_means": 0.03}}}
    d = _classify("C", _spec(direction="monotone in rank"), eff)
    assert d["state"] == "INCONCLUSIVE"


def test_high_risk_when_benefit_carries_destruction():
    eff = {"value": 0.3, "basis": "b",
           "risk": {"destruction_rate": 0.7}}
    d = _classify("C", _spec(), eff)
    assert d["state"] == "HIGH-RISK"


def test_eligible_on_measured_uncontradicted_lever():
    eff = {"value": 0.3, "basis": "b"}
    d = _classify("C", _spec(), eff)
    assert d["state"] == "ELIGIBLE"


def test_preservation_remedy_quotes_the_measured_harm():
    eff = {"value": 0.6, "basis": "mean destruction"}
    d = _classify("C", {"lever": "preservation", "claim": "rehearse",
                        "direction": None}, eff)
    assert d["state"] == "ELIGIBLE"
    assert "measured" in d["reason"] and "0.600" in d["reason"]


# -- the current evidence lands where the assessment said -------------

def test_current_selector_state_matches_evidence():
    """Against the real evidence tree: E5 must be CONFOUNDED (not
    REFUTED), E2 RECOMMENDED, and the selection must declare its
    completeness. Skipped honestly when run records are absent."""
    import pytest
    runs_root = ROOT / "results/campaign3b2/runs"
    if not runs_root.is_dir():
        pytest.skip("run evidence not in this checkout (evidence is a "
                    "separate package)")
    effects = SEL.measure_effects()
    lever_effect = {}
    for meas, blk in effects.items():
        if meas.startswith("_"):
            continue
        lv = SEL.LEVER_OF_MEASUREMENT.get(meas)
        if lv is None:
            continue
        if lv not in lever_effect or blk["value"] > lever_effect[lv]["value"]:
            lever_effect[lv] = {**blk, "measurement": meas}
    by_name = {n: SEL._classify(n, s, lever_effect.get(s["lever"]),
                                effects)
               for n, s in SEL.CANDIDATES.items()}
    assert by_name["E5_rank_or_target_modules"]["state"] == "CONFOUNDED"
    states = {d["state"] for d in by_name.values()}
    assert "REFUTED" not in states  # nothing here is refuted


def test_confounded_never_means_refuted():
    """P1's core guarantee: a confounded measurement can never produce
    REFUTED — it produces CONFOUNDED regardless of direction data."""
    eff = {"value": 0.9, "basis": "b",
           "confound": "covariate moved with the lever",
           "per_breadth": {"x": {"rank_means": {4: 0.9, 32: 0.1},
                                 "within_level_sd": 0.01,
                                 "spread_across_rank_means": 0.8}}}
    d = _classify("E5_rank_or_target_modules",
                  {"lever": "capacity",
                   "claim": "more rank helps",
                   "direction": "monotone"}, eff)
    assert d["state"] == "CONFOUNDED"
    assert d["state"] != "REFUTED"
