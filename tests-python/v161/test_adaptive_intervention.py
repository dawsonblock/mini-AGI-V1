"""v4.2.0 — adaptive intervention selection (Phase 6).

Covers the outcome record schema, receipt-gated verification (no
verifier => nothing counts), the empirical selector's scoring
(shrinkage, cost, regression, bounded exploration), and the policy
comparison harness — the adaptive policy must beat fixed/random
baselines on a synthetic stream before an adaptive-benefit claim.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.adaptive_intervention import (  # noqa: E402
    EmpiricalSelector, FixedPolicy, OutcomeRecord, RandomPolicy,
    bandit_simulation, load_outcomes, append_outcome, verify_outcomes)


def _rec(intervention, benefit, regression=0.0, cost=60.0,
         qpass=True):
    return OutcomeRecord(
        intervention=intervention, target_capability="string_transformation",
        evaluation={"within_family_delta": benefit,
                    "cross_family_delta": 0.0,
                    "retention_regression": regression,
                    "qualification_passed": qpass},
        resources={"training_seconds": cost, "gpu_seconds": 0.0,
                   "evaluation_seconds": 10.0},
        evidence_receipt_digest="sha256:" + "ab" * 32)


def test_outcome_record_requires_evaluation_fields():
    with pytest.raises(ValueError, match="within_family_delta"):
        OutcomeRecord(
            intervention="fixed_lora", target_capability="t",
            evaluation={}, resources={"training_seconds": 1,
                                      "gpu_seconds": 0,
                                      "evaluation_seconds": 0},
            evidence_receipt_digest="sha256:x")
    with pytest.raises(ValueError, match="receipt"):
        OutcomeRecord(
            intervention="fixed_lora", target_capability="t",
            evaluation={"within_family_delta": 0,
                        "cross_family_delta": 0,
                        "retention_regression": 0,
                        "qualification_passed": False},
            resources={"training_seconds": 1, "gpu_seconds": 0,
                       "evaluation_seconds": 0},
            evidence_receipt_digest="")


def test_verification_fails_closed_without_verifier():
    recs = [_rec("fixed_lora", 0.3)]
    v = verify_outcomes(recs, None)
    assert not v["verified"] and len(v["unverified"]) == 1
    v = verify_outcomes(recs, lambda r: True)
    assert len(v["verified"]) == 1
    v = verify_outcomes(recs, lambda r: (_ for _ in ()).throw(
        RuntimeError("boom")))          # a broken verifier is not evidence
    assert len(v["verified"]) == 0


def test_selector_prefers_verified_good_arm():
    sel = EmpiricalSelector()
    verified = [_rec("fixed_lora", 0.30) for _ in range(4)] + \
               [_rec("rehearsal", 0.02) for _ in range(4)]
    out = sel.select(["fixed_lora", "rehearsal", "no_intervention"],
                     verified)
    assert out["selected"] == "fixed_lora"
    ests = {e["intervention"]: e for e in out["estimates"]}
    assert ests["fixed_lora"]["score"] > ests["rehearsal"]["score"]


def test_selector_cost_normalization():
    sel = EmpiricalSelector(cost_weight=0.5, cost_scale=1000.0)
    cheap = [_rec("cheap", 0.2, cost=10.0) for _ in range(3)]
    costly = [_rec("costly", 0.2, cost=10000.0) for _ in range(3)]
    out = sel.select(["cheap", "costly"], cheap + costly)
    assert out["selected"] == "cheap"


def test_selector_penalizes_regression():
    sel = EmpiricalSelector(risk_penalty=2.0, beta=0.0)
    clean = [_rec("clean", 0.15, regression=0.0) for _ in range(4)]
    dirty = [_rec("dirty", 0.15, regression=0.5) for _ in range(4)]
    out = sel.select(["clean", "dirty"], clean + dirty)
    assert out["selected"] == "clean"


def test_selector_explores_untried_but_bounded():
    sel = EmpiricalSelector(beta=0.1)
    bad = [_rec("bad", -0.1) for _ in range(6)]
    out = sel.select(["bad", "untried"], bad)
    ests = {e["intervention"]: e for e in out["estimates"]}
    # untried arm gets a finite exploration bonus, not infinity — and
    # its shrunken benefit pulls toward the (0) prior, so a mildly bad
    # measured arm still loses to an untried one but not by blowup
    assert ests["untried"]["explore_bonus"] > 0
    assert ests["untried"]["n_verified"] == 0
    assert out["selected"] == "untried"


def test_selector_refuses_when_nothing_beats_no_intervention():
    sel = EmpiricalSelector(beta=0.0)
    bad = [_rec("worse", -0.2) for _ in range(5)]
    out = sel.select(["worse", "no_intervention"], bad)
    assert out["selected"] == "no_intervention"


def test_outcome_store_roundtrip(tmp_path):
    p = tmp_path / "outcomes.jsonl"
    append_outcome(p, _rec("fixed_lora", 0.3))
    append_outcome(p, _rec("rehearsal", 0.1))
    recs = load_outcomes(p)
    assert len(recs) == 2
    assert recs[0].intervention == "fixed_lora"
    assert recs[0].benefit == pytest.approx(0.3)


def test_adaptive_beats_baselines_on_synthetic_stream():
    """G9-shaped evidence: on an environment where one arm dominates,
    the adaptive policy converges to it; fixed-on-worst and random do
    not."""
    true_model = {
        "good": {"benefit": 0.30, "regression": 0.0,
                 "cost_seconds": 300.0, "noise_sd": 0.05},
        "mediocre": {"benefit": 0.05, "regression": 0.0,
                     "cost_seconds": 300.0, "noise_sd": 0.05},
        "harmful": {"benefit": -0.10, "regression": 0.3,
                    "cost_seconds": 300.0, "noise_sd": 0.05},
    }
    res = bandit_simulation(
        true_model,
        {"adaptive": EmpiricalSelector(beta=0.15),
         "fixed_worst": FixedPolicy("harmful"),
         "random": RandomPolicy(0)},
        rounds=60, seed=7)
    rep = res["policies"]
    assert rep["adaptive"]["mean_realized_benefit"] > \
        rep["random"]["mean_realized_benefit"]
    assert rep["adaptive"]["mean_realized_benefit"] > \
        rep["fixed_worst"]["mean_realized_benefit"]
    assert rep["adaptive"]["distinct_arms_tried"] >= 1


def test_decision_records_proposal_only_flag():
    out = EmpiricalSelector().select(
        ["fixed_lora"], [_rec("fixed_lora", 0.5)])
    assert out["candidate_controlled"] is True
