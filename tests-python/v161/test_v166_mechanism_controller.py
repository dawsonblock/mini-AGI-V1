"""v16.6.0 Phase 5 — evidence-driven, cost-aware mechanism selection.

The controller ranks candidate mechanisms by a frozen objective
U = ΔQ - λ_C·C - λ_R·R - λ_L·L, applies hard safety constraints before
ranking, refuses weight-adaptation estimates without recorded cheaper
attempts, falls back to a diagnostic experiment when nothing clears the
confidence floor, and records every attempt as a digest-chained
AttemptReceipt.
"""
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer, Ed25519Verifier  # noqa: E402
from minagi.v161.plasticity import (  # noqa: E402
    DIAGNOSTIC_EXPERIMENT, AttemptOutcome, AttemptReceipt,
    FailureEvidence, FailureKind, Mechanism, MechanismDecision,
    MechanismEstimate, ObjectiveWeights, SignedAttemptReceipt, propose,
    rank_candidates, select_mechanism)

D = lambda s: digest(s)  # noqa: E731


def _evaluator():
    """A trusted evaluator: signer + a verifier that knows its key."""
    signer = Ed25519Signer.generate()
    verifier = Ed25519Verifier()
    verifier.register(signer.key_id, signer.public_bytes())
    return signer, verifier


def _attempt(signer, ev, mechanism, *, outcome=AttemptOutcome.FAILED.value,
             attempt_id="a1"):
    """A signed attempt receipt bound to `ev` for one ladder rung."""
    return SignedAttemptReceipt.sign(
        signer=signer, attempt_id=attempt_id,
        proposal_digest=D("p"), mechanism=mechanism,
        evidence_digest=ev.digest, task_family_digest=D("family"),
        candidate_digest=D("cand"), outcome=outcome,
        observed_gain=0.0, observed_cost=0.1,
        evaluation_bundle_digest=D("bundle"),
        evaluator_identity="evaluator-1")


def _ev(**kw):
    base = dict(task_digest=D("task"), failure_signature=D("sig"),
                retrieval_found=False, retrieval_helped=False,
                repeats_prior_signature=False, harness_error=False)
    base.update(kw)
    return FailureEvidence(**base)


def _est(mechanism, gain, cost, confidence=1.0, risk=0.0, latency=0.0):
    return MechanismEstimate(
        mechanism=mechanism, predicted_gain=gain, expected_cost=cost,
        confidence=confidence, regression_risk=risk, latency_cost=latency)


W = ObjectiveWeights(lambda_cost=1.0, lambda_risk=1.0, lambda_latency=1.0)


# ---------- estimates and objective -------------------------------------

def test_utility_arithmetic():
    est = _est(Mechanism.RETRIEVAL.value, gain=0.30, cost=0.05,
               risk=0.02, latency=0.01)
    assert W.utility(est) == pytest.approx(0.30 - 0.05 - 0.02 - 0.01)


def test_estimate_validation():
    with pytest.raises(ValueError, match="not a valid Mechanism"):
        _est("telepathy", 1.0, 0.0)
    with pytest.raises(ValueError, match="confidence"):
        _est(Mechanism.RETRIEVAL.value, 1.0, 0.0, confidence=1.5)
    with pytest.raises(ValueError, match="regression_risk"):
        _est(Mechanism.RETRIEVAL.value, 1.0, 0.0, risk=-0.1)
    with pytest.raises(ValueError, match=">= 0"):
        _est(Mechanism.RETRIEVAL.value, 1.0, -0.5)
    with pytest.raises(ValueError, match="finite"):
        _est(Mechanism.RETRIEVAL.value, float("nan"), 0.0)


def test_objective_weights_are_frozen_and_validated():
    with pytest.raises(ValueError, match="lambda"):
        ObjectiveWeights(lambda_cost=-1.0)
    with pytest.raises(FrozenInstanceError):
        W.lambda_cost = 5.0  # type: ignore[misc]
    assert W.digest.startswith("sha256:")


# ---------- ranking ------------------------------------------------------

def test_ranking_orders_by_utility_with_deterministic_ties():
    low = _est(Mechanism.SKILL.value, gain=0.10, cost=0.10)
    high = _est(Mechanism.RETRIEVAL.value, gain=0.40, cost=0.05)
    tie_a = _est(Mechanism.HARNESS.value, gain=0.20, cost=0.10)
    tie_b = _est(Mechanism.SKILL.value, gain=0.20, cost=0.10)
    ranked, excluded = rank_candidates([low, tie_a, high, tie_b], W)
    assert excluded == []
    assert [e.mechanism for e in ranked] == [
        Mechanism.RETRIEVAL.value,      # U = 0.35
        Mechanism.SKILL.value,          # U = 0.10, ladder before harness
        Mechanism.HARNESS.value,        # U = 0.10, ladder after skill
        Mechanism.SKILL.value]          # U = 0.00 (duplicate mech, lowest)
    # rerunning yields the identical order (deterministic tie-break)
    ranked2, _ = rank_candidates([low, tie_a, high, tie_b], W)
    assert [e.digest for e in ranked] == [e.digest for e in ranked2]


def test_hard_safety_cap_excludes_before_ranking():
    """A high-utility candidate above the risk cap is excluded — safety
    is not traded for utility."""
    risky = _est(Mechanism.WEIGHTS.value, gain=5.0, cost=0.0, risk=0.9)
    safe = _est(Mechanism.RETRIEVAL.value, gain=0.05, cost=0.01)
    ranked, excluded = rank_candidates([risky, safe], W,
                                       max_regression_risk=0.25)
    assert [e.mechanism for e in ranked] == [Mechanism.RETRIEVAL.value]
    assert excluded == [(Mechanism.WEIGHTS.value,
                         "regression_risk above hard cap")]


def test_confidence_floor_excludes():
    unsure = _est(Mechanism.SKILL.value, gain=1.0, cost=0.0, confidence=0.2)
    sure = _est(Mechanism.RETRIEVAL.value, gain=0.1, cost=0.0,
                confidence=0.9)
    ranked, excluded = rank_candidates([unsure, sure], W,
                                       min_confidence=0.5)
    assert [e.mechanism for e in ranked] == [Mechanism.RETRIEVAL.value]
    assert excluded == [(Mechanism.SKILL.value, "confidence below floor")]


# ---------- selection ----------------------------------------------------

def test_weights_estimates_require_prior_attempts():
    ev = _ev()  # diagnoses to WEIGHTS
    signer, verifier = _evaluator()
    estimates = [_est(Mechanism.WEIGHTS.value, 0.5, 0.2)]
    with pytest.raises(ValueError, match="verified attempts"):
        select_mechanism(ev, estimates, W)
    decision = select_mechanism(
        ev, estimates, W,
        attempts=(_attempt(signer, ev, Mechanism.RETRIEVAL.value,
                           attempt_id="a1"),
                  _attempt(signer, ev, Mechanism.SKILL.value,
                           attempt_id="a2")),
        evaluator_verifier=verifier)
    assert decision.chosen == Mechanism.WEIGHTS.value


def test_diagnostic_selection_when_nothing_clears_floor():
    ev = _ev(repeats_prior_signature=True)
    unsure = _est(Mechanism.SKILL.value, gain=1.0, cost=0.0, confidence=0.1)
    decision = select_mechanism(ev, [unsure], W, min_confidence=0.5)
    assert decision.chosen == DIAGNOSTIC_EXPERIMENT
    assert decision.ranked == ()
    assert decision.excluded == ((Mechanism.SKILL.value,
                                  "confidence below floor"),)
    assert "diagnostic experiment" in decision.rationale


def test_decision_is_deterministic_and_digest_bound():
    ev = _ev(retrieval_found=True, retrieval_helped=True)
    estimates = [_est(Mechanism.RETRIEVAL.value, 0.3, 0.05),
                 _est(Mechanism.SKILL.value, 0.2, 0.10)]
    d1 = select_mechanism(ev, estimates, W)
    d2 = select_mechanism(ev, estimates, W)
    assert d1.digest == d2.digest
    assert d1.chosen == Mechanism.RETRIEVAL.value
    assert d1.failure_kind == FailureKind.MISSING_INFORMATION.value
    assert d1.utility == pytest.approx(0.25)
    assert isinstance(d1, MechanismDecision)


def test_selection_binds_evidence_and_ladder_for_proposals():
    """The controller decision and the ladder-enforcing proposal agree on
    the same evidence, and the attempt receipt chains both by digest."""
    ev = _ev()
    signer, verifier = _evaluator()
    attempts = (_attempt(signer, ev, Mechanism.RETRIEVAL.value,
                       attempt_id="a1"),
                _attempt(signer, ev, Mechanism.SKILL.value,
                         attempt_id="a2"))
    proposal = propose(ev, proposal_id="p1",
                       candidate_config_digest=D("cfg"),
                       baseline_digest=D("base"),
                       prior_attempts=tuple(r.digest for r in attempts))
    decision = select_mechanism(
        ev, [_est(Mechanism.WEIGHTS.value, 0.4, 0.3)], W,
        attempts=attempts, evaluator_verifier=verifier)
    assert decision.chosen == proposal.mechanism
    receipt = AttemptReceipt(
        attempt_id="a1", proposal_digest=proposal.digest,
        mechanism=decision.chosen, evidence_digest=ev.digest,
        outcome=AttemptOutcome.IMPROVED.value, observed_gain=0.21,
        cost=0.30, evaluation_bundle_digest=D("bundle"))
    assert receipt.digest.startswith("sha256:")
    assert receipt.proposal_digest == proposal.digest
    assert receipt.evidence_digest == ev.digest


# ---------- attempt receipts --------------------------------------------

def test_attempt_receipt_validation():
    with pytest.raises(ValueError, match="attempt_id"):
        AttemptReceipt(attempt_id="", proposal_digest=D("p"),
                       mechanism=Mechanism.RETRIEVAL.value,
                       evidence_digest=D("e"), outcome="improved",
                       observed_gain=0.1, cost=0.1,
                       evaluation_bundle_digest=D("b"))
    with pytest.raises(ValueError, match="outcome"):
        AttemptReceipt(attempt_id="a", proposal_digest=D("p"),
                       mechanism=Mechanism.RETRIEVAL.value,
                       evidence_digest=D("e"), outcome="vibes",
                       observed_gain=0.1, cost=0.1,
                       evaluation_bundle_digest=D("b"))
    with pytest.raises(ValueError, match="invalid sha256 digest"):
        AttemptReceipt(attempt_id="a", proposal_digest="not-a-digest",
                       mechanism=Mechanism.RETRIEVAL.value,
                       evidence_digest=D("e"), outcome="improved",
                       observed_gain=0.1, cost=0.1,
                       evaluation_bundle_digest=D("b"))
    with pytest.raises(ValueError, match="cost"):
        AttemptReceipt(attempt_id="a", proposal_digest=D("p"),
                       mechanism=Mechanism.RETRIEVAL.value,
                       evidence_digest=D("e"), outcome="failed",
                       observed_gain=0.0, cost=-1.0,
                       evaluation_bundle_digest=D("b"))
    diag = AttemptReceipt(
        attempt_id="d", proposal_digest=D("p"),
        mechanism=DIAGNOSTIC_EXPERIMENT, evidence_digest=D("e"),
        outcome=AttemptOutcome.UNCHANGED.value, observed_gain=0.0,
        cost=0.01, evaluation_bundle_digest=D("b"))
    assert diag.mechanism == DIAGNOSTIC_EXPERIMENT
    regressed = AttemptReceipt(
        attempt_id="r", proposal_digest=D("p"),
        mechanism=Mechanism.WEIGHTS.value, evidence_digest=D("e"),
        outcome=AttemptOutcome.REGRESSED.value, observed_gain=-0.4,
        cost=0.5, evaluation_bundle_digest=D("b"))
    assert regressed.observed_gain < 0
