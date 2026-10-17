"""v16.4.5 — controller gate (SEC-005/SEC-006 + WP11).

The adversarial acceptance tests the plan requires:

  * fabricated prior learning attempts -> LoRA escalation rejected
    (SEC-005): raw digests, unsigned receipts, untrusted signers,
    receipts bound to other evidence, INVALID outcomes, and wrong
    ladder rungs all fail the verified-attempt gate;
  * all intervention utilities negative -> NO_CHANGE (SEC-006): the
    selector never picks a below-threshold intervention;
  * exceeded budgets -> rejected (WP11): an exhausted experiment
    ledger yields NO_CHANGE, refused charges consume nothing, and a
    breach leaves a signed ExhaustionRecord;
  * determinism: identical frozen evidence + policy produce identical
    decision digests.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer, Ed25519Verifier  # noqa: E402
from minagi.v161.experiment_budget import (  # noqa: E402
    AttemptExecutor, BudgetExhausted, ExperimentBudgetLedger,
    ExperimentBudgetPolicy)
from minagi.v161.plasticity import (  # noqa: E402
    DIAGNOSTIC_EXPERIMENT, NO_CHANGE, AttemptOutcome, AttemptReceipt,
    FailureEvidence, Mechanism, MechanismEstimate, ObjectiveWeights,
    SignedAttemptReceipt, select_mechanism)

D = lambda s: digest(s)  # noqa: E731


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


def _evaluator():
    signer = Ed25519Signer.generate()
    verifier = Ed25519Verifier()
    verifier.register(signer.key_id, signer.public_bytes())
    return signer, verifier


def _attempt(signer, ev, mechanism, *,
             outcome=AttemptOutcome.FAILED.value, attempt_id="a1",
             evidence_digest=None):
    return SignedAttemptReceipt.sign(
        signer=signer, attempt_id=attempt_id,
        proposal_digest=D("p"), mechanism=mechanism,
        evidence_digest=evidence_digest or ev.digest,
        task_family_digest=D("family"), candidate_digest=D("cand"),
        outcome=outcome, observed_gain=0.0, observed_cost=0.1,
        evaluation_bundle_digest=D("bundle"),
        evaluator_identity="evaluator-1")


def _verified_ladder(signer, ev):
    return (_attempt(signer, ev, Mechanism.RETRIEVAL.value,
                     attempt_id="a1"),
            _attempt(signer, ev, Mechanism.SKILL.value,
                     attempt_id="a2"))


W = ObjectiveWeights(lambda_cost=1.0, lambda_risk=1.0, lambda_latency=1.0)


# ---------- SEC-006: safe abstention --------------------------------------

def test_all_negative_utilities_select_no_change():
    """Every candidate has U < 0 — the controller must abstain, not pick
    the least-bad intervention (the SEC-006 defect)."""
    ev = _ev(retrieval_found=True, retrieval_helped=True)
    estimates = [_est(Mechanism.RETRIEVAL.value, gain=0.05, cost=0.10),
                 _est(Mechanism.SKILL.value, gain=0.02, cost=0.20)]
    decision = select_mechanism(ev, estimates, W)
    assert decision.chosen == NO_CHANGE
    assert decision.utility == 0.0
    assert "does not exceed frozen threshold" in decision.rationale


def test_threshold_boundary_is_strict():
    """U == τ does not clear the gate — strictly greater is required."""
    ev = _ev(retrieval_found=True, retrieval_helped=True)
    w = ObjectiveWeights(lambda_cost=1.0, lambda_risk=1.0,
                         lambda_latency=1.0, utility_threshold=0.25)
    at = _est(Mechanism.RETRIEVAL.value, gain=0.30, cost=0.05)  # U = 0.25
    over = _est(Mechanism.SKILL.value, gain=0.40, cost=0.10)    # U = 0.30
    assert select_mechanism(ev, [at], w).chosen == NO_CHANGE
    decision = select_mechanism(ev, [at, over], w)
    assert decision.chosen == Mechanism.SKILL.value
    assert decision.utility == pytest.approx(0.30)


def test_positive_utility_still_selects():
    ev = _ev(retrieval_found=True, retrieval_helped=True)
    estimates = [_est(Mechanism.RETRIEVAL.value, gain=0.40, cost=0.05)]
    assert select_mechanism(ev, estimates, W).chosen == \
        Mechanism.RETRIEVAL.value


def test_no_change_outranks_nothing_cleared_diagnostic():
    """All excluded -> diagnostic (uncertainty); admissible but below τ
    -> NO_CHANGE. The two abstain paths stay distinct."""
    ev = _ev(repeats_prior_signature=True)
    unsure = _est(Mechanism.SKILL.value, gain=1.0, cost=0.0,
                  confidence=0.1)
    assert select_mechanism(ev, [unsure], W,
                            min_confidence=0.5).chosen == \
        DIAGNOSTIC_EXPERIMENT
    weak = _est(Mechanism.SKILL.value, gain=0.01, cost=0.50,
                confidence=0.9)
    assert select_mechanism(ev, [weak], W).chosen == NO_CHANGE


# ---------- SEC-005: verified prerequisite attempts ------------------------

def test_fabricated_digest_strings_do_not_count():
    """A tuple of sha256-looking strings is not an attempt — the exact
    adversarial case from the plan."""
    ev = _ev()
    estimates = [_est(Mechanism.WEIGHTS.value, 0.9, 0.1)]
    _, verifier = _evaluator()
    with pytest.raises(ValueError, match="verified attempts"):
        select_mechanism(ev, estimates, W,
                         attempts=(D("retrieval-attempt"),
                                   D("skill-attempt")),
                         evaluator_verifier=verifier)


def test_unsigned_receipt_does_not_count():
    """A structurally valid but UNSIGNED AttemptReceipt is not evidence."""
    ev = _ev()
    _, verifier = _evaluator()
    unsigned = AttemptReceipt(
        attempt_id="a1", proposal_digest=D("p"),
        mechanism=Mechanism.RETRIEVAL.value, evidence_digest=ev.digest,
        outcome=AttemptOutcome.FAILED.value, observed_gain=0.0, cost=0.1,
        evaluation_bundle_digest=D("b"))
    with pytest.raises(ValueError, match="verified attempts"):
        select_mechanism(ev, [_est(Mechanism.WEIGHTS.value, 0.9, 0.1)], W,
                         attempts=(unsigned,), evaluator_verifier=verifier)


def test_untrusted_evaluator_does_not_count():
    """Signed by an evaluator whose key the verifier does not know."""
    ev = _ev()
    signer, _ = _evaluator()
    other = Ed25519Signer.generate()             # a verifier for a
    other_verifier = Ed25519Verifier()           # DIFFERENT key — the
    other_verifier.register(other.key_id, other.public_bytes())
    attempts = _verified_ladder(signer, ev)
    with pytest.raises(ValueError, match="verified attempts"):
        select_mechanism(ev, [_est(Mechanism.WEIGHTS.value, 0.9, 0.1)], W,
                         attempts=attempts,
                         evaluator_verifier=other_verifier)


def test_receipt_for_other_evidence_does_not_count():
    """A verified attempt on a DIFFERENT failure is not a prerequisite
    for this one — relevance is bound by evidence_digest."""
    ev = _ev()
    other_ev = _ev(task_digest=D("other-task"))
    signer, verifier = _evaluator()
    attempts = (_attempt(signer, ev, Mechanism.RETRIEVAL.value,
                         attempt_id="a1",
                         evidence_digest=other_ev.digest),
                _attempt(signer, ev, Mechanism.SKILL.value,
                         attempt_id="a2",
                         evidence_digest=other_ev.digest))
    with pytest.raises(ValueError, match="verified attempts"):
        select_mechanism(ev, [_est(Mechanism.WEIGHTS.value, 0.9, 0.1)], W,
                         attempts=attempts, evaluator_verifier=verifier)


def test_invalid_outcome_does_not_count():
    """An INVALID attempt is rejected evidence, not an attempt."""
    ev = _ev()
    signer, verifier = _evaluator()
    attempts = (_attempt(signer, ev, Mechanism.RETRIEVAL.value,
                         attempt_id="a1",
                         outcome=AttemptOutcome.INVALID.value),
                _attempt(signer, ev, Mechanism.SKILL.value,
                         attempt_id="a2"))
    with pytest.raises(ValueError, match="verified attempts"):
        select_mechanism(ev, [_est(Mechanism.WEIGHTS.value, 0.9, 0.1)], W,
                         attempts=attempts, evaluator_verifier=verifier)


def test_missing_rung_and_wrong_mechanism_rejected():
    """Coverage is per-rung: a harness receipt cannot stand in for the
    required retrieval/skill rungs, and one rung missing still fails."""
    ev = _ev()
    signer, verifier = _evaluator()
    w_est = [_est(Mechanism.WEIGHTS.value, 0.9, 0.1)]
    wrong = (_attempt(signer, ev, Mechanism.HARNESS.value,
                      attempt_id="a1"),
             _attempt(signer, ev, Mechanism.RETRIEVAL.value,
                      attempt_id="a2"))
    with pytest.raises(ValueError, match="verified attempts"):
        select_mechanism(ev, w_est, W, attempts=wrong,
                         evaluator_verifier=verifier)
    partial = _verified_ladder(signer, ev)[:1]   # retrieval only
    with pytest.raises(ValueError, match="verified attempts"):
        select_mechanism(ev, w_est, W, attempts=partial,
                         evaluator_verifier=verifier)


def test_no_verifier_means_no_weights():
    """Without a trusted evaluator registry the gate cannot verify —
    fail closed, not open."""
    ev = _ev()
    signer, _ = _evaluator()
    with pytest.raises(ValueError, match="verified attempts"):
        select_mechanism(ev, [_est(Mechanism.WEIGHTS.value, 0.9, 0.1)], W,
                         attempts=_verified_ladder(signer, ev),
                         evaluator_verifier=None)


def test_verified_attempts_admit_weights():
    """The honest path: both cheaper rungs verified on this evidence ->
    the weights estimate is admissible and wins on utility."""
    ev = _ev()
    signer, verifier = _evaluator()
    decision = select_mechanism(
        ev, [_est(Mechanism.WEIGHTS.value, 0.9, 0.1)], W,
        attempts=_verified_ladder(signer, ev),
        evaluator_verifier=verifier)
    assert decision.chosen == Mechanism.WEIGHTS.value


def test_signed_receipt_roundtrip_and_tamper():
    """The receipt verifies over its full signed body — any mutated
    field breaks verification."""
    ev = _ev()
    signer, verifier = _evaluator()
    r = _attempt(signer, ev, Mechanism.RETRIEVAL.value)
    assert r.verify(verifier)
    forged = SignedAttemptReceipt(
        attempt_id=r.attempt_id, proposal_digest=r.proposal_digest,
        mechanism=r.mechanism, evidence_digest=r.evidence_digest,
        task_family_digest=r.task_family_digest,
        candidate_digest=r.candidate_digest,
        outcome=AttemptOutcome.IMPROVED.value,   # tampered
        observed_gain=0.9, cost=r.cost,
        evaluation_bundle_digest=r.evaluation_bundle_digest,
        evaluator_identity=r.evaluator_identity,
        signer_key_id=r.signer_key_id, signature_b64=r.signature_b64)
    assert not forged.verify(verifier)


# ---------- WP11: enforceable experiment budgets ---------------------------

def _policy(**kw):
    base = dict(max_attempts=4, max_evaluation_calls=8,
                wall_clock_seconds=600.0, gpu_seconds=300.0,
                memory_bytes=1 << 30, spend_ceiling=10.0)
    base.update(kw)
    return ExperimentBudgetPolicy(**base)


def test_exhausted_budget_forces_no_change():
    """A high-utility intervention cannot run when the experiment
    budget is spent — budgets are enforced, not reported."""
    ev = _ev(retrieval_found=True, retrieval_helped=True)
    ledger = ExperimentBudgetLedger(_policy(max_attempts=1))
    ledger.charge(attempts=1)
    assert ledger.exhausted
    decision = select_mechanism(
        ev, [_est(Mechanism.RETRIEVAL.value, 0.9, 0.01)], W,
        experiment_budget=ledger)
    assert decision.chosen == NO_CHANGE
    assert "budget exhausted" in decision.rationale


def test_charge_is_atomic_on_breach():
    """A refused charge consumes NOTHING — retries cannot build on
    partial spend."""
    ledger = ExperimentBudgetLedger(_policy(max_evaluation_calls=4))
    ledger.charge(evaluation_calls=3, gpu_seconds=10.0)
    with pytest.raises(BudgetExhausted):
        ledger.charge(evaluation_calls=2, gpu_seconds=50.0)
    assert ledger.spent("evaluation_calls") == 3.0
    assert ledger.spent("gpu_seconds") == 10.0


def test_executor_refuses_when_attempts_spent():
    """Attempt #5 under a 4-attempt policy: refused before work starts
    and leaves a signed ExhaustionRecord, not a silent retry."""
    signer = Ed25519Signer.generate()
    ledger = ExperimentBudgetLedger(_policy(max_attempts=4))
    ex = AttemptExecutor(ledger, signer=signer)
    ran = []
    for i in range(4):
        result, record = ex.execute(f"a{i}", lambda: ("done", {}))
        assert result == "done" and record is None
        ran.append(i)
    assert len(ran) == 4
    result, record = ex.execute("a5", lambda: ("done", {}), at=42.0)
    assert result is None
    assert record.dimension == "attempts"
    assert record.limit == 4.0 and record.consumed == 4.0
    assert record.observed_at == 42.0
    verifier = Ed25519Verifier()
    verifier.register(signer.key_id, signer.public_bytes())
    assert record.verify(verifier)


def test_executor_records_measured_overrun():
    """An attempt that overruns on measured actuals yields the signed
    record and its result does not count."""
    signer = Ed25519Signer.generate()
    ledger = ExperimentBudgetLedger(_policy(gpu_seconds=10.0))
    ex = AttemptExecutor(ledger, signer=signer)
    result, record = ex.execute(
        "a1", lambda: ("model", {"gpu_seconds": 20.0}))
    assert result is None and record is not None
    assert record.dimension == "gpu_seconds"


def test_budget_policy_validation():
    with pytest.raises(ValueError, match="finite"):
        _policy(gpu_seconds=float("inf"))
    with pytest.raises(ValueError, match=">= 0"):
        _policy(max_attempts=-1)


# ---------- determinism ----------------------------------------------------

def test_frozen_inputs_produce_identical_decisions():
    """The controller gate: same evidence + policy -> same decision."""
    ev = _ev(retrieval_found=True, retrieval_helped=True)
    estimates = [_est(Mechanism.RETRIEVAL.value, 0.30, 0.05),
                 _est(Mechanism.SKILL.value, 0.10, 0.20)]
    d1 = select_mechanism(ev, estimates, W)
    d2 = select_mechanism(ev, list(estimates), W)
    assert d1.digest == d2.digest
    d3 = select_mechanism(ev, estimates, W,
                          evaluator_verifier=Ed25519Verifier())
    assert d1.digest == d3.digest
