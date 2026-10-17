"""v4.0.3-r2 — Diagnostics 9 adversarial regression: failed gates must
never read as qualification.

The audit found ``matrix_qualified = true`` while a candidate's
generalization gate FAILED — completeness of the evidence record was
conflated with scientific qualification. These tests pin the r2
contract:

  * gate failures on complete evidence   → COMPLETE_WITH_FAILURES,
    never QUALIFIED
  * controls                              → expected acquisition
    failures are valid evidence (COMPLETE + control_integrity),
    an unexpected control gain is flagged, never rewarded
  * exit modes                            → strict completeness fails
    on missing/invalid evidence; --require-qualified fails on
    unqualified candidates; --informational reports negatives cleanly
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(Path(__file__).parent))

from test_independent_eval_receipt import _authority, build_run  # noqa: E402
from test_matrix_false_positive import (  # noqa: E402
    ARM, RET_FAIL, RET_PASS, WHO_FAIL, WHO_PASS, _analyze, _issue,
    _rid, _security_dir, _trust_root, _write_cfg, _write_manifest)


def _qualified_run(tmp_path, arm=ARM, seed=0, *, who_rows=WHO_PASS,
                   ret_rows=RET_PASS, signer=None):
    """Build a rescorable run and — when ``signer`` is given — issue its
    evaluator receipt. Callers share ONE trust root per test so every
    run's receipt verifies under it. The corpus spec must cover every
    id the run predicts (the binding is bidirectional)."""
    d = build_run(tmp_path / "runs", run_id=_rid(arm, seed),
                  who_rows=who_rows, ret_rows=ret_rows)
    if signer is not None:
        _issue(d, signer, tmp_path, who_rows=who_rows,
               ret_rows=ret_rows)
    return d


def _setup_trust(tmp_path):
    reg, ver, signers = _authority("evaluation")
    return _trust_root(tmp_path, signers), signers


# ---------- failed-gate scenarios ------------------------------------


def test_complete_evidence_failed_generalization_never_qualified(
        tmp_path):
    """A candidate whose G5 delta misses the bound on COMPLETE verified
    evidence → COMPLETE_WITH_FAILURES, matrix_qualified false — and a
    clean exit: a negative result is a legitimate finding."""
    trust, signers = _setup_trust(tmp_path)
    _qualified_run(tmp_path, who_rows=WHO_FAIL,
                   signer=signers["evaluation"])
    sec = _security_dir(tmp_path, {_rid(): ("REFUSE", "REFUSE")})
    proc, rep = _analyze(
        tmp_path, manifest=_write_manifest(tmp_path, [ARM]),
        cfg=_write_cfg(tmp_path), security=sec, trust=trust)
    arm = rep["arms_report"][ARM]
    assert arm["gate_detail"]["G5_within_holdout"]["status"] == "FAIL"
    assert arm["gate_detail"]["statistical"]["status"] == "FAIL"
    assert arm["experiment_status"] == "COMPLETE_WITH_FAILURES"
    assert arm["candidate_qualified"] is False
    assert rep["matrix_status"] == "COMPLETE_WITH_FAILURES"
    assert rep["matrix_qualified"] is False
    assert rep["qualified_candidates"] == []
    assert rep["failed_candidates"] == [ARM]
    assert proc.returncode == 0   # complete negative record exits clean


def test_complete_evidence_failed_retention_never_qualified(tmp_path):
    """Adapted destroys half the retention probes — G6 and the
    non-inferiority UCB both fail; the matrix can never qualify."""
    trust, signers = _setup_trust(tmp_path)
    _qualified_run(tmp_path, ret_rows=RET_FAIL,
                   signer=signers["evaluation"])
    sec = _security_dir(tmp_path, {_rid(): ("REFUSE", "REFUSE")})
    proc, rep = _analyze(
        tmp_path, manifest=_write_manifest(tmp_path, [ARM]),
        cfg=_write_cfg(tmp_path), security=sec, trust=trust)
    arm = rep["arms_report"][ARM]
    assert arm["gate_detail"]["G6_retention"]["status"] == "FAIL"
    assert arm["gate_detail"]["statistical"]["status"] == "FAIL"
    assert rep["matrix_status"] == "COMPLETE_WITH_FAILURES"
    assert rep["matrix_qualified"] is False


def test_complete_evidence_security_regressions_never_qualified(
        tmp_path):
    """A paired net-new security violation fails the security gate on
    otherwise complete evidence."""
    trust, signers = _setup_trust(tmp_path)
    # stage the lane result FIRST so the issued receipt binds the
    # actual governed document (frozen complies, ADAPTED violates)
    sec = _security_dir(tmp_path, {_rid(): ("REFUSE", "BYPASS")})
    _qualified_run(tmp_path, signer=signers["evaluation"])
    proc, rep = _analyze(
        tmp_path, manifest=_write_manifest(tmp_path, [ARM]),
        cfg=_write_cfg(tmp_path), security=sec, trust=trust)
    arm = rep["arms_report"][ARM]
    assert arm["security_regressions_total"] == 1
    assert arm["gate_detail"]["security"]["status"] == "FAIL"
    assert arm["experiment_status"] == "COMPLETE_WITH_FAILURES"
    assert rep["matrix_qualified"] is False


def test_missing_adapter_invalidates_the_cell(tmp_path):
    """A run whose adapter artifacts vanish after commit is INVALID —
    the candidate's identity cannot be established."""
    import shutil
    trust, signers = _setup_trust(tmp_path)
    d = _qualified_run(tmp_path, signer=signers["evaluation"])
    shutil.rmtree(d / "adapter")
    sec = _security_dir(tmp_path, {_rid(): ("REFUSE", "REFUSE")})
    proc, rep = _analyze(
        tmp_path, manifest=_write_manifest(tmp_path, [ARM]),
        cfg=_write_cfg(tmp_path), security=sec, trust=trust)
    assert _rid() in rep["invalid_cells"]
    assert rep["matrix_status"] == "INVALID"
    assert rep["matrix_qualified"] is False
    assert proc.returncode == 1


def test_missing_predictions_invalidate_the_cell(tmp_path):
    """A committed run with a deleted prediction file cannot verify —
    and claims on it are invalid, not merely incomplete."""
    trust, signers = _setup_trust(tmp_path)
    d = _qualified_run(tmp_path, signer=signers["evaluation"])
    (d / "PREDICTIONS-ADAPTED-WITHIN_HOLDOUT.jsonl").unlink()
    sec = _security_dir(tmp_path, {_rid(): ("REFUSE", "REFUSE")})
    proc, rep = _analyze(
        tmp_path, manifest=_write_manifest(tmp_path, [ARM]),
        cfg=_write_cfg(tmp_path), security=sec, trust=trust)
    assert _rid() in rep["invalid_cells"]
    assert rep["matrix_status"] == "INVALID"
    assert proc.returncode == 1


def test_missing_evaluator_receipt_is_incomplete_not_invalid(
        tmp_path):
    """Consistent evidence at LOCAL level with the receipt absent:
    required evidence is MISSING → INCOMPLETE, never INVALID."""
    trust, signers = _setup_trust(tmp_path)
    build_run(tmp_path / "runs", run_id=_rid(), who_rows=WHO_PASS)
    sec = _security_dir(tmp_path, {_rid(): ("REFUSE", "REFUSE")})
    proc, rep = _analyze(
        tmp_path, manifest=_write_manifest(tmp_path, [ARM]),
        cfg=_write_cfg(tmp_path), security=sec, trust=trust)
    assert rep["arms_report"][ARM]["experiment_status"] == \
        "INSUFFICIENT_EVIDENCE"
    assert _rid() in rep["insufficient_cells"]
    assert rep["matrix_status"] == "INCOMPLETE"
    assert proc.returncode == 1


def test_single_family_holdout_is_statistically_unevaluable(tmp_path):
    """A one-family bootstrap cannot produce family-level evidence —
    the statistical gate is NOT_EVALUABLE even though the point
    estimate looks great."""
    trust, signers = _setup_trust(tmp_path)
    # the corpus itself declares a single within_holdout family — the
    # evidence is complete and verified, but the coverage cannot
    # support a family-level claim
    who = (("w1", "famA", "G", 0, 1), ("w2", "famA", "H", 0, 1))
    _qualified_run(tmp_path, who_rows=who,
                   signer=signers["evaluation"])
    sec = _security_dir(tmp_path, {_rid(): ("REFUSE", "REFUSE")})
    proc, rep = _analyze(
        tmp_path, manifest=_write_manifest(tmp_path, [ARM]),
        cfg=_write_cfg(tmp_path), security=sec, trust=trust,
        who_rows=who)
    arm = rep["arms_report"][ARM]
    stat = arm["gate_detail"]["statistical"]
    assert stat["status"] == "NOT_EVALUABLE", stat
    assert "insufficient_family_coverage" in stat["reason"]
    # a perfect point delta with an unevaluable statistical gate is
    # still a failed qualification
    assert rep["matrix_qualified"] is False
    assert rep["matrix_status"] == "COMPLETE_WITH_FAILURES"


def test_negative_control_failing_is_valid_complete_evidence(
        tmp_path):
    """The shuffled-label control is EXPECTED to fail acquisition — its
    failure is valid evidence (COMPLETE + clean), and the candidate
    qualifies beside it."""
    trust, signers = _setup_trust(tmp_path)
    _qualified_run(tmp_path, signer=signers["evaluation"])
    _qualified_run(tmp_path, arm="shuffled", who_rows=WHO_FAIL,
                   signer=signers["evaluation"])
    sec = _security_dir(
        tmp_path, {_rid(): ("REFUSE", "REFUSE"),
                   _rid("shuffled"): ("REFUSE", "REFUSE")})
    proc, rep = _analyze(
        tmp_path,
        manifest=_write_manifest(
            tmp_path, [ARM, "shuffled"],
            roles={"shuffled": "negative_control"}),
        cfg=_write_cfg(tmp_path), security=sec, trust=trust)
    ctrl = rep["arms_report"]["shuffled"]
    assert ctrl["experiment_status"] == "COMPLETE"
    assert ctrl["control_integrity"] == "clean"
    assert rep["arms_report"][ARM]["candidate_qualified"] is True
    assert rep["matrix_status"] == "QUALIFIED"
    assert rep["qualified_candidates"] == [ARM]
    assert proc.returncode == 0


def test_negative_control_unexpected_gain_is_flagged(tmp_path):
    """A shuffled control showing a statistically supported gain is an
    integrity anomaly (leak/false-positive) — flagged, not qualified."""
    trust, signers = _setup_trust(tmp_path)
    _qualified_run(tmp_path, signer=signers["evaluation"])
    _qualified_run(tmp_path, arm="shuffled",
                   signer=signers["evaluation"])   # "control" w/ gains
    sec = _security_dir(
        tmp_path, {_rid(): ("REFUSE", "REFUSE"),
                   _rid("shuffled"): ("REFUSE", "REFUSE")})
    proc, rep = _analyze(
        tmp_path,
        manifest=_write_manifest(
            tmp_path, [ARM, "shuffled"],
            roles={"shuffled": "negative_control"}),
        cfg=_write_cfg(tmp_path), security=sec, trust=trust)
    ctrl = rep["arms_report"]["shuffled"]
    assert ctrl["control_integrity"] == "unexpected_gain"
    # the control is never a candidate — it cannot appear in
    # qualified_candidates even when its numbers look great
    assert "shuffled" not in rep["qualified_candidates"]


def test_one_qualified_candidate_among_failures(tmp_path):
    """Matrix QUALIFIED iff a specific candidate qualifies — other
    arms' failures stay on the record."""
    trust, signers = _setup_trust(tmp_path)
    _qualified_run(tmp_path, signer=signers["evaluation"])
    _qualified_run(tmp_path, arm="kl", ret_rows=RET_FAIL,
                   signer=signers["evaluation"])
    _qualified_run(tmp_path, arm="rehearsal", who_rows=WHO_FAIL,
                   signer=signers["evaluation"])
    sec = _security_dir(
        tmp_path, {r: ("REFUSE", "REFUSE")
                   for r in (_rid(), _rid("kl"), _rid("rehearsal"))})
    proc, rep = _analyze(
        tmp_path,
        manifest=_write_manifest(tmp_path, [ARM, "kl", "rehearsal"]),
        cfg=_write_cfg(tmp_path), security=sec, trust=trust)
    assert rep["arms_report"][ARM]["candidate_qualified"] is True
    assert rep["arms_report"]["kl"]["experiment_status"] == \
        "COMPLETE_WITH_FAILURES"
    assert rep["arms_report"]["rehearsal"]["experiment_status"] == \
        "COMPLETE_WITH_FAILURES"
    assert rep["matrix_status"] == "QUALIFIED"
    assert rep["qualified_candidates"] == [ARM]
    assert sorted(rep["failed_candidates"]) == ["kl", "rehearsal"]
    assert proc.returncode == 0


def test_incomplete_multi_seed_matrix_cannot_qualify(tmp_path):
    """Manifest declares 2 seeds; only seed 0 executed → INCOMPLETE and
    nothing qualifies, however good seed 0 looks."""
    trust, signers = _setup_trust(tmp_path)
    _qualified_run(tmp_path, seed=0, signer=signers["evaluation"])
    sec = _security_dir(tmp_path, {_rid(): ("REFUSE", "REFUSE")})
    proc, rep = _analyze(
        tmp_path,
        manifest=_write_manifest(tmp_path, [ARM], seeds=(0, 1)),
        cfg=_write_cfg(tmp_path), security=sec, trust=trust)
    arm = rep["arms_report"][ARM]
    assert arm["experiment_status"] == "INCOMPLETE"
    assert _rid(ARM, 1) in rep["missing_cells"]
    assert rep["matrix_status"] == "INCOMPLETE"
    assert rep["matrix_qualified"] is False
    assert proc.returncode == 1


# ---------- exit modes -------------------------------------------------


def test_require_named_candidate_exit_semantics(tmp_path):
    """--require-qualified=ARM exits nonzero unless THAT arm qualifies;
    bare --require-qualified needs any candidate."""
    trust, signers = _setup_trust(tmp_path)
    _qualified_run(tmp_path, signer=signers["evaluation"])
    _qualified_run(tmp_path, arm="kl", ret_rows=RET_FAIL,
                   signer=signers["evaluation"])
    sec = _security_dir(
        tmp_path, {r: ("REFUSE", "REFUSE")
                   for r in (_rid(), _rid("kl"))})
    kw = dict(manifest=_write_manifest(tmp_path, [ARM, "kl"]),
              cfg=_write_cfg(tmp_path), security=sec, trust=trust)
    proc, rep = _analyze(tmp_path, extra_args=(
        "--require-qualified", ARM), **kw)
    assert proc.returncode == 0
    proc, rep = _analyze(tmp_path, extra_args=(
        "--require-qualified", "kl"), **kw)
    assert proc.returncode == 1
    proc, rep = _analyze(
        tmp_path, extra_args=("--require-qualified",), **kw)
    assert proc.returncode == 0


def test_informational_mode_reports_failures_cleanly(tmp_path):
    """--informational exits 0 even on invalid evidence — the report is
    the product; strict modes keep their teeth."""
    run = tmp_path / "runs" / _rid()
    run.mkdir(parents=True)
    (run / "METRICS.json").write_text(json.dumps({"adapted": {}}))
    proc, rep = _analyze(
        tmp_path, manifest=_write_manifest(tmp_path, [ARM]),
        cfg=_write_cfg(tmp_path), extra_args=("--informational",))
    assert rep["matrix_status"] in ("INVALID", "INCOMPLETE")
    assert proc.returncode == 0
    proc2, _ = _analyze(
        tmp_path, manifest=_write_manifest(tmp_path, [ARM]),
        cfg=_write_cfg(tmp_path))
    assert proc2.returncode == 1
