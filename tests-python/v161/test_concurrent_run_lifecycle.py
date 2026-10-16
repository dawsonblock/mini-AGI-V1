"""V4-26 — atomic, ownership-bound run reservation.

Acceptance (from the v4.0.2 recovery plan): launch two attempts with
the same requested run id. Exactly one may claim the canonical run;
both attempts must retain their own evidence; neither may delete the
other's files; and no committed run may be overwritten.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.run_reservation import (DestinationCommitted,  # noqa: E402
                                         RunReservation, reserve)

SCRIPT = ROOT / "scripts" / "positive_control_fit.py"


def _commit(res: RunReservation, status="COMPLETE"):
    (res.staging / "RUN_COMMIT.json").write_text(json.dumps({
        "run_id": res.run_id, "status": status,
        "attempt_id": res.attempt_id}, indent=2))


# ---------------------------------------------------------------- claims

def test_concurrent_attempts_exactly_one_claims(tmp_path):
    root = tmp_path / "runs"
    a = reserve(root, "exp1", "attempt-A")
    b = reserve(root, "exp1", "attempt-B")
    assert a.owns_claim != b.owns_claim          # exactly one wins
    loser = b if a.owns_claim else a
    assert (loser.staging / "CLAIM_LOST.json").is_file()
    # both attempts retain their own staging evidence
    assert a.staging.is_dir() and b.staging.is_dir()
    assert a.staging != b.staging


def test_publish_requires_claim_and_commit(tmp_path):
    root = tmp_path / "runs"
    winner = reserve(root, "exp1", "attempt-A")
    loser = reserve(root, "exp1", "attempt-B")
    if not winner.owns_claim:
        winner, loser = loser, winner
    # the loser cannot publish
    _commit(loser)
    with pytest.raises(Exception):
        loser.publish()
    # the winner cannot publish without a commit record
    with pytest.raises(Exception):
        winner.publish()
    _commit(winner)
    final = winner.publish()
    assert final == root / "exp1" and final.is_dir()
    assert (final / "RUN_COMMIT.json").is_file()
    # the loser's evidence is still on disk, untouched
    assert loser.staging.is_dir()


def test_committed_run_is_never_replaced(tmp_path):
    root = tmp_path / "runs"
    r1 = reserve(root, "exp1", "attempt-A")
    _commit(r1)
    final = r1.publish()
    marker = final / "marker.txt"
    marker.write_text("committed evidence")
    # a new reservation for the same run id refuses before any work
    with pytest.raises(DestinationCommitted):
        reserve(root, "exp1", "attempt-B")
    assert marker.read_text() == "committed evidence"
    # a late publish attempt against an existing dir refuses even with
    # a forged claim — committed runs are immutable
    stale = reserve(root, "exp2", "attempt-C")
    _commit(stale)
    stale.final_dir = final            # simulate a retargeted attempt
    stale.claim_dir = root / ".claims" / "exp2"
    with pytest.raises(Exception):
        stale.publish()
    assert marker.read_text() == "committed evidence"


def test_failed_attempt_preserves_evidence_and_releases_claim(tmp_path):
    root = tmp_path / "runs"
    a = reserve(root, "exp1", "attempt-A")
    _commit(a, status="FAILED")
    failed_dir = a.fail()
    assert failed_dir.name == "exp1.FAILED-attempt-A"
    assert (failed_dir / "RUN_COMMIT.json").is_file()
    # the claim is released: a later attempt may claim the canonical name
    b = reserve(root, "exp1", "attempt-B")
    assert b.owns_claim
    _commit(b)
    final = b.publish()
    # success and failure coexist — neither deletes the other
    assert final.is_dir() and failed_dir.is_dir()
    assert json.loads(
        (failed_dir / "RUN_COMMIT.json").read_text())["status"] == "FAILED"


def test_staging_dirs_are_unique_per_attempt(tmp_path):
    root = tmp_path / "runs"
    a = reserve(root, "exp1", "attempt-A")
    # an attempt id reuse collision refuses rather than reusing
    with pytest.raises(FileExistsError):
        reserve(root, "exp1", "attempt-A")
    assert (a.staging / "ATTEMPT.json").is_file()


# ------------------------------------------------------- CLI end-to-end

def test_cli_attempt_with_held_claim_exits_without_model_load(tmp_path):
    """A contested run id must be refused before torch/model init.

    We pre-seed a claim held by another attempt; the runner must notice
    and stop while everything is still cheap."""
    run_root = tmp_path / "runs"
    claim = run_root / ".claims" / "taken"
    claim.mkdir(parents=True)
    (claim / "CLAIM.json").write_text(json.dumps(
        {"run_id": "taken", "attempt_id": "other-attempt"}))
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--families", "pc-upper,pc-rev",
         "--steps", "1", "--run-root", str(run_root),
         "--run-id", "taken"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert proc.returncode == 3
    out = proc.stdout + proc.stderr
    assert "claim held" in out
    # the loser retains its own attempt evidence and touches nothing else
    stagings = list((run_root / ".staging").glob("taken.*"))
    assert stagings, "loser attempt evidence missing"
    assert (stagings[0] / "CLAIM_LOST.json").is_file()
    assert not (run_root / "taken").exists()
