"""v4.0.3 Phase 3 — filesystem hardening and evidence completeness.

Two surfaces under test:

* ``run_reservation`` — run/attempt identifiers are filename-safe
  single components (path traversal, separators, dots-only and
  overlong ids refused), symlinked infrastructure is refused, and
  publication is a true no-replace operation under the atomic
  ``os.mkdir`` nonce.
* ``evidence_reconciliation`` — a COMPLETE run must report every
  measurement its prediction surfaces make resolvable: missing
  required metrics, missing/duplicated/unexpected predictions,
  wrong corpus digests, truncated commits and extra evidence all
  resolve to INVALID/INCONSISTENT, never VERIFIED.
* ``requalify_executed_campaign`` — ``--expected-manifest`` /
  ``--strict-completeness`` / ``--require-evidence`` gates and the
  empty-campaign refusal.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(Path(__file__).parent))

from minagi.v161.evidence_reconciliation import (  # noqa: E402
    INCONSISTENT, INVALID, VERIFIED, verify_run)
from minagi.v161.run_reservation import (  # noqa: E402
    ReservationError, UnsafeIdentifier, reserve)
from test_qualified_outcomes import (  # noqa: E402
    _rec, _recommit, _write_jsonl, build_run)

REQUALIFY = ROOT / "scripts" / "requalify_executed_campaign.py"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------- identifier / path hardening --------------------------------

@pytest.mark.parametrize("bad", [
    "../escape", "..", ".", "a/b", "a\\b", "/abs/path", "", "-lead",
    ".hidden", "run id", "run\0id", "run\nid", "x" * 129,
])
def test_unsafe_run_identifiers_refused(tmp_path, bad):
    with pytest.raises(UnsafeIdentifier):
        reserve(tmp_path, bad)


def test_unsafe_attempt_identifier_refused(tmp_path):
    with pytest.raises(UnsafeIdentifier):
        reserve(tmp_path, "ok-run", attempt_id="../x")


def test_valid_identifiers_accepted(tmp_path):
    r = reserve(tmp_path, "v401-3e-8fam-e8-s0",
                attempt_id="1734567890-ab12cd34")
    assert r.owns_claim
    assert r.staging.parent.name == ".staging"


def test_symlinked_claims_dir_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / ".claims").symlink_to(outside)
    with pytest.raises(UnsafeIdentifier):
        reserve(tmp_path / "root", "run-1")


def test_publish_is_no_replace_under_planted_dir(tmp_path):
    """A planted EMPTY directory at the canonical name cannot be
    silently replaced by a racing publish — mkdir(EEXIST) refuses."""
    r = reserve(tmp_path, "run-1")
    (r.staging / "RUN_COMMIT.json").write_text(
        json.dumps({"run_id": "run-1", "status": "COMPLETE",
                    "files": {}}))
    planted = tmp_path / "run-1"
    planted.mkdir()
    with pytest.raises(ReservationError, match="never replaced"):
        r.publish()
    assert planted.exists() and not (planted / "RUN_COMMIT.json") \
        .exists()          # the plant survives — nothing was replaced


def test_publish_requires_commit_and_claim(tmp_path):
    r = reserve(tmp_path, "run-1")
    with pytest.raises(ReservationError, match="RUN_COMMIT"):
        r.publish()
    (r.staging / "RUN_COMMIT.json").write_text(
        json.dumps({"run_id": "run-1", "status": "COMPLETE",
                    "files": {}}))
    p = r.publish()
    assert (p / "RUN_COMMIT.json").is_file()
    # a second reservation for the same committed run is refused
    with pytest.raises(ReservationError):
        reserve(tmp_path, "run-1")


# ---------- required-metric schema --------------------------------------

def _verify(d: Path):
    return verify_run(d)


def test_missing_required_accuracy_field_invalidates(tmp_path):
    d = build_run(tmp_path)
    m = json.loads((d / "METRICS.json").read_text())
    del m["adapted"]["hidden"]                # required field removed
    (d / "METRICS.json").write_text(json.dumps(m))
    _recommit(d)
    r = _verify(d)
    assert r["status"] == INCONSISTENT
    assert any("adapted.hidden: required metric absent" in x
               for x in r["mismatches"])


def test_missing_capability_destruction_invalidates(tmp_path):
    d = build_run(tmp_path)
    m = json.loads((d / "METRICS.json").read_text())
    del m["capability_destruction"]
    (d / "METRICS.json").write_text(json.dumps(m))
    _recommit(d)
    r = _verify(d)
    assert r["status"] == INCONSISTENT
    assert any("capability_destruction" in x for x in r["mismatches"])


def test_out_of_range_metric_invalidates(tmp_path):
    d = build_run(tmp_path)
    m = json.loads((d / "METRICS.json").read_text())
    m["adapted"]["hidden"] = 1.5              # impossible accuracy
    (d / "METRICS.json").write_text(json.dumps(m))
    _recommit(d)
    r = _verify(d)
    assert r["status"] == INCONSISTENT
    assert any("outside [0,1]" in x for x in r["mismatches"])


def test_non_numeric_metric_invalidates(tmp_path):
    d = build_run(tmp_path)
    m = json.loads((d / "METRICS.json").read_text())
    m["frozen"]["hidden"] = "pretty good"
    (d / "METRICS.json").write_text(json.dumps(m))
    _recommit(d)
    assert _verify(d)["status"] == INCONSISTENT


def test_missing_committed_prediction_file_invalid(tmp_path):
    d = build_run(tmp_path)
    (d / "PREDICTIONS-ADAPTED.jsonl").unlink()
    # keep the stale commit — the inventory names the deleted file
    r = _verify(d)
    assert r["status"] == INVALID            # inventory mismatch


def test_duplicated_task_id_invalidates(tmp_path):
    d = build_run(tmp_path)
    p = d / "PREDICTIONS-FROZEN.jsonl"
    rows = [json.loads(ln) for ln in p.read_text().splitlines()]
    rows.append(dict(rows[0]))               # duplicate id
    _write_jsonl(p, rows)
    _recommit(d)
    r = _verify(d)
    assert r["status"] == INCONSISTENT
    assert any("duplicated task ids" in x for x in r["mismatches"])


def test_unexpected_task_id_invalidates(tmp_path):
    d = build_run(tmp_path)
    p = d / "PREDICTIONS-FROZEN.jsonl"
    rows = [json.loads(ln) for ln in p.read_text().splitlines()]
    rows.append(_rec("h999", "fx", "Z", "Z", "hidden"))
    _write_jsonl(p, rows)
    _recommit(d)
    r = _verify(d)
    assert r["status"] == INCONSISTENT


def test_wrong_corpus_expected_invalidates(tmp_path):
    d = build_run(tmp_path)
    corpus = [{"id": "h1", "split": "hidden", "family": "fx",
               "prompt": "prompt-h1", "expected": "DIFFERENT"}]
    r = verify_run(d, corpus_rows=corpus)
    assert r["status"] == INCONSISTENT
    assert any("expected answer differs" in x for x in r["mismatches"])


def test_truncated_commit_invalid(tmp_path):
    d = build_run(tmp_path)
    (d / "RUN_COMMIT.json").write_text('{"run_id": "run-ok", "status"')
    r = _verify(d)
    assert r["status"] in (INVALID, INCONSISTENT)
    # and no commit at all is INVALID outright
    (d / "RUN_COMMIT.json").unlink()
    assert _verify(d)["status"] == INVALID


def test_unexpected_prediction_files_flagged(tmp_path):
    d = build_run(tmp_path)
    _write_jsonl(d / "PREDICTIONS-EXTRA.jsonl",
                 [_rec("x1", "fx", "Q", "Q", "hidden")])
    _recommit(d)
    r = _verify(d)
    assert r["status"] == INCONSISTENT
    assert any("unexpected prediction files" in x
               for x in r["mismatches"])


def test_complete_run_still_verifies(tmp_path):
    assert _verify(build_run(tmp_path))["status"] == VERIFIED


# ---------- campaign completeness ---------------------------------------

def _run_requalify(*argv) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REQUALIFY), *argv],
        capture_output=True, text=True, cwd=str(ROOT))


def test_empty_campaign_is_not_a_qualification(tmp_path):
    proc = _run_requalify("--results", str(tmp_path / "nothing"),
                          "--require-evidence", "--output",
                          str(tmp_path / "OUT.md"))
    assert proc.returncode == 2
    assert "empty" in proc.stderr


def test_missing_campaign_run_fails_manifest(tmp_path):
    d = build_run(tmp_path / "runs", run_id="seed0")
    manifest = tmp_path / "expected.json"
    manifest.write_text(json.dumps({"runs": [
        {"run_id": "seed0", "expect": "VERIFIED"},
        {"run_id": "seed1", "expect": "VERIFIED"},
        {"run_id": "seed2-failed", "expect": "FAILED_RUN"}]}))
    proc = _run_requalify("--results", str(d),
                          "--expected-manifest", str(manifest),
                          "--strict-completeness", "--require-evidence",
                          "--output", str(tmp_path / "OUT.md"))
    assert proc.returncode == 2
    assert "expected run missing: seed1" in proc.stderr
    assert "expected run missing: seed2-failed" in proc.stderr


def test_complete_manifest_passes(tmp_path):
    d = build_run(tmp_path / "runs", run_id="seed0")
    manifest = tmp_path / "expected.json"
    manifest.write_text(json.dumps({"runs": [
        {"run_id": "seed0", "expect": "VERIFIED"}]}))
    proc = _run_requalify("--results", str(d),
                          "--expected-manifest", str(manifest),
                          "--strict-completeness", "--require-evidence",
                          "--output", str(tmp_path / "OUT.md"))
    assert proc.returncode == 0, proc.stderr
    assert "VERIFIED" in (tmp_path / "OUT.md").read_text()


def test_status_mismatch_fails_strict(tmp_path):
    d = build_run(tmp_path / "runs", run_id="seed0")
    m = json.loads((d / "METRICS.json").read_text())
    del m["adapted"]["hidden"]
    (d / "METRICS.json").write_text(json.dumps(m))
    _recommit(d)
    manifest = tmp_path / "expected.json"
    manifest.write_text(json.dumps({"runs": [
        {"run_id": "seed0", "expect": "VERIFIED"}]}))
    proc = _run_requalify("--results", str(d),
                          "--expected-manifest", str(manifest),
                          "--strict-completeness",
                          "--output", str(tmp_path / "OUT.md"))
    assert proc.returncode == 2
    assert "expected VERIFIED, got INCONSISTENT" in proc.stderr
