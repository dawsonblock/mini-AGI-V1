"""Campaign 3B.2 Step 20 — fail-closed production vs synthetic confirmation.

Verifies that:
  1. Synthetic test fixtures are explicitly tagged mode='synthetic' and
     production_valid=False so they can never be used as production
     qualification receipts.
  2. Production mode requires verified candidate artifacts, enforced
     independent evaluator custody (distinct OS identity + mode 0600 +
     outside storage), approved scoring configuration, and complete
     authority signers.
  3. If any required component is missing in production mode,
     run_confirmation returns decision='NOT_QUALIFIED' (never 'QUALIFIED')
     and production_valid=False.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))


def _fixture():
    spec = importlib.util.spec_from_file_location(
        "sealed_confirmation_fixture",
        Path(__file__).parent / "test_sealed_confirmation_execution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_F = _fixture()
Custody, TOOL, _executor = _F.Custody, _F.TOOL, _F._executor
_bind_candidate_artifacts = _F._bind_candidate_artifacts


def _write_sealed(c, path: Path, *, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, sort_keys=True)
                              for r in c.rows) + "\n")
    path.chmod(mode)
    return path


def test_synthetic_mode_marks_receipt_non_production(tmp_path):
    c = Custody(tmp_path)
    record = c.run(_executor())
    val = record["value"]
    assert val["decision"] == "QUALIFIED"
    assert val["mode"] == "synthetic"
    assert val["production_valid"] is False
    assert "SYNTHETIC TEST RECORD" in val["note"]


def test_production_mode_fails_closed_when_artifacts_or_custody_missing(
        tmp_path):
    c = Custody(tmp_path)
    # 1. No artifact verification and no custody -> NOT_QUALIFIED
    rec1 = c.run(_executor(), mode="production")
    assert rec1["value"]["decision"] == "NOT_QUALIFIED"
    assert rec1["value"]["production_valid"] is False
    assert any("verified candidate artifacts" in r
               for r in rec1["value"]["reasons"])
    assert any("enforced independent evaluator custody" in r
               for r in rec1["value"]["reasons"])

    # 2. Artifacts verified, but same-UID custody -> still NOT_QUALIFIED
    _bind_candidate_artifacts(c)
    sealed = _write_sealed(c, tmp_path / "eval" / "sealed.jsonl", mode=0o600)
    same_uid_custody = TOOL.sealed_custody_report(
        sealed, storage=c.storage, environ={})
    rec2 = c.run(
        _executor(), mode="production", custody=same_uid_custody,
        verify_artifacts=lambda: TOOL.verify_candidate_artifacts(
            c.storage, _F.CID, c.plan))
    assert rec2["value"]["decision"] == "NOT_QUALIFIED"
    assert rec2["value"]["production_valid"] is False
    assert any("enforced independent evaluator custody" in r
               for r in rec2["value"]["reasons"])


def test_production_mode_qualifies_only_when_all_requirements_met(tmp_path):
    c = Custody(tmp_path)
    _bind_candidate_artifacts(c)
    sealed = _write_sealed(c, tmp_path / "eval" / "sealed.jsonl", mode=0o600)
    isolated_custody = TOOL.sealed_custody_report(
        sealed, storage=c.storage, environ={},
        research_uid=os.geteuid() + 1000)
    rec = c.run(
        _executor(), mode="production", custody=isolated_custody,
        verify_artifacts=lambda: TOOL.verify_candidate_artifacts(
            c.storage, _F.CID, c.plan))
    assert rec["value"]["decision"] == "QUALIFIED"
    assert rec["value"]["mode"] == "production"
    assert rec["value"]["production_valid"] is True


def test_invalid_mode_is_rejected(tmp_path):
    c = Custody(tmp_path)
    with pytest.raises(ValueError, match="mode must be"):
        c.run(_executor(), mode="bogus")
