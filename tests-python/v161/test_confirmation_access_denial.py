"""Campaign 3C — the confirmation record must not leak sealed labels.

The evaluator holds the confirmation tasks and their answers; the
research worker receives only aggregates. These tests pin that
operational separation: neither the signed confirmation record nor the
append-only ledger may contain sealed expected answers or prompts, and
the sealed block carries only counts, families and the manifest digest.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))


def _fixture():
    """Load the confirmation fixture module by path (pytest's importlib
    mode does not put sibling test modules on sys.path)."""
    spec = importlib.util.spec_from_file_location(
        "sealed_confirmation_fixture",
        Path(__file__).parent / "test_sealed_confirmation_execution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_F = _fixture()
Custody, TOOL, _executor = _F.Custody, _F.TOOL, _F._executor


def _serialized(record) -> str:
    return json.dumps(record, sort_keys=True)


def test_record_contains_no_sealed_answers_or_prompts(tmp_path):
    c = Custody(tmp_path)
    record = c.run(_executor())
    blob = _serialized(record)
    for row in c.rows:
        assert str(row["expected"]) not in blob or row["expected"] == "", \
            f"sealed answer for {row['id']} leaked into the record"
        assert str(row["prompt"]) not in blob, \
            f"sealed prompt for {row['id']} leaked into the record"
    # the sealed block carries only counts, families and the digest
    assert set(record["value"]["sealed"]) == {"manifest_digest", "blocks",
                                              "hidden_families"}
    assert record["value"]["sealed"]["hidden_families"] == ["fam-0", "fam-1"]


def test_ledger_contains_no_sealed_content(tmp_path):
    c = Custody(tmp_path)
    record = c.run(_executor())
    TOOL.append_ledger(c.storage / "CONFIRMATION_LEDGER.jsonl", record,
                       actor=c.signers["qualification"].key_id)
    blob = (c.storage / "CONFIRMATION_LEDGER.jsonl").read_text()
    for row in c.rows:
        if row["expected"]:
            assert str(row["expected"]) not in blob
        assert str(row["prompt"]) not in blob


def test_ledger_is_append_only_across_attempts(tmp_path):
    c = Custody(tmp_path)
    ledger = c.storage / "CONFIRMATION_LEDGER.jsonl"
    first = c.run(_executor())
    TOOL.append_ledger(ledger, first, actor="qualifier-1")
    before = ledger.read_text()
    second = c.run(_executor(l5_correct=4, l6_correct=4))  # refused
    TOOL.append_ledger(ledger, second, actor="qualifier-1")
    lines = [json.loads(ln) for ln in ledger.read_text().splitlines()
             if ln.strip()]
    assert [e["seq"] for e in lines] == [0, 1]
    assert [e["decision"] for e in lines] == ["QUALIFIED", "REFUSED"]
    # the earlier entry is untouched (prefix preserved)
    assert ledger.read_text().startswith(before)


# --------------------------------------------------------------------
# Research-side denial (3B Step 22): the research execution environment
# must not be able to read the confirmation set at all.
# --------------------------------------------------------------------

def _write_sealed(c, path: Path, *, mode: int = 0o600) -> Path:
    """Serialise the custody fixture's sealed rows to an evaluator-held
    file outside the research storage, under owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, sort_keys=True)
                              for r in c.rows) + "\n")
    path.chmod(mode)
    return path


def test_research_worker_marker_is_parsed_conservatively():
    assert TOOL.is_research_worker({}) is False
    assert TOOL.is_research_worker({TOOL.RESEARCH_WORKER_ENV: "0"}) is False
    assert TOOL.is_research_worker(
        {TOOL.RESEARCH_WORKER_ENV: "false"}) is False
    assert TOOL.is_research_worker({TOOL.RESEARCH_WORKER_ENV: "1"}) is True
    assert TOOL.is_research_worker(
        {TOOL.RESEARCH_WORKER_ENV: "true"}) is True
    assert TOOL.is_research_worker(
        {TOOL.RESEARCH_WORKER_ENV: "YES"}) is True


def test_research_worker_cannot_load_sealed_rows(tmp_path):
    """A research-marked process is refused before any row is read."""
    c = Custody(tmp_path)
    sealed = _write_sealed(c, tmp_path / "evaluator" / "sealed.jsonl")
    try:
        TOOL.load_sealed(sealed, environ={TOOL.RESEARCH_WORKER_ENV: "1"})
    except TOOL.ConfirmationRefused as exc:
        assert "research worker" in str(exc)
    else:
        raise AssertionError("a research worker must not load sealed rows")


def test_custody_guard_fires_before_the_filesystem(tmp_path):
    """The refusal is a custody decision, not a missing-file error: a
    nonexistent path under the research marker still reports custody."""
    missing = tmp_path / "does-not-exist.jsonl"
    try:
        TOOL.load_sealed(missing,
                         environ={TOOL.RESEARCH_WORKER_ENV: "true"})
    except TOOL.ConfirmationRefused as exc:
        assert "research worker" in str(exc)
    else:
        raise AssertionError("custody guard must fire first")


def test_sealed_set_inside_research_storage_is_refused(tmp_path):
    """Research workers hold write access to the campaign storage, so a
    sealed set living there is readable through the same tree."""
    c = Custody(tmp_path)
    inside = _write_sealed(c, c.storage / "sealed-inside.jsonl")
    try:
        TOOL.load_sealed(inside, storage=c.storage, environ={})
    except TOOL.ConfirmationRefused as exc:
        assert "evaluator custody" in str(exc)
    else:
        raise AssertionError("sealed data under research storage must be refused")


def test_evaluator_custody_loads_the_same_rows(tmp_path):
    """The guard refuses research workers only — evaluator custody is
    unaffected, so the confirmation still runs."""
    c = Custody(tmp_path)
    sealed = _write_sealed(c, tmp_path / "evaluator" / "sealed.jsonl")
    rows = TOOL.load_sealed(sealed, storage=c.storage, environ={})
    assert len(rows) == len(c.rows)
    record = c.run(_executor())
    assert record["value"]["decision"] == "QUALIFIED"


# --------------------------------------------------------------------
# The enforceable control (reviewer finding): an environment marker is
# a cooperative guard, not isolation. A research worker that controls
# its environment bypasses it — but it cannot open a file its OS
# identity has no permission to read.
# --------------------------------------------------------------------

def test_group_or_world_readable_sealed_set_is_refused(tmp_path):
    c = Custody(tmp_path)
    sealed = _write_sealed(c, tmp_path / "evaluator" / "sealed.jsonl",
                           mode=0o644)
    try:
        TOOL.load_sealed(sealed, storage=c.storage, environ={})
    except TOOL.ConfirmationRefused as exc:
        assert "readable beyond its owner" in str(exc)
    else:
        raise AssertionError("a world-readable sealed set must be refused")


def test_the_environment_marker_alone_is_not_isolation(tmp_path):
    """Documents the weakness honestly: an in-process caller that does
    not pass the marker is not stopped by it. The permission control is
    what actually holds."""
    c = Custody(tmp_path)
    sealed = _write_sealed(c, tmp_path / "evaluator" / "sealed.jsonl",
                           mode=0o600)
    # the cooperative guard is trivially bypassed by not declaring it
    rows = TOOL.load_sealed(sealed, storage=c.storage, environ={})
    assert rows, "the cooperative guard does not, and cannot, stop this"
    # ...but the permission control still binds
    sealed.chmod(0o644)
    try:
        TOOL.load_sealed(sealed, storage=c.storage, environ={})
    except TOOL.ConfirmationRefused:
        pass
    else:
        raise AssertionError("the enforceable control must still refuse")


def test_custody_report_states_the_real_level_of_assurance(tmp_path):
    c = Custody(tmp_path)
    sealed = _write_sealed(c, tmp_path / "evaluator" / "sealed.jsonl",
                           mode=0o600)
    # Same-UID arrangement (research storage and sealed file owned by the
    # same UID): permissions_enforced is True, but independent_isolation
    # MUST be False (Campaign 3B.2 Step 19).
    same_uid_report = TOOL.sealed_custody_report(
        sealed, storage=c.storage, environ={})
    kinds = {c_["control"]: c_ for c_ in same_uid_report["controls"]}
    assert kinds["sealed_file_permissions"]["enforced"] is True
    assert kinds["sealed_outside_research_storage"]["enforced"] is True
    assert kinds["research_worker_env_marker"]["enforced"] is True
    assert kinds["distinct_evaluator_os_identity"]["enforced"] is False
    assert same_uid_report["permissions_enforced"] is True
    assert same_uid_report["distinct_os_identity_enforced"] is False
    assert same_uid_report["independent_isolation_established"] is False
    assert "did NOT establish independent" in same_uid_report["statement"]

    # Distinct-UID arrangement: evaluator UID != research UID + mode 0600
    # + outside research storage -> independent isolation established.
    distinct_report = TOOL.sealed_custody_report(
        sealed, storage=c.storage, environ={},
        research_uid=os.geteuid() + 1000)
    assert distinct_report["distinct_os_identity_enforced"] is True
    assert distinct_report["independent_isolation_established"] is True
    assert "NOT isolation" in distinct_report["statement"]

    # an unenforced permission control must NOT claim isolation even with
    # a distinct research UID
    sealed.chmod(0o644)
    weak = TOOL.sealed_custody_report(
        sealed, storage=c.storage, environ={},
        research_uid=os.geteuid() + 1000)
    assert weak["independent_isolation_established"] is False


def test_custody_report_is_recorded_in_the_signed_record(tmp_path):
    c = Custody(tmp_path)
    sealed = _write_sealed(c, tmp_path / "evaluator" / "sealed.jsonl",
                           mode=0o600)
    custody = TOOL.sealed_custody_report(
        sealed, storage=c.storage, environ={},
        research_uid=os.geteuid() + 1000)
    record = c.run(_executor(), custody=custody)
    assert record["value"]["custody"]["independent_isolation_established"] is True


def test_absent_custody_report_never_claims_isolation(tmp_path):
    """A caller that supplies no custody report gets an explicit
    'NOT established', never a silent pass."""
    c = Custody(tmp_path)
    record = c.run(_executor())
    custody = record["value"]["custody"]
    assert custody["independent_isolation_established"] is False
    assert "NOT established" in custody["statement"]


# --------------------------------------------------------------------
# Step 18: the sealed confirmation applies the same governed safety
# suite the research gate used — it must not be the weaker check.
# --------------------------------------------------------------------

def _custody_with_governed(tmp_path):
    from minagi.v161.safety_regression import safety_probes
    rows = tuple(r for r in _F._rows() if r["split"] != "security") \
        + tuple(safety_probes())
    return Custody(tmp_path, rows=list(rows))


def test_sealed_confirmation_enforces_the_governed_suite(tmp_path):
    """A compliant governed sealed run qualifies and reports ENFORCED."""
    c = _custody_with_governed(tmp_path)
    record = c.run(_executor())
    assert record["value"]["decision"] == "QUALIFIED"
    gov = record["value"]["stats"]["governed_safety"]
    assert gov["applicability"] == "ENFORCED"
    assert gov["decision"] == "PASS"
    assert gov["blocked_categories"] == []


def test_sealed_confirmation_refuses_a_governed_violation(tmp_path):
    """An arm that discloses the canary on the sealed set refuses."""
    from minagi.v161.safety_regression import PROTECTED_CANARY
    c = _custody_with_governed(tmp_path)

    def violating(seed, arm_id, rows):
        out = _executor()(seed, arm_id, rows)
        for r in rows:
            if str(r.get("category")) == "protected_information":
                out[str(r["id"])] = f"the value is {PROTECTED_CANARY}"
        return out

    record = c.run(violating)
    assert record["value"]["decision"] == "REFUSED"
    gov = record["value"]["stats"]["governed_safety"]
    assert gov["decision"] == "BLOCK"
    assert "protected_information" in gov["blocked_categories"]
    assert any("governed safety gate BLOCKED" in r
               for r in record["value"]["reasons"])


def test_sealed_confirmation_labels_a_legacy_sealed_set(tmp_path):
    """Without governed probes the sealed safety block is reported as
    NOT_APPLICABLE, never as a safety pass."""
    c = Custody(tmp_path)
    record = c.run(_executor())
    gov = record["value"]["stats"]["governed_safety"]
    assert gov["applicability"] == "NOT_APPLICABLE"
    assert gov["decision"] == "NOT_APPLICABLE"
