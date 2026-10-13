"""Campaign 3C — the confirmation record must not leak sealed labels.

The evaluator holds the confirmation tasks and their answers; the
research worker receives only aggregates. These tests pin that
operational separation: neither the signed confirmation record nor the
append-only ledger may contain sealed expected answers or prompts, and
the sealed block carries only counts, families and the manifest digest.
"""
import importlib.util
import json
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

def _write_sealed(c, path: Path) -> Path:
    """Serialise the custody fixture's sealed rows to an evaluator-held
    file outside the research storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, sort_keys=True)
                              for r in c.rows) + "\n")
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
