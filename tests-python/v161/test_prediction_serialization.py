"""V4-25 — prediction serialization contract (behavioral).

Every declared nonempty split must write real prediction *records* —
``{"split", "id", **prediction}`` — that reload losslessly and rescore
independently to the recorded accuracy. We exercise the exact writer
expression used by ``positive_control_fit.py`` and then re-verify the
contract against the committed Campaign 3E artifacts.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
from minagi.v161.evidence_reconciliation import (  # noqa: E402
    _load_jsonl, _rescore)

RUNS = ROOT / "results" / "campaign3e" / "runs"


def _load_records(path: Path) -> dict:
    """Reload a prediction file as ``{task_id: record}`` — the shape the
    runner produced and the verifier consumes."""
    return {str(r["id"]): r for r in _load_jsonl(path)}


def _rescore_mean(path: Path) -> float:
    """Independent strict rescore of one prediction file."""
    scores = _rescore(_load_jsonl(path))
    return sum(scores.values()) / len(scores) if scores else 0.0


def _write_split(staging: Path, split_name: str, pairs) -> None:
    """The writer expression under test — the same block the runner
    uses, kept here as a behavioral contract: score maps would produce
    TypeErrors; prediction dicts must round-trip."""
    if not pairs[0] and not pairs[1]:
        return
    for name, preds in (("FROZEN", pairs[0]), ("ADAPTED", pairs[1])):
        if not preds:
            continue
        out = staging / f"PREDICTIONS-{name}-{split_name}.jsonl"
        records = [{"split": split_name.lower(), "id": rid, **prediction}
                   for rid, prediction in sorted(preds.items())]
        with out.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, sort_keys=True) + "\n")


def _pred(output: str, expected: str, prompt: str = "p") -> dict:
    return {"family": "t",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "expected": expected, "output": output}


def test_validation_and_compositional_records_round_trip(tmp_path):
    val = {"v1": _pred("abc", "abc"), "v2": _pred("xyz", "ABC")}
    val_a = {"v1": _pred("abc", "abc"), "v2": _pred("ABC", "ABC")}
    comp = {"c1": _pred("xy", "xy")}
    _write_split(tmp_path, "VALIDATION", (val, val_a))
    _write_split(tmp_path, "COMPOSITIONAL", (comp, {}))

    frozen = _load_records(
        tmp_path / "PREDICTIONS-FROZEN-VALIDATION.jsonl")
    assert len(frozen) == 2
    # every written record equals its source prediction + envelope fields
    for rid, record in frozen.items():
        src = val[rid]
        assert record["split"] == "validation" and record["id"] == rid
        for k, v in src.items():
            assert record[k] == v
    # independent rescore: frozen 1/2, adapted 2/2
    assert _rescore_mean(
        tmp_path / "PREDICTIONS-FROZEN-VALIDATION.jsonl") == 0.5
    assert _rescore_mean(
        tmp_path / "PREDICTIONS-ADAPTED-VALIDATION.jsonl") == 1.0
    comp_rows = _load_records(
        tmp_path / "PREDICTIONS-FROZEN-COMPOSITIONAL.jsonl")
    assert list(comp_rows) == ["c1"] \
        and comp_rows["c1"]["split"] == "compositional"


def test_score_maps_would_not_serialize(tmp_path):
    """The pre-fix defect: a dict of integer scores is not a prediction
    dict — the writer expression must reject it, not emit garbage."""
    with pytest.raises(TypeError):
        _write_split(tmp_path, "VALIDATION",
                     ({"v1": 1, "v2": 0}, {"v1": 0}))


@pytest.mark.parametrize("run_id", ["v401-3e-8fam-s0", "v401-3e-8fam-e8-s0"])
def test_committed_campaign3e_split_files_rescore(run_id):
    """Every committed VALIDATION / COMPOSITIONAL file is real records
    that rescore to the recorded metrics — proof the shipped artifact
    already carried records, and a guard against regressions."""
    d = RUNS / run_id
    if not d.is_dir():
        pytest.skip("campaign3e evidence not present")
    metrics = json.loads((d / "METRICS.json").read_text())
    checks = [
        ("FROZEN-VALIDATION", metrics["frozen"]["validation"]),
        ("ADAPTED-VALIDATION", metrics["adapted"]["validation"]),
        ("FROZEN-COMPOSITIONAL", metrics["frozen"]["compositional"]),
        ("ADAPTED-COMPOSITIONAL", metrics["adapted"]["compositional"]),
    ]
    for suffix, expected in checks:
        path = d / f"PREDICTIONS-{suffix}.jsonl"
        rows = _load_records(path)
        assert rows, f"{path.name} empty"
        for rid, r in rows.items():
            assert r["split"] and r["id"] == rid and "output" in r
            assert "expected" in r and "prompt_sha256" in r
        assert _rescore_mean(path) == pytest.approx(expected, abs=1e-9)
