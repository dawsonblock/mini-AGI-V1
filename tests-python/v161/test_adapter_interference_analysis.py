"""Campaign 3B.2 Step 7 — the interference analyser.

The Campaign 3B.1 claim (L7 loses six, wins none, against L5b) must be
reproducible from preserved per-row predictions rather than from a
summary table. These tests pin the analyser's classification rules, the
undefined-denominator rule, and the two Step 9 diagnostics, using
synthetic prediction files so they need no model.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

SCRIPT = ROOT / "scripts" / "analysis" / "adapter_interference.py"


def _write_corpus(tmp_path, rows):
    p = tmp_path / "corpus.jsonl"
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return p


def _write_preds(tmp_path, arm, recs):
    p = tmp_path / f"PREDICTIONS-{arm}.jsonl"
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))
    return p


def _rec(rid, output, input_sha="sha-x", gen=2, mx=16):
    return {"id": rid, "block": "hidden", "input": "i", "input_sha256": input_sha,
            "output": output, "generated_tokens": gen, "max_new_tokens": mx,
            "status": "ok"}


def _rows():
    return [
        {"id": "t1", "split": "hidden", "family": "pc-len",
         "prompt": "Task: Return the number of characters.\nInput: egg\nAnswer:",
         "expected": "3"},
        {"id": "t2", "split": "hidden", "family": "pc-len",
         "prompt": "Task: Return the number of characters.\nInput: ice\nAnswer:",
         "expected": "3"},
        {"id": "t3", "split": "hidden", "family": "pc-dup",
         "prompt": "Task: Repeat the word twice.\nInput: egg\nAnswer:",
         "expected": "eggegg"},
    ]


def _run(tmp_path, corpus, base_recs, treat_recs, treatment="L7",
         baseline="L5b"):
    _write_preds(tmp_path, baseline, base_recs)
    _write_preds(tmp_path, treatment, treat_recs)
    out = tmp_path / "out.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--predictions-dir", str(tmp_path), "--corpus", str(corpus),
         "--treatment", treatment, "--baseline", baseline,
         "--output", str(out)],
        capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


def test_four_way_outcome_classification(tmp_path):
    corpus = _write_corpus(tmp_path, _rows())
    base = [_rec("t1", "3"), _rec("t2", "3"), _rec("t3", "wrong")]
    treat = [_rec("t1", "3"),          # preserved-correct
             _rec("t2", "eeg"),        # destroyed-correct
             _rec("t3", "eggegg")]     # newly-correct
    rep = _run(tmp_path, corpus, base, treat)
    assert rep["outcomes"] == {"preserved-correct": 1, "newly-correct": 1,
                               "destroyed-correct": 1, "still-incorrect": 0}
    assert rep["paired"]["wins"] == 1 and rep["paired"]["losses"] == 1
    assert rep["paired"]["delta"] == 0.0


def test_destruction_rate_and_undefined_denominator(tmp_path):
    corpus = _write_corpus(tmp_path, _rows())
    base = [_rec("t1", "3"), _rec("t2", "3"), _rec("t3", "wrong")]
    treat = [_rec("t1", "eeg"), _rec("t2", "owe"), _rec("t3", "wrong")]
    rep = _run(tmp_path, corpus, base, treat)
    d = rep["capability_destruction_rate"]
    assert d["numerator"] == 2 and d["denominator"] == 2 and d["D"] == 1.0
    assert d["status"] == "ESTIMATED"

    # baseline never correct -> the rate cannot be estimated
    base2 = [_rec("t1", "x"), _rec("t2", "x"), _rec("t3", "x")]
    rep2 = _run(tmp_path, corpus, base2, treat)
    d2 = rep2["capability_destruction_rate"]
    assert d2["denominator"] == 0
    assert d2["D"] is None
    assert d2["status"] == "NOT_ESTIMABLE"


def test_missing_predictions_are_reported_not_ignored(tmp_path):
    corpus = _write_corpus(tmp_path, _rows())
    base = [_rec("t1", "3")]                       # t2, t3 absent
    treat = [_rec("t1", "3")]
    rep = _run(tmp_path, corpus, base, treat)
    assert rep["n_missing_predictions"] == 2
    assert rep["missing_task_ids"] == ["t2", "t3"]
    assert rep["n_tasks"] == 1, "only scored tasks count toward the report"


def test_input_identity_is_detected(tmp_path):
    corpus = _write_corpus(tmp_path, _rows())
    base = [_rec("t1", "3", "same"), _rec("t2", "3", "a"), _rec("t3", "x", "b")]
    treat = [_rec("t1", "3", "same"), _rec("t2", "eeg", "c"),
             _rec("t3", "x", "b")]
    rep = _run(tmp_path, corpus, base, treat)
    assert rep["input_identity"]["n_identical"] == 2
    assert rep["input_identity"]["all_identical"] is False


def test_family_confusion_detects_a_trained_transformation(tmp_path):
    """'egg' uppercased is pc-upper — a TRAINED family, not the asked
    pc-dup task. That is one learned transformation applied elsewhere."""
    corpus = _write_corpus(tmp_path, _rows())
    base = [_rec("t1", "3"), _rec("t2", "3"), _rec("t3", "wrong")]
    # t2 asks pc-dup for "ice" and answers "ICE" (pc-upper of ice);
    # t3 asks pc-dup for "egg" and answers "EGG" (pc-upper of egg).
    # Both are the SAME trained transformation applied to an unrelated
    # task, so both count as exact matches.
    treat = [_rec("t1", "3"), _rec("t2", "ICE"), _rec("t3", "EGG")]
    rep = _run(tmp_path, corpus, base, treat)
    fc = rep["family_confusion"]
    assert fc["n_confused"] == 2
    assert fc["by_family"]["pc-upper"] == 2
    # pc-sort("egg") is "egg", which CASEFOLDS equal to "EGG". Scoring
    # casefolds, but the diagnosis must not: reporting pc-sort as an
    # exact confusion would over-count. It is separated out instead.
    assert fc["by_family"]["pc-sort"] == 0
    assert fc["by_family_casefold_only"]["pc-sort"] == 1
    assert fc["n_casefold_only"] == 1


def test_format_drift_is_detected(tmp_path):
    corpus = _write_corpus(tmp_path, _rows())
    base = [_rec("t1", "3"), _rec("t2", "3"), _rec("t3", "eggegg")]
    treat = [_rec("t1", "3"), _rec("t2", "Sure! The answer is three."),
             _rec("t3", "eggegg")]
    rep = _run(tmp_path, corpus, base, treat)
    assert rep["format_drift"]["n_broke"] == 1


def test_counterfactual_script_refuses_bad_adapter_arguments():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "analysis" /
                             "adapter_counterfactual.py"),
         "--output", "/tmp/never.json"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode != 0
    assert "at least one --adapter is required" in (proc.stdout + proc.stderr)

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "analysis" /
                             "adapter_counterfactual.py"),
         "--adapter", "NOPE", "--output", "/tmp/never.json"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode != 0
    assert "NAME=PATH" in (proc.stdout + proc.stderr)
