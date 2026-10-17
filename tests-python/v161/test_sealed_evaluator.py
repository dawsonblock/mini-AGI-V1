"""v4.0.3-r2 Phase 5 — evaluator-owned sealed holdout.

The shipped campaign4a holdout was reclassified as a public dev
benchmark — final confirmation needs evaluator-custody tasks whose
answers the research agent cannot inspect. These tests pin the seal:
answers are permission-sealed, the dataset commitment binds receipts,
repeated evaluations are refused, and the confirmation result exposes
only aggregates.
"""
from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(Path(__file__).parent))

from test_independent_eval_receipt import _authority  # noqa: E402
from test_model_execution_provenance import _loader  # noqa: E402

from minagi.v161.sealed_evaluator import (  # noqa: E402
    ANSWERS_FILE, SealedHoldoutStore, generate_fresh_tasks,
    run_sealed_confirmation)


def _adapter(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": "model/m1"}))
    (d / "adapter_model.bin").write_bytes(b"adapter-bytes")
    return d


def test_fresh_tasks_have_no_corpus_overlap(tmp_path):
    """Fresh families are generated per-eval — distinct ids, sealed
    answers, public prompts."""
    tasks = generate_fresh_tasks(42)
    assert len(tasks) >= 8
    fams = {t["family"] for t in tasks}
    assert len(fams) >= 4
    store = SealedHoldoutStore(tmp_path / "sealed")
    meta = store.create_eval_set(tasks)
    assert meta["n_tasks"] == len(tasks)
    # public surface carries no answers
    pub = store.public_tasks(meta["eval_id"])
    assert all("expected" not in t for t in pub)
    # answers exist only in the sealed file
    assert sorted(store.load_answers(meta["eval_id"])) == \
        sorted(t["id"] for t in tasks)


def test_answers_file_is_owner_only(tmp_path):
    """The sealed answers carry filesystem custody — mode 0600 inside
    a 0700 store."""
    store = SealedHoldoutStore(tmp_path / "sealed")
    meta = store.create_eval_set(generate_fresh_tasks(7))
    apath = store._dir(meta["eval_id"]) / ANSWERS_FILE
    assert stat.S_IMODE(apath.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE((tmp_path / "sealed").stat().st_mode) \
        & 0o077 == 0


def test_dataset_commitment_binds_execution(tmp_path):
    """The sealed set's commitment is exactly what the execution
    receipt binds — the evaluator ran THIS task set."""
    reg, ver, signers = _authority("evaluation")
    store = SealedHoldoutStore(tmp_path / "sealed")
    tasks = generate_fresh_tasks(11)
    meta = store.create_eval_set(tasks)
    outs = [t["expected"] for t in tasks]
    res = run_sealed_confirmation(
        store, meta["eval_id"], signers["evaluation"],
        model_id="model/m1", model_revision="rev-1",
        adapter_dir=_adapter(tmp_path),
        gates={"min_accuracy": 0.5},
        generation_config={"max_new_tokens": 8},
        model_loader=_loader(outs))
    assert res["dataset_digest"] == meta["dataset_digest"]
    assert res["n_correct"] == res["n_tasks"]
    assert res["gate"]["status"] == "PASS"
    # the receipt's dataset_digest equals the sealed commitment — the
    # executed tasks ARE the sealed set
    assert res["execution_receipt"]["dataset_digest"] == \
        meta["dataset_digest"]


def test_result_leaks_no_answers(tmp_path):
    """The returned confirmation carries aggregates + receipt — no
    expected answers, no per-row labels."""
    reg, ver, signers = _authority("evaluation")
    store = SealedHoldoutStore(tmp_path / "sealed")
    tasks = generate_fresh_tasks(13)
    meta = store.create_eval_set(tasks)
    outs = [t["expected"] for t in tasks]
    res = run_sealed_confirmation(
        store, meta["eval_id"], signers["evaluation"],
        model_id="model/m1", model_revision="rev-1",
        adapter_dir=_adapter(tmp_path),
        generation_config={}, model_loader=_loader(outs))
    blob = json.dumps(res)
    assert "answers" not in res
    assert "scores" not in res
    # multi-char answers must not appear as plaintext anywhere in the
    # result (digests commit to them — plaintext never leaves)
    long_answers = {t["expected"] for t in tasks
                    if len(str(t["expected"])) > 3}
    for a in long_answers:
        assert a not in blob
    # the receipt binds commitments (sha256 fields), not task text
    assert "expected" not in blob


def test_repeated_evaluation_is_refused(tmp_path):
    """A final holdout is single-use — a second confirmation attempt
    on the same eval id is refused (no adaptive tuning oracle)."""
    reg, ver, signers = _authority("evaluation")
    store = SealedHoldoutStore(tmp_path / "sealed")
    tasks = generate_fresh_tasks(17)
    meta = store.create_eval_set(tasks)
    outs = [t["expected"] for t in tasks]
    kw = dict(model_id="model/m1", model_revision="rev-1",
              adapter_dir=_adapter(tmp_path), generation_config={},
              model_loader=_loader(outs))
    run_sealed_confirmation(store, meta["eval_id"],
                            signers["evaluation"], **kw)
    with pytest.raises(PermissionError):
        run_sealed_confirmation(store, meta["eval_id"],
                                signers["evaluation"], **kw)


def test_tampered_answers_change_commitment(tmp_path):
    """Mutating the sealed answers changes the dataset commitment — a
    receipt bound to the original set no longer attests it."""
    store = SealedHoldoutStore(tmp_path / "sealed")
    tasks = generate_fresh_tasks(19)
    meta = store.create_eval_set(tasks)
    d0 = meta["dataset_digest"]
    apath = store._dir(meta["eval_id"]) / ANSWERS_FILE
    answers = json.loads(apath.read_text())
    answers[tasks[0]["id"]] = "FORGED"
    apath.write_text(json.dumps(answers))
    assert store.dataset_digest(meta["eval_id"]) != d0
