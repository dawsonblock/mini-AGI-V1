"""V4-23 — campaign3e corpus: within-family validation invariants.

The campaign3b defect: the split named "validation" carried families
disjoint from train, so no partition measured unseen inputs of TRAINED
families. campaign3e's validation is tr3-* families on word-disjoint
inputs — these tests pin that property and the committed file's
fidelity to the deterministic generator.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

GEN_PATH = ROOT / "scripts" / "gen_campaign3e.py"
CORPUS = ROOT / "configs" / "campaign3e_tasks.jsonl"
HOLDOUT = ROOT / "configs" / "holdout" / "campaign3e_holdout.jsonl"


def _load_gen():
    spec = importlib.util.spec_from_file_location("gen_campaign3e",
                                                  GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_gen()


def test_within_family_validation_invariants(gen):
    rows, _ = gen.build()
    by = {s: [r for r in rows if r["split"] == s]
          for s in ("train", "validation")}
    train_fams = {r["family"] for r in by["train"]}
    val_fams = {r["family"] for r in by["validation"]}
    # every validation family is a TRAINED family
    assert val_fams <= train_fams
    # and every trained family is validated
    assert val_fams == train_fams
    train_words = {gen._prompt_word(r) for r in by["train"]}
    val_words = {gen._prompt_word(r) for r in by["validation"]}
    # inputs are unseen — the within-family guarantee is a new
    # (function, input) pair, not a replayed training row
    assert not (val_words & train_words)
    # and no sample-id reuse
    assert not ({r["id"] for r in by["validation"]}
                & {r["id"] for r in by["train"]})


def test_check_passes_and_reports_invariants(gen):
    rows, holdout = gen.build()
    res = gen.check(rows, holdout)
    wf = res["within_family"]
    assert wf["all_under_train"]
    assert wf["unseen_words"] == res["counts"]["validation"]


def test_committed_files_match_generator(gen):
    """The committed corpus/holdout are exactly what the deterministic
    generator produces — no hand-edited drift."""
    if not CORPUS.is_file() or not HOLDOUT.is_file():
        pytest.skip("generated files not materialized")
    rows, holdout = gen.build()
    on_disk = [json.loads(ln) for ln in CORPUS.read_text().splitlines()
               if ln.strip()]
    on_disk_ho = [json.loads(ln)
                  for ln in HOLDOUT.read_text().splitlines() if ln.strip()]
    assert on_disk == rows
    assert on_disk_ho == holdout


def test_campaign3b_corpus_defect_is_real_and_confined(gen):
    """Pins that the SHIPPED campaign3b corpus does carry the defect
    (validation families disjoint from train) — so a future silent
    'fix' of that file is itself flagged as an evidence mutation."""
    b_rows = [json.loads(ln) for ln in
              (ROOT / "configs" / "campaign3b_tasks.jsonl")
              .read_text().splitlines() if ln.strip()]
    b_train = {r["family"] for r in b_rows if r["split"] == "train"}
    b_val = {r["family"] for r in b_rows if r["split"] == "validation"}
    assert b_val and not (b_val & b_train)  # the defect, documented
