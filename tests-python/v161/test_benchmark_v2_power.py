"""Campaign 3B.2 — the expanded held-out benchmark.

The v1 held-out block is 12 tasks, so one answer moves accuracy by
8.33 pp and a paired comparison needs roughly a 44 pp effect to be
detectable at 80% power. That is enough for the large composition effect
(29 pp) and useless for a rank ladder, where effects are expected to be
a few points.

v2 keeps the TRAINING block byte-identical to v1 so that a run trained
on v2 sees the same training condition as one trained on v1, and only
the evaluation changes. These tests pin that invariant, the benchmark
size, and determinism — the properties that make a v1/v2 comparison
meaningful.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_positive_control import build_corpus  # noqa: E402
from generate_positive_control_v2 import (  # noqa: E402
    HELDOUT_FAMILIES_V2, build_corpus_v2)


def _train(rows):
    return [r for r in rows if r["split"] == "train"]


def test_training_block_is_byte_identical_to_v1():
    """Without this, a v1/v2 comparison confounds training with evaluation."""
    v1 = json.dumps(_train(build_corpus()), sort_keys=True)
    v2 = json.dumps(_train(build_corpus_v2()), sort_keys=True)
    assert v1 == v2, (
        "the v2 training block drifted from v1; runs on the two corpora "
        "would no longer be comparable")


def test_hidden_block_is_large_enough_to_be_useful():
    rows = [r for r in build_corpus_v2() if r["split"] == "hidden"]
    assert len(rows) >= 100, (
        f"hidden block is {len(rows)} tasks; a rank ladder needs ~105 for "
        "a 15 pp paired effect at 80% power")
    families = sorted({r["family"] for r in rows})
    assert len(families) >= 8, f"only {len(families)} held-out families"
    # granularity, stated as the thing that matters
    assert 100 / len(rows) <= 1.0, (
        "one answer should move accuracy by at most 1 pp")


def test_held_out_families_are_disjoint_from_the_trained_ones():
    train_fams = {r["family"] for r in build_corpus() if r["split"] == "train"}
    held = set(HELDOUT_FAMILIES_V2)
    assert not (train_fams & held), (
        f"held-out families overlap the trained ones: {train_fams & held}")


def test_every_held_out_family_has_its_own_transformation():
    fns = [fn for _line, fn in HELDOUT_FAMILIES_V2.values()]
    assert len(fns) == len({id(f) for f in fns})
    # and each transformation is actually distinct on a sample word
    outs = {fn("cat") for _line, fn in HELDOUT_FAMILIES_V2.values()}
    assert len(outs) >= len(HELDOUT_FAMILIES_V2) - 1, (
        "two held-out families produce identical outputs, so they are not "
        "independent evidence")


def test_expected_answers_rederive_from_the_family_function():
    for row in build_corpus_v2():
        if row["split"] != "hidden":
            continue
        _line, fn = HELDOUT_FAMILIES_V2[row["family"]]
        payload = row["prompt"].split("Input: ", 1)[1] \
            .rsplit("\nAnswer:", 1)[0]
        assert fn(payload) == row["expected"], row["id"]


def test_generator_is_deterministic(tmp_path):
    out = tmp_path / "v2.jsonl"
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "generate_positive_control_v2.py"),
             "--out", str(out)],
            capture_output=True, text=True, cwd=str(ROOT))
        assert proc.returncode == 0, proc.stderr
        digest = __import__("hashlib").sha256(out.read_bytes()).hexdigest()
        if _ == 0:
            first = digest
    assert digest == first, "regenerating the corpus produced a different file"


#: KNOWN v1 CORPUS DEFECT, inherited by v2 by design.
#:
#: Every v1 training word is three letters long, and for a three-letter
#: word "swap the first and last characters" IS "reverse the word". So
#: `pc-rev` and `pc-swap` are the same function on the training data: the
#: corpus declares six train families but carries five distinct
#: functions.
#:
#: v2 keeps the training block byte-identical to v1 deliberately, so that
#: a v1-trained and a v2-trained run see the same training condition and
#: are comparable. Fixing the collision here would break that invariant
#: and silently change every training condition.
#:
#: Consequence for interpretation, recorded rather than hidden: any
#: "breadth" count over the train families overstates by one, and
#: `pc-rev` and `pc-swap` are not independent evidence about each other.
#: A future corpus should use varied word lengths.
KNOWN_FUNCTION_COLLISIONS = {("pc-rev", "pc-swap")}


def _family_function_collisions(rows):
    by_word: dict = {}
    for r in rows:
        word = r["prompt"].split("Input: ", 1)[1].rsplit("\nAnswer:", 1)[0]
        by_word.setdefault(word, {})[r["family"]] = r["expected"]
    families = sorted({r["family"] for r in rows})
    found = set()
    for i, a in enumerate(families):
        for b in families[i + 1:]:
            shared = [w for w in by_word if a in by_word[w] and b in by_word[w]]
            if shared and all(by_word[w][a] == by_word[w][b] for w in shared):
                found.add((a, b))
    return found


def test_family_function_collisions_are_exactly_the_known_ones():
    """Pins the defect: no NEW collision, and the known one not silently
    fixed (which would break v1/v2 comparability)."""
    found = _family_function_collisions(
        [r for r in build_corpus() if r["split"] == "train"])
    assert found == KNOWN_FUNCTION_COLLISIONS, (
        f"family-function collisions changed: found {sorted(found)}, "
        f"expected {sorted(KNOWN_FUNCTION_COLLISIONS)}. A new collision "
        "means the corpus overstates its breadth further; a missing one "
        "means the training block was altered and v1/v2 comparability is "
        "broken")


def test_v2_inherits_the_collision_so_the_two_corpora_stay_comparable():
    v1 = _family_function_collisions(
        [r for r in build_corpus() if r["split"] == "train"])
    v2 = _family_function_collisions(
        [r for r in build_corpus_v2() if r["split"] == "train"])
    assert v1 == v2, "v2's training block no longer matches v1's"
