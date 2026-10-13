"""SCI-305 — an undefined false-activation rate is never reported as zero.

The conditional rate requires a baseline-correct population. Campaign
3A's zero-denominator case must surface as NOT_ESTIMABLE (None), and
the qualification record must treat that as insufficient evidence —
absence of off-target flips is not demonstrated by an unmeasurable
metric.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.evaluators import exact_match  # noqa: E402
from minagi.v161.stats import false_activation_rate  # noqa: E402


def test_zero_denominator_is_not_estimable_not_zero():
    base = {"a": "wrong", "b": "wrong"}
    arm = {"a": "also wrong", "b": "still wrong"}
    golds = {"a": "right", "b": "right"}
    fa = false_activation_rate(base, arm, golds, exact_match)
    assert fa["n_baseline_correct"] == 0
    assert fa["false_activation_rate"] is None
    assert fa["conditional_regression_rate"] is None
    assert fa["status"] == "NOT_ESTIMABLE"
    # counts are still honest
    assert fa["n_pairs"] == 2
    assert fa["n_false_activation"] == 0


def test_estimable_rate_is_computed_and_statused():
    base = {"a": "right", "b": "right", "c": "wrong"}
    arm = {"a": "right", "b": "flipped", "c": "wrong"}
    golds = {"a": "right", "b": "right", "c": "right"}
    fa = false_activation_rate(base, arm, golds, exact_match)
    assert fa["n_baseline_correct"] == 2
    assert fa["status"] == "ESTIMATED"
    assert fa["n_false_activation"] == 1
    assert fa["false_activation_rate"] == 0.5
    assert fa["n_conditional_regression"] == 1
    assert fa["conditional_regression_rate"] == 0.5
    assert fa["changed_ids"] == ["b"]


def test_conditional_regression_is_a_superset_of_flips():
    """Every discrete flip is also a conditional regression; a row that
    regresses without changing its output is counted only by score."""
    base = {"a": "right", "b": "right"}
    arm = {"a": "wrong", "b": "right"}
    golds = {"a": "right", "b": "right"}
    fa = false_activation_rate(base, arm, golds, exact_match)
    assert fa["n_baseline_correct"] == 2
    assert fa["n_conditional_regression"] == 1
    assert fa["n_false_activation"] == 1
    assert set(fa["changed_ids"]) <= set(["a"])
    assert fa["false_activation_rate"] <= fa["conditional_regression_rate"]


def test_empty_overlap_is_not_estimable():
    fa = false_activation_rate({}, {}, {}, exact_match)
    assert fa["status"] == "NOT_ESTIMABLE"
    assert fa["false_activation_rate"] is None
    assert fa["n_pairs"] == 0
