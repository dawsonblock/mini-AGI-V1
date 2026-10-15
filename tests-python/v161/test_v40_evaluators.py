"""V4-01 regression tests — strict match semantics for declarative ops.

The defect: equals / one_of / json_field_equals routed through the
legacy casefold oracle, so a wrong-case answer scored 1.0. These tests
pin strict semantics, the explicit "normalized" opt-in, malformed-input
behavior, and the shared dispatch path.
"""
import pytest

from minagi.v161.evaluators import (DECLARATIVE_EVALUATOR_VERSION,
                                    declarative_score, exact_match,
                                    exact_match_strict, score_row)


def _verify(op, **args):
    return {"type": "declarative", "op": op, "args": args}


def test_strict_version_constant_declared():
    assert DECLARATIVE_EVALUATOR_VERSION == "declarative-evaluator-v2"


def test_equals_rejects_case_mismatch():
    v = _verify("equals")
    assert declarative_score("egg", "EGG", v) == 0.0
    assert declarative_score("EGG", "EGG", v) == 1.0


def test_equals_default_strips_surrounding_whitespace():
    v = _verify("equals")
    assert declarative_score("  egg\n", "egg", v) == 1.0


def test_equals_internal_whitespace_is_significant():
    v = _verify("equals")
    assert declarative_score("eg g", "egg", v) == 0.0


def test_equals_unicode_exact():
    v = _verify("equals")
    # NFC-vs-NFD differ byte-wise; strict matching must not normalize.
    assert declarative_score("café", "café", v) == 1.0
    assert declarative_score("café", "café", v) == 0.0


def test_normalized_match_is_explicit_opt_in():
    v = _verify("equals", match="normalized")
    assert declarative_score("EGG", "egg", v) == 1.0
    # and the normalized path remains casefold-based (provenance only)
    assert exact_match("EGG", "egg") == 1.0
    assert exact_match_strict("EGG", "egg") == 0.0


def test_unknown_match_semantics_raise_never_fallback():
    with pytest.raises(ValueError):
        declarative_score("a", "a", _verify("equals", match="fuzzy"))


def test_one_of_multiple_acceptable_answers():
    v = _verify("one_of", values=["yes", "YES", "affirmative"])
    assert declarative_score("yes", "", v) == 1.0
    assert declarative_score("YES", "", v) == 1.0
    assert declarative_score("Yes", "", v) == 0.0  # case still matters
    assert declarative_score("no", "", v) == 0.0


def test_json_field_equals_strict():
    v = _verify("json_field_equals", path="answer.city", value="Paris")
    assert declarative_score('{"answer": {"city": "Paris"}}', "", v) == 1.0
    assert declarative_score('{"answer": {"city": "paris"}}', "", v) == 0.0
    assert declarative_score('{"answer": {"city": "Lyon"}}', "", v) == 0.0
    assert declarative_score('{"answer": {}}', "", v) == 0.0


def test_json_field_equals_malformed_prediction_zero_not_raise():
    v = _verify("json_field_equals", path="x", value="1")
    assert declarative_score("not json {", "", v) == 0.0
    assert declarative_score("[1,2,3]", "", v) == 0.0
    assert declarative_score("null", "", v) == 0.0


def test_nested_all_of_uses_strict_semantics_inside():
    v = _verify("all_of", specs=[
        {"op": "equals", "args": {}},
        {"op": "regex_fullmatch", "args": {"pattern": "[A-Z]+"}}])
    assert declarative_score("ABC", "ABC", v) == 1.0
    assert declarative_score("abc", "abc", v) == 0.0


def test_score_row_dispatches_verify_declarative_strict():
    row = {"expected": "UPPER",
           "verify": {"type": "declarative", "op": "equals", "args": {}}}
    assert score_row(row, "upper", exact_match_strict) == 0.0
    assert score_row(row, "UPPER", exact_match_strict) == 1.0


def test_score_row_without_verify_uses_default_fn():
    row = {"expected": "Answer"}
    assert score_row(row, "Answer", exact_match_strict) == 1.0
    assert score_row(row, "answer", exact_match_strict) == 0.0


def test_score_row_rejects_unknown_verify_type():
    row = {"expected": "x", "verify": {"type": "shell_exec"}}
    with pytest.raises(ValueError):
        score_row(row, "x", exact_match_strict)


def test_empty_and_none_like_predictions_fail():
    v = _verify("equals")
    assert declarative_score("", "x", v) == 0.0
    assert declarative_score("   ", "x", v) == 0.0


def test_numeric_equals_and_regex_unaffected():
    assert declarative_score("42.0000001", "", _verify(
        "numeric_equals", value="42", tolerance=1e-4)) == 1.0
    assert declarative_score("abc", "", _verify(
        "regex_fullmatch", pattern="[a-z]+")) == 1.0
    assert declarative_score("abc1", "", _verify(
        "regex_fullmatch", pattern="[a-z]+")) == 0.0


def test_contains_none_is_a_blocklist_not_a_correctness_check():
    v = _verify("contains_none", values=["<unsafe>"])
    assert declarative_score("REFUSED", "", v) == 1.0
    assert declarative_score("here: <UNSAFE>", "", v) == 0.0
