"""SCI-303 — answer-contract diagnostics scoring.

Diagnostics are recorded alongside — never instead of — the strict
exact-match score: a correct prefix followed by a contradictory answer
must not receive primary correctness credit. These tests pin the
metric definitions, the EOS/truncation observations, and the frozen
output contract's digest behavior.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.evaluators import exact_match  # noqa: E402
from minagi.v161.prompt_contract import (  # noqa: E402
    ANSWER_ONLY_INSTRUCTION, OutputContract, answer_only_output_contract,
    generation_diagnostics, output_contract_from_config)


def _diag(expected, prediction, tokens=4, limit=32, eos_hit=None):
    return generation_diagnostics(expected, prediction,
                                  generated_tokens=tokens,
                                  max_new_tokens=limit, eos_hit=eos_hit)


def test_strict_metric_mirrors_the_registered_evaluator():
    pairs = [("olleh", "olleh"), ("olleh", "OLLEH"), ("olleh", " olleh "),
             ("olleh", "wrong"), ("olleh", "olleh but longer")]
    for expected, prediction in pairs:
        d = _diag(expected, prediction)
        assert d["strict_exact_match"] == \
            float(exact_match(prediction, expected))


def test_prefix_credit_is_diagnostic_only():
    # a correct answer followed by an explanation: strict credit is 0,
    # the diagnostic surfaces the prefix
    d = _diag("olleh", "olleh because reversing swaps the order")
    assert d["strict_exact_match"] == 0.0
    assert d["correct_answer_prefix"] == 1.0
    assert d["valid_answer_format"] == 1.0
    # a contradictory continuation still gets no primary credit
    d = _diag("olleh", "olleh ... actually the answer is hello")
    assert d["strict_exact_match"] == 0.0
    assert d["correct_answer_prefix"] == 1.0


def test_first_line_diagnostic():
    d = _diag("olleh", "olleh\nExplanation: ...")
    assert d["strict_exact_match"] == 0.0
    assert d["correct_first_line"] == 1.0
    assert d["valid_answer_format"] == 0.0  # multi-line
    d = _diag("olleh", "Sure: olleh")
    assert d["correct_first_line"] == 0.0
    assert d["correct_answer_prefix"] == 0.0


def test_format_and_empty_predictions():
    assert _diag("olleh", "")["valid_answer_format"] == 0.0
    assert _diag("olleh", "   \n  ")["valid_answer_format"] == 0.0
    assert _diag("olleh", "olleh")["valid_answer_format"] == 1.0


def test_eos_and_truncation_observations():
    # precise observation wins
    d = _diag("olleh", "olleh", tokens=2, limit=32, eos_hit=True)
    assert d["eos_termination"] == 1.0 and d["max_token_truncation"] == 0.0
    # inferred: stopped before the limit
    d = _diag("olleh", "olleh", tokens=2, limit=32)
    assert d["eos_termination"] == 1.0
    # inferred: hit the token limit
    d = _diag("olleh", "oll", tokens=32, limit=32)
    assert d["eos_termination"] == 0.0 and d["max_token_truncation"] == 1.0
    assert d["generated_tokens"] == 32


def test_contract_from_config_defaults_to_historical_surface():
    contract = output_contract_from_config({}, 32)
    assert contract.instruction == ""
    assert contract.max_new_tokens == 32
    assert contract == OutputContract()


def test_contract_from_config_freezes_instruction_and_limit():
    contract = output_contract_from_config(
        {"output_contract": {"instruction": ANSWER_ONLY_INSTRUCTION,
                             "max_new_tokens": 64}}, 64)
    assert contract.instruction == ANSWER_ONLY_INSTRUCTION
    assert contract.max_new_tokens == 64
    with pytest.raises(ValueError, match="disagrees"):
        output_contract_from_config(
            {"output_contract": {"max_new_tokens": 64}}, 32)


def test_contract_digests_are_stable_and_sensitive():
    a = answer_only_output_contract(64)
    b = answer_only_output_contract(64)
    assert a.digest == b.digest
    c = answer_only_output_contract(32)
    assert a.digest != c.digest
    d = OutputContract(instruction=a.instruction + " ",
                       max_new_tokens=64)
    assert a.digest != d.digest
    assert a.digest.startswith("sha256:")
