"""SCI-302 — the verified replay baseline must actually operate.

Campaign 3A's L5 replay buffer contained zero traces across all ten
seeds, so the primary comparison had no functioning non-neural
comparator. These tests pin the repaired L5b mechanism (Step 9 table):

  * verified training examples supplied  -> buffer becomes nonempty
  * relevant evaluation prompt supplied  -> eligible examples retrieved
  * no eligible example                  -> explicit no-hit recorded
  * example exceeds the context budget   -> deterministically trimmed
  * retrieval state tampered with        -> digest verification fails
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.arms import (  # noqa: E402
    OracleVerifiedReplayArm)

TRAIN_ROWS = [
    {"id": "t-1", "family": "rev", "split": "train",
     "prompt": "Reverse the string 'hello'.", "expected": "olleh"},
    {"id": "t-2", "family": "sort", "split": "train",
     "prompt": "Sort the characters of 'banana'.", "expected": "aaabnn"},
    {"id": "t-3", "family": "upper", "split": "train",
     "prompt": "Uppercase the word 'test'.", "expected": "TEST"},
]


def _arm(tmp_path, rows=None, **kwargs):
    arm = OracleVerifiedReplayArm(**kwargs)
    arm.build_state(tmp_path / "L5b", rows if rows is not None else TRAIN_ROWS,
                    split_digest="sha256:" + "a" * 64)
    arm.persist()
    return arm


def test_verified_examples_make_the_buffer_nonempty(tmp_path):
    arm = _arm(tmp_path)
    assert len(arm.records) == len(TRAIN_ROWS)
    assert arm.entries
    doc = json.loads((tmp_path / "L5b" / "REPLAY.json").read_text())
    assert len(doc["records"]) == len(TRAIN_ROWS)
    assert arm.verify_state() == len(TRAIN_ROWS)


def test_relevant_prompt_retrieves_eligible_examples(tmp_path):
    arm = _arm(tmp_path)
    prompt = "Reverse the string 'world'."
    augmented = arm.augment(prompt, "rev")
    assert augmented != prompt
    assert "Verified experience:" in augmented
    stats = arm.retrieval_stats()
    assert stats["records"] == len(TRAIN_ROWS)
    assert stats["queries"] == 1
    assert stats["hit_queries"] == 1
    assert stats["no_hit_queries"] == 0
    assert stats["retrieved_task_ids"], "no retrieval ids recorded"
    assert stats["injected_chars"] > 0
    # the injected block carries an admitted training example
    assert "olleh" in augmented


def test_no_eligible_example_records_an_explicit_no_hit(tmp_path):
    arm = _arm(tmp_path, rows=[])
    assert arm.records == []
    augmented = arm.augment("Anything at all.", "x")
    assert augmented == "Anything at all."
    stats = arm.retrieval_stats()
    assert stats["records"] == 0
    assert stats["hit_queries"] == 0
    assert stats["no_hit_queries"] == 1


def test_context_budget_trims_deterministically(tmp_path):
    arm = _arm(tmp_path, k=3, max_block_chars=40)
    first = arm.augment("Sort the characters of 'banana'.", "sort")
    second = arm.augment("Sort the characters of 'banana'.", "sort")
    assert first == second, "retrieval is not deterministic"
    log = arm.retrieval_log[-1]
    assert log["injected"] <= 40
    assert log["trimmed"] > 0
    assert log["augmented_chars"] == len(first)


def test_tampered_state_fails_digest_verification(tmp_path):
    arm = _arm(tmp_path)
    path = tmp_path / "L5b" / "REPLAY.json"
    doc = json.loads(path.read_text())
    doc["entries"][0]["answer"] = "tampered"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    try:
        arm.verify_state()
    except ValueError as exc:
        assert "answer digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered replay state was accepted")


def test_token_costs_are_recorded_when_a_counter_is_set(tmp_path):
    arm = _arm(tmp_path)
    arm.token_counter = len  # characters as a stand-in token count
    arm.augment("Reverse the string 'world'.", "rev")
    stats = arm.retrieval_stats()
    assert stats["augmented_tokens"] >= stats["prompt_tokens"] > 0
    assert stats["injected_tokens"] == (stats["augmented_tokens"]
                                        - stats["prompt_tokens"])
