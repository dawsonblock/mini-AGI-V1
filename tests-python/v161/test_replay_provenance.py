"""SCI-302 — verified replay provenance and leakage refusal.

The replay baseline must prove what it admitted: every record binds the
prompt/answer content digests, the train-split membership digest and
the verification policy; nothing from a hidden/validation/holdout split
may ever enter the buffer, and duplicate or conflicting rows are
refused with reasons.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import sha256_bytes  # noqa: E402
from minagi.v161.arms import (  # noqa: E402
    LABEL_VERIFIER_DIGEST, OracleVerifiedReplayArm)

SPLIT_DIGEST = "sha256:" + "b" * 64
TRAIN = {"id": "t-1", "family": "rev", "split": "train",
         "prompt": "Reverse the string 'hello'.", "expected": "olleh"}
HIDDEN = {"id": "h-1", "family": "rev", "split": "hidden",
          "prompt": "Reverse the string 'secret'.", "expected": "terces"}
VALIDATION = {"id": "v-1", "family": "rev", "split": "validation",
              "prompt": "Reverse the string 'val'.", "expected": "lav"}


def _build(tmp_path, rows, **kwargs):
    arm = OracleVerifiedReplayArm()
    arm.build_state(tmp_path / "L5b", rows, split_digest=SPLIT_DIGEST,
                    **kwargs)
    arm.persist()
    return arm


def test_records_bind_content_split_and_verifier(tmp_path):
    arm = _build(tmp_path, [TRAIN])
    rec = arm.records[0]
    assert rec.task_id == "t-1"
    assert rec.family_id == "rev"
    assert rec.prompt_digest == sha256_bytes(TRAIN["prompt"].encode())
    assert rec.answer_digest == sha256_bytes(TRAIN["expected"].encode())
    assert rec.split_digest == SPLIT_DIGEST
    assert rec.verifier_digest == LABEL_VERIFIER_DIGEST
    assert rec.verification_status == "PASS"


def test_non_train_rows_are_refused_and_never_persisted(tmp_path):
    arm = _build(tmp_path, [TRAIN, HIDDEN, VALIDATION])
    assert [r.task_id for r in arm.records] == ["t-1"]
    reasons = {r["task_id"]: r["reason"] for r in arm.rejected}
    assert "non-train split" in reasons["h-1"]
    assert "non-train split" in reasons["v-1"]
    doc = json.loads((tmp_path / "L5b" / "REPLAY.json").read_text())
    ids = {e["id"] for e in doc["entries"]}
    assert ids == {"t-1"}
    # a hidden label can never be retrieved into an augmented prompt
    augmented = arm.augment("Reverse the string 'x'.", "rev")
    assert "terces" not in augmented


def test_duplicates_and_conflicts_are_refused(tmp_path):
    dup = dict(TRAIN)
    conflict = {"id": "t-9", "family": "rev", "split": "train",
                "prompt": TRAIN["prompt"], "expected": "WRONG"}
    arm = _build(tmp_path, [TRAIN, dup, conflict])
    assert [r.task_id for r in arm.records] == ["t-1"]
    reasons = [r["reason"] for r in arm.rejected]
    assert "duplicate task id" in reasons
    assert "conflicting answer for prompt" in reasons


def test_excluded_ids_and_empty_rows_are_refused(tmp_path):
    empty = {"id": "t-2", "family": "rev", "split": "train",
             "prompt": "", "expected": "x"}
    arm = _build(tmp_path, [TRAIN, empty], excluded_ids=("t-1",))
    assert arm.records == []
    reasons = [r["reason"] for r in arm.rejected]
    assert "excluded id" in reasons
    assert "empty prompt or answer" in reasons


def test_custom_verifier_is_enforced_and_recorded(tmp_path):
    calls = []

    def verifier(row):
        calls.append(row["id"])
        return row["id"] != "t-1"

    arm = _build(tmp_path, [TRAIN], verifier=verifier,
                 verifier_digest="sha256:" + "c" * 64)
    assert arm.records == []
    assert calls == ["t-1"]
    assert arm.rejected[0]["reason"] == "verifier rejected"
    doc = json.loads((tmp_path / "L5b" / "REPLAY.json").read_text())
    assert doc["verifier_digest"] == "sha256:" + "c" * 64
