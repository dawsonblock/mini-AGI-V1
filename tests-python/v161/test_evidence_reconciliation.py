"""V4-27 — behavioral coverage for the independent run verifier.

Builds a complete synthetic run directory (predictions + metrics +
commit inventory + adapter/receipt identities) and then mutates it —
the verifier must catch each tamper class the plan enumerates: missing
or extra prediction files, wrong recorded metrics, inventory drift,
identity violations, and the FAILED-run carve-out.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from minagi.v161.evidence_reconciliation import (  # noqa: E402
    FAILED_RUN, INCONSISTENT, INVALID, VERIFIED,
    adapter_artifact_digest_of, verify_run)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rec(rid, family, expected, output, split):
    return {"split": split, "id": rid, "family": family,
            "prompt_sha256": hashlib.sha256(
                f"prompt-{rid}".encode()).hexdigest(),
            "expected": expected, "output": output}


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n"
                            for r in rows))


def build_run(root: Path, run_id="test-run") -> Path:
    """A minimal COMPLETE run that must come back VERIFIED."""
    d = root / run_id
    (d / "adapter").mkdir(parents=True)

    hidden = [
        _rec("h1", "fx", "A", "A", "hidden"),    # frozen right, adapt breaks
        _rec("h2", "fx", "B", "X", "hidden"),    # frozen wrong, adapt gains
        _rec("h3", "fx", "C", "X", "hidden"),
        _rec("h4", "fx", "D", "X", "hidden"),
    ]
    hidden_a = [
        _rec("h1", "fx", "A", "X", "hidden"),
        _rec("h2", "fx", "B", "B", "hidden"),
        _rec("h3", "fx", "C", "X", "hidden"),
        _rec("h4", "fx", "D", "X", "hidden"),
    ]
    subset = [_rec("s1", "famA", "E", "E", "train-subset"),
              _rec("s2", "famA", "F", "X", "train-subset")]
    val_f = [_rec("v1", "famA", "G", "G", "validation"),
             _rec("v2", "famB", "H", "X", "validation"),
             _rec("v3", "famA", "I", "X", "validation")]
    val_a = [_rec("v1", "famA", "G", "G", "validation"),
             _rec("v2", "famB", "H", "H", "validation"),
             _rec("v3", "famA", "I", "X", "validation")]
    comp_f = [_rec("c1", "famC", "J", "X", "compositional"),
              _rec("c2", "famC", "K", "X", "compositional")]
    comp_a = [_rec("c1", "famC", "J", "J", "compositional"),
              _rec("c2", "famC", "K", "X", "compositional")]

    _write_jsonl(d / "PREDICTIONS-FROZEN.jsonl", hidden)
    _write_jsonl(d / "PREDICTIONS-ADAPTED.jsonl", hidden_a)
    _write_jsonl(d / "PREDICTIONS-SUBSET.jsonl", subset)
    _write_jsonl(d / "PREDICTIONS-FROZEN-VALIDATION.jsonl", val_f)
    _write_jsonl(d / "PREDICTIONS-ADAPTED-VALIDATION.jsonl", val_a)
    _write_jsonl(d / "PREDICTIONS-FROZEN-COMPOSITIONAL.jsonl", comp_f)
    _write_jsonl(d / "PREDICTIONS-ADAPTED-COMPOSITIONAL.jsonl", comp_a)

    # adapter artifact + identities
    (d / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": "model/m1"}))
    (d / "adapter" / "adapter_model.bin").write_bytes(b"weights")
    spec = {"rank": 16, "seed": 0}
    (d / "adapter" / "TRAINING_SPEC.json").write_text(json.dumps(spec))

    config = {
        "families": ["famA"], "model_id": "model/m1",
        "model_revision": "rev-1", "score_oracle": "exact-match-v2",
        "subset_rows": 2, "hidden_rows": 4,
        "validation_rows": 3, "compositional_rows": 2,
        "lora_spec_digest": digest(spec),
        "prompt_contract_digest": "sha256:" + "1" * 64,
        "output_contract_digest": "sha256:" + "2" * 64,
    }
    config_digest = digest(config)
    config["config_digest"] = config_digest
    config["run_id"] = run_id
    config["attempt_id"] = "a1"
    (d / "CONFIG.json").write_text(json.dumps(config, sort_keys=True))

    metrics = {
        "fit_gate": {"threshold": 0.95, "adapted_subset_accuracy": 0.5,
                     "passed": False},
        "frozen": {"subset": 0.0, "hidden": 0.25, "validation": 1 / 3,
                   "compositional": 0.0},
        "adapted": {"subset": 0.5, "hidden": 0.25, "validation": 2 / 3,
                    "compositional": 0.5},
        "validation": {"rows": 3,
                       "frozen": 1 / 3, "adapted": 2 / 3,
                       "within_family": {"frozen": 0.5, "adapted": 0.5},
                       "untrained_family": {"frozen": 0.0,
                                            "adapted": 1.0}},
        "compositional": {"rows": 2, "frozen": 0.0, "adapted": 0.5},
        "capability_destruction": {"n_frozen_correct_hidden": 1,
                                   "destroyed_ids": ["h1"],
                                   "n_destroyed": 1,
                                   "gained_ids": ["h2"], "n_gained": 1},
    }
    (d / "METRICS.json").write_text(json.dumps(metrics, sort_keys=True))

    (d / "TRAINING_RECEIPT.json").write_text(json.dumps({
        "adapter_artifact_digest": adapter_artifact_digest_of(
            d / "adapter")}))
    (d / "ADAPTER_MANIFEST.json").write_text(json.dumps({
        "run_id": run_id, "adapter_dir": "adapter",
        "files": {p.name: _sha(p)
                  for p in sorted((d / "adapter").iterdir())
                  if p.is_file()},
        "reload_verified": True,
        "prompt_contract_digest": config["prompt_contract_digest"],
        "output_contract_digest": config["output_contract_digest"]}))

    inventory = {p.relative_to(d).as_posix(): _sha(p)
                 for p in sorted(d.rglob("*")) if p.is_file()}
    (d / "RUN_COMMIT.json").write_text(json.dumps({
        "run_id": run_id, "status": "COMPLETE",
        "config_digest": config_digest, "attempt_id": "a1",
        "files": inventory}))
    return d


def _recommit(d: Path):
    """Re-hash the inventory after a mutation, keeping status COMPLETE
    — isolates consistency failures from integrity failures."""
    commit = json.loads((d / "RUN_COMMIT.json").read_text())
    commit["files"] = {p.relative_to(d).as_posix(): _sha(p)
                       for p in sorted(d.rglob("*"))
                       if p.is_file() and p.name != "RUN_COMMIT.json"}
    (d / "RUN_COMMIT.json").write_text(json.dumps(commit))


def test_clean_run_verifies(tmp_path):
    res = verify_run(build_run(tmp_path))
    assert res["status"] == VERIFIED, res["mismatches"]
    assert len(res["rescore"]) == 7      # every split file was rescored


def test_missing_prediction_file_is_inconsistent(tmp_path):
    d = build_run(tmp_path)
    (d / "PREDICTIONS-ADAPTED-VALIDATION.jsonl").unlink()
    _recommit(d)
    res = verify_run(d)
    assert res["status"] == INCONSISTENT
    assert any("absent" in m for m in res["mismatches"])


def test_recorded_metric_that_does_not_rescore_fails(tmp_path):
    d = build_run(tmp_path)
    m = json.loads((d / "METRICS.json").read_text())
    m["adapted"]["validation"] = 0.99       # a claim the rows can't back
    (d / "METRICS.json").write_text(json.dumps(m))
    _recommit(d)
    res = verify_run(d)
    assert res["status"] == INCONSISTENT
    assert any("adapted.validation" in m for m in res["mismatches"])


def test_inventory_tamper_is_invalid(tmp_path):
    d = build_run(tmp_path)
    (d / "PREDICTIONS-FROZEN.jsonl").write_text("tampered\n")
    res = verify_run(d)                      # inventory still stale
    assert res["status"] == INVALID


def test_duplicate_prediction_ids_are_inconsistent(tmp_path):
    d = build_run(tmp_path)
    p = d / "PREDICTIONS-FROZEN-VALIDATION.jsonl"
    rows = [json.loads(ln) for ln in p.read_text().splitlines()]
    _write_jsonl(p, rows + [rows[0]])        # duplicate v1
    _recommit(d)
    res = verify_run(d)
    assert res["status"] == INCONSISTENT
    assert any("duplicated" in m for m in res["mismatches"])


def test_wrong_split_labels_are_inconsistent(tmp_path):
    d = build_run(tmp_path)
    p = d / "PREDICTIONS-FROZEN-VALIDATION.jsonl"
    rows = [json.loads(ln) for ln in p.read_text().splitlines()]
    rows[0]["split"] = "hidden"              # mislabeled record
    _write_jsonl(p, rows)
    _recommit(d)
    res = verify_run(d)
    assert res["status"] == INCONSISTENT
    assert any("split labels" in m for m in res["mismatches"])


def test_unexpected_prediction_file_is_inconsistent(tmp_path):
    d = build_run(tmp_path)
    _write_jsonl(d / "PREDICTIONS-FROZEN-RETENTION.jsonl",
                 [_rec("r1", "fx", "A", "A", "retention")])
    _recommit(d)
    res = verify_run(d)
    assert res["status"] == INCONSISTENT
    assert any("unexpected prediction files" in m
               for m in res["mismatches"])


def test_missing_model_revision_is_an_identity_violation(tmp_path):
    d = build_run(tmp_path)
    c = json.loads((d / "CONFIG.json").read_text())
    c.pop("model_revision")
    c["config_digest"] = digest(
        {k: v for k, v in c.items()
         if k not in ("config_digest", "attempt_id", "run_id")})
    (d / "CONFIG.json").write_text(json.dumps(c))
    _recommit(d)
    commit = json.loads((d / "RUN_COMMIT.json").read_text())
    commit["config_digest"] = c["config_digest"]
    (d / "RUN_COMMIT.json").write_text(json.dumps(commit))
    res = verify_run(d)
    assert res["status"] == INCONSISTENT
    assert any("model_revision" in m for m in res["mismatches"])


def test_adapter_bytes_mismatch_is_an_identity_violation(tmp_path):
    d = build_run(tmp_path)
    (d / "adapter" / "adapter_model.bin").write_bytes(b"EVIL-WEIGHTS")
    # keep the commit inventory consistent so ONLY the identity chain
    # fires — the manifest's own per-file hash catches it
    _recommit(d)
    res = verify_run(d)
    assert res["status"] == INCONSISTENT
    assert any("adapter file hash mismatch" in m
               for m in res["mismatches"])


def test_failed_run_is_preserved_not_verified(tmp_path):
    d = build_run(tmp_path)
    commit = json.loads((d / "RUN_COMMIT.json").read_text())
    commit["status"] = "FAILED"
    (d / "RUN_COMMIT.json").write_text(json.dumps(commit))
    res = verify_run(d)
    assert res["status"] == FAILED_RUN


def test_missing_commit_is_invalid(tmp_path):
    d = build_run(tmp_path)
    (d / "RUN_COMMIT.json").unlink()
    assert verify_run(d)["status"] == INVALID
