"""v4.0.3-r1 — the independent evaluation receipt.

An ``independent-evaluation-v1`` receipt is the artifact that lifts a
locally recomputed run to INDEPENDENTLY_VERIFIED: an evaluation-role
signer recomputed the evidence itself and bound the exact identities it
measured. These tests pin the acceptance contract — signature,
role, and *exact evidence identities* — and the forgeries the audit
requires to fail: self-issued evidence, wrong-role signatures, altered
predictions, replayed receipts and incompatible model identities.
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.v161.authority import (  # noqa: E402
    AuthorityRecord, AuthorityRegistry)
from minagi.v161.evidence_reconciliation import (  # noqa: E402
    adapter_artifact_digest_of)
from minagi.v161.independent_eval_receipt import (  # noqa: E402
    BINDING_FIELDS, INDEPENDENT_STATUS, RECEIPT_FILE, RECEIPT_SCHEMA,
    evidence_binding, issue_evaluation_receipt,
    verify_evaluation_receipt, write_receipt_file)


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


def build_run(root: Path, run_id="run-ok", *,
              frozen_hidden=("h1", "h2", "h3", "h4"),
              adapted_hidden_correct=("h2",)) -> Path:
    """A minimal COMPLETE, rescorable run (same shape the reconciliation
    and outcome tests use), with a within_holdout and a retention
    surface so every binding field is exercised."""
    d = root / run_id
    (d / "adapter").mkdir(parents=True)

    hid_f = [_rec(r, "fx", r.upper(), r.upper() if r in frozen_hidden[:1]
                  else "X", "hidden") for r in frozen_hidden]
    hid_a = [_rec(r, "fx", r.upper(),
                  r.upper() if r in adapted_hidden_correct else "X",
                  "hidden") for r in frozen_hidden]
    sub = [_rec("s1", "famA", "E", "E", "train-subset"),
           _rec("s2", "famA", "F", "F", "train-subset")]
    who_f = [_rec("w1", "famA", "G", "G", "within_holdout"),
             _rec("w2", "famA", "H", "X", "within_holdout")]
    who_a = [_rec("w1", "famA", "G", "G", "within_holdout"),
             _rec("w2", "famA", "H", "H", "within_holdout")]
    ret_f = [_rec("r1", "famR", "K", "K", "retention"),
             _rec("r2", "famR", "L", "L", "retention")]
    ret_a = [_rec("r1", "famR", "K", "K", "retention"),
             _rec("r2", "famR", "L", "L", "retention")]

    _write_jsonl(d / "PREDICTIONS-FROZEN.jsonl", hid_f)
    _write_jsonl(d / "PREDICTIONS-ADAPTED.jsonl", hid_a)
    _write_jsonl(d / "PREDICTIONS-SUBSET.jsonl", sub)
    _write_jsonl(d / "PREDICTIONS-FROZEN-WITHIN_HOLDOUT.jsonl", who_f)
    _write_jsonl(d / "PREDICTIONS-ADAPTED-WITHIN_HOLDOUT.jsonl", who_a)
    _write_jsonl(d / "PREDICTIONS-FROZEN-RETENTION.jsonl", ret_f)
    _write_jsonl(d / "PREDICTIONS-ADAPTED-RETENTION.jsonl", ret_a)

    (d / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": "model/m1"}))
    (d / "adapter" / "adapter_model.bin").write_bytes(b"weights")
    spec = {"rank": 16, "seed": 0}
    (d / "adapter" / "TRAINING_SPEC.json").write_text(json.dumps(spec))

    config = {
        "families": ["famA"], "model_id": "model/m1",
        "model_revision": "rev-1", "score_oracle": "exact-match-v2",
        "subset_rows": 2, "hidden_rows": len(frozen_hidden),
        "within_holdout_rows": 2, "retention_rows": 2,
        "seed": 0, "lora_spec_digest": digest(spec),
        "prompt_contract_digest": "sha256:" + "1" * 64,
        "output_contract_digest": "sha256:" + "2" * 64,
    }
    config["config_digest"] = digest(config)
    config["run_id"] = run_id
    config["attempt_id"] = "a1"
    (d / "CONFIG.json").write_text(json.dumps(config, sort_keys=True))

    n_frozen_correct = len(frozen_hidden[:1])
    destroyed = sorted(set(frozen_hidden[:1])
                       - set(adapted_hidden_correct))
    gained = sorted(set(adapted_hidden_correct)
                    - set(frozen_hidden[:1]))
    metrics = {
        "fit_gate": {"adapted_subset_accuracy": 1.0, "threshold": 0.95,
                     "passed": True},
        "frozen": {"subset": 1.0, "hidden": n_frozen_correct / 4,
                   "within_holdout": 0.5, "retention": 1.0},
        "adapted": {"subset": 1.0,
                    "hidden": len(adapted_hidden_correct) / 4,
                    "within_holdout": 1.0, "retention": 1.0},
        "within_holdout": {"rows": 2, "frozen": 0.5, "adapted": 1.0},
        "retention": {"rows": 2, "frozen": 1.0, "adapted": 1.0,
                      "n_frozen_correct": 2, "n_destroyed": 0,
                      "destroyed_ids": [], "gained_ids": [],
                      "destruction_rate": 0.0},
        "capability_destruction": {
            "n_frozen_correct_hidden": n_frozen_correct,
            "destroyed_ids": destroyed, "n_destroyed": len(destroyed),
            "gained_ids": gained, "n_gained": len(gained)},
        "training": {"train_seconds": 12.5},
    }
    (d / "METRICS.json").write_text(json.dumps(metrics, sort_keys=True))
    (d / "TRAINING_RECEIPT.json").write_text(json.dumps({
        "adapter_artifact_digest": adapter_artifact_digest_of(
            d / "adapter"),
        "train_seconds": 12.5}))
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
        "config_digest": config["config_digest"], "attempt_id": "a1",
        "files": inventory}))
    return d


def _authority(*roles):
    signers = {r: Ed25519Signer.generate() for r in roles}
    recs = [AuthorityRecord(
        role=r, key_id=s.key_id,
        public_key_b64=base64.b64encode(s.public_bytes()).decode())
        for r, s in signers.items()]
    reg = AuthorityRegistry(recs)
    return reg, reg.verifier(), signers


def _tamper(d: Path, fn):
    """Mutate a file then re-issue a consistent commit inventory so the
    run stays hash-consistent (integrity holds; the CONTENT changed)."""
    fn(d)
    commit = json.loads((d / "RUN_COMMIT.json").read_text())
    commit["files"] = {p.relative_to(d).as_posix(): _sha(p)
                       for p in sorted(d.rglob("*"))
                       if p.is_file() and p.name != "RUN_COMMIT.json"}
    (d / "RUN_COMMIT.json").write_text(json.dumps(commit))


# ---------- issuance -----------------------------------------------------

def test_issue_binds_every_evidence_identity(tmp_path):
    d = build_run(tmp_path)
    _, _, signers = _authority("evaluation")
    receipt = issue_evaluation_receipt(signers["evaluation"], d)
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["evaluation_status"] == INDEPENDENT_STATUS
    assert receipt["evaluator_authority_id"] == \
        signers["evaluation"].key_id
    for f in BINDING_FIELDS:
        assert f in receipt
    # the binding recomputes deterministically
    assert evidence_binding(d) == {
        f: receipt[f] for f in BINDING_FIELDS}
    assert receipt["experiment_id"] == d.name
    assert receipt["candidate_digest"].startswith("sha256:")
    assert receipt["predictions_digest"].startswith("sha256:")
    assert receipt["retention_metrics_digest"].startswith("sha256:")
    assert receipt["signature"]


def test_issue_refuses_unverifiable_evidence(tmp_path):
    """The evaluator signs only what its own recomputation verifies —
    an inconsistent run gets no receipt."""
    d = build_run(tmp_path)
    _tamper(d, lambda d: (d / "METRICS.json").write_text(
        json.dumps({"adapted": {"hidden": 0.99}})))
    _, _, signers = _authority("evaluation")
    with pytest.raises(ValueError, match="VERIFIED"):
        issue_evaluation_receipt(signers["evaluation"], d)


# ---------- verification -------------------------------------------------

def test_issued_receipt_verifies_against_its_run(tmp_path):
    d = build_run(tmp_path)
    reg, ver, signers = _authority("evaluation")
    receipt = issue_evaluation_receipt(signers["evaluation"], d)
    write_receipt_file(d, receipt)
    res = verify_evaluation_receipt(
        receipt, verifier=ver, authority=reg, run_dir=d)
    assert res["verified"], res
    assert res["binding_checked"]


def test_forged_signature_rejected(tmp_path):
    d = build_run(tmp_path)
    reg, ver, signers = _authority("evaluation")
    receipt = issue_evaluation_receipt(signers["evaluation"], d)
    forger = Ed25519Signer.generate()
    forged = dict(receipt)
    forged["evaluator_authority_id"] = forger.key_id
    forged["signature"] = forger.sign(
        {k: v for k, v in forged.items()
         if k != "signature"}).signature_b64
    res = verify_evaluation_receipt(
        forged, verifier=ver, authority=reg, run_dir=d)
    assert not res["verified"]
    assert "not authorized" in res["reason"]


def test_wrong_role_signature_rejected(tmp_path):
    """A valid signature from a plan-role key carries no evaluation
    authority — roles are separated."""
    d = build_run(tmp_path)
    reg, ver, signers = _authority("evaluation", "plan")
    receipt = issue_evaluation_receipt(signers["evaluation"], d)
    wrong = dict(receipt)
    wrong["evaluator_authority_id"] = signers["plan"].key_id
    wrong["signature"] = signers["plan"].sign(
        {k: v for k, v in wrong.items()
         if k != "signature"}).signature_b64
    res = verify_evaluation_receipt(
        wrong, verifier=ver, authority=reg, run_dir=d)
    assert not res["verified"]
    assert "plan" in res["reason"]


def test_expired_and_revoked_evaluators_rejected(tmp_path):
    d = build_run(tmp_path)
    ev = Ed25519Signer.generate()
    other = Ed25519Signer.generate()
    reg = AuthorityRegistry([
        AuthorityRecord(
            role="evaluation", key_id=ev.key_id,
            public_key_b64=base64.b64encode(
                ev.public_bytes()).decode(),
            not_after="2000-01-01T00:00:00+00:00"),
        AuthorityRecord(
            role="evaluation", key_id=other.key_id,
            public_key_b64=base64.b64encode(
                other.public_bytes()).decode(), status="revoked")])
    ver = reg.verifier()
    receipt = issue_evaluation_receipt(ev, d)
    res = verify_evaluation_receipt(
        receipt, verifier=ver, authority=reg, run_dir=d)
    assert not res["verified"] and "not authorized" in res["reason"]
    receipt2 = issue_evaluation_receipt(other, d)
    res = verify_evaluation_receipt(
        receipt2, verifier=ver, authority=reg, run_dir=d)
    assert not res["verified"]


def test_altered_predictions_break_the_binding(tmp_path):
    """Recomputed binding catches post-receipt tampering: the receipt
    attested DIFFERENT prediction bytes."""
    d = build_run(tmp_path)
    reg, ver, signers = _authority("evaluation")
    receipt = issue_evaluation_receipt(signers["evaluation"], d)

    def _alter(d):
        p = d / "PREDICTIONS-ADAPTED.jsonl"
        rows = [json.loads(ln) for ln in p.read_text().splitlines()]
        rows[0]["output"] = "FORGED"
        _write_jsonl(p, rows)
    _tamper(d, _alter)
    res = verify_evaluation_receipt(
        receipt, verifier=ver, authority=reg, run_dir=d)
    assert not res["verified"]
    assert res["mismatches"] or "reconcile" in res["reason"]


def test_replayed_receipt_does_not_cover_another_run(tmp_path):
    """A receipt issued for run A must not stand in for run B — every
    binding field names run A's evidence."""
    d_a = build_run(tmp_path, run_id="run-a")
    d_b = build_run(tmp_path, run_id="run-b")
    reg, ver, signers = _authority("evaluation")
    receipt_a = issue_evaluation_receipt(signers["evaluation"], d_a)
    res = verify_evaluation_receipt(
        receipt_a, verifier=ver, authority=reg, run_dir=d_b)
    assert not res["verified"]
    assert "experiment_id" in res["reason"] or res["mismatches"]


def test_incompatible_model_identity_rejected(tmp_path):
    """A receipt bound to model/m1 cannot attest a run declaring a
    different base model — even when that run itself verifies."""
    d = build_run(tmp_path)
    reg, ver, signers = _authority("evaluation")
    receipt = issue_evaluation_receipt(signers["evaluation"], d)
    # a second, self-consistent run declaring a different base model
    d2 = build_run(tmp_path / "b", run_id="run-ok")
    cfg = json.loads((d2 / "CONFIG.json").read_text())
    cfg["model_id"] = "model/other"
    cfg["config_digest"] = digest(
        {k: v for k, v in cfg.items()
         if k not in ("config_digest", "attempt_id", "run_id")})
    (d2 / "CONFIG.json").write_text(json.dumps(cfg))
    (d2 / "adapter" / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "model/other"}))
    (d2 / "ADAPTER_MANIFEST.json").write_text(json.dumps({
        "run_id": "run-ok", "adapter_dir": "adapter",
        "files": {p.name: _sha(p)
                  for p in sorted((d2 / "adapter").iterdir())
                  if p.is_file()},
        "reload_verified": True,
        "prompt_contract_digest": cfg["prompt_contract_digest"],
        "output_contract_digest": cfg["output_contract_digest"]}))
    (d2 / "TRAINING_RECEIPT.json").write_text(json.dumps({
        "adapter_artifact_digest": adapter_artifact_digest_of(
            d2 / "adapter"),
        "train_seconds": 12.5}))
    _tamper(d2, lambda d: None)
    commit = json.loads((d2 / "RUN_COMMIT.json").read_text())
    commit["config_digest"] = cfg["config_digest"]
    (d2 / "RUN_COMMIT.json").write_text(json.dumps(commit))
    res = verify_evaluation_receipt(
        receipt, verifier=ver, authority=reg, run_dir=d2)
    assert not res["verified"]


def test_detached_receipt_refused_for_trusted_import(tmp_path):
    """A receipt that cannot be bound to its evidence is not
    independent verification — signature alone is insufficient."""
    d = build_run(tmp_path)
    reg, ver, signers = _authority("evaluation")
    receipt = issue_evaluation_receipt(signers["evaluation"], d)
    res = verify_evaluation_receipt(
        receipt, verifier=ver, authority=reg)   # no run_dir
    assert not res["verified"]
    assert "evidence identities" in res["reason"]
    # allow_detached exists for tooling introspection only — it never
    # claims binding
    res = verify_evaluation_receipt(
        receipt, verifier=ver, authority=reg, allow_detached=True)
    assert res["verified"] and not res["binding_checked"]


def test_no_authority_no_verification(tmp_path):
    """Without a trust root the signer's role cannot be established —
    even a perfect signature is not independent evaluation."""
    d = build_run(tmp_path)
    _, ver, signers = _authority("evaluation")
    receipt = issue_evaluation_receipt(signers["evaluation"], d)
    res = verify_evaluation_receipt(
        receipt, verifier=ver, authority=None, run_dir=d)
    assert not res["verified"]
    assert "authority" in res["reason"]


def test_receipt_file_lifecycle(tmp_path):
    """EVALUATION_RECEIPT.json is post-commit evidence: it lands beside
    the committed inventory and stays verifiable."""
    d = build_run(tmp_path)
    reg, ver, signers = _authority("evaluation")
    receipt = issue_evaluation_receipt(signers["evaluation"], d)
    p = write_receipt_file(d, receipt)
    assert p.name == RECEIPT_FILE
    doc = json.loads(p.read_text())
    res = verify_evaluation_receipt(
        doc, verifier=ver, authority=reg, run_dir=d)
    assert res["verified"]
