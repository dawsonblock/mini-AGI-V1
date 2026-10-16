"""v4.0.3 Phase 1 — the acquisition controller's evidence boundary.

Adversarial coverage for the fabricated-outcome defect: a self-authored
run containing METRICS.json and a hash-consistent RUN_COMMIT.json must
NEVER enter verified history, and the CLI must never accept an
operator-supplied improvement value for a verified record. The three
verification states (INTEGRITY_VERIFIED / EVALUATION_VERIFIED /
QUALIFIED) and the signed-qualification path are pinned end to end.
"""
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.v161.authority import (  # noqa: E402
    AuthorityRecord, AuthorityRegistry, TRUST_ROOT_SCHEMA)
from minagi.v161.capability_controller import (  # noqa: E402
    intervention_history_load)
from minagi.v161.capability_outcomes import (  # noqa: E402
    verify_outcome_receipt)
from minagi.v161.evidence_reconciliation import (  # noqa: E402
    adapter_artifact_digest_of)
from minagi.v161.qualified_outcome import (  # noqa: E402
    NOT_QUALIFIED, QUALIFIED, QUALIFICATION_SCHEMA,
    STATE_EVALUATION, STATE_INTEGRITY, STATE_QUALIFIED,
    derive_qualified_outcome, qualification_decision)

CLI = ROOT / "scripts" / "capability_control.py"


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
    """A minimal COMPLETE+rescorable run the evaluator can verify.

    Defaults: frozen gets h1 right; adapted gets h2 right — a clean
    destroy-1/gain-1 run with a within-family validation surface that
    improves 0.5 -> 1.0.
    """
    d = root / run_id
    (d / "adapter").mkdir(parents=True)

    hid_f = [_rec(r, "fx", r.upper(), r.upper() if r in frozen_hidden[:1]
                  else "X", "hidden") for r in frozen_hidden]
    hid_a = [_rec(r, "fx", r.upper(),
                  r.upper() if r in adapted_hidden_correct else "X",
                  "hidden") for r in frozen_hidden]
    sub = [_rec("s1", "famA", "E", "E", "train-subset"),
           _rec("s2", "famA", "F", "X", "train-subset")]
    val_f = [_rec("v1", "famA", "G", "G", "validation"),
             _rec("v2", "famA", "H", "X", "validation")]
    val_a = [_rec("v1", "famA", "G", "G", "validation"),
             _rec("v2", "famA", "H", "H", "validation")]
    comp_f = [_rec("c1", "famC", "J", "X", "compositional")]
    comp_a = [_rec("c1", "famC", "J", "J", "compositional")]

    _write_jsonl(d / "PREDICTIONS-FROZEN.jsonl", hid_f)
    _write_jsonl(d / "PREDICTIONS-ADAPTED.jsonl", hid_a)
    _write_jsonl(d / "PREDICTIONS-SUBSET.jsonl", sub)
    _write_jsonl(d / "PREDICTIONS-FROZEN-VALIDATION.jsonl", val_f)
    _write_jsonl(d / "PREDICTIONS-ADAPTED-VALIDATION.jsonl", val_a)
    _write_jsonl(d / "PREDICTIONS-FROZEN-COMPOSITIONAL.jsonl", comp_f)
    _write_jsonl(d / "PREDICTIONS-ADAPTED-COMPOSITIONAL.jsonl", comp_a)

    (d / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": "model/m1"}))
    (d / "adapter" / "adapter_model.bin").write_bytes(b"weights")
    spec = {"rank": 16, "seed": 0}
    (d / "adapter" / "TRAINING_SPEC.json").write_text(json.dumps(spec))

    config = {
        "families": ["famA"], "model_id": "model/m1",
        "model_revision": "rev-1", "score_oracle": "exact-match-v2",
        "subset_rows": 2, "hidden_rows": len(frozen_hidden),
        "validation_rows": 2, "compositional_rows": 1,
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
        "fit_gate": {"adapted_subset_accuracy": 0.5, "threshold": 0.95,
                     "passed": False},
        "frozen": {"subset": 0.5, "hidden": n_frozen_correct / 4,
                   "validation": 0.5, "compositional": 0.0},
        "adapted": {"subset": 0.5,
                    "hidden": len(adapted_hidden_correct) / 4,
                    "validation": 1.0, "compositional": 1.0},
        "validation": {"rows": 2, "frozen": 0.5, "adapted": 1.0,
                       "within_family": {"frozen": 0.5, "adapted": 1.0},
                       "untrained_family": {"frozen": None,
                                            "adapted": None}},
        "compositional": {"rows": 1, "frozen": 0.0, "adapted": 1.0},
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


def _recommit(d: Path):
    commit = json.loads((d / "RUN_COMMIT.json").read_text())
    commit["files"] = {p.relative_to(d).as_posix(): _sha(p)
                       for p in sorted(d.rglob("*"))
                       if p.is_file() and p.name != "RUN_COMMIT.json"}
    (d / "RUN_COMMIT.json").write_text(json.dumps(commit))


def _authority(*roles):
    signers = {r: Ed25519Signer.generate() for r in roles}
    recs = [AuthorityRecord(
        role=r, key_id=s.key_id,
        public_key_b64=base64.b64encode(s.public_bytes()).decode())
        for r, s in signers.items()]
    reg = AuthorityRegistry(recs)
    return reg, reg.verifier(), signers


def _trust_root_file(tmp_path, signers) -> Path:
    doc = {"schema": TRUST_ROOT_SCHEMA,
           "authorities": [{"role": r, "key_id": s.key_id,
                            "public_key_b64": base64.b64encode(
                                s.public_bytes()).decode(),
                            "status": "active",
                            "not_before": "", "not_after": ""}
                           for r, s in signers.items()]}
    p = tmp_path / "TRUST_ROOT.json"
    p.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return p


def _cli(*argv) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *argv],
        capture_output=True, text=True, cwd=str(ROOT))


# ---------- derived-outcome states -------------------------------------

def test_rescored_run_is_evaluation_verified_and_derives_numbers(
        tmp_path):
    d = build_run(tmp_path)
    oc = derive_qualified_outcome(d)
    assert oc.verification_state == STATE_EVALUATION
    assert oc.admissible
    assert oc.outcome_surface == "validation_within"
    assert oc.baseline_accuracy == 0.5 and oc.adapted_accuracy == 1.0
    assert oc.outcome_delta == 0.5
    # hidden surface: frozen 1/4 correct, adapted 1/4 (destroyed h1,
    # gained h2 — net zero, but the within-family surface is primary)
    assert oc.retention_regression == 1.0   # 1 of 1 frozen-correct lost
    assert oc.training_seconds == 12.5
    assert oc.intervention == "fixed_lora"
    assert oc.model_id == "model/m1" and oc.model_revision == "rev-1"
    assert oc.evidence_digest.startswith("sha256:")
    # immutability: the record is a frozen value object
    with pytest.raises(Exception):
        oc.outcome_delta = 9.9


def test_self_authored_commit_is_integrity_only(tmp_path):
    """THE defect: METRICS.json + hash-consistent RUN_COMMIT.json in a
    self-built directory was accepted as verified. Now it is
    INTEGRITY_VERIFIED at best — no rescorable evidence, no outcome."""
    run = tmp_path / "self-authored"
    run.mkdir()
    (run / "METRICS.json").write_text(json.dumps(
        {"adapted": {"hidden": 0.999}, "frozen": {"hidden": 0.0}}))
    (run / "RUN_COMMIT.json").write_text(json.dumps({
        "run_id": "self-authored", "status": "COMPLETE",
        "files": {"METRICS.json": _sha(run / "METRICS.json")}}))
    oc = derive_qualified_outcome(run)
    assert oc.verification_state == STATE_INTEGRITY
    assert not oc.admissible
    assert oc.outcome_delta is None          # no numbers leak out
    res = verify_outcome_receipt(run)
    assert not res["verified"]


def test_self_authored_qualification_does_not_qualify(tmp_path):
    """A run cannot qualify itself: unsigned / self-signed decision
    docs are NOT_QUALIFIED even when evaluation verifies."""
    d = build_run(tmp_path)
    # unsigned self-assertion inside the run dir
    (d / "QUALIFICATION.json").write_text(json.dumps({
        "schema": QUALIFICATION_SCHEMA, "run_id": d.name,
        "evidence_digest": "sha256:" + "0" * 64,
        "decision": "QUALIFIED"}))
    oc = derive_qualified_outcome(d)
    assert oc.verification_state == STATE_EVALUATION
    assert oc.qualification_status == NOT_QUALIFIED
    # a forged signature from an unregistered key is likewise refused
    forger = Ed25519Signer.generate()
    oc = derive_qualified_outcome(d)
    forged = qualification_decision(
        forger, run_id=oc.run_id, evidence_digest=oc.evidence_digest)
    (d / "QUALIFICATION.json").write_text(json.dumps(forged))
    reg, ver, _ = _authority("qualification")
    oc2 = derive_qualified_outcome(d, verifier=ver, authority=reg)
    assert oc2.qualification_status == NOT_QUALIFIED
    assert oc2.verification_state == STATE_EVALUATION


def test_authorized_qualification_reaches_qualified(tmp_path):
    """The happy path: evaluation-verified run + a qualification-role
    signature binding run_id and evidence digest => QUALIFIED."""
    d = build_run(tmp_path)
    reg, ver, signers = _authority("qualification", "evaluation")
    oc = derive_qualified_outcome(d, verifier=ver, authority=reg)
    qdoc = qualification_decision(
        signers["qualification"], run_id=oc.run_id,
        evidence_digest=oc.evidence_digest)
    (d / "QUALIFICATION.json").write_text(json.dumps(qdoc))
    oc = derive_qualified_outcome(d, verifier=ver, authority=reg)
    assert oc.qualification_status == QUALIFIED
    assert oc.verification_state == STATE_QUALIFIED
    assert oc.qualification_digest == digest(qdoc)
    assert oc.qualification_signer == signers["qualification"].key_id


def test_qualification_binding_to_other_evidence_is_rejected(tmp_path):
    """A QUALIFIED decision over different bytes is not a qualification
    for this run — the evidence digest must match."""
    d = build_run(tmp_path)
    reg, ver, signers = _authority("qualification")
    qdoc = qualification_decision(
        signers["qualification"], run_id=d.name,
        evidence_digest="sha256:" + "f" * 64)    # wrong bytes
    (d / "QUALIFICATION.json").write_text(json.dumps(qdoc))
    oc = derive_qualified_outcome(d, verifier=ver, authority=reg)
    assert oc.qualification_status == NOT_QUALIFIED
    assert any("evidence_digest" in r for r in oc.reasons)


def test_wrong_role_signature_is_rejected(tmp_path):
    """A signature from a key authorized only for 'plan' is not a
    qualification — roles are separated."""
    d = build_run(tmp_path)
    reg, ver, signers = _authority("plan")
    oc = derive_qualified_outcome(d, verifier=ver, authority=reg)
    qdoc = qualification_decision(
        signers["plan"], run_id=oc.run_id,
        evidence_digest=oc.evidence_digest)
    (d / "QUALIFICATION.json").write_text(json.dumps(qdoc))
    oc = derive_qualified_outcome(d, verifier=ver, authority=reg)
    assert oc.qualification_status == NOT_QUALIFIED
    assert any("not authorized" in r for r in oc.reasons)


# ---------- evidence tampering fails evaluation ------------------------

def test_incorrect_reported_metrics_fail_rescoring(tmp_path):
    d = build_run(tmp_path)
    m = json.loads((d / "METRICS.json").read_text())
    m["adapted"]["validation"] = 0.99           # claim the rows deny
    (d / "METRICS.json").write_text(json.dumps(m))
    _recommit(d)
    oc = derive_qualified_outcome(d)
    assert oc.verification_state == STATE_INTEGRITY
    assert not oc.admissible
    assert oc.outcome_delta is None


def test_missing_prediction_records_are_rejected(tmp_path):
    d = build_run(tmp_path)
    (d / "PREDICTIONS-ADAPTED-VALIDATION.jsonl").unlink()
    _recommit(d)
    oc = derive_qualified_outcome(d)
    assert not oc.admissible
    assert oc.verification_state == STATE_INTEGRITY


def test_altered_prediction_record_fails(tmp_path):
    d = build_run(tmp_path)
    p = d / "PREDICTIONS-FROZEN.jsonl"
    rows = [json.loads(ln) for ln in p.read_text().splitlines()]
    rows[0]["output"] = "FORGED"               # flip a frozen answer
    _write_jsonl(p, rows)
    _recommit(d)
    oc = derive_qualified_outcome(d)
    assert not oc.admissible


def test_wrong_frozen_model_identity_fails(tmp_path):
    """The outcome must derive against the declared frozen baseline —
    a config claiming a different base model than the adapter was
    built on breaks the identity chain and cannot verify."""
    d = build_run(tmp_path)
    c = json.loads((d / "CONFIG.json").read_text())
    c["model_id"] = "model/WRONG"            # adapter binds model/m1
    c["config_digest"] = digest(
        {k: v for k, v in c.items()
         if k not in ("config_digest", "attempt_id", "run_id")})
    (d / "CONFIG.json").write_text(json.dumps(c))
    _recommit(d)
    commit = json.loads((d / "RUN_COMMIT.json").read_text())
    commit["config_digest"] = c["config_digest"]
    (d / "RUN_COMMIT.json").write_text(json.dumps(commit))
    oc = derive_qualified_outcome(d)
    assert not oc.admissible
    assert any("base model" in r for r in oc.reasons)


def test_signed_receipt_from_unauthorized_role_rejected(tmp_path):
    """A validly-signed receipt from a signer authorized only for a
    non-evaluation role is not evaluation evidence."""
    reg, ver, signers = _authority("plan")
    receipt = {"schema": "mini-agi-v16.1-executed-run-receipt-v1",
               "run_id": "r", "signer_key_id": signers["plan"].key_id}
    receipt["signature_b64"] = signers["plan"].sign(receipt).signature_b64
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps(receipt))
    res = verify_outcome_receipt(p, ver, authority=reg)
    assert not res["verified"]
    assert "not authorized" in res["reason"]


def test_evaluator_signed_receipt_is_evaluation_verified(tmp_path):
    reg, ver, signers = _authority("evaluation")
    receipt = {"schema": "mini-agi-v16.1-executed-run-receipt-v1",
               "run_id": "r",
               "signer_key_id": signers["evaluation"].key_id}
    receipt["signature_b64"] = signers["evaluation"].sign(
        receipt).signature_b64
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps(receipt))
    res = verify_outcome_receipt(p, ver, authority=reg)
    assert res["verified"]
    assert res["verification_state"] == STATE_EVALUATION


def test_receipt_signature_cannot_be_checked_is_not_verified(tmp_path):
    s = Ed25519Signer.generate()
    receipt = {"schema": "mini-agi-v16.1-executed-run-receipt-v1",
               "run_id": "r", "signer_key_id": s.key_id}
    receipt["signature_b64"] = s.sign(receipt).signature_b64
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps(receipt))
    assert not verify_outcome_receipt(p)["verified"]


# ---------- CLI contract ------------------------------------------------

def test_cli_record_derives_numbers_from_evidence(tmp_path):
    d = build_run(tmp_path)
    hist = tmp_path / "history.jsonl"
    proc = _cli("record", "--receipt", str(d), "--history", str(hist))
    assert proc.returncode == 0, proc.stderr
    rec = json.loads(proc.stdout)
    assert rec["verified"] is True
    assert rec["outcome_delta"] == 0.5       # derived, not asserted
    assert rec["intervention"] == "fixed_lora"
    assert rec["cost"] == 12.5
    loaded = intervention_history_load(hist)
    assert len(loaded) == 1 and loaded[0].verified
    assert loaded[0].receipt_digest


def test_cli_record_refuses_caller_supplied_delta(tmp_path):
    """--delta 999 with a receipt: verified records cannot be steered."""
    d = build_run(tmp_path)
    hist = tmp_path / "history.jsonl"
    proc = _cli("record", "--receipt", str(d), "--history", str(hist),
                "--intervention", "fixed_lora", "--delta", "999",
                "--verified")
    assert proc.returncode == 2
    assert "derive" in proc.stderr.lower()
    assert not hist.exists()


def test_cli_record_refuses_unverifiable_receipt(tmp_path):
    d = build_run(tmp_path)
    m = json.loads((d / "METRICS.json").read_text())
    m["adapted"]["hidden"] = 0.999           # fabricated, unreconcilable
    (d / "METRICS.json").write_text(json.dumps(m))
    _recommit(d)
    proc = _cli("record", "--receipt", str(d),
                "--history", str(tmp_path / "h.jsonl"))
    assert proc.returncode == 2
    assert "did not verify" in proc.stderr


def test_cli_record_requires_evidence_or_diagnostic(tmp_path):
    proc = _cli("record", "--history", str(tmp_path / "h.jsonl"),
                "--intervention", "fixed_lora", "--delta", "0.5")
    assert proc.returncode == 2
    assert "--receipt" in proc.stderr and "--diagnostic" in proc.stderr


def test_diagnostic_records_never_promote(tmp_path):
    """Manual numbers are allowed ONLY as an explicitly unverified
    diagnostic record — it cannot claim verified status."""
    hist = tmp_path / "history.jsonl"
    proc = _cli("record", "--diagnostic", "--history", str(hist),
                "--intervention", "fixed_lora", "--delta", "999")
    assert proc.returncode == 0, proc.stderr
    rec = json.loads(proc.stdout)
    assert rec["verified"] is False and rec["outcome_delta"] == 999
    assert rec["provenance"] == "diagnostic"
    loaded = intervention_history_load(hist)
    assert len(loaded) == 1 and not loaded[0].verified
    # and --verified without evidence is still refused outright
    proc = _cli("record", "--verified", "--history", str(hist),
                "--intervention", "fixed_lora", "--delta", "999")
    assert proc.returncode == 2


def test_cli_record_outcome_derives_from_evidence(tmp_path):
    d = build_run(tmp_path)
    out = tmp_path / "outcomes.jsonl"
    proc = _cli("record-outcome", "--receipt", str(d),
                "--outcomes", str(out), "--intervention", "fixed_lora")
    assert proc.returncode == 0, proc.stderr
    rec = json.loads(proc.stdout)
    ev = rec["evaluation"]
    assert ev["within_family_delta"] == 0.5
    assert ev["qualification_passed"] is False
    assert rec["resources"]["training_seconds"] == 12.5
    # caller numbers are refused on the verified path
    proc = _cli("record-outcome", "--receipt", str(d),
                "--outcomes", str(out), "--intervention", "fixed_lora",
                "--within-delta", "999")
    assert proc.returncode == 2
    # and an unqualified assertion cannot be made
    proc = _cli("record-outcome", "--receipt", str(d),
                "--outcomes", str(out), "--intervention", "fixed_lora",
                "--qualified")
    assert proc.returncode == 2
    assert "qualification" in proc.stderr


def test_cli_record_outcome_qualified_when_signed(tmp_path):
    d = build_run(tmp_path)
    reg, ver, signers = _authority("qualification")
    trust = _trust_root_file(tmp_path, signers)
    oc = derive_qualified_outcome(d, verifier=ver, authority=reg)
    qdoc = qualification_decision(signers["qualification"],
                                  run_id=oc.run_id,
                                  evidence_digest=oc.evidence_digest)
    # QUALIFICATION.json is written by the INDEPENDENT qualifier after
    # the run commits — it is deliberately not part of the run's own
    # inventory (the reconciler records it as uncommitted, not as a
    # violation).
    (d / "QUALIFICATION.json").write_text(json.dumps(qdoc))
    out = tmp_path / "outcomes.jsonl"
    proc = _cli("record-outcome", "--receipt", str(d),
                "--outcomes", str(out), "--intervention", "fixed_lora",
                "--qualified", "--trust-root", str(trust))
    assert proc.returncode == 0, proc.stderr
    rec = json.loads(proc.stdout)
    assert rec["evaluation"]["qualification_passed"] is True
    assert rec["verification_state"] == "QUALIFIED"
