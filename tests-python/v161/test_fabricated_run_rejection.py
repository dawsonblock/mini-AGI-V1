"""v4.0.3-r1 adversarial regression — fabricated run rejection.

Reproduces the audit's central finding end to end:

  * a synthetic directory carrying self-authored predictions, metrics
    and a MATCHING commit inventory recomputes cleanly — that is
    LOCALLY_RECOMPUTED, the highest grade local evidence can reach;
  * it must NEVER become INDEPENDENTLY_VERIFIED — no evaluator with
    independent authority ever saw it;
  * it must NEVER enter the controller's trusted learning history —
    ``capability_control.py record`` must deny the import.

Also covered: forged evaluator signatures, signatures from
unauthorized roles, altered predictions after receipt, replayed
receipts, and incompatible model identities — plus the complementary
requirement that a VERIFIED NEGATIVE outcome IS admissible (the
controller must not learn only from successes).
"""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(Path(__file__).parent))

from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.v161.authority import (  # noqa: E402
    AuthorityRecord, AuthorityRegistry, TRUST_ROOT_SCHEMA)
from minagi.v161.capability_outcomes import (  # noqa: E402
    verify_outcome_receipt)
from minagi.v161.independent_eval_receipt import (  # noqa: E402
    RECEIPT_FILE, issue_evaluation_receipt, write_receipt_file)
from minagi.v161.qualified_outcome import (  # noqa: E402
    STATE_INDEPENDENT, STATE_LOCAL, derive_qualified_outcome)
from test_independent_eval_receipt import _authority, _sha, _tamper, \
    _write_jsonl, build_run  # noqa: E402

CLI = ROOT / "scripts" / "capability_control.py"


def _cli(*argv) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *argv],
        capture_output=True, text=True, cwd=str(ROOT))


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


def test_fabricated_complete_run_is_denied(tmp_path):
    """THE audit scenario: self-authored predictions + metrics + a
    matching commit inventory, no authorized evaluator receipt."""
    # The fabricator authors everything — predictions, metrics and the
    # commit — so local recomputation reproduces exactly.
    d = build_run(tmp_path, run_id="fabricated")

    # LOCAL_RECOMPUTATION: PASS
    oc = derive_qualified_outcome(d)
    assert oc.verification_state == STATE_LOCAL
    assert oc.evaluation_status == STATE_LOCAL
    assert not oc.admissible

    # INDEPENDENT_VERIFICATION: FAIL
    res = verify_outcome_receipt(d)
    assert not res["verified"]
    assert res["verification_state"] == STATE_LOCAL

    # CONTROLLER_TRUSTED_IMPORT: DENIED
    hist = tmp_path / "history.jsonl"
    proc = _cli("record", "--receipt", str(d), "--history", str(hist))
    assert proc.returncode == 2
    assert "did not verify" in proc.stderr
    assert not hist.exists() or not hist.read_text().strip()


def test_authorized_receipt_admits_independent_history(tmp_path):
    """The same run WITH an authorized evaluation receipt is
    INDEPENDENTLY_VERIFIED and may enter trusted history — including a
    receipt the fabricator could not have issued itself."""
    d = build_run(tmp_path)
    reg, ver, signers = _authority("evaluation")
    trust = _trust_root_file(tmp_path, signers)
    write_receipt_file(
        d, issue_evaluation_receipt(signers["evaluation"], d))

    oc = derive_qualified_outcome(d, verifier=ver, authority=reg)
    assert oc.verification_state == STATE_INDEPENDENT
    assert oc.admissible
    assert oc.evaluation_signer == signers["evaluation"].key_id

    hist = tmp_path / "history.jsonl"
    proc = _cli("record", "--receipt", str(d), "--history", str(hist),
                "--trust-root", str(trust))
    assert proc.returncode == 0, proc.stderr
    rec = json.loads(proc.stdout)
    assert rec["verified"] is True
    assert rec["verification_state"] == STATE_INDEPENDENT


def test_receipt_signed_by_research_agent_key_denied(tmp_path):
    """A receipt signed by a key with no evaluation authority — e.g. a
    key the training pipeline controls — cannot self-verify."""
    d = build_run(tmp_path)
    reg, ver, signers = _authority("execution_witness", "plan")
    trust = _trust_root_file(tmp_path, signers)
    # the research agent holds the execution_witness key: it CAN sign,
    # but its role carries no evaluation authority
    write_receipt_file(
        d, issue_evaluation_receipt(signers["execution_witness"], d))
    proc = _cli("record", "--receipt", str(d),
                "--history", str(tmp_path / "h.jsonl"),
                "--trust-root", str(trust))
    assert proc.returncode == 2


def test_forged_receipt_denied(tmp_path):
    d = build_run(tmp_path)
    reg, ver, signers = _authority("evaluation")
    trust = _trust_root_file(tmp_path, signers)
    receipt = issue_evaluation_receipt(signers["evaluation"], d)
    forged = dict(receipt)
    forged["signature"] = "AAAA"          # garbage signature
    (d / RECEIPT_FILE).write_text(json.dumps(forged))
    res = verify_outcome_receipt(d, ver, authority=reg)
    assert not res["verified"]
    proc = _cli("record", "--receipt", str(d),
                "--history", str(tmp_path / "h.jsonl"),
                "--trust-root", str(trust))
    assert proc.returncode == 2


def test_altered_predictions_after_receipt_denied(tmp_path):
    """Evidence mutated after the evaluator signed breaks the binding
    at import time — the receipt attested different bytes."""
    d = build_run(tmp_path)
    reg, ver, signers = _authority("evaluation")
    write_receipt_file(
        d, issue_evaluation_receipt(signers["evaluation"], d))

    def _alter(d):
        p = d / "PREDICTIONS-ADAPTED-WITHIN_HOLDOUT.jsonl"
        rows = [json.loads(ln) for ln in p.read_text().splitlines()]
        rows[0]["output"] = "FORGED"
        _write_jsonl(p, rows)
    _tamper(d, _alter)
    res = verify_outcome_receipt(d, ver, authority=reg)
    assert not res["verified"]


def test_replayed_receipt_denied(tmp_path):
    """A valid receipt copied into a different run binds the first
    run's evidence — replay is a binding mismatch, not a signature
    problem."""
    d_a = build_run(tmp_path, run_id="run-a")
    d_b = build_run(tmp_path / "b", run_id="run-b")
    reg, ver, signers = _authority("evaluation")
    receipt_a = issue_evaluation_receipt(signers["evaluation"], d_a)
    (d_b / RECEIPT_FILE).write_text(json.dumps(receipt_a))
    res = verify_outcome_receipt(d_b, ver, authority=reg)
    assert not res["verified"]


def test_incompatible_model_identity_denied(tmp_path):
    """A receipt bound to a different declared base model cannot be
    replayed onto this run."""
    d = build_run(tmp_path)
    reg, ver, signers = _authority("evaluation")
    write_receipt_file(
        d, issue_evaluation_receipt(signers["evaluation"], d))
    # swap in a receipt issued against a same-layout run whose config
    # declares a different base model
    d2 = build_run(tmp_path / "other", run_id="run-ok")
    cfg = json.loads((d2 / "CONFIG.json").read_text())
    cfg["model_id"] = "model/other"
    from egai.common.canonical import digest as _d
    cfg["config_digest"] = _d(
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
    from minagi.v161.evidence_reconciliation import (
        adapter_artifact_digest_of as _aad)
    (d2 / "TRAINING_RECEIPT.json").write_text(json.dumps({
        "adapter_artifact_digest": _aad(d2 / "adapter"),
        "train_seconds": 12.5}))
    commit = json.loads((d2 / "RUN_COMMIT.json").read_text())
    commit["config_digest"] = cfg["config_digest"]
    commit["files"] = {p.relative_to(d2).as_posix(): _sha(p)
                       for p in sorted(d2.rglob("*"))
                       if p.is_file() and p.name != "RUN_COMMIT.json"}
    (d2 / "RUN_COMMIT.json").write_text(json.dumps(commit))
    receipt2 = issue_evaluation_receipt(signers["evaluation"], d2)
    (d / RECEIPT_FILE).write_text(json.dumps(receipt2))
    res = verify_outcome_receipt(d, ver, authority=reg)
    assert not res["verified"]


def test_verified_negative_outcome_is_admissible(tmp_path):
    """Anti-selection-bias: a candidate whose independently verified
    measurements are NEGATIVE must still enter trusted history —
    dropping failures would teach the controller only from wins."""
    # adapted destroys all hidden correctness: an honest negative run
    d = build_run(tmp_path, run_id="run-negative",
                  adapted_hidden_correct=())
    reg, ver, signers = _authority("evaluation")
    trust = _trust_root_file(tmp_path, signers)
    write_receipt_file(
        d, issue_evaluation_receipt(signers["evaluation"], d))
    oc = derive_qualified_outcome(d, verifier=ver, authority=reg)
    assert oc.verification_state == STATE_INDEPENDENT
    assert oc.admissible                       # negative, but real
    hist = tmp_path / "history.jsonl"
    proc = _cli("record", "--receipt", str(d), "--history", str(hist),
                "--trust-root", str(trust))
    assert proc.returncode == 0, proc.stderr
    rec = json.loads(proc.stdout)
    assert rec["verified"] is True
    # the measured outcome was honestly NEGATIVE — the adapted model
    # lost previously-correct work on the hidden surface — and it is
    # still admitted, because trusted history must keep failures
    assert rec["outcome_delta"] < 0
    # and the recorded history entry carries the measured (negative)
    # evidence — not a filtered "success only" view
    lines = [json.loads(ln) for ln in hist.read_text().splitlines()
             if ln.strip()]
    assert lines and lines[0]["verified"] is True


def test_integrity_only_fabrication_denied(tmp_path):
    """METRICS.json + self-consistent commit but no rescorable
    predictions: INTEGRITY_VERIFIED at best — never trusted."""
    run = tmp_path / "self-authored"
    run.mkdir()
    (run / "METRICS.json").write_text(json.dumps(
        {"adapted": {"hidden": 0.999}, "frozen": {"hidden": 0.0},
         "training": {"train_seconds": 1}}))
    (run / "RUN_COMMIT.json").write_text(json.dumps({
        "run_id": "self-authored", "status": "COMPLETE",
        "files": {"METRICS.json": _sha(run / "METRICS.json")}}))
    oc = derive_qualified_outcome(run)
    assert oc.verification_state == "INTEGRITY_VERIFIED"
    assert not oc.admissible
    proc = _cli("record", "--receipt", str(run),
                "--history", str(tmp_path / "h.jsonl"))
    assert proc.returncode == 2
