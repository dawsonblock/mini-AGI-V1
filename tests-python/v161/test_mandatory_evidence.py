"""v4.0.3-r2 Phase 3 — mandatory evidence identities + trusted
resolver + custody.

Diagnostics 9 requires every final-evaluation identity to be bound —
base model, tokenizer, adapter (or declared adapter-free), dataset,
scorer, protocol, generation config, predictions, security and
retention evidence, authorized receipts. Null is not permitted except
for declared adapter-free experiment types. These tests pin the
resolver's fail-closed contract and the signing-key custody checks.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(Path(__file__).parent))

from test_independent_eval_receipt import (  # noqa: E402
    _authority, _sha, _tamper, build_run)
from test_matrix_false_positive import _corpus  # noqa: E402

from minagi.v161.evaluator_custody import (  # noqa: E402
    check_signing_key_custody)
from minagi.v161.model_execution_receipt import (  # noqa: E402
    EXECUTION_FILE, issue_execution_receipt)
from minagi.v161.qualified_outcome import (  # noqa: E402
    STATE_INDEPENDENT, STATE_INVALID, STATE_LOCAL)
from minagi.v161.trusted_evidence import (  # noqa: E402
    resolve_identities, resolve_trusted_evidence)


def _rows(tmp_path, **kw):
    corpus = _corpus(tmp_path, **kw)
    return [json.loads(ln) for ln in corpus.read_text().splitlines()
            if ln.strip()]


def _patch_config(d: Path, **updates):
    """Rewrite CONFIG.json and keep the commit self-consistent —
    integrity holds; the declared content changed."""
    cfg = json.loads((d / "CONFIG.json").read_text())
    cfg.pop("config_digest", None)
    cfg.update(updates)
    from egai.common.canonical import digest
    # config_digest covers the body without the post-digest identity
    # keys (build_run appends run_id/attempt_id after digesting)
    body = {k: v for k, v in cfg.items()
            if k not in ("run_id", "attempt_id")}
    cfg["config_digest"] = digest(body)
    (d / "CONFIG.json").write_text(json.dumps(cfg, sort_keys=True))
    commit = json.loads((d / "RUN_COMMIT.json").read_text())
    commit["config_digest"] = cfg["config_digest"]
    commit["files"]["CONFIG.json"] = _sha(d / "CONFIG.json")
    (d / "RUN_COMMIT.json").write_text(json.dumps(commit))


# ---------- mandatory identities ----------------------------------------


def test_missing_adapter_is_blocking_not_silent_baseline(tmp_path):
    """No adapter + no declared adapter-free type = missing identity,
    never an implicit baseline."""
    d = build_run(tmp_path)
    _tamper(d, lambda d: __import__("shutil").rmtree(d / "adapter"))
    ident = resolve_identities(d, corpus_rows=_rows(tmp_path))
    assert ident["identities"]["adapter"]["status"] == "missing"
    assert "adapter" in ident["blocking"]


def test_declared_adapter_free_type_resolves(tmp_path):
    """An explicit experiment_type=frozen_baseline declaration is the
    only valid null-adapter path."""
    d = build_run(tmp_path)
    _tamper(d, lambda d: __import__("shutil").rmtree(d / "adapter"))
    _patch_config(d, experiment_type="frozen_baseline")
    ident = resolve_identities(d, corpus_rows=_rows(tmp_path))
    assert ident["identities"]["adapter"]["status"] == \
        "declared_absent"
    assert "adapter" not in ident["blocking"]


def test_missing_predictions_blocking(tmp_path):
    d = build_run(tmp_path)
    _tamper(d, lambda d: [
        p.unlink() for p in d.glob("PREDICTIONS-*.jsonl")])
    ident = resolve_identities(d, corpus_rows=_rows(tmp_path))
    assert ident["identities"]["predictions"]["status"] == "missing"
    assert "predictions" in ident["blocking"]


def test_unsupported_scorer_is_unbound(tmp_path):
    """A scoring version the evaluator doesn't support cannot resolve
    — measurements under unknown semantics are not evidence."""
    d = build_run(tmp_path)
    _patch_config(d, score_oracle="mystery-scorer-v9")
    ident = resolve_identities(d, corpus_rows=_rows(tmp_path))
    assert ident["identities"]["scorer"]["status"] == "unbound"
    assert "scorer" in ident["blocking"]


def test_security_rows_require_governed_sidecar(tmp_path):
    """The corpus carries a security row → mandatory governed security
    evidence; no sidecar supplied = missing."""
    d = build_run(tmp_path)
    ident = resolve_identities(d, corpus_rows=_rows(tmp_path))
    assert ident["identities"]["security"]["status"] == "missing"
    assert "security" in ident["blocking"]
    sec = {"schema": "gov-sec-v1", "runs": {}}
    ident2 = resolve_identities(d, corpus_rows=_rows(tmp_path),
                                security_doc=sec)
    assert ident2["identities"]["security"]["status"] == "resolved"


def test_retention_rows_require_paired_files(tmp_path):
    d = build_run(tmp_path)
    _tamper(d, lambda d: (d / "PREDICTIONS-ADAPTED-RETENTION.jsonl")
            .unlink())
    ident = resolve_identities(d, corpus_rows=_rows(tmp_path))
    assert ident["identities"]["retention"]["status"] == "missing"
    assert "retention" in ident["blocking"]


# ---------- trusted evidence resolution ---------------------------------


def test_clean_run_resolves_to_local_without_receipt(tmp_path):
    """Self-authored evidence tops out at LOCALLY_RECOMPUTED — always."""
    d = build_run(tmp_path)
    res = resolve_trusted_evidence(d, corpus_rows=_rows(tmp_path))
    assert res["state"] == STATE_LOCAL
    assert any("no evaluation receipt" in r for r in res["reasons"])


def test_receipted_run_resolves_independent(tmp_path):
    from minagi.v161.independent_eval_receipt import (
        issue_evaluation_receipt, write_receipt_file)
    reg, ver, signers = _authority("evaluation")
    d = build_run(tmp_path)
    _patch_config(d, tokenizer_digest="sha256:" + "3" * 64)
    rows = _rows(tmp_path)
    sec = {"schema": "gov-sec-v1", "runs": {}}
    write_receipt_file(d, issue_evaluation_receipt(
        signers["evaluation"], d, corpus_rows=rows, security_doc=sec))
    res = resolve_trusted_evidence(
        d, corpus_rows=rows, verifier=ver, authority=reg,
        security_doc=sec)
    assert res["state"] == STATE_INDEPENDENT
    assert res["resolved"]


def test_forged_receipt_makes_state_invalid(tmp_path):
    """A receipt that fails binding/authorization is not merely absent
    — present-but-refused evidence is INVALID."""
    reg, ver, signers = _authority("evaluation", "plan")
    d = build_run(tmp_path)
    rows = _rows(tmp_path)
    from minagi.v161.independent_eval_receipt import (
        issue_evaluation_receipt, write_receipt_file)
    write_receipt_file(d, issue_evaluation_receipt(
        signers["plan"], d, corpus_rows=rows))
    res = resolve_trusted_evidence(
        d, corpus_rows=rows, verifier=ver, authority=reg)
    assert res["state"] == STATE_INVALID
    assert not res["resolved"]
    assert any("receipt refused" in r for r in res["reasons"])


def test_execution_required_without_receipt_fails_closed(tmp_path):
    """require_execution=True with no EXECUTION_RECEIPT = unresolved."""
    d = build_run(tmp_path)
    res = resolve_trusted_evidence(
        d, corpus_rows=_rows(tmp_path), require_execution=True)
    assert not res["resolved"]
    assert "execution_receipt" in res["identities"] or \
        any("execution" in r for r in res["reasons"])


def test_execution_receipt_over_other_bytes_is_invalid(tmp_path):
    """An exec receipt attesting DIFFERENT adapter bytes than the
    candidate's is cross-binding evidence — INVALID, not merely
    insufficient."""
    reg, ver, signers = _authority("evaluation")
    d = build_run(tmp_path)
    other = tmp_path / "other-adapter"
    other.mkdir()
    (other / "adapter_config.json").write_text(json.dumps({}))
    (other / "adapter_model.bin").write_bytes(b"OTHER-bytes")
    doc = issue_execution_receipt(
        signers["evaluation"], execution_id="exec-x",
        execution_mode="real_model", model_id="model/m1",
        model_revision="rev-1", tokenizer_digest="sha256:" + "0" * 64,
        adapter_dir=other, generation_config={"max_new_tokens": 4})
    (d / EXECUTION_FILE).write_text(json.dumps(doc))
    res = resolve_trusted_evidence(
        d, corpus_rows=_rows(tmp_path), verifier=ver, authority=reg,
        require_execution=True)
    assert res["state"] == STATE_INVALID
    assert not res["resolved"]
    assert any("execution receipt refused" in r
               for r in res["reasons"])


def test_execution_receipt_wrong_model_refused(tmp_path):
    """An exec receipt attesting a different model revision than the
    run's committed identity is refused."""
    reg, ver, signers = _authority("evaluation")
    d = build_run(tmp_path)
    doc = issue_execution_receipt(
        signers["evaluation"], execution_id="exec-y",
        execution_mode="real_model", model_id="model/m1",
        model_revision="rev-WRONG",
        tokenizer_digest="sha256:" + "0" * 64,
        adapter_dir=d / "adapter",
        generation_config={"max_new_tokens": 4})
    (d / EXECUTION_FILE).write_text(json.dumps(doc))
    res = resolve_trusted_evidence(
        d, corpus_rows=_rows(tmp_path), verifier=ver, authority=reg,
        require_execution=True)
    assert res["state"] == STATE_INVALID
    assert not res["resolved"]


# ---------- custody ------------------------------------------------------


def test_key_custody_rejects_group_readable_key(tmp_path):
    key = tmp_path / "eval.key"
    key.write_bytes(b"private")
    key.chmod(0o640)
    res = check_signing_key_custody(key)
    assert not res["custody_ok"]
    assert any("group/other" in f for f in res["failures"])


def test_key_custody_rejects_workspace_resident_key(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    key = ws / "eval.key"
    key.write_bytes(b"private")
    key.chmod(0o600)
    res = check_signing_key_custody(key, workspace_root=ws)
    assert not res["custody_ok"]
    assert any("workspace" in f for f in res["failures"])


def test_key_custody_accepts_owner_only_external_key(tmp_path):
    key = tmp_path / "keys" / "eval.key"
    key.parent.mkdir()
    key.write_bytes(b"private")
    key.chmod(0o600)
    res = check_signing_key_custody(key, workspace_root=tmp_path / "ws")
    assert res["custody_ok"], res
