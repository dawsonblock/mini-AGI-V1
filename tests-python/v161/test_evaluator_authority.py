"""v4.0.3-r1 — evaluator authorization helpers.

``evaluator_authority`` is the narrow surface that establishes *who may
sign evaluation evidence*: an ``evaluation``-role key that is active in
the campaign trust root at verification time. These tests pin that a
signature is evaluation-authorized iff the signing identity is
registered for exactly that role — revoked, expired, not-yet-valid and
wrong-role keys must not verify as evaluators.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.crypto import Ed25519Signer, SignedEnvelope  # noqa: E402
from minagi.v161.authority import (  # noqa: E402
    AuthorityRecord, AuthorityRegistry, TRUST_ROOT_SCHEMA)
from minagi.v161.evaluator_authority import (  # noqa: E402
    EVALUATOR_ROLE, authorization_failure, evaluation_verifier,
    is_authorized_evaluator, load_trust_root)


def _rec(role, signer, **kw):
    return AuthorityRecord(
        role=role, key_id=signer.key_id,
        public_key_b64=base64.b64encode(
            signer.public_bytes()).decode(), **kw)


def _signed(signer, payload):
    env = signer.sign(payload)
    return SignedEnvelope(key_id=env.key_id,
                          signature_b64=env.signature_b64)


def test_load_trust_root_roundtrip(tmp_path):
    ev, q = Ed25519Signer.generate(), Ed25519Signer.generate()
    p = tmp_path / "TRUST_ROOT.json"
    p.write_text(json.dumps({
        "schema": TRUST_ROOT_SCHEMA,
        "authorities": [_rec("evaluation", ev).__dict__,
                        _rec("qualification", q).__dict__]}))
    reg, ver = load_trust_root(p)
    assert reg.record_for(ev.key_id).role == "evaluation"
    assert reg.record_for(q.key_id).role == "qualification"
    # the returned verifier covers every currently-valid identity —
    # role checks are the caller's job (or evaluation_verifier's)
    payload = {"k": "v"}
    assert ver.verify(payload, _signed(q, payload))


def test_load_trust_root_rejects_bad_schema_and_missing(tmp_path):
    p = tmp_path / "TRUST_ROOT.json"
    p.write_text(json.dumps({"schema": "other", "authorities": []}))
    with pytest.raises(ValueError):
        load_trust_root(p)
    with pytest.raises(OSError):
        load_trust_root(tmp_path / "absent.json")


def test_only_evaluation_role_is_evaluator_authorized():
    ev, plan = Ed25519Signer.generate(), Ed25519Signer.generate()
    reg = AuthorityRegistry(
        [_rec("evaluation", ev), _rec("plan", plan)])
    assert is_authorized_evaluator(reg, ev.key_id)
    assert not is_authorized_evaluator(reg, plan.key_id)
    assert "plan" in authorization_failure(reg, plan.key_id)
    assert "unregistered" in authorization_failure(reg, "ghost-key")


def test_revoked_and_expired_evaluators_carry_no_authority():
    ev, old = Ed25519Signer.generate(), Ed25519Signer.generate()
    reg = AuthorityRegistry([
        _rec("evaluation", ev, status="revoked"),
        _rec("evaluation", old,
             not_after="2000-01-01T00:00:00+00:00")])
    assert not is_authorized_evaluator(reg, ev.key_id)
    assert not is_authorized_evaluator(reg, old.key_id)
    assert "revoked" in authorization_failure(reg, ev.key_id)
    assert "expired" in authorization_failure(reg, old.key_id)


def test_evaluation_verifier_registers_only_evaluators():
    """The role-restricted verifier is the enforcement point: a
    wrong-role signature cannot verify under it AT ALL — role and
    signature are checked in one step, not two that can drift apart."""
    ev, plan, ghost = (Ed25519Signer.generate() for _ in range(3))
    reg = AuthorityRegistry(
        [_rec("evaluation", ev), _rec("plan", plan)])
    ev_ver = evaluation_verifier(reg)
    payload = {"result": "x"}
    assert ev_ver.verify(payload, _signed(ev, payload))
    assert not ev_ver.verify(payload, _signed(plan, payload))
    assert not ev_ver.verify(payload, _signed(ghost, payload))
    # the full verifier accepts all of them — authority lives in the
    # role check, not in the signature's math
    assert reg.verifier().verify(payload, _signed(plan, payload))


def test_revoked_key_falls_out_of_evaluation_verifier():
    ev = Ed25519Signer.generate()
    live = Ed25519Signer.generate()
    reg = AuthorityRegistry([
        _rec("evaluation", ev, status="revoked"),
        _rec("evaluation", live)])
    ev_ver = evaluation_verifier(reg)
    payload = {"result": "x"}
    assert not ev_ver.verify(payload, _signed(ev, payload))
    assert ev_ver.verify(payload, _signed(live, payload))


def test_evaluator_role_constant_is_evaluation():
    # role separation is name-level: the string must stay distinct from
    # the roles a training pipeline legitimately holds
    assert EVALUATOR_ROLE == "evaluation"
    assert EVALUATOR_ROLE not in ("plan", "execution_witness",
                                  "promotion", "qualification")
