"""FIX-005 — authority key validity intervals, revocation, and time.

Reproduces the v16.2.1 defect: `AuthorityRecord.not_before/not_after`
were stored but never enforced — an expired key verified exactly like a
valid one, and `verifier()` registered every active key regardless of
its window. Every check here must fail on the original implementation
and pass on the repaired one.
"""
import base64
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.crypto import Ed25519Signer, SignedEnvelope  # noqa: E402
from minagi.v161.authority import (AuthorityLedger,  # noqa: E402
                                   AuthorityRecord, AuthorityRegistry)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
BEFORE = (NOW - timedelta(days=30)).isoformat()
AFTER = (NOW + timedelta(days=30)).isoformat()
PAST = (NOW - timedelta(days=1)).isoformat()
FUTURE = (NOW + timedelta(days=1)).isoformat()


def _record(role, signer, **kw):
    return AuthorityRecord(
        role=role, key_id=signer.key_id,
        public_key_b64=base64.b64encode(signer.public_bytes()).decode(),
        **kw)


def _registry(*records):
    return AuthorityRegistry(records)


def test_expired_key_is_unauthorized():
    s = Ed25519Signer.generate()
    reg = _registry(_record("promotion", s, not_before=BEFORE,
                            not_after=PAST))
    assert reg.is_authorized("promotion", s.key_id, now=NOW) is False
    # the original implementation had no `now` and returned True


def test_not_yet_valid_key_is_unauthorized():
    s = Ed25519Signer.generate()
    reg = _registry(_record("promotion", s, not_before=FUTURE,
                            not_after=AFTER))
    assert reg.is_authorized("promotion", s.key_id, now=NOW) is False


def test_window_is_inclusive_at_both_bounds():
    s = Ed25519Signer.generate()
    reg = _registry(_record("promotion", s, not_before=BEFORE,
                            not_after=AFTER))
    assert reg.is_authorized("promotion", s.key_id, now=NOW) is True
    assert reg.is_authorized(
        "promotion", s.key_id,
        now=datetime.fromisoformat(BEFORE)) is True
    assert reg.is_authorized(
        "promotion", s.key_id,
        now=datetime.fromisoformat(AFTER)) is True


def test_revoked_key_is_unauthorized_inside_its_window():
    s = Ed25519Signer.generate()
    reg = _registry(_record("promotion", s, status="revoked",
                            not_before=BEFORE, not_after=AFTER))
    assert reg.is_authorized("promotion", s.key_id, now=NOW) is False


def test_open_bounds_remain_valid_for_legacy_trust_roots():
    s = Ed25519Signer.generate()
    reg = _registry(_record("promotion", s))
    assert reg.is_authorized("promotion", s.key_id, now=NOW) is True


def test_z_suffix_and_naive_bounds_parse():
    s = Ed25519Signer.generate()
    reg = _registry(_record("promotion", s,
                            not_before="2026-10-07T00:00:00Z",
                            not_after="2026-10-09T00:00:00Z"))
    assert reg.is_authorized("promotion", s.key_id, now=NOW) is True
    naive = _registry(_record("promotion", s,
                              not_before="2026-10-07T00:00:00",
                              not_after="2026-10-09T00:00:00"))
    assert naive.is_authorized("promotion", s.key_id, now=NOW) is True


def test_malformed_bound_fails_closed():
    s = Ed25519Signer.generate()
    reg = _registry(_record("promotion", s, not_after="not-a-timestamp"))
    with pytest.raises(ValueError, match="not_after"):
        reg.is_authorized("promotion", s.key_id, now=NOW)


def test_naive_now_is_rejected():
    s = Ed25519Signer.generate()
    reg = _registry(_record("promotion", s))
    with pytest.raises(ValueError, match="naive"):
        reg.is_authorized("promotion", s.key_id,
                          now=datetime(2026, 10, 8, 12, 0))


def test_verifier_excludes_out_of_window_keys():
    """A signature by an expired key must not verify through the
    registry's verifier, even though the key is 'active'."""
    s = Ed25519Signer.generate()
    reg = _registry(_record("promotion", s, not_before=BEFORE,
                            not_after=PAST))
    env = s.sign({"decision": "promote"})
    assert reg.verifier(now=NOW).verify(
        {"decision": "promote"}, env) is False
    # inside the window the same signature verifies
    inside = reg.verifier(now=datetime.fromisoformat(BEFORE))
    assert inside.verify({"decision": "promote"}, env) is True


def test_assert_authorized_explains_expiry():
    s = Ed25519Signer.generate()
    reg = _registry(_record("promotion", s, not_after=PAST))
    with pytest.raises(PermissionError, match="expired"):
        reg.assert_authorized("promotion", s.key_id, now=NOW)


def test_assert_authorized_explains_not_yet_valid():
    s = Ed25519Signer.generate()
    reg = _registry(_record("promotion", s, not_before=FUTURE))
    with pytest.raises(PermissionError, match="not valid before"):
        reg.assert_authorized("promotion", s.key_id, now=NOW)


def test_ledger_verify_reports_expired_signer(tmp_path):
    """Ledger integrity is time-aware: a record signed by a key whose
    window has passed fails verification at `now`."""
    plan = Ed25519Signer.generate()
    promo = Ed25519Signer.generate()
    reg = _registry(
        _record("plan", plan),
        _record("promotion", promo, not_before=BEFORE, not_after=AFTER))
    ledger = AuthorityLedger(tmp_path / "AUTHORITY_LEDGER.jsonl")
    ledger.append(plan, "experiment_preregistration", {"plan": 1})
    ledger.append(promo, "promotion_decision", {"decision": 1})
    assert ledger.verify(reg, now=NOW) == []
    after = datetime.fromisoformat(AFTER) + timedelta(seconds=1)
    failures = ledger.verify(reg, now=after)
    assert any("not authorized" in f for f in failures)
    assert any("signature invalid" in f for f in failures)
