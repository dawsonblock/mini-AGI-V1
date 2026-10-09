"""v16.4.2 — authenticated, monotonic revocation snapshots.

UPGRADE_PLAN §3.4: revocation evidence must be signed by the dedicated
`revocation` role, freshness-and-future bounded, epoch-monotonic, and
stored atomically. Every failure mode fails closed.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.security.signed_revocations import (  # noqa: E402
    RevocationRefused, RevocationSnapshotV2, RevocationStore,
    verify_snapshot)
from minagi.v161.authority import (AUTHORITY_ROLES, AuthorityRegistry,  # noqa: E402
                                   write_trust_root)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
TS = int(NOW.timestamp())
DECISION = "sha256:" + "a" * 64


def _chain(tmp_path):
    storage = tmp_path / "storage"
    write_trust_root(storage / ".keys", storage / "trust_root.json")
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    signers = {r: Ed25519Signer.from_private_bytes(
        (storage / ".keys" / f"{r}.pem").read_bytes())
        for r in AUTHORITY_ROLES}
    return registry, signers


def _snap_doc(signer, *, epoch=0, issued_at=TS - 60,
              valid_until=TS + 3600, revoked=(DECISION,), **kw):
    return RevocationSnapshotV2(
        epoch=epoch, issued_at=issued_at, valid_until=valid_until,
        revoked_decision_digests=tuple(revoked), **kw
    ).to_doc(signer=signer)


# ---------- happy path --------------------------------------------------

def test_signed_snapshot_verifies_and_contains_revocation(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = _snap_doc(signers["revocation"])
    snap = verify_snapshot(doc, registry, now=NOW)
    assert snap.epoch == 0
    assert snap.contains(DECISION)
    assert not snap.contains("sha256:" + "b" * 64)


# ---------- refusal rows -------------------------------------------------

def test_unsigned_revocation_list_refused(tmp_path):
    """A bare v16.4.1 revocation list is not authenticated evidence."""
    registry, _ = _chain(tmp_path)
    bare = {"schema": "mini-agi-v16.4.1-revocation-list-v1",
            "digests": [], "generated_at": TS}
    with pytest.raises(RevocationRefused, match="envelope"):
        verify_snapshot(bare, registry, now=NOW)


def test_snapshot_signed_by_wrong_role_refused(tmp_path):
    """The revocation role is dedicated — a promotion or runtime
    signature does not authorize revocation evidence."""
    registry, signers = _chain(tmp_path)
    for role in ("promotion", "runtime", "plan", "admission"):
        doc = _snap_doc(signers[role])
        with pytest.raises(RevocationRefused,
                           match="revocation authority"):
            verify_snapshot(doc, registry, now=NOW)


def test_forged_signature_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = _snap_doc(signers["revocation"])
    forged = dict(doc)
    forged["value"] = dict(doc["value"])
    forged["value"]["revoked_decision_digests"] = ["sha256:" + "c" * 64]
    # fix the digest so only the signature is stale
    from egai.common.canonical import digest
    forged["digest"] = digest(forged["value"])
    with pytest.raises(RevocationRefused, match="signature invalid"):
        verify_snapshot(forged, registry, now=NOW)


def test_envelope_digest_mismatch_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = dict(_snap_doc(signers["revocation"]))
    doc["digest"] = "sha256:" + "0" * 64
    with pytest.raises(RevocationRefused, match="digest mismatch"):
        verify_snapshot(doc, registry, now=NOW)


def test_future_dated_snapshot_refused(tmp_path):
    """A revocation list 'from 2100' is evidence of forgery, not
    freshness — the v16.4.1 list could not express this refusal."""
    registry, signers = _chain(tmp_path)
    doc = _snap_doc(signers["revocation"], issued_at=4102444800,
                    valid_until=4102444800 + 3600)
    with pytest.raises(RevocationRefused, match="future-dated"):
        verify_snapshot(doc, registry, now=NOW)


def test_stale_snapshot_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = _snap_doc(signers["revocation"], issued_at=TS - 10**7,
                    valid_until=TS + 3600)
    with pytest.raises(RevocationRefused, match="stale"):
        verify_snapshot(doc, registry, now=NOW, max_age_seconds=86400)


def test_expired_valid_until_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = _snap_doc(signers["revocation"], issued_at=TS - 7200,
                    valid_until=TS - 60)
    with pytest.raises(RevocationRefused, match="valid_until"):
        verify_snapshot(doc, registry, now=NOW)


def test_replayed_older_epoch_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = _snap_doc(signers["revocation"], epoch=3)
    with pytest.raises(RevocationRefused, match="epoch"):
        verify_snapshot(doc, registry, now=NOW, min_epoch=5)


# ---------- the durable store -------------------------------------------

def test_store_publishes_and_selects_newest(tmp_path):
    registry, signers = _chain(tmp_path)
    store = RevocationStore(tmp_path / "rev")
    store.publish(_snap_doc(signers["revocation"], epoch=0))
    store.publish(_snap_doc(signers["revocation"], epoch=1,
                            revoked=()))
    snap = store.latest_valid(registry, now=NOW)
    assert snap.epoch == 1
    assert not snap.contains(DECISION)


def test_store_refuses_epoch_regression(tmp_path):
    registry, signers = _chain(tmp_path)
    store = RevocationStore(tmp_path / "rev")
    store.publish(_snap_doc(signers["revocation"], epoch=5))
    with pytest.raises(RevocationRefused, match="regresses"):
        store.publish(_snap_doc(signers["revocation"], epoch=4))
    with pytest.raises(RevocationRefused, match="write-once"):
        store.publish(_snap_doc(signers["revocation"], epoch=5))


def test_store_refuses_unsigned_publish(tmp_path):
    _, signers = _chain(tmp_path)
    store = RevocationStore(tmp_path / "rev")
    with pytest.raises(RevocationRefused, match="envelope"):
        store.publish({"digests": []})
    assert store.epochs() == []


def test_store_fails_closed_when_empty(tmp_path):
    registry, _ = _chain(tmp_path)
    store = RevocationStore(tmp_path / "rev")
    with pytest.raises(RevocationRefused, match="fails closed"):
        store.latest_valid(registry, now=NOW)


def test_newest_invalid_snapshot_not_silently_skipped(tmp_path):
    """A newer-but-invalid snapshot must be reported, not skipped in
    favour of an older valid one — a forged newest file may be an
    attempt to hide fresh revocations."""
    registry, signers = _chain(tmp_path)
    store = RevocationStore(tmp_path / "rev")
    store.publish(_snap_doc(signers["revocation"], epoch=0))
    # a snapshot signed by the wrong role can be *published* (storage
    # is transport; verification happens on read) but never served
    store.publish(_snap_doc(signers["promotion"], epoch=1))
    with pytest.raises(RevocationRefused, match="revocation authority"):
        store.latest_valid(registry, now=NOW)


def test_published_files_are_atomic_and_listed(tmp_path):
    _, signers = _chain(tmp_path)
    store = RevocationStore(tmp_path / "rev")
    store.publish(_snap_doc(signers["revocation"], epoch=0))
    store.publish(_snap_doc(signers["revocation"], epoch=3))
    assert store.epochs() == [0, 3]
    assert store.load(0) is not None
    assert store.load(99) is None
