"""v16.4.3 — verified event log (SEC-205 / WP4).

`store.verify_chain` checks hash-chain continuity and digests;
`verify_event_log` adds signature/role binding and transition grammar;
`anchor_checkpoint` bounds tail tampering; `migrate_v1_journal` imports
verified V1 JSONL history behind a signed checkpoint.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.runtime.authority_store import (  # noqa: E402
    AuthorityStore, StoreCorrupt)
from minagi.runtime.journal_v2 import (  # noqa: E402
    JournalRefused, anchor_checkpoint, migrate_v1_journal,
    verify_event_log)
from minagi.v161.authority import (AUTHORITY_ROLES, AuthorityRegistry,  # noqa: E402
                                   write_trust_root)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
TS = int(NOW.timestamp())


def _chain(tmp_path):
    storage = tmp_path / "storage"
    write_trust_root(storage / ".keys", storage / "trust_root.json")
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    signers = {r: Ed25519Signer.from_private_bytes(
        (storage / ".keys" / f"{r}.pem").read_bytes())
        for r in AUTHORITY_ROLES}
    return registry, signers


def _history(store, aid, states, *, signer=None, kind="transition"):
    src = ""
    for st in states:
        store.append_event(activation_id=aid, event_type=kind,
                           from_state=src, to_state=st, at=TS,
                           signer=signer)
        src = st


def _commit(store, aid, *, signer):
    store.commit_with_pointer(
        activation_id=aid, at=TS, artifact_root_digest=digest({"r": aid}),
        backend_id="hf-peft", detail={"kind": "activation_intent"})
    store.append_event(
        activation_id=aid, event_type="activation_completion",
        from_state="COMMITTED", to_state="ACTIVE", at=TS,
        detail={"kind": "activation_completion"}, signer=signer)


# ---------- hash chain + digest ------------------------------------------

def _lifecycle_through_ready(store, aid):
    _history(store, aid,
             ["REQUESTED", "AUTHORIZED", "STAGED", "PREPARED", "READY"])


def test_clean_log_verifies(tmp_path):
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    _lifecycle_through_ready(store, "aa" * 16)
    _commit(store, "aa" * 16, signer=signers["runtime"])
    events = verify_event_log(store, registry, now=NOW)
    assert len(events) == 7


def test_modified_record_detected(tmp_path):
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    _history(store, "aa" * 16, ["REQUESTED", "AUTHORIZED"])
    store._db.execute(
        "UPDATE runtime_events SET detail_json = '{\"x\": 1}' "
        "WHERE sequence = 1")
    store._db.commit()
    with pytest.raises(StoreCorrupt):
        verify_event_log(store, registry, now=NOW)


def test_truncated_tail_detected(tmp_path):
    """Deleting the last event breaks the anchor — a truncation cannot
    pass silently once a checkpoint exists."""
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    _lifecycle_through_ready(store, "aa" * 16)
    _commit(store, "aa" * 16, signer=signers["runtime"])
    anchor_checkpoint(store, signers["runtime"], now=NOW)
    store._db.execute("DELETE FROM runtime_events WHERE sequence = 7")
    store._db.commit()
    with pytest.raises((StoreCorrupt, JournalRefused)):
        verify_event_log(store, registry, now=NOW)


def test_spliced_middle_detected(tmp_path):
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    _history(store, "aa" * 16, ["REQUESTED", "AUTHORIZED", "STAGED"])
    store._db.execute("DELETE FROM runtime_events WHERE sequence = 2")
    store._db.commit()
    with pytest.raises(StoreCorrupt):
        verify_event_log(store, registry, now=NOW)


# ---------- signatures + grammar ------------------------------------------

def test_unsigned_completion_refused(tmp_path):
    """An authority-bearing event without a signature is not
    evidence — even if the hash chain is intact."""
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    _history(store, "aa" * 16,
             ["REQUESTED", "AUTHORIZED", "STAGED", "PREPARED", "READY"])
    store.commit_with_pointer(
        activation_id="aa" * 16, at=TS,
        artifact_root_digest=digest({"r": 1}), backend_id="hf-peft",
        detail={"kind": "activation_intent"})
    store.append_event(                       # UNSIGNED completion
        activation_id="aa" * 16, event_type="activation_completion",
        from_state="COMMITTED", to_state="ACTIVE", at=TS,
        detail={"kind": "activation_completion"})
    with pytest.raises(JournalRefused, match="unsigned"):
        verify_event_log(store, registry, now=NOW)


def test_wrong_role_signature_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    _history(store, "aa" * 16,
             ["REQUESTED", "AUTHORIZED", "STAGED", "PREPARED", "READY"])
    store.commit_with_pointer(
        activation_id="aa" * 16, at=TS,
        artifact_root_digest=digest({"r": 1}), backend_id="hf-peft",
        detail={"kind": "activation_intent"})
    store.append_event(
        activation_id="aa" * 16, event_type="activation_completion",
        from_state="COMMITTED", to_state="ACTIVE", at=TS,
        detail={"kind": "activation_completion"},
        signer=signers["promotion"])          # wrong role
    with pytest.raises(JournalRefused, match="not an authorized"):
        verify_event_log(store, registry, now=NOW)


def test_illegal_transition_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    _history(store, "aa" * 16, ["REQUESTED"])
    store.append_event(                       # REQUESTED -> ACTIVE hop
        activation_id="aa" * 16, event_type="transition",
        from_state="REQUESTED", to_state="ACTIVE", at=TS)
    with pytest.raises(JournalRefused, match="illegal transition"):
        verify_event_log(store, registry, now=NOW)


def test_forged_state_history_refused(tmp_path):
    """An event claiming a from_state that doesn't match the recorded
    state is a fabricated history."""
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    _history(store, "aa" * 16, ["REQUESTED", "AUTHORIZED"])
    store.append_event(
        activation_id="aa" * 16, event_type="transition",
        from_state="READY", to_state="COMMITTED", at=TS)
    with pytest.raises(JournalRefused, match="recorded state"):
        verify_event_log(store, registry, now=NOW)


# ---------- V1 -> V2 migration --------------------------------------------

def _v1_journal(path, records):
    """Write a v16.4.2-format JSONL journal."""
    lines = []
    for rec in records:
        body = {"schema": "mini-agi-v16.4.2-activation-journal-v1",
                "activation_id": rec["aid"],
                "from_state": rec.get("from", ""),
                "to_state": rec["to"], "at": rec.get("at", TS),
                "detail": rec.get("detail", {})}
        entry = dict(body)
        if rec.get("signer") is not None:
            env = rec["signer"].sign(body)
            entry["signer_key_id"] = env.key_id
            entry["signature_b64"] = env.signature_b64
        entry["record_digest"] = digest(body)
        lines.append(json.dumps(entry, sort_keys=True,
                                separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def test_v1_journal_migrates_with_checkpoint(tmp_path):
    registry, signers = _chain(tmp_path)
    jdir = tmp_path / "journal"
    aid = "cc" * 16
    _v1_journal(jdir / "activation_journal.jsonl", [
        {"aid": aid, "to": "REQUESTED"},
        {"aid": aid, "from": "REQUESTED", "to": "AUTHORIZED"},
        {"aid": aid, "from": "AUTHORIZED", "to": "STAGED"},
        {"aid": aid, "from": "STAGED", "to": "PREPARED"},
        {"aid": aid, "from": "PREPARED", "to": "READY"},
        {"aid": aid, "from": "READY", "to": "COMMITTED",
         "detail": {"kind": "activation_intent"}},
        {"aid": aid, "from": "COMMITTED", "to": "ACTIVE",
         "detail": {"kind": "activation_completion"},
         "signer": signers["runtime"]},
    ])
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    report = migrate_v1_journal(jdir, store, registry=registry,
                                migration_signer=signers["runtime"])
    assert report["migrated"] == 7
    events = verify_event_log(store, registry, now=NOW)
    assert len(events) == 8   # 7 migrated + checkpoint event
    assert store.get_meta("migration_checkpoint")


def test_v1_tampered_record_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    jdir = tmp_path / "journal"
    aid = "dd" * 16
    _v1_journal(jdir / "activation_journal.jsonl", [
        {"aid": aid, "to": "REQUESTED"},
        {"aid": aid, "from": "REQUESTED", "to": "AUTHORIZED"},
    ])
    # rewrite history in place
    lines = (jdir / "activation_journal.jsonl").read_text().splitlines()
    rec = json.loads(lines[1])
    rec["to_state"] = "ACTIVE"     # without recomputing record_digest
    lines[1] = json.dumps(rec)
    (jdir / "activation_journal.jsonl").write_text("\n".join(lines))
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    with pytest.raises(JournalRefused, match="digest mismatch"):
        migrate_v1_journal(jdir, store, registry=registry,
                           migration_signer=signers["runtime"])


def test_v1_migration_refuses_nonempty_store(tmp_path):
    registry, signers = _chain(tmp_path)
    jdir = tmp_path / "journal"
    _v1_journal(jdir / "activation_journal.jsonl",
                [{"aid": "ee" * 16, "to": "REQUESTED"}])
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    _history(store, "ee" * 16, ["REQUESTED"])
    with pytest.raises(JournalRefused, match="not empty"):
        migrate_v1_journal(jdir, store, registry=registry,
                           migration_signer=signers["runtime"])


def test_migrated_legacy_rollback_grammar_lenient(tmp_path):
    """The V1 journal wrote rollback as ABORTED -> ACTIVE — migrated
    events carry event_type='migrated' and are checked leniently."""
    registry, signers = _chain(tmp_path)
    jdir = tmp_path / "journal"
    aid = "ff" * 16
    _v1_journal(jdir / "activation_journal.jsonl", [
        {"aid": aid, "to": "REQUESTED"},
        {"aid": aid, "from": "REQUESTED", "to": "AUTHORIZED"},
        {"aid": aid, "from": "AUTHORIZED", "to": "ABORTED"},
        {"aid": aid, "from": "ABORTED", "to": "ACTIVE",
         "detail": {"kind": "rollback"}, "signer": signers["runtime"]},
    ])
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    migrate_v1_journal(jdir, store, registry=registry,
                       migration_signer=signers["runtime"])
    events = verify_event_log(store, registry, now=NOW)
    assert any(e["event_type"] == "migrated" for e in events)
