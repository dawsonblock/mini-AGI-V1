"""v16.4.3 — durable, atomic admission-grant ledger (SEC-201/SEC-207).

The spec's acceptance gate: exactly-once grant reservation must hold
across concurrent requests, process restarts, and injected storage
failures. Every case below exercises `AuthorityStore.reserve_grant`
(and the supervisor's `authorize` path that wraps it).
"""
import multiprocessing
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.runtime.authority_store import (  # noqa: E402
    AuthorityStore, AuthorityStoreError, GrantConsumed)
from minagi.runtime.supervisor import (  # noqa: E402
    ServingSupervisor)
from minagi.security.admission_grants import (  # noqa: E402
    AdmissionGrantV1, issue_grant)
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


def _grant_doc(signers, **over):
    kw = dict(decision_digest=digest({"d": 1}),
              qualification_digest=digest({"q": 1}),
              runtime_manifest_digest=digest({"m": 1}),
              artifact_root_digest=digest({"r": 1}),
              backend_id="hf-peft",
              audience_runtime_identity="local-supervisor", now=NOW)
    kw.update(over)
    return issue_grant(signers["admission"], **kw)


def _grant_obj(doc):
    return AdmissionGrantV1.from_value(doc["value"])


def _reserve(store, doc, aid):
    return store.reserve_grant(_grant_obj(doc), activation_id=aid,
                               at=TS)


# ---------- replay --------------------------------------------------------

def test_reuse_same_process_refused(tmp_path):
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    doc = _grant_doc(_chain(tmp_path)[1])
    _reserve(store, doc, "aa" * 16)
    with pytest.raises(GrantConsumed):
        _reserve(store, doc, "bb" * 16)


def test_reuse_after_restart_refused(tmp_path):
    """The reservation is durable: a fresh store on the same database
    still refuses the consumed grant."""
    db = tmp_path / "state" / "authority.sqlite"
    doc = _grant_doc(_chain(tmp_path)[1])
    _reserve(AuthorityStore(db), doc, "aa" * 16)
    with pytest.raises(GrantConsumed):
        _reserve(AuthorityStore(db), doc, "cc" * 16)


def test_identical_nonce_different_grant_id_refused(tmp_path):
    """A grant is identified by id AND nonce AND digest — a replay
    with the same nonce under a different grant id still collides."""
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    doc_a = _grant_doc(_chain(tmp_path)[1])
    _reserve(store, doc_a, "aa" * 16)
    forged = dict(doc_a["value"])
    forged["grant_id"] = "grant-" + "f" * 16   # different id, same nonce
    forged_obj = AdmissionGrantV1(**{k: v for k, v in forged.items()})
    with pytest.raises(GrantConsumed):
        store.reserve_grant(forged_obj, activation_id="dd" * 16, at=TS)


def test_activation_id_uniqueness_refused(tmp_path):
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    signers = _chain(tmp_path)[1]
    _reserve(store, _grant_doc(signers), "ee" * 16)
    with pytest.raises(GrantConsumed):
        _reserve(store, _grant_doc(signers), "ee" * 16)


# ---------- concurrency ----------------------------------------------------

def test_two_threads_exactly_one_wins(tmp_path):
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    doc = _grant_doc(_chain(tmp_path)[1])
    results = {"ok": 0, "refused": 0}
    barrier = threading.Barrier(2)

    def try_reserve(aid):
        barrier.wait()
        try:
            _reserve(store, doc, aid)
            results["ok"] += 1
        except (GrantConsumed, AuthorityStoreError):
            results["refused"] += 1

    threads = [threading.Thread(target=try_reserve, args=(f"{i:016x}" * 2,))
               for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results["ok"] == 1
    assert results["refused"] == 1


def _proc_reserve(db_path: str, grant_value: dict, aid: str, q):
    store = AuthorityStore(db_path)
    try:
        store.reserve_grant(AdmissionGrantV1.from_value(grant_value),
                            activation_id=aid, at=TS)
        q.put("ok")
    except GrantConsumed:
        q.put("refused")
    except Exception as exc:  # noqa: BLE001
        q.put(f"error:{exc}")


def test_two_processes_exactly_one_wins(tmp_path):
    """BEGIN IMMEDIATE serializes writers across processes — exactly
    one reservation commits."""
    db = tmp_path / "state" / "authority.sqlite"
    AuthorityStore(db)  # create schema first
    doc = _grant_doc(_chain(tmp_path)[1])
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_proc_reserve,
                         args=(str(db), doc["value"], f"{i:016x}" * 2, q))
             for i in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
    outcomes = [q.get(timeout=30) for _ in procs]
    assert outcomes.count("ok") == 1
    assert outcomes.count("refused") == 1


# ---------- crash + failure ------------------------------------------------

def test_crash_after_reservation_commit_keeps_grant_consumed(tmp_path):
    """A failure after the reservation commit does not release the
    grant — the row is durable the moment COMMIT lands."""
    db = tmp_path / "state" / "authority.sqlite"
    doc = _grant_doc(_chain(tmp_path)[1])
    _reserve(AuthorityStore(db), doc, "aa" * 16)
    # simulate the crash: drop the connection without cleanup, reopen
    reopened = AuthorityStore(db)
    assert reopened.grant_reserved(_grant_obj(doc).grant_id)
    with pytest.raises(GrantConsumed):
        _reserve(reopened, doc, "bb" * 16)


def test_unavailable_database_refuses_activation(tmp_path):
    """No activation is permitted when the authority store cannot be
    opened — fail closed, never a degraded in-memory fallback."""
    bad = tmp_path / "notadir"
    bad.write_text("blocking")
    with pytest.raises(AuthorityStoreError):
        AuthorityStore(bad / "authority.sqlite")


def test_corrupt_database_fails_closed(tmp_path):
    db = tmp_path / "state"
    db.mkdir(parents=True)
    (db / "authority.sqlite").write_bytes(b"not a sqlite database" * 64)
    with pytest.raises(AuthorityStoreError):
        AuthorityStore(db / "authority.sqlite")


# ---------- idempotent request outcomes ------------------------------------

def test_retry_same_request_returns_recorded_outcome(tmp_path):
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    store.record_request(request_id="req-1", activation_id="aa" * 16,
                         outcome="committed",
                         outcome_digest=digest({"r": 1}), at=TS)
    hit = store.lookup_request("req-1")
    assert hit["outcome"] == "committed"
    assert hit["activation_id"] == "aa" * 16
    assert store.lookup_request("req-absent") is None


def test_supervisor_authorize_reserves_through_store(tmp_path):
    """The supervisor's authorize() reserves in the store — the
    in-memory set is gone."""
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    sup = ServingSupervisor(store, runtime_signer=signers["runtime"],
                            registry=registry, now=NOW)
    assert not hasattr(sup, "_consumed_grants")
    doc = _grant_doc(signers)
    aid = sup.request().activation_id
    sup.authorize(aid, doc)
    assert store.grant_reserved(_grant_obj(doc).grant_id)
