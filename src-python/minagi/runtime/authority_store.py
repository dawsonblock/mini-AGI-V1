"""v16.4.3 single transactional authority store (HARDENING_PLAN WP1).

The v16.4.2 supervisor enforced single-use admission grants with an
in-memory ``set()``: the check and the state transition were not atomic,
and a restart forgot every consumed grant (SEC-201/SEC-207). This module
makes grant reservations, lifecycle events, and the serving pointer one
transactional authority — a SQLite database in WAL mode with
``synchronous=FULL`` and explicit ``BEGIN IMMEDIATE`` write transactions:

  * ``reserve_grant`` verifies nothing itself (the caller verifies the
    signature/window first); it atomically checks grant id, nonce, and
    digest against *persisted* reservations and inserts the reservation
    row plus the authorization event in one committed transaction. Two
    threads or two processes reserving the same grant serialize on the
    database write lock — exactly one wins.
  * ``runtime_events`` is the hash-chained durable event log: each event
    binds the previous event's digest, its payload digest, and (for
    authority-bearing kinds) a signature. The JSONL journal remains as a
    compatibility input for migration only; it is not a second
    authority.
  * The serving pointer lives in the same database and moves in the
    same transaction as the COMMITTED event — a crash cannot leave a
    committed pointer without its commit event.
  * ``launch_requests`` gives idempotent retry: a repeated client
    request identity returns the recorded outcome instead of consuming
    a second grant.

The database file must sit on protected storage owned by the runtime
service identity (``secure_dir`` in ``access_policy`` enforces this).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer, SignedEnvelope

STORE_SCHEMA_VERSION = "mini-agi-v16.4.3-authority-store-v1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS admission_grants (
    grant_id TEXT PRIMARY KEY,
    grant_nonce TEXT NOT NULL UNIQUE,
    grant_digest TEXT NOT NULL UNIQUE,
    activation_id TEXT NOT NULL UNIQUE,
    runtime_identity TEXT NOT NULL,
    reservation_state TEXT NOT NULL,
    reserved_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS runtime_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    activation_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    at INTEGER NOT NULL,
    detail_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    previous_digest TEXT NOT NULL,
    event_digest TEXT NOT NULL UNIQUE,
    signer_key_id TEXT NOT NULL,
    signature_b64 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS launch_requests (
    request_id TEXT PRIMARY KEY,
    activation_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    outcome_digest TEXT NOT NULL,
    recorded_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS serving_pointer (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    activation_id TEXT NOT NULL,
    artifact_root_digest TEXT NOT NULL,
    backend_id TEXT NOT NULL,
    committed_at INTEGER NOT NULL,
    event_sequence INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class AuthorityStoreError(RuntimeError):
    """The transactional authority could not be consulted — fail closed."""


class GrantConsumed(AuthorityStoreError):
    """The grant id, nonce, or digest is already reserved — replay."""


class StoreCorrupt(AuthorityStoreError):
    """Durable state failed integrity checks — no activation permitted."""


def _payload_digest(activation_id: str, event_type: str, from_state: str,
                    to_state: str, at: int, detail: dict) -> str:
    return digest({"activation_id": activation_id,
                   "event_type": event_type, "from_state": from_state,
                   "to_state": to_state, "at": int(at),
                   "detail": dict(detail or {})})


class AuthorityStore:
    """Transactional authority for grants, lifecycle events, and the
    serving pointer. One instance is thread-safe; multiple instances
    (processes) serialize on the SQLite write lock."""

    def __init__(self, db_path, *, busy_timeout_ms: int = 5000):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(
                str(self.db_path), timeout=max(1, busy_timeout_ms // 1000),
                check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            with self._db:
                self._db.executescript(_SCHEMA)
                self._db.execute(
                    "INSERT OR IGNORE INTO meta(key, value) "
                    "VALUES('schema_version', ?)",
                    (STORE_SCHEMA_VERSION,))
        except (sqlite3.Error, OSError) as exc:
            raise AuthorityStoreError(
                f"authority store unavailable at {self.db_path}: {exc}") \
                from exc

    def close(self) -> None:
        with self._lock:
            try:
                self._db.close()
            except sqlite3.Error:
                pass

    # --- grants -----------------------------------------------------
    def reserve_grant(self, grant, *, activation_id: str, at: int,
                      from_state: str = "", to_state: str = "AUTHORIZED",
                      detail: dict | None = None) -> dict:
        """Atomically reserve a verified grant and journal the
        AUTHORIZED event. Raises GrantConsumed on id/nonce/digest
        collision — the grant consumed stays consumed even if the
        activation later fails."""
        try:
            with self._lock:
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    clash = self._db.execute(
                        "SELECT grant_id FROM admission_grants WHERE "
                        "grant_id = ? OR grant_nonce = ? OR "
                        "grant_digest = ? OR activation_id = ?",
                        (grant.grant_id, grant.nonce,
                         grant.digest, activation_id)).fetchone()
                    if clash is not None:
                        self._db.execute("ROLLBACK")
                        raise GrantConsumed(
                            "grant id, nonce, digest, or activation id "
                            "is already reserved — a grant authorizes "
                            "exactly one activation")
                    self._db.execute(
                        "INSERT INTO admission_grants("
                        "grant_id, grant_nonce, grant_digest, "
                        "activation_id, runtime_identity, "
                        "reservation_state, reserved_at, expires_at) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (grant.grant_id, grant.nonce, grant.digest,
                         activation_id, grant.audience_runtime_identity,
                         "reserved", int(at), int(grant.expires_at)))
                    event = self._append_event_txn(
                        activation_id=activation_id,
                        event_type="authorized",
                        from_state=from_state, to_state=to_state, at=at,
                        detail=dict(detail or {}))
                    self._db.execute("COMMIT")
                    return event
                except Exception:
                    try:
                        self._db.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise
        except GrantConsumed:
            raise
        except sqlite3.Error as exc:
            raise AuthorityStoreError(
                f"grant reservation failed: {exc}") from exc

    def grant_reserved(self, grant_id: str) -> bool:
        row = self._db.execute(
            "SELECT 1 FROM admission_grants WHERE grant_id = ?",
            (grant_id,)).fetchone()
        return row is not None

    def set_grant_state(self, grant_id: str, state: str) -> None:
        """Update the reservation outcome (reserved -> committed /
        aborted / quarantined). The row is never deleted."""
        with self._lock, self._db:
            self._db.execute(
                "UPDATE admission_grants SET reservation_state = ? "
                "WHERE grant_id = ?", (state, grant_id))

    # --- events -----------------------------------------------------
    def _append_event_txn(self, *, activation_id: str, event_type: str,
                          from_state: str, to_state: str, at: int,
                          detail: dict | None = None,
                          signer: Ed25519Signer | None = None) -> dict:
        """Append one hash-chained event. Must run inside BEGIN
        IMMEDIATE so the sequence it computes is the sequence assigned."""
        row = self._db.execute(
            "SELECT COALESCE(MAX(sequence), 0), "
            "COALESCE((SELECT event_digest FROM runtime_events "
            "         ORDER BY sequence DESC LIMIT 1), '') "
            "FROM runtime_events").fetchone()
        seq, prev = int(row[0]) + 1, str(row[1])
        pdigest = _payload_digest(activation_id, event_type, from_state,
                                  to_state, at, detail)
        body = {"sequence": seq, "activation_id": activation_id,
                "event_type": event_type, "from_state": from_state,
                "to_state": to_state, "at": int(at),
                "payload_digest": pdigest, "previous_digest": prev}
        env = signer.sign(body) if signer is not None else \
            SignedEnvelope("", "")
        event_digest = digest(dict(body, signer_key_id=env.key_id,
                                   signature_b64=env.signature_b64))
        detail_json = json.dumps(dict(detail or {}), sort_keys=True,
                                 separators=(",", ":"))
        self._db.execute(
            "INSERT INTO runtime_events(sequence, activation_id, "
            "event_type, from_state, to_state, at, detail_json, "
            "payload_digest, previous_digest, event_digest, "
            "signer_key_id, signature_b64) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (seq, activation_id, event_type, from_state, to_state,
             int(at), detail_json, pdigest, prev, event_digest,
             env.key_id, env.signature_b64))
        return {"sequence": seq, "activation_id": activation_id,
                "event_type": event_type, "from_state": from_state,
                "to_state": to_state, "at": int(at),
                "payload_digest": pdigest, "previous_digest": prev,
                "event_digest": event_digest,
                "signer_key_id": env.key_id,
                "signature_b64": env.signature_b64,
                "detail": dict(detail or {})}

    def append_event(self, *, activation_id: str, event_type: str,
                     from_state: str, to_state: str, at: int,
                     detail: dict | None = None,
                     signer: Ed25519Signer | None = None) -> dict:
        """Durably append one event in its own IMMEDIATE transaction."""
        try:
            with self._lock:
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    event = self._append_event_txn(
                        activation_id=activation_id,
                        event_type=event_type, from_state=from_state,
                        to_state=to_state, at=at, detail=detail,
                        signer=signer)
                    self._db.execute("COMMIT")
                    return event
                except Exception:
                    try:
                        self._db.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise
        except sqlite3.Error as exc:
            raise AuthorityStoreError(
                f"event append failed: {exc}") from exc

    def commit_with_pointer(self, *, activation_id: str, at: int,
                            artifact_root_digest: str, backend_id: str,
                            event_type: str = "commit_intent",
                            from_state: str = "READY",
                            to_state: str = "COMMITTED",
                            detail: dict | None = None,
                            signer: Ed25519Signer | None = None) -> dict:
        """Atomically append the commit event AND move the serving
        pointer — the two can never diverge across a crash."""
        try:
            with self._lock:
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    event = self._append_event_txn(
                        activation_id=activation_id,
                        event_type=event_type, from_state=from_state,
                        to_state=to_state, at=at, detail=detail,
                        signer=signer)
                    self._db.execute(
                        "INSERT INTO serving_pointer(id, activation_id, "
                        "artifact_root_digest, backend_id, committed_at,"
                        " event_sequence) VALUES(1,?,?,?,?,?) ON "
                        "CONFLICT(id) DO UPDATE SET activation_id="
                        "excluded.activation_id, artifact_root_digest="
                        "excluded.artifact_root_digest, backend_id="
                        "excluded.backend_id, committed_at=excluded."
                        "committed_at, event_sequence=excluded."
                        "event_sequence",
                        (activation_id, artifact_root_digest, backend_id,
                         int(at), int(event["sequence"])))
                    self._db.execute("COMMIT")
                    return event
                except Exception:
                    try:
                        self._db.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise
        except sqlite3.Error as exc:
            raise AuthorityStoreError(
                f"commit-with-pointer transaction failed: {exc}") from exc

    def clear_pointer(self, *, at: int, because: str = "") -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE serving_pointer SET activation_id = '', "
                "artifact_root_digest = '', backend_id = '', "
                "committed_at = ? WHERE id = 1", (int(at),))

    def read_pointer(self) -> dict | None:
        row = self._db.execute(
            "SELECT activation_id, artifact_root_digest, backend_id, "
            "committed_at, event_sequence FROM serving_pointer "
            "WHERE id = 1").fetchone()
        if row is None:
            return None
        return {"activation_id": row[0], "artifact_root_digest": row[1],
                "backend_id": row[2], "committed_at": row[3],
                "event_sequence": row[4]}

    def events(self, activation_id: str | None = None) -> list[dict]:
        """The durable event log (all activations or one), in order."""
        q = ("SELECT sequence, activation_id, event_type, from_state, "
             "to_state, at, detail_json, payload_digest, "
             "previous_digest, event_digest, signer_key_id, "
             "signature_b64 FROM runtime_events")
        rows = (self._db.execute(q + " WHERE activation_id = ? ORDER BY "
                                 "sequence", (activation_id,)).fetchall()
                if activation_id else
                self._db.execute(q + " ORDER BY sequence").fetchall())
        return [{"sequence": r[0], "activation_id": r[1],
                 "event_type": r[2], "from_state": r[3], "to_state": r[4],
                 "at": r[5], "detail": json.loads(r[6] or "{}"),
                 "payload_digest": r[7],
                 "previous_digest": r[8], "event_digest": r[9],
                 "signer_key_id": r[10], "signature_b64": r[11]}
                for r in rows]

    # --- idempotent request outcomes --------------------------------
    def lookup_request(self, request_id: str) -> dict | None:
        row = self._db.execute(
            "SELECT activation_id, outcome, outcome_digest, recorded_at "
            "FROM launch_requests WHERE request_id = ?",
            (request_id,)).fetchone()
        if row is None:
            return None
        return {"activation_id": row[0], "outcome": row[1],
                "outcome_digest": row[2], "recorded_at": row[3]}

    def record_request(self, *, request_id: str, activation_id: str,
                       outcome: str, outcome_digest: str, at: int) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO launch_requests(request_id, "
                "activation_id, outcome, outcome_digest, recorded_at) "
                "VALUES(?,?,?,?,?)",
                (request_id, activation_id, outcome, outcome_digest,
                 int(at)))

    # --- integrity --------------------------------------------------
    def verify_chain(self) -> list[dict]:
        """Re-read every event and verify the hash chain and payload
        digests. Returns the verified events; raises StoreCorrupt on
        any discontinuity (tamper, truncation, rewrite)."""
        events = self.events()
        prev = ""
        for e in events:
            if e["previous_digest"] != prev:
                raise StoreCorrupt(
                    f"event chain broken at sequence {e['sequence']}: "
                    "previous_digest mismatch — the durable log was "
                    "altered or truncated")
            body = {"sequence": e["sequence"],
                    "activation_id": e["activation_id"],
                    "event_type": e["event_type"],
                    "from_state": e["from_state"],
                    "to_state": e["to_state"], "at": e["at"],
                    "payload_digest": e["payload_digest"],
                    "previous_digest": e["previous_digest"]}
            expect = digest(dict(body, signer_key_id=e["signer_key_id"],
                                 signature_b64=e["signature_b64"]))
            if e["event_digest"] != expect:
                raise StoreCorrupt(
                    f"event digest mismatch at sequence "
                    f"{e['sequence']} — record was modified")
            recomp = _payload_digest(
                e["activation_id"], e["event_type"], e["from_state"],
                e["to_state"], e["at"], e["detail"])
            if e["payload_digest"] != recomp:
                raise StoreCorrupt(
                    f"payload digest mismatch at sequence "
                    f"{e['sequence']} — record was modified")
            prev = e["event_digest"]
        return events

    def last_event_sequence(self) -> int:
        row = self._db.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM runtime_events"
        ).fetchone()
        return int(row[0])

    # --- meta -------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self._db.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO meta(key, value) VALUES(?,?) ON CONFLICT"
                "(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value) if not isinstance(value, str)
                 else value))
