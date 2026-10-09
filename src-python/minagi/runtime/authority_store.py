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

v16.4.4 (WP-A/WP-D) adds the deployment-generation transition record
and the mandatory administrative audit chain:

  * ``deployment`` is a single-row, monotonically versioned record of
    which activation SHOULD serve: ``deployment_generation``,
    ``desired_activation_id``, ``previous_activation_id``,
    ``transition_id``, ``transition_phase``, ``policy_epoch``, and
    ``last_committed_event_digest``. Durable intent commits BEFORE any
    traffic is published (authorization is not activation); the
    observed routing outcome is recorded afterwards as separate,
    truthful evidence — never described as pre-traffic authorization.
    Every transition carries the expected generation so a stale
    transition cannot overwrite a completed newer one (compare-and-
    swap on ``deployment_generation``).
  * ``admin_audit`` is a hash-chained, signed log of security-
    sensitive administrative decisions — a failed audit write refuses
    the operation (emergency traffic shutdown excepted by policy).

The database file must sit on protected storage owned by the runtime
service identity (``secure_dir`` in ``access_policy`` enforces this).
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from pathlib import Path

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer, SignedEnvelope

STORE_SCHEMA_VERSION = "mini-agi-v16.4.4-authority-store-v2"

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
CREATE TABLE IF NOT EXISTS deployment (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    deployment_generation INTEGER NOT NULL,
    desired_activation_id TEXT NOT NULL,
    previous_activation_id TEXT NOT NULL,
    transition_id TEXT NOT NULL,
    transition_phase TEXT NOT NULL,
    policy_epoch INTEGER NOT NULL,
    last_committed_event_digest TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS admin_audit (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    principal_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    target_activation TEXT NOT NULL,
    policy_digest TEXT NOT NULL,
    decision TEXT NOT NULL,
    before_state TEXT NOT NULL,
    after_state TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    at INTEGER NOT NULL,
    previous_event_digest TEXT NOT NULL,
    event_digest TEXT NOT NULL UNIQUE,
    signer_key_id TEXT NOT NULL,
    signature_b64 TEXT NOT NULL
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


class GenerationConflict(AuthorityStoreError):
    """A transition based on a superseded deployment generation — the
    stale transition loses; it does not overwrite a completed newer
    generation (compare-and-swap)."""


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
                    "INSERT INTO meta(key, value) "
                    "VALUES('schema_version', ?) ON CONFLICT(key) DO "
                    "UPDATE SET value = excluded.value",
                    (STORE_SCHEMA_VERSION,))
                self._migrate_deployment_row()
        except (sqlite3.Error, OSError) as exc:
            raise AuthorityStoreError(
                f"authority store unavailable at {self.db_path}: {exc}") \
                from exc

    def _migrate_deployment_row(self) -> None:
        """v16.4.3 -> v16.4.4: a pre-existing ``serving_pointer`` seeds
        the single deployment row at generation 1. The migrated phase
        is ROUTED when a signed completion/observation event exists for
        the pointed activation, COMMITTED otherwise — durable intent
        and live state remain distinguishable."""
        row = self._db.execute(
            "SELECT COUNT(*) FROM deployment").fetchone()
        if int(row[0]) > 0:
            return
        ptr = self._db.execute(
            "SELECT activation_id, committed_at, event_sequence FROM "
            "serving_pointer WHERE id = 1").fetchone()
        if ptr is None or not str(ptr[0]):
            return
        observed = self._db.execute(
            "SELECT COUNT(*) FROM runtime_events WHERE activation_id = "
            "? AND to_state = 'ACTIVE' AND event_type IN "
            "('activation_completion', 'routing_observed', "
            "'rollback_completion')", (str(ptr[0]),)).fetchone()
        phase = "ROUTED" if int(observed[0]) > 0 else "COMMITTED"
        last = self._db.execute(
            "SELECT event_digest FROM runtime_events WHERE sequence = ?",
            (int(ptr[2]),)).fetchone()
        self._db.execute(
            "INSERT INTO deployment(id, deployment_generation, "
            "desired_activation_id, previous_activation_id, "
            "transition_id, transition_phase, policy_epoch, "
            "last_committed_event_digest, updated_at) "
            "VALUES(1,?,?,?,?,?,?,?,?)",
            (1, str(ptr[0]), "", f"migrated-{secrets.token_hex(8)}",
             phase, 0, str(last[0]) if last else "", int(ptr[1])))

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

    # --- deployment generations (v16.4.4 WP-A) ------------------------
    def deployment(self) -> dict | None:
        """The current deployment transition record: the durable
        intent (which activation should serve), its monotonically
        increasing generation, and the transition phase."""
        row = self._db.execute(
            "SELECT deployment_generation, desired_activation_id, "
            "previous_activation_id, transition_id, transition_phase, "
            "policy_epoch, last_committed_event_digest, updated_at "
            "FROM deployment WHERE id = 1").fetchone()
        if row is None:
            return None
        return {"deployment_generation": int(row[0]),
                "desired_activation_id": row[1],
                "previous_activation_id": row[2],
                "transition_id": row[3],
                "transition_phase": row[4],
                "policy_epoch": int(row[5]),
                "last_committed_event_digest": row[6],
                "updated_at": int(row[7])}

    def commit_activation_intent(self, *, candidate_id: str,
                                 expected_generation: int, at: int,
                                 artifact_root_digest: str,
                                 backend_id: str, policy_epoch: int = 0,
                                 event_type: str = "commit_intent",
                                 from_state: str = "READY",
                                 to_state: str = "COMMITTED",
                                 detail: dict | None = None,
                                 signer: Ed25519Signer | None = None
                                 ) -> dict:
        """Stage one of the two-stage activation protocol: durably
        record WHICH activation is authorized to serve, at the NEXT
        deployment generation, BEFORE any traffic may reach it.

        Compare-and-swap: the transaction refuses unless the current
        deployment generation equals ``expected_generation`` — a
        transition based on generation N cannot overwrite a completed
        generation N+1. The commit event, the serving pointer, and the
        deployment row move in ONE transaction."""
        try:
            with self._lock:
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    dep = self._deployment_txn()
                    current = int(dep["deployment_generation"]) \
                        if dep is not None else 0
                    if current != int(expected_generation):
                        self._db.execute("ROLLBACK")
                        raise GenerationConflict(
                            f"deployment generation is {current}, not "
                            f"{expected_generation} — the transition is "
                            "stale and loses")
                    generation = current + 1
                    transition_id = secrets.token_hex(16)
                    previous = str(dep["desired_activation_id"]) \
                        if dep is not None else ""
                    detail = dict(detail or {})
                    detail.update({
                        "transition_id": transition_id,
                        "deployment_generation": generation,
                        "previous_desired": previous})
                    event = self._append_event_txn(
                        activation_id=candidate_id,
                        event_type=event_type, from_state=from_state,
                        to_state=to_state, at=at, detail=detail,
                        signer=signer)
                    self._db.execute(
                        "INSERT INTO serving_pointer(id, activation_id, "
                        "artifact_root_digest, backend_id, "
                        "committed_at, event_sequence) "
                        "VALUES(1,?,?,?,?,?) ON CONFLICT(id) DO UPDATE "
                        "SET activation_id=excluded.activation_id, "
                        "artifact_root_digest=excluded."
                        "artifact_root_digest, backend_id=excluded."
                        "backend_id, committed_at=excluded.committed_at,"
                        " event_sequence=excluded.event_sequence",
                        (candidate_id, artifact_root_digest, backend_id,
                         int(at), int(event["sequence"])))
                    self._db.execute(
                        "INSERT INTO deployment(id, "
                        "deployment_generation, desired_activation_id, "
                        "previous_activation_id, transition_id, "
                        "transition_phase, policy_epoch, "
                        "last_committed_event_digest, updated_at) "
                        "VALUES(1,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO "
                        "UPDATE SET deployment_generation=excluded."
                        "deployment_generation, desired_activation_id="
                        "excluded.desired_activation_id, "
                        "previous_activation_id=excluded."
                        "previous_activation_id, transition_id=excluded."
                        "transition_id, transition_phase=excluded."
                        "transition_phase, policy_epoch=excluded."
                        "policy_epoch, last_committed_event_digest="
                        "excluded.last_committed_event_digest, "
                        "updated_at=excluded.updated_at",
                        (generation, candidate_id, previous,
                         transition_id, "COMMITTED",
                         int(policy_epoch), event["event_digest"],
                         int(at)))
                    self._db.execute("COMMIT")
                    return {"generation": generation,
                            "transition_id": transition_id,
                            "previous_desired": previous,
                            "event": event}
                except Exception:
                    try:
                        self._db.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise
        except GenerationConflict:
            raise
        except sqlite3.Error as exc:
            raise AuthorityStoreError(
                f"activation-intent transaction failed: {exc}") from exc

    def _deployment_txn(self) -> dict | None:
        row = self._db.execute(
            "SELECT deployment_generation, desired_activation_id, "
            "previous_activation_id, transition_id, transition_phase, "
            "policy_epoch, last_committed_event_digest, updated_at "
            "FROM deployment WHERE id = 1").fetchone()
        if row is None:
            return None
        return {"deployment_generation": int(row[0]),
                "desired_activation_id": row[1],
                "previous_activation_id": row[2],
                "transition_id": row[3],
                "transition_phase": row[4],
                "policy_epoch": int(row[5]),
                "last_committed_event_digest": row[6],
                "updated_at": int(row[7])}

    def record_routing_observation(self, *, activation_id: str,
                                   generation: int, transition_id: str,
                                   at: int,
                                   event_type: str = "routing_observed",
                                   from_state: str = "COMMITTED",
                                   to_state: str = "ACTIVE",
                                   detail: dict | None = None,
                                   signer: Ed25519Signer | None = None
                                   ) -> dict:
        """Stage two: the supervisor's OBSERVED report that the
        authorized generation was actually published to routing. This
        is post-traffic evidence — it exists only because routing
        happened, and it is never described as pre-traffic
        authorization. Refuses if the durable intent has moved on."""
        try:
            with self._lock:
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    dep = self._deployment_txn()
                    if dep is None or \
                            dep["desired_activation_id"] != activation_id \
                            or dep["deployment_generation"] != \
                            int(generation) or \
                            dep["transition_id"] != transition_id:
                        self._db.execute("ROLLBACK")
                        raise GenerationConflict(
                            "the durable deployment intent no longer "
                            f"names {activation_id!r} at generation "
                            f"{generation} — the observation belongs to "
                            "a superseded transition")
                    detail = dict(detail or {})
                    detail.update({
                        "transition_id": transition_id,
                        "deployment_generation": int(generation)})
                    event = self._append_event_txn(
                        activation_id=activation_id,
                        event_type=event_type, from_state=from_state,
                        to_state=to_state, at=at, detail=detail,
                        signer=signer)
                    self._db.execute(
                        "UPDATE deployment SET transition_phase = "
                        "'ROUTED', last_committed_event_digest = ?, "
                        "updated_at = ? WHERE id = 1",
                        (event["event_digest"], int(at)))
                    self._db.execute("COMMIT")
                    return event
                except Exception:
                    try:
                        self._db.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise
        except GenerationConflict:
            raise
        except sqlite3.Error as exc:
            raise AuthorityStoreError(
                f"routing-observation transaction failed: {exc}") \
                from exc

    def reconcile_deployment(self, *, expected_generation: int,
                             desired_id: str, phase: str, at: int,
                             event_activation_id: str,
                             event_type: str, event_from_state: str,
                             event_to_state: str, because: str = "",
                             artifact_root_digest: str = "",
                             backend_id: str = "",
                             detail: dict | None = None,
                             signer: Ed25519Signer | None = None) -> dict:
        """Generation-checked reconciliation after a failed or
        withdrawn transition: move the durable intent to ``desired_id``
        (a live committed predecessor, or ``""`` for durably
        unavailable) at a NEW generation, in one transaction with the
        reconcile event. The in-memory router and durable state
        converge — a predecessor is never restored only in memory."""
        try:
            with self._lock:
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    dep = self._deployment_txn()
                    current = int(dep["deployment_generation"]) \
                        if dep is not None else 0
                    if current != int(expected_generation):
                        self._db.execute("ROLLBACK")
                        raise GenerationConflict(
                            f"deployment generation is {current}, not "
                            f"{expected_generation} — reconciliation "
                            "refused against a newer transition")
                    generation = current + 1
                    transition_id = secrets.token_hex(16)
                    previous = str(dep["desired_activation_id"]) \
                        if dep is not None else ""
                    detail = dict(detail or {})
                    detail.update({
                        "transition_id": transition_id,
                        "deployment_generation": generation,
                        "previous_desired": previous,
                        "because": because})
                    event = self._append_event_txn(
                        activation_id=event_activation_id,
                        event_type=event_type,
                        from_state=event_from_state,
                        to_state=event_to_state, at=at, detail=detail,
                        signer=signer)
                    if desired_id:
                        self._db.execute(
                            "INSERT INTO serving_pointer(id, "
                            "activation_id, artifact_root_digest, "
                            "backend_id, committed_at, event_sequence) "
                            "VALUES(1,?,?,?,?,?) ON CONFLICT(id) DO "
                            "UPDATE SET activation_id=excluded."
                            "activation_id, artifact_root_digest="
                            "excluded.artifact_root_digest, backend_id="
                            "excluded.backend_id, committed_at=excluded."
                            "committed_at, event_sequence=excluded."
                            "event_sequence",
                            (desired_id, artifact_root_digest,
                             backend_id, int(at),
                             int(event["sequence"])))
                    else:
                        self._db.execute(
                            "UPDATE serving_pointer SET activation_id = "
                            "'', artifact_root_digest = '', backend_id "
                            "= '', committed_at = ? WHERE id = 1",
                            (int(at),))
                    self._db.execute(
                        "INSERT INTO deployment(id, "
                        "deployment_generation, desired_activation_id, "
                        "previous_activation_id, transition_id, "
                        "transition_phase, policy_epoch, "
                        "last_committed_event_digest, updated_at) "
                        "VALUES(1,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO "
                        "UPDATE SET deployment_generation=excluded."
                        "deployment_generation, desired_activation_id="
                        "excluded.desired_activation_id, "
                        "previous_activation_id=excluded."
                        "previous_activation_id, transition_id=excluded."
                        "transition_id, transition_phase=excluded."
                        "transition_phase, policy_epoch=excluded."
                        "policy_epoch, last_committed_event_digest="
                        "excluded.last_committed_event_digest, "
                        "updated_at=excluded.updated_at",
                        (generation, desired_id, previous,
                         transition_id, phase,
                         int(dep["policy_epoch"]) if dep else 0,
                         event["event_digest"], int(at)))
                    self._db.execute("COMMIT")
                    return {"generation": generation,
                            "transition_id": transition_id,
                            "event": event}
                except Exception:
                    try:
                        self._db.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise
        except GenerationConflict:
            raise
        except sqlite3.Error as exc:
            raise AuthorityStoreError(
                f"deployment reconciliation failed: {exc}") from exc

    # --- mandatory administrative audit (v16.4.4 WP-D) ----------------
    def append_admin_audit(self, *, event_id: str | None = None,
                           principal_id: str, operation: str,
                           target_activation: str, policy_digest: str,
                           decision: str, before_state: dict | str,
                           after_state: dict | str, at: int,
                           detail: dict | None = None,
                           signer: Ed25519Signer) -> dict:
        """Durably record one security-sensitive administrative
        decision in the hash-chained admin log. A failure here means
        there is NO durable evidence — callers must refuse ordinary
        administrative changes on AuthorityStoreError (emergency
        traffic shutdown is the explicit policy exception)."""
        event_id = event_id or f"audit-{secrets.token_hex(12)}"
        before = (json.dumps(before_state, sort_keys=True,
                             separators=(",", ":"))
                  if not isinstance(before_state, str) else before_state)
        after = (json.dumps(after_state, sort_keys=True,
                            separators=(",", ":"))
                 if not isinstance(after_state, str) else after_state)
        try:
            with self._lock:
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    row = self._db.execute(
                        "SELECT COALESCE((SELECT event_digest FROM "
                        "admin_audit ORDER BY sequence DESC LIMIT 1), "
                        "'')").fetchone()
                    prev = str(row[0])
                    pdigest = digest({
                        "event_id": event_id,
                        "principal_id": principal_id,
                        "operation": operation,
                        "target_activation": target_activation,
                        "policy_digest": policy_digest,
                        "decision": decision,
                        "before_state": before, "after_state": after,
                        "at": int(at), "detail": dict(detail or {})})
                    body = {"event_id": event_id,
                            "principal_id": principal_id,
                            "operation": operation,
                            "target_activation": target_activation,
                            "policy_digest": policy_digest,
                            "decision": decision,
                            "before_state": before,
                            "after_state": after, "at": int(at),
                            "payload_digest": pdigest,
                            "previous_event_digest": prev}
                    env = signer.sign(body)
                    event_digest = digest(
                        dict(body, signer_key_id=env.key_id,
                             signature_b64=env.signature_b64))
                    self._db.execute(
                        "INSERT INTO admin_audit(event_id, "
                        "principal_id, operation, target_activation, "
                        "policy_digest, decision, before_state, "
                        "after_state, detail_json, at, "
                        "previous_event_digest, event_digest, "
                        "signer_key_id, signature_b64) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (event_id, principal_id, operation,
                         target_activation, policy_digest, decision,
                         before, after,
                         json.dumps(dict(detail or {}), sort_keys=True,
                                    separators=(",", ":")),
                         int(at), prev, event_digest, env.key_id,
                         env.signature_b64))
                    self._db.execute("COMMIT")
                    return {"event_id": event_id,
                            "event_digest": event_digest,
                            "previous_event_digest": prev,
                            "signer_key_id": env.key_id}
                except Exception:
                    try:
                        self._db.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise
        except sqlite3.Error as exc:
            raise AuthorityStoreError(
                f"administrative audit append failed: {exc}") from exc

    def admin_events(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT sequence, event_id, principal_id, operation, "
            "target_activation, policy_digest, decision, before_state, "
            "after_state, detail_json, at, previous_event_digest, "
            "event_digest, signer_key_id, signature_b64 FROM "
            "admin_audit ORDER BY sequence").fetchall()
        return [{"sequence": r[0], "event_id": r[1],
                 "principal_id": r[2], "operation": r[3],
                 "target_activation": r[4], "policy_digest": r[5],
                 "decision": r[6], "before_state": r[7],
                 "after_state": r[8], "detail": json.loads(r[9] or "{}"),
                 "at": r[10], "previous_event_digest": r[11],
                 "event_digest": r[12], "signer_key_id": r[13],
                 "signature_b64": r[14]} for r in rows]

    def verify_admin_chain(self) -> list[dict]:
        """Verify the admin-audit hash chain — tampering or truncation
        fails closed with StoreCorrupt."""
        events = self.admin_events()
        prev = ""
        for e in events:
            if e["previous_event_digest"] != prev:
                raise StoreCorrupt(
                    f"admin audit chain broken at sequence "
                    f"{e['sequence']} — the durable decision log was "
                    "altered or truncated")
            body = {"event_id": e["event_id"],
                    "principal_id": e["principal_id"],
                    "operation": e["operation"],
                    "target_activation": e["target_activation"],
                    "policy_digest": e["policy_digest"],
                    "decision": e["decision"],
                    "before_state": e["before_state"],
                    "after_state": e["after_state"], "at": e["at"],
                    "payload_digest": digest({
                        "event_id": e["event_id"],
                        "principal_id": e["principal_id"],
                        "operation": e["operation"],
                        "target_activation": e["target_activation"],
                        "policy_digest": e["policy_digest"],
                        "decision": e["decision"],
                        "before_state": e["before_state"],
                        "after_state": e["after_state"], "at": e["at"],
                        "detail": e["detail"]}),
                    "previous_event_digest": e["previous_event_digest"]}
            expect = digest(dict(body, signer_key_id=e["signer_key_id"],
                                 signature_b64=e["signature_b64"]))
            if e["event_digest"] != expect:
                raise StoreCorrupt(
                    f"admin audit digest mismatch at sequence "
                    f"{e['sequence']} — record was modified")
            prev = e["event_digest"]
        return events

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
