from __future__ import annotations

import json
import time
import uuid
from typing import Iterable

from egai.common.canonical import digest, validate_digest
from .storage_v144 import GovernanceDBV144


class GovernanceDBV145(GovernanceDBV144):
    """Authority-converged persistence additions for alpha5.

    V144 remains the atomic mutation/activation engine.  V145 adds an explicit
    epistemic dependency graph, terminal revocation/quarantine controls, and a
    serving StateEpoch registry with request leases.  These tables deliberately
    share the same SQLite authority database so activation and all later serving
    decisions have one durable source of truth.
    """

    EPOCH_STATES = {
        "PREPARED", "LOCALLY_COMMITTED", "EXTERNALLY_WITNESSED",
        "ATTESTED", "SERVABLE", "RETIRED",
    }

    def __init__(self, path):
        super().__init__(path)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS dependency_edges_v145(
              parent_digest TEXT NOT NULL,
              child_digest TEXT NOT NULL,
              relation TEXT NOT NULL,
              created_at REAL NOT NULL,
              PRIMARY KEY(parent_digest,child_digest,relation)
            );
            CREATE INDEX IF NOT EXISTS idx_dependency_v145_child
              ON dependency_edges_v145(child_digest);
            CREATE TABLE IF NOT EXISTS object_controls_v145(
              object_digest TEXT PRIMARY KEY,
              state TEXT NOT NULL CHECK(state IN ('ACTIVE','QUARANTINED','REVOKED')),
              reason TEXT NOT NULL,
              authority_digest TEXT,
              updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS object_control_history_v145(
              event_digest TEXT PRIMARY KEY,
              object_digest TEXT NOT NULL,
              state TEXT NOT NULL,
              reason TEXT NOT NULL,
              authority_digest TEXT,
              created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS state_epochs_v145(
              epoch_digest TEXT PRIMARY KEY,
              epoch_number INTEGER NOT NULL UNIQUE,
              predecessor_digest TEXT,
              runtime_manifest_digest TEXT NOT NULL UNIQUE,
              qualification_digest TEXT NOT NULL,
              authorization_digest TEXT NOT NULL UNIQUE,
              activation_digest TEXT,
              state TEXT NOT NULL,
              witness_digest TEXT,
              attestation_digest TEXT,
              body_json TEXT NOT NULL,
              created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS serving_epoch_head_v145(
              singleton INTEGER PRIMARY KEY CHECK(singleton=1),
              epoch_digest TEXT NOT NULL,
              updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS epoch_transitions_v145(
              seq INTEGER PRIMARY KEY AUTOINCREMENT,
              transition_digest TEXT NOT NULL UNIQUE,
              epoch_digest TEXT NOT NULL,
              from_state TEXT NOT NULL,
              to_state TEXT NOT NULL,
              artifact_digest TEXT,
              signer_key_id TEXT,
              body_json TEXT NOT NULL,
              created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS epoch_leases_v145(
              lease_id TEXT PRIMARY KEY,
              epoch_digest TEXT NOT NULL,
              request_id TEXT NOT NULL,
              acquired_at REAL NOT NULL,
              expires_at REAL NOT NULL,
              released_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_epoch_leases_v145_live
              ON epoch_leases_v145(epoch_digest,released_at,expires_at);
            """
        )

    # ------------------------- epistemic dependency graph ------------------
    def dependency_descendants_v145(self, root_digest: str) -> tuple[str, ...]:
        root_digest = validate_digest(root_digest)
        seen: set[str] = set()
        frontier = [root_digest]
        while frontier:
            parent = frontier.pop()
            rows = self.conn.execute(
                "SELECT child_digest FROM dependency_edges_v145 WHERE parent_digest=?",
                (parent,),
            ).fetchall()
            for row in rows:
                child = str(row["child_digest"])
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        return tuple(sorted(seen))

    def add_dependencies_v145(self, *, child_digest: str, parent_digests: Iterable[str],
                              relation: str = "depends_on", actor: str = "dependency-authority"):
        child_digest = validate_digest(child_digest)
        parents = tuple(sorted({validate_digest(x) for x in parent_digests}))
        if child_digest in parents:
            raise ValueError("self dependency forbidden")
        descendants = set(self.dependency_descendants_v145(child_digest))
        if any(parent in descendants for parent in parents):
            raise RuntimeError("dependency cycle forbidden")
        now = time.time()
        with self.transaction() as db:
            for parent in parents:
                db.execute(
                    "INSERT OR IGNORE INTO dependency_edges_v145(parent_digest,child_digest,relation,created_at) VALUES(?,?,?,?)",
                    (parent, child_digest, str(relation), now),
                )
            if parents:
                self._audit(db, event_type="dependency.register.v145", object_digest=child_digest,
                            actor=actor, payload={"parents": parents, "relation": str(relation)})

    def control_state_v145(self, object_digest: str) -> str:
        object_digest = validate_digest(object_digest)
        row = self.conn.execute(
            "SELECT state FROM object_controls_v145 WHERE object_digest=?", (object_digest,)
        ).fetchone()
        return "ACTIVE" if row is None else str(row["state"])

    def _set_control_v145(self, *, object_digest: str, state: str, reason: str,
                          authority_digest: str | None = None, actor: str = "epistemic-control") -> str:
        object_digest = validate_digest(object_digest)
        state = str(state).upper()
        if state not in {"ACTIVE", "QUARANTINED", "REVOKED"}:
            raise ValueError("invalid object-control state")
        if authority_digest is not None:
            authority_digest = validate_digest(authority_digest)
        prior = self.control_state_v145(object_digest)
        if prior == "REVOKED" and state != "REVOKED":
            raise RuntimeError("revocation is terminal; publish a new revision instead")
        now = time.time()
        event = digest({"object_digest": object_digest, "state": state, "reason": str(reason),
                        "authority_digest": authority_digest, "time": now})
        with self.transaction() as db:
            db.execute(
                "INSERT INTO object_controls_v145(object_digest,state,reason,authority_digest,updated_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(object_digest) DO UPDATE SET state=excluded.state,reason=excluded.reason,authority_digest=excluded.authority_digest,updated_at=excluded.updated_at",
                (object_digest, state, str(reason), authority_digest, now),
            )
            db.execute(
                "INSERT INTO object_control_history_v145(event_digest,object_digest,state,reason,authority_digest,created_at) VALUES(?,?,?,?,?,?)",
                (event, object_digest, state, str(reason), authority_digest, now),
            )
            self._audit(db, event_type=f"object-control.{state.lower()}.v145", object_digest=object_digest,
                        actor=actor, payload={"reason": str(reason), "authority_digest": authority_digest})
        return event

    def invalidate_cascade_v145(self, *, root_digest: str, reason: str,
                                authority_digest: str | None = None, actor: str = "epistemic-control") -> tuple[str, ...]:
        """Permanently revoke root evidence and quarantine all derived objects."""
        root_digest = validate_digest(root_digest)
        descendants = self.dependency_descendants_v145(root_digest)
        self._set_control_v145(object_digest=root_digest, state="REVOKED", reason=reason,
                               authority_digest=authority_digest, actor=actor)
        for child in descendants:
            if self.control_state_v145(child) != "REVOKED":
                self._set_control_v145(object_digest=child, state="QUARANTINED",
                                       reason=f"dependency invalidated by {root_digest}: {reason}",
                                       authority_digest=authority_digest, actor=actor)
        return descendants

    def require_usable_v145(self, object_digest: str) -> None:
        state = self.control_state_v145(object_digest)
        if state != "ACTIVE":
            raise PermissionError(f"object {object_digest} is {state.lower()}")

    def dependency_snapshot_digest_v145(self) -> str:
        rows = self.conn.execute(
            "SELECT parent_digest,child_digest,relation FROM dependency_edges_v145 ORDER BY parent_digest,child_digest,relation"
        ).fetchall()
        return digest({"schema": "mini-agi-v14.1-alpha5-dependency-snapshot-v1",
                       "edges": [tuple(r) for r in rows]})

    # ------------------------- StateEpoch primitives -----------------------
    def next_epoch_number_v145(self) -> int:
        row = self.conn.execute("SELECT MAX(epoch_number) AS n FROM state_epochs_v145").fetchone()
        return int(row["n"] or 0) + 1

    def state_epoch_row_v145(self, epoch_digest: str):
        return self.conn.execute("SELECT * FROM state_epochs_v145 WHERE epoch_digest=?",
                                 (validate_digest(epoch_digest),)).fetchone()

    def serving_epoch_v145(self):
        row = self.conn.execute("SELECT epoch_digest FROM serving_epoch_head_v145 WHERE singleton=1").fetchone()
        return None if row is None else str(row["epoch_digest"])

    def live_epoch_leases_v145(self, epoch_digest: str) -> int:
        now = time.time()
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM epoch_leases_v145 WHERE epoch_digest=? AND released_at IS NULL AND expires_at>=?",
            (validate_digest(epoch_digest), now),
        ).fetchone()
        return int(row["n"])
