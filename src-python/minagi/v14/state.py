from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import time

from .models import CandidateStateEvent


LEGAL_TRANSITIONS = {
    None: {"REGISTERED"},
    "REGISTERED": {"BUILT", "REJECTED"},
    "BUILT": {"EVALUATED", "REJECTED"},
    "EVALUATED": {"QUALIFIED", "REJECTED"},
    "QUALIFIED": {"AUTHORIZED", "REJECTED"},
    "AUTHORIZED": {"ACTIVE", "REJECTED"},
    "ACTIVE": {"ROLLED_BACK", "RETIRED"},
    "ROLLED_BACK": {"RETIRED"},
    "REJECTED": set(),
    "RETIRED": set(),
}


class CandidateStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS candidate_state(
          candidate_digest TEXT PRIMARY KEY,
          state TEXT NOT NULL,
          last_event_digest TEXT NOT NULL,
          authority_generation INTEGER NOT NULL,
          policy_generation INTEGER NOT NULL,
          updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS candidate_events(
          seq INTEGER PRIMARY KEY AUTOINCREMENT,
          event_digest TEXT NOT NULL UNIQUE,
          candidate_digest TEXT NOT NULL,
          previous_state TEXT,
          new_state TEXT NOT NULL,
          event_json TEXT NOT NULL,
          created_at REAL NOT NULL
        );
        """)

    def current(self, candidate_digest: str) -> str | None:
        row = self.conn.execute("SELECT state FROM candidate_state WHERE candidate_digest=?", (candidate_digest,)).fetchone()
        return None if row is None else str(row["state"])

    def transition(self, *, candidate_digest: str, new_state: str, actor: str,
                   authority_generation: int, policy_generation: int,
                   authorization_digest: str = "") -> CandidateStateEvent:
        previous = self.current(candidate_digest)
        allowed = LEGAL_TRANSITIONS.get(previous, set())
        if new_state not in allowed:
            raise PermissionError(f"illegal candidate transition {previous!r} -> {new_state!r}")
        if new_state in {"AUTHORIZED", "ACTIVE"} and not authorization_digest.startswith("sha256:"):
            raise PermissionError("authorization digest required for privileged transition")
        ev = CandidateStateEvent(
            candidate_digest=candidate_digest,
            previous_state=previous,
            new_state=new_state,
            actor=actor,
            authority_generation=int(authority_generation),
            policy_generation=int(policy_generation),
            authorization_digest=authorization_digest,
        )
        payload = json.dumps(asdict(ev), sort_keys=True, separators=(",", ":"))
        with self.conn:
            self.conn.execute(
                "INSERT INTO candidate_events(event_digest,candidate_digest,previous_state,new_state,event_json,created_at) VALUES(?,?,?,?,?,?)",
                (ev.digest, candidate_digest, previous, new_state, payload, ev.created_at),
            )
            self.conn.execute(
                "INSERT INTO candidate_state(candidate_digest,state,last_event_digest,authority_generation,policy_generation,updated_at) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(candidate_digest) DO UPDATE SET state=excluded.state,last_event_digest=excluded.last_event_digest,authority_generation=excluded.authority_generation,policy_generation=excluded.policy_generation,updated_at=excluded.updated_at",
                (candidate_digest, new_state, ev.digest, int(authority_generation), int(policy_generation), time.time()),
            )
        return ev
