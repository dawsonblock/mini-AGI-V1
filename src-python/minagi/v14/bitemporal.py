from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sqlite3
import time
import uuid

from egai.common.canonical import digest, validate_digest


@dataclass(frozen=True)
class BeliefRevision:
    belief_id: str
    revision: int
    subject: str
    predicate: str
    object_json: str
    valid_from: float
    valid_to: float | None
    evidence_digests: tuple[str, ...]
    confidence: float
    supersedes_digest: str = ""
    created_at: float = 0.0
    schema: str = "mini-agi-v14.1-belief-revision-v1"

    def __post_init__(self) -> None:
        if not self.belief_id or not self.subject or not self.predicate:
            raise ValueError("belief identity and relation are required")
        if self.revision < 1:
            raise ValueError("revision must be >=1")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if self.valid_to is not None and float(self.valid_to) <= float(self.valid_from):
            raise ValueError("valid_to must be after valid_from")
        for d in self.evidence_digests:
            validate_digest(d)
        if self.supersedes_digest:
            validate_digest(self.supersedes_digest)
        if not self.created_at:
            object.__setattr__(self, "created_at", time.time())

    @property
    def digest(self) -> str:
        return digest(self)


class BiTemporalBeliefStore:
    """Immutable belief revisions plus an atomic head pointer.

    Transaction time is represented by revision creation and the head history;
    old revisions are never overwritten. Valid time is explicit in each object.
    """

    def __init__(self, path: str | Path, authorization_gate=None):
        self.authorization_gate = authorization_gate
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS belief_revisions(
          belief_digest TEXT PRIMARY KEY,
          belief_id TEXT NOT NULL,
          revision INTEGER NOT NULL,
          body_json TEXT NOT NULL,
          created_at REAL NOT NULL,
          UNIQUE(belief_id,revision)
        );
        CREATE TABLE IF NOT EXISTS belief_heads(
          belief_id TEXT PRIMARY KEY,
          belief_digest TEXT NOT NULL,
          updated_at REAL NOT NULL,
          FOREIGN KEY(belief_digest) REFERENCES belief_revisions(belief_digest) ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS belief_head_history(
          seq INTEGER PRIMARY KEY AUTOINCREMENT,
          belief_id TEXT NOT NULL,
          belief_digest TEXT NOT NULL,
          transaction_time REAL NOT NULL,
          authorization_digest TEXT NOT NULL,
          FOREIGN KEY(belief_digest) REFERENCES belief_revisions(belief_digest) ON DELETE RESTRICT
        );
        """)

    def promote(self, *, subject: str, predicate: str, object_value, valid_from: float,
                evidence_digests: tuple[str, ...], confidence: float,
                authorization_digest: str, belief_id: str | None = None,
                valid_to: float | None = None) -> BeliefRevision:
        validate_digest(authorization_digest)
        if self.authorization_gate is not None:
            self.authorization_gate.require(authorization_digest, mutation_scope="belief.promote")
        belief_id = belief_id or "BEL-" + uuid.uuid4().hex
        self.db.execute("BEGIN IMMEDIATE")
        try:
            head = self.db.execute("SELECT belief_digest FROM belief_heads WHERE belief_id=?", (belief_id,)).fetchone()
            previous = "" if head is None else str(head["belief_digest"])
            row = self.db.execute("SELECT MAX(revision) AS r FROM belief_revisions WHERE belief_id=?", (belief_id,)).fetchone()
            revision = int(row["r"] or 0) + 1
            obj_json = json.dumps(object_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            belief = BeliefRevision(belief_id, revision, subject, predicate, obj_json, float(valid_from), valid_to,
                                    tuple(evidence_digests), float(confidence), previous)
            body = json.dumps(asdict(belief), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            self.db.execute("INSERT INTO belief_revisions(belief_digest,belief_id,revision,body_json,created_at) VALUES(?,?,?,?,?)",
                            (belief.digest, belief_id, revision, body, belief.created_at))
            now = time.time()
            self.db.execute("INSERT INTO belief_heads(belief_id,belief_digest,updated_at) VALUES(?,?,?) "
                            "ON CONFLICT(belief_id) DO UPDATE SET belief_digest=excluded.belief_digest,updated_at=excluded.updated_at",
                            (belief_id, belief.digest, now))
            self.db.execute("INSERT INTO belief_head_history(belief_id,belief_digest,transaction_time,authorization_digest) VALUES(?,?,?,?)",
                            (belief_id, belief.digest, now, authorization_digest))
            self.db.execute("COMMIT")
            return belief
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def current(self, belief_id: str) -> BeliefRevision | None:
        row = self.db.execute("SELECT r.body_json FROM belief_heads h JOIN belief_revisions r ON r.belief_digest=h.belief_digest WHERE h.belief_id=?", (belief_id,)).fetchone()
        if row is None:
            return None
        body = json.loads(row["body_json"])
        body["evidence_digests"] = tuple(body["evidence_digests"])
        return BeliefRevision(**body)

    def history(self, belief_id: str) -> tuple[BeliefRevision, ...]:
        out=[]
        for row in self.db.execute("SELECT body_json FROM belief_revisions WHERE belief_id=? ORDER BY revision", (belief_id,)):
            body=json.loads(row["body_json"]); body["evidence_digests"]=tuple(body["evidence_digests"]); out.append(BeliefRevision(**body))
        return tuple(out)
