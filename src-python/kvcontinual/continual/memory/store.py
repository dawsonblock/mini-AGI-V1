from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid


@dataclass
class MemoryRecord:
    text: str
    memory_type: str
    source_type: str
    source_id: str
    confidence: float = 0.5
    utility: float = 0.0
    novelty: float = 0.0
    importance: float = 0.0
    recurrence: float = 0.0
    redundancy: float = 0.0
    future_utility: float = 0.0
    valid_from: str | None = None
    valid_until: str | None = None
    supersedes: str | None = None
    contradicted_by: list[str] = field(default_factory=list)
    verified: bool = False
    evidence_blocks: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MemoryStore:
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        if path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS memory (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            confidence REAL NOT NULL,
            utility REAL NOT NULL,
            novelty REAL NOT NULL,
            importance REAL NOT NULL,
            recurrence REAL NOT NULL,
            redundancy REAL NOT NULL,
            future_utility REAL NOT NULL,
            valid_from TEXT,
            valid_until TEXT,
            supersedes TEXT,
            contradicted_by TEXT NOT NULL,
            verified INTEGER NOT NULL,
            evidence_blocks TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(supersedes) REFERENCES memory(id)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(memory_type);
        CREATE INDEX IF NOT EXISTS idx_memory_valid ON memory(valid_from, valid_until);
        """)

    @staticmethod
    def _values(r: MemoryRecord) -> tuple[object, ...]:
        return (
            r.id, r.text, r.memory_type, r.source_type, r.source_id,
            r.confidence, r.utility, r.novelty, r.importance, r.recurrence,
            r.redundancy, r.future_utility, r.valid_from, r.valid_until,
            r.supersedes, json.dumps(r.contradicted_by), int(r.verified),
            json.dumps(r.evidence_blocks), r.created_at,
        )

    def _insert(self, r: MemoryRecord) -> None:
        self.conn.execute(
            """INSERT INTO memory VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            self._values(r),
        )

    def put(self, r: MemoryRecord) -> str:
        with self.conn:
            self._insert(r)
        return r.id

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self.conn.execute("SELECT * FROM memory WHERE id=?", (memory_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["contradicted_by"] = json.loads(d["contradicted_by"])
        d["evidence_blocks"] = json.loads(d["evidence_blocks"])
        d["verified"] = bool(d["verified"])
        return MemoryRecord(**d)

    def list_current(self, memory_type: str | None = None, at: str | None = None) -> list[MemoryRecord]:
        at = at or datetime.now(timezone.utc).isoformat()
        q = """SELECT id FROM memory
               WHERE (valid_from IS NULL OR valid_from <= ?)
                 AND (valid_until IS NULL OR valid_until > ?)"""
        args: list[object] = [at, at]
        if memory_type:
            q += " AND memory_type=?"
            args.append(memory_type)
        ids = [r[0] for r in self.conn.execute(q, args).fetchall()]
        return [r for i in ids if (r := self.get(i)) is not None]

    def supersede(self, old_id: str, new_record: MemoryRecord, at: str) -> str:
        # One transaction: the old fact is never expired unless the replacement
        # is successfully installed too.
        with self.conn:
            row = self.conn.execute("SELECT id FROM memory WHERE id=?", (old_id,)).fetchone()
            if row is None:
                raise KeyError(old_id)
            new_record.supersedes = old_id
            if new_record.valid_from is None:
                new_record.valid_from = at
            self.conn.execute("UPDATE memory SET valid_until=? WHERE id=?", (at, old_id))
            self._insert(new_record)
        return new_record.id
