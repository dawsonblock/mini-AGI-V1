from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json, re, sqlite3, time
from .util import canonical_json, sha256_bytes

_WORD = re.compile(r"[A-Za-z0-9_]{2,}")

def _terms(s: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD.finditer(s or "")}

@dataclass(frozen=True)
class CanonicalMemoryRecord:
    record_id: str
    revision: int
    type: str
    content: str
    source: str
    source_hash: str
    recorded_at: float
    valid_from: float | None
    valid_until: float | None
    confidence: float
    salience: float
    usefulness: float
    state: str
    provenance: dict[str, Any]
    links: dict[str, list[str]]
    digest: str

class CanonicalMemoryStore:
    """Append-only bitemporal canonical memory.

    Neural cache tensors never live here. Facts are revised by appending a new
    record revision, preserving transaction history and rollback evidence.
    """
    STATES = {"candidate", "active", "superseded", "retracted", "archived"}
    TYPES = {"episodic", "semantic", "procedural", "user", "failure", "solution"}

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS memory_events(
              seq INTEGER PRIMARY KEY AUTOINCREMENT,
              record_id TEXT NOT NULL,
              revision INTEGER NOT NULL,
              type TEXT NOT NULL,
              content TEXT NOT NULL,
              source TEXT NOT NULL,
              source_hash TEXT NOT NULL,
              recorded_at REAL NOT NULL,
              valid_from REAL,
              valid_until REAL,
              confidence REAL NOT NULL,
              salience REAL NOT NULL,
              usefulness REAL NOT NULL,
              state TEXT NOT NULL,
              provenance_json TEXT NOT NULL,
              links_json TEXT NOT NULL,
              digest TEXT NOT NULL UNIQUE,
              UNIQUE(record_id,revision)
            );
            CREATE INDEX IF NOT EXISTS mem_record ON memory_events(record_id,revision);
            CREATE INDEX IF NOT EXISTS mem_type ON memory_events(type);
            CREATE INDEX IF NOT EXISTS mem_time ON memory_events(recorded_at);
            CREATE TABLE IF NOT EXISTS access_events(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              record_id TEXT NOT NULL,
              ts REAL NOT NULL,
              useful REAL NOT NULL
            );
            """)

    def _db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def _latest_revision(self, record_id: str) -> int | None:
        with self._db() as db:
            row = db.execute(
                "SELECT revision FROM memory_events WHERE record_id=? ORDER BY revision DESC LIMIT 1",
                (record_id,),
            ).fetchone()
        return None if row is None else int(row[0])

    def append(self, *, type: str, content: str, source: str,
               source_hash: str | None = None, record_id: str | None = None,
               valid_from: float | None = None, valid_until: float | None = None,
               confidence: float = 1.0, salience: float = 0.5,
               usefulness: float = 0.5, state: str = "active",
               provenance: dict[str, Any] | None = None,
               links: dict[str, Iterable[str]] | None = None) -> CanonicalMemoryRecord:
        type = type.strip(); content = content.strip(); source = source.strip()
        if type not in self.TYPES:
            raise ValueError(f"invalid memory type: {type}")
        if state not in self.STATES:
            raise ValueError(f"invalid memory state: {state}")
        if not content or not source:
            raise ValueError("content and source are required")
        now = time.time()
        source_hash = source_hash or sha256_bytes(source.encode())
        if record_id is None:
            record_id = "mem-" + sha256_bytes(
                f"{type}\0{source}\0{content}\0{now}".encode()
            )[:24]
            revision = 1
        else:
            last = self._latest_revision(record_id)
            revision = 1 if last is None else last + 1
        provenance = dict(provenance or {})
        links2 = {k: sorted(set(map(str, v))) for k, v in (links or {}).items()}
        confidence = max(0.0, min(1.0, float(confidence)))
        salience = max(0.0, min(1.0, float(salience)))
        usefulness = max(0.0, min(1.0, float(usefulness)))
        base = dict(
            record_id=record_id, revision=revision, type=type, content=content,
            source=source, source_hash=source_hash, recorded_at=now,
            valid_from=valid_from, valid_until=valid_until,
            confidence=confidence, salience=salience, usefulness=usefulness,
            state=state, provenance=provenance, links=links2,
        )
        digest = sha256_bytes(canonical_json(base).encode())
        with self._db() as db:
            db.execute("""INSERT INTO memory_events(
                record_id,revision,type,content,source,source_hash,recorded_at,
                valid_from,valid_until,confidence,salience,usefulness,state,
                provenance_json,links_json,digest)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (record_id, revision, type, content, source, source_hash, now,
                 valid_from, valid_until, confidence, salience, usefulness, state,
                 canonical_json(provenance), canonical_json(links2), digest))
        return CanonicalMemoryRecord(**base, digest=digest)

    def revise(self, record_id: str, *, content: str | None = None,
               state: str | None = None, valid_from: float | None = None,
               valid_until: float | None = None, confidence: float | None = None,
               salience: float | None = None, usefulness: float | None = None,
               source: str | None = None, provenance: dict[str, Any] | None = None,
               links: dict[str, Iterable[str]] | None = None) -> CanonicalMemoryRecord:
        cur = self.get(record_id)
        if cur is None:
            raise KeyError(record_id)
        return self.append(
            type=cur.type, content=cur.content if content is None else content,
            source=source or cur.source, source_hash=cur.source_hash,
            record_id=record_id,
            valid_from=cur.valid_from if valid_from is None else valid_from,
            valid_until=cur.valid_until if valid_until is None else valid_until,
            confidence=cur.confidence if confidence is None else confidence,
            salience=cur.salience if salience is None else salience,
            usefulness=cur.usefulness if usefulness is None else usefulness,
            state=state or cur.state,
            provenance=provenance or cur.provenance,
            links=links or cur.links,
        )

    def supersede(self, old_id: str, *, new_content: str, source: str,
                  valid_from: float | None = None, confidence: float = 1.0,
                  provenance: dict[str, Any] | None = None):
        old = self.get(old_id)
        if old is None:
            raise KeyError(old_id)
        closed = self.revise(old_id, state="superseded", valid_until=valid_from)
        new = self.append(
            type=old.type, content=new_content, source=source,
            valid_from=valid_from, confidence=confidence,
            salience=old.salience, usefulness=old.usefulness,
            provenance=provenance, links={"supersedes": [old_id]},
        )
        return closed, new

    def retract(self, record_id: str, *, source: str | None = None, reason: str = ""):
        cur = self.get(record_id)
        if cur is None:
            raise KeyError(record_id)
        provenance = dict(cur.provenance)
        provenance["retraction_reason"] = reason
        return self.revise(record_id, state="retracted", source=source or cur.source,
                           provenance=provenance)

    def _row(self, row) -> CanonicalMemoryRecord:
        keys = (
            "record_id", "revision", "type", "content", "source", "source_hash",
            "recorded_at", "valid_from", "valid_until", "confidence", "salience",
            "usefulness", "state", "provenance", "links", "digest",
        )
        d = dict(zip(keys, row))
        d["provenance"] = json.loads(d["provenance"])
        d["links"] = json.loads(d["links"])
        return CanonicalMemoryRecord(**d)

    def get(self, record_id: str, revision: int | None = None):
        sql = """SELECT record_id,revision,type,content,source,source_hash,recorded_at,
                 valid_from,valid_until,confidence,salience,usefulness,state,
                 provenance_json,links_json,digest FROM memory_events WHERE record_id=? """
        args: list[Any] = [record_id]
        if revision is None:
            sql += "ORDER BY revision DESC LIMIT 1"
        else:
            sql += "AND revision=?"
            args.append(int(revision))
        with self._db() as db:
            row = db.execute(sql, tuple(args)).fetchone()
        return None if row is None else self._row(row)

    def current(self, *, valid_at: float | None = None,
                types: Iterable[str] | None = None,
                include_states: Iterable[str] = ("active",)) -> list[CanonicalMemoryRecord]:
        at = time.time() if valid_at is None else float(valid_at)
        states = tuple(include_states)
        types2 = tuple(types or ())
        clauses = [
            "e.revision=(SELECT MAX(e2.revision) FROM memory_events e2 WHERE e2.record_id=e.record_id)",
            "(e.valid_from IS NULL OR e.valid_from<=?)",
            "(e.valid_until IS NULL OR e.valid_until>?)",
        ]
        args: list[Any] = [at, at]
        if states:
            clauses.append("e.state IN (%s)" % ",".join("?" * len(states)))
            args.extend(states)
        if types2:
            clauses.append("e.type IN (%s)" % ",".join("?" * len(types2)))
            args.extend(types2)
        sql = """SELECT e.record_id,e.revision,e.type,e.content,e.source,e.source_hash,
                 e.recorded_at,e.valid_from,e.valid_until,e.confidence,e.salience,
                 e.usefulness,e.state,e.provenance_json,e.links_json,e.digest
                 FROM memory_events e WHERE """ + " AND ".join(clauses) + " ORDER BY e.recorded_at"
        with self._db() as db:
            rows = db.execute(sql, tuple(args)).fetchall()
        return [self._row(r) for r in rows]

    def history(self, record_id: str) -> list[CanonicalMemoryRecord]:
        with self._db() as db:
            rows = db.execute("""SELECT record_id,revision,type,content,source,source_hash,
                recorded_at,valid_from,valid_until,confidence,salience,usefulness,state,
                provenance_json,links_json,digest FROM memory_events
                WHERE record_id=? ORDER BY revision""", (record_id,)).fetchall()
        return [self._row(r) for r in rows]

    def search(self, query: str, *, limit: int = 8, valid_at: float | None = None):
        q = _terms(query)
        scored = []
        for rec in self.current(valid_at=valid_at):
            t = _terms(rec.content + " " + rec.source)
            jaccard = len(q & t) / max(1, len(q | t))
            score = jaccard * (0.4 + 0.3 * rec.confidence + 0.15 * rec.salience + 0.15 * rec.usefulness)
            if score > 0:
                scored.append((score, rec))
        scored.sort(key=lambda x: (-x[0], -x[1].recorded_at))
        return scored[:max(1, int(limit))]

    def record_access(self, record_id: str, useful: float = 1.0) -> None:
        if self.get(record_id) is None:
            raise KeyError(record_id)
        with self._db() as db:
            db.execute("INSERT INTO access_events(record_id,ts,useful) VALUES(?,?,?)",
                       (record_id, time.time(), float(useful)))
