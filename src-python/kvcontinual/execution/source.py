from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
import uuid


def _sha256_json(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass
class SourceSegment:
    tokens: list[int]
    source_locator: str
    observed_from: str | None = None
    observed_until: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    semantic_refs: list[str] = field(default_factory=list)
    canonical_stream: str | None = None
    canonical_start: int | None = None
    canonical_end: int | None = None
    tokenizer_digest: str = "unknown"
    normalization_digest: str = "none"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def token_digest(self) -> str:
        # Legacy token-only digest retained for compatibility and diagnostics.
        return _sha256_json(self.tokens)

    @property
    def content_digest(self) -> str:
        """Cryptographic identity of the canonical token representation.

        Unlike the random segment id, this digest is safe to bind into derived
        execution artifacts.  Tokenizer and normalization identities are part of
        the digest so equal integer ids from incompatible tokenizers cannot alias.
        """
        return _sha256_json({
            "tokens": self.tokens,
            "tokenizer_digest": self.tokenizer_digest,
            "normalization_digest": self.normalization_digest,
        })


class SourceSegmentStore:
    """Append-only source-of-truth token store.

    Segment IDs are immutable aliases. Re-inserting an identical record is
    idempotent; attempting to change any identity-bearing content under an
    existing ID fails closed.
    """

    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS source_segment (
                id TEXT PRIMARY KEY,
                tokens TEXT NOT NULL,
                source_locator TEXT NOT NULL,
                observed_from TEXT,
                observed_until TEXT,
                evidence_ids TEXT NOT NULL,
                semantic_refs TEXT NOT NULL,
                canonical_stream TEXT,
                canonical_start INTEGER,
                canonical_end INTEGER,
                tokenizer_digest TEXT NOT NULL,
                normalization_digest TEXT NOT NULL,
                token_digest TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        # Forward-migrate RC10.16 databases created before tokenizer/content binding.
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(source_segment)")}
        for name, default in (("tokenizer_digest", "unknown"), ("normalization_digest", "none"), ("content_digest", "")):
            if name not in cols:
                literal = "'" + default.replace("'", "''") + "'"
                self.conn.execute(f"ALTER TABLE source_segment ADD COLUMN {name} TEXT NOT NULL DEFAULT {literal}")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_source_stream ON source_segment(canonical_stream, canonical_start)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_source_content_digest ON source_segment(content_digest)")
        self.conn.commit()

    @staticmethod
    def _serialized(s: SourceSegment) -> tuple:
        return (
            s.id, json.dumps(s.tokens, separators=(",", ":")), s.source_locator,
            s.observed_from, s.observed_until,
            json.dumps(s.evidence_ids, separators=(",", ":")),
            json.dumps(s.semantic_refs, separators=(",", ":")),
            s.canonical_stream, s.canonical_start, s.canonical_end,
            s.tokenizer_digest, s.normalization_digest,
            s.token_digest, s.content_digest, s.created_at,
        )

    def put(self, s: SourceSegment) -> str:
        existing = self.conn.execute("SELECT * FROM source_segment WHERE id=?", (s.id,)).fetchone()
        if existing is not None:
            # Idempotent retry is allowed only when the immutable content identity
            # and canonical placement match the already-persisted record.
            checks = {
                "token_digest": s.token_digest,
                "content_digest": s.content_digest,
                "source_locator": s.source_locator,
                "canonical_stream": s.canonical_stream,
                "canonical_start": s.canonical_start,
                "canonical_end": s.canonical_end,
                "tokenizer_digest": s.tokenizer_digest,
                "normalization_digest": s.normalization_digest,
            }
            for key, expected in checks.items():
                if existing[key] != expected:
                    raise ValueError(f"immutable source segment {s.id!r} cannot be rewritten: {key} changed")
            return s.id
        self.conn.execute(
            """INSERT INTO source_segment
               (id,tokens,source_locator,observed_from,observed_until,evidence_ids,semantic_refs,
                canonical_stream,canonical_start,canonical_end,tokenizer_digest,normalization_digest,
                token_digest,content_digest,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            self._serialized(s),
        )
        self.conn.commit()
        return s.id

    def get(self, segment_id: str) -> SourceSegment | None:
        row = self.conn.execute("SELECT * FROM source_segment WHERE id=?", (segment_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d.pop("token_digest", None)
        d.pop("content_digest", None)
        d["tokens"] = json.loads(d["tokens"])
        d["evidence_ids"] = json.loads(d["evidence_ids"])
        d["semantic_refs"] = json.loads(d["semantic_refs"])
        return SourceSegment(**d)

    def content_digest(self, segment_id: str) -> str:
        row = self.conn.execute("SELECT content_digest FROM source_segment WHERE id=?", (segment_id,)).fetchone()
        if row is None:
            raise KeyError(segment_id)
        digest = str(row[0])
        if not digest:
            # Lazy compatibility repair for a migrated legacy row.
            s = self.get(segment_id)
            if s is None:
                raise KeyError(segment_id)
            digest = s.content_digest
            self.conn.execute("UPDATE source_segment SET content_digest=? WHERE id=?", (digest, segment_id))
            self.conn.commit()
        return digest

    def classify_topology(self, segment_ids: list[str]):
        from kvcontinual.execution.types import AssemblyTopology
        segments = [self.get(x) for x in segment_ids]
        if not segments or any(s is None for s in segments):
            return AssemblyTopology.UNKNOWN
        assert all(s is not None for s in segments)
        stream = segments[0].canonical_stream
        if stream is None or any(s.canonical_stream != stream for s in segments):
            return AssemblyTopology.ARBITRARY
        positions = [(s.canonical_start, s.canonical_end) for s in segments]
        if any(a is None or b is None for a, b in positions):
            return AssemblyTopology.ARBITRARY
        contiguous = all(positions[i][1] == positions[i + 1][0] for i in range(len(positions) - 1))
        monotone = all(positions[i][0] <= positions[i + 1][0] for i in range(len(positions) - 1))
        if contiguous and monotone:
            # A contiguous interior slice is canonical, but it is not a prefix.
            return AssemblyTopology.EXACT_PREFIX if positions[0][0] == 0 else AssemblyTopology.CANONICAL
        return AssemblyTopology.ARBITRARY
