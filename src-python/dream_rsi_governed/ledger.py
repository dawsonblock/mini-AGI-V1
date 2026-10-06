from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import canonical_json_bytes, sha256_digest


class AppendOnlyLedger:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._connect() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("CREATE TABLE IF NOT EXISTS ledger (seq INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, payload BLOB NOT NULL, prev_digest TEXT, digest TEXT UNIQUE NOT NULL)")
            c.execute("CREATE TRIGGER IF NOT EXISTS ledger_no_update BEFORE UPDATE ON ledger BEGIN SELECT RAISE(ABORT, 'append-only ledger'); END")
            c.execute("CREATE TRIGGER IF NOT EXISTS ledger_no_delete BEFORE DELETE ON ledger BEGIN SELECT RAISE(ABORT, 'append-only ledger'); END")

    def _connect(self):
        return sqlite3.connect(self.path)

    def append(self, kind: str, payload: Any) -> str:
        data = canonical_json_bytes(payload)
        with self._connect() as c:
            row = c.execute("SELECT seq,digest FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
            seq = (row[0] + 1) if row else 1
            prev = row[1] if row else None
            ts = datetime.now(timezone.utc).isoformat()
            digest = sha256_digest({"seq": seq, "ts": ts, "kind": kind, "payload": data.decode(), "prev_digest": prev})
            c.execute("INSERT INTO ledger(seq,ts,kind,payload,prev_digest,digest) VALUES(?,?,?,?,?,?)", (seq, ts, kind, data, prev, digest))
            return digest

    def verify(self) -> bool:
        with self._connect() as c:
            rows = c.execute("SELECT seq,ts,kind,payload,prev_digest,digest FROM ledger ORDER BY seq").fetchall()
        expected_prev = None
        for seq, ts, kind, payload, prev, digest in rows:
            if prev != expected_prev:
                return False
            calc = sha256_digest({"seq": seq, "ts": ts, "kind": kind, "payload": payload.decode(), "prev_digest": prev})
            if calc != digest:
                return False
            expected_prev = digest
        return True
