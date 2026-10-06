from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Iterable

from minagi.v14.verification import EpisodeVerificationReceipt, EpisodeVerificationValidator

from kvcontinual.continual.experience import Episode


@dataclass
class ExperienceRow:
    id: str
    prompt: str
    response: str
    timestamp: str
    reward: float | None
    verified: bool
    importance: float
    learning_status: str
    retrieved_memories: list[str]
    evidence: list[str]
    errors: list[str]


class ExperienceStore:
    """Authoritative SQLite experience ledger for continual learning.

    JSONL episode logging remains supported for append-only diagnostics, while
    this store provides queryable state for verification and dataset export.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
          id TEXT PRIMARY KEY,
          prompt TEXT NOT NULL,
          response TEXT NOT NULL,
          timestamp TEXT NOT NULL,
          reward REAL,
          user_feedback TEXT,
          verified INTEGER NOT NULL DEFAULT 0,
          importance REAL NOT NULL DEFAULT 0.5,
          learning_status TEXT NOT NULL DEFAULT 'NEW',
          retrieved_memories TEXT NOT NULL,
          source_blocks TEXT NOT NULL,
          tools_used TEXT NOT NULL,
          evidence TEXT NOT NULL,
          errors TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(learning_status, verified);
        CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(timestamp);

        CREATE TABLE IF NOT EXISTS verification_receipts (
          receipt_digest TEXT PRIMARY KEY,
          episode_id TEXT NOT NULL UNIQUE,
          verifier_key_id TEXT NOT NULL,
          receipt_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          FOREIGN KEY(episode_id) REFERENCES episodes(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS learning_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL,
          dataset_digest TEXT NOT NULL,
          example_count INTEGER NOT NULL,
          adapter_candidate_id TEXT,
          status TEXT NOT NULL,
          metadata TEXT NOT NULL
        );
        """)

    def append(self, e: Episode, *, verified: bool = False, importance: float = 0.5) -> str:
        if verified:
            raise PermissionError('v14 forbids caller-supplied verified=True; install a signed verification receipt')
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO episodes
                (id,prompt,response,timestamp,reward,user_feedback,verified,importance,learning_status,
                 retrieved_memories,source_blocks,tools_used,evidence,errors)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    e.id, e.prompt, e.response, e.timestamp, e.reward, e.user_feedback,
                    int(verified), float(importance), "NEW",
                    json.dumps(e.retrieved_memories), json.dumps(e.source_blocks),
                    json.dumps(e.tools_used), json.dumps(e.evidence), json.dumps(e.errors),
                ),
            )
        return e.id

    def mark_verified(self, episode_id: str, *, verified: bool = True, importance: float | None = None) -> None:
        raise PermissionError(
            "v14 removed boolean verification authority; use install_verification_receipt() with a trusted verifier"
        )

    def install_verification_receipt(
        self, receipt: EpisodeVerificationReceipt, validator: EpisodeVerificationValidator
    ) -> str:
        row = self.conn.execute("SELECT * FROM episodes WHERE id=?", (receipt.episode_id,)).fetchone()
        if row is None:
            raise KeyError(receipt.episode_id)
        validator.validate(
            receipt,
            episode_id=receipt.episode_id,
            prompt=row["prompt"],
            attempted_output=row["response"],
        )
        payload = asdict(receipt)
        with self.conn:
            self.conn.execute(
                "INSERT INTO verification_receipts(receipt_digest,episode_id,verifier_key_id,receipt_json,created_at) VALUES(?,?,?,?,?)",
                (
                    receipt.digest, receipt.episode_id, receipt.verifier_key_id,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self.conn.execute(
                "UPDATE episodes SET verified=1, importance=? WHERE id=?",
                (float(receipt.importance), receipt.episode_id),
            )
        return receipt.digest

    def verification_receipt_digest(self, episode_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT receipt_digest FROM verification_receipts WHERE episode_id=?", (episode_id,)
        ).fetchone()
        return None if row is None else str(row[0])

    def list_training_ready(self, limit: int = 1000) -> list[ExperienceRow]:
        rows = self.conn.execute(
            """SELECT e.* FROM episodes e
               WHERE e.verified=1 AND e.learning_status='NEW' AND e.response<>''
                 AND EXISTS (SELECT 1 FROM verification_receipts v WHERE v.episode_id=e.id)
               ORDER BY e.importance DESC, e.timestamp ASC LIMIT ?""",
            (int(limit),),
        ).fetchall()
        return [self._row(r) for r in rows]

    def mark_exported(self, episode_ids: Iterable[str]) -> None:
        ids = list(episode_ids)
        if not ids:
            return
        with self.conn:
            self.conn.executemany(
                "UPDATE episodes SET learning_status='EXPORTED' WHERE id=? AND verified=1",
                [(x,) for x in ids],
            )

    def create_learning_run(self, dataset_digest: str, count: int, metadata: dict) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO learning_runs(created_at,dataset_digest,example_count,status,metadata) VALUES(?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), dataset_digest, int(count), "DATASET_READY", json.dumps(metadata, sort_keys=True)),
            )
            return int(cur.lastrowid)

    def bind_candidate(self, run_id: int, candidate_id: str) -> None:
        with self.conn:
            cur = self.conn.execute(
                "UPDATE learning_runs SET adapter_candidate_id=?, status='CANDIDATE_REGISTERED' WHERE id=?",
                (candidate_id, int(run_id)),
            )
            if cur.rowcount != 1:
                raise KeyError(run_id)

    @staticmethod
    def _row(r: sqlite3.Row) -> ExperienceRow:
        return ExperienceRow(
            id=r["id"], prompt=r["prompt"], response=r["response"], timestamp=r["timestamp"],
            reward=r["reward"], verified=bool(r["verified"]), importance=float(r["importance"]),
            learning_status=r["learning_status"], retrieved_memories=json.loads(r["retrieved_memories"]),
            evidence=json.loads(r["evidence"]), errors=json.loads(r["errors"]),
        )
