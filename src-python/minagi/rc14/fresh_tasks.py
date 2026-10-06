from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import secrets
import sqlite3
import stat
import time
import uuid

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from kvcontinual.execution.durability.file_lock import ProcessFileLock
from minagi.egai.canonical import sha256_json


@dataclass(frozen=True)
class HiddenTaskCommitment:
    task_id: str
    generation: int
    commitment_digest: str
    created_ns: int
    @property
    def digest(self) -> str: return sha256_json(asdict(self))


@dataclass(frozen=True)
class FreshTaskLease:
    lease_id: str
    task_id: str
    generation: int
    consumer_id: str
    commitment_digest: str
    issued_ns: int
    expires_ns: int
    @property
    def digest(self) -> str: return sha256_json(asdict(self))


class DurableFreshTaskAuthority:
    """Encrypted-at-rest one-shot fresh-task authority.

    The SQLite database never stores plaintext task payloads or salts. The AES key is
    kept in a separate mode-0600 file so deployments can place it behind an OS/service
    boundary. True hidden-evaluation secrecy still requires running this authority under
    a principal the research plane cannot read.
    """

    def __init__(self, root: str | Path, *, key_path: str | Path | None = None):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "fresh_tasks.sqlite3"; self.lock_path = self.root / "fresh_tasks.lock"
        self.key_path = Path(key_path) if key_path is not None else self.root / ".fresh_tasks_aes256.key"
        self._key = self._load_or_create_key()
        self._init_db()

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            if self.key_path.is_symlink():
                raise RuntimeError("fresh-task key must not be a symlink")
            st = self.key_path.stat()
            if not stat.S_ISREG(st.st_mode):
                raise RuntimeError("fresh-task key must be a regular file")
            if stat.S_IMODE(st.st_mode) & 0o077:
                raise RuntimeError("fresh-task key permissions must not grant group/other access")
            key = self.key_path.read_bytes()
            if len(key) != 32: raise RuntimeError("fresh-task key must be 32 bytes")
            return key
        key = AESGCM.generate_key(bit_length=256)
        fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try: os.write(fd, key)
        finally: os.close(fd)
        try: os.chmod(self.key_path, 0o600)
        except OSError: pass
        return key

    def _connect(self):
        c = sqlite3.connect(self.path, timeout=30.0, isolation_level=None); c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA synchronous=FULL"); c.execute("PRAGMA busy_timeout=30000"); return c

    def _init_db(self):
        with self._connect() as c:
            cols = {r[1] for r in c.execute("PRAGMA table_info(tasks)").fetchall()}
            if cols and "task_json" in cols:
                # Migrate RC14.0 plaintext rows in-place through a temporary encrypted table.
                rows = c.execute("SELECT * FROM tasks").fetchall()
                c.execute("ALTER TABLE tasks RENAME TO tasks_plaintext_legacy")
                self._create_table(c)
                aes = AESGCM(self._key)
                for row in rows:
                    payload = json.dumps({"task": json.loads(row["task_json"]), "salt": row["salt"]}, sort_keys=True, separators=(",", ":")).encode()
                    nonce = secrets.token_bytes(12); aad = self._aad(row["task_id"], int(row["generation"]), row["commitment_digest"])
                    cipher = aes.encrypt(nonce, payload, aad)
                    c.execute("INSERT INTO tasks(task_id,generation,commitment_digest,payload_ciphertext,payload_nonce,created_ns,state,lease_id,consumer_id,issued_ns,expires_ns,consumed_ns) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                              (row["task_id"], row["generation"], row["commitment_digest"], cipher, nonce, row["created_ns"], row["state"], row["lease_id"], row["consumer_id"], row["issued_ns"], row["expires_ns"], row["consumed_ns"]))
                c.execute("DROP TABLE tasks_plaintext_legacy")
            else:
                self._create_table(c)

    @staticmethod
    def _create_table(c):
        c.executescript("""
        CREATE TABLE IF NOT EXISTS tasks(
          task_id TEXT PRIMARY KEY,
          generation INTEGER NOT NULL,
          commitment_digest TEXT NOT NULL UNIQUE,
          payload_ciphertext BLOB NOT NULL,
          payload_nonce BLOB NOT NULL,
          created_ns INTEGER NOT NULL,
          state TEXT NOT NULL CHECK(state IN ('SEALED','LEASED','CONSUMED','CLOSED')),
          lease_id TEXT UNIQUE,
          consumer_id TEXT,
          issued_ns INTEGER,
          expires_ns INTEGER,
          consumed_ns INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_generation_state ON tasks(generation,state);
        """)

    @staticmethod
    def commitment(task, salt: str) -> str: return sha256_json({"task": task, "salt": str(salt)})

    @staticmethod
    def _aad(task_id: str, generation: int, commitment_digest: str) -> bytes:
        return json.dumps({"task_id": task_id, "generation": generation, "commitment_digest": commitment_digest}, sort_keys=True, separators=(",", ":")).encode()

    def _encrypt(self, *, task_id: str, generation: int, commitment_digest: str, task, salt: str) -> tuple[bytes, bytes]:
        payload = json.dumps({"task": task, "salt": salt}, sort_keys=True, separators=(",", ":")).encode()
        nonce = secrets.token_bytes(12)
        return AESGCM(self._key).encrypt(nonce, payload, self._aad(task_id, generation, commitment_digest)), nonce

    def _decrypt_row(self, row):
        payload = AESGCM(self._key).decrypt(bytes(row["payload_nonce"]), bytes(row["payload_ciphertext"]), self._aad(str(row["task_id"]), int(row["generation"]), str(row["commitment_digest"])))
        return json.loads(payload)

    def seal(self, task, *, generation: int) -> HiddenTaskCommitment:
        task_id = "FT-" + uuid.uuid4().hex; salt = secrets.token_hex(32); created = time.time_ns(); commitment = self.commitment(task, salt)
        cipher, nonce = self._encrypt(task_id=task_id, generation=int(generation), commitment_digest=commitment, task=task, salt=salt)
        with ProcessFileLock(self.lock_path, timeout=30.0):
            with self._connect() as c:
                c.execute("BEGIN IMMEDIATE")
                c.execute("INSERT INTO tasks(task_id,generation,commitment_digest,payload_ciphertext,payload_nonce,created_ns,state) VALUES(?,?,?,?,?,?,?)", (task_id, int(generation), commitment, cipher, nonce, created, "SEALED"))
                c.execute("COMMIT")
        return HiddenTaskCommitment(task_id, int(generation), commitment, created)

    def lease(self, *, task_id: str, consumer_id: str, ttl_seconds: float = 300.0) -> FreshTaskLease:
        now = time.time_ns(); expires = now + int(float(ttl_seconds) * 1_000_000_000); lease_id = "LEASE-" + uuid.uuid4().hex
        with ProcessFileLock(self.lock_path, timeout=30.0):
            c = self._connect(); c.execute("BEGIN IMMEDIATE")
            try:
                row = c.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
                if row is None: raise KeyError(task_id)
                if row["state"] != "SEALED": raise PermissionError("hidden task is not available for first use")
                cur = c.execute("UPDATE tasks SET state='LEASED',lease_id=?,consumer_id=?,issued_ns=?,expires_ns=? WHERE task_id=? AND state='SEALED'", (lease_id, str(consumer_id), now, expires, task_id))
                if cur.rowcount != 1: raise PermissionError("hidden task lease race lost")
                c.execute("COMMIT")
            except Exception:
                if c.in_transaction: c.execute("ROLLBACK")
                raise
            finally: c.close()
        return FreshTaskLease(lease_id, task_id, int(row["generation"]), str(consumer_id), str(row["commitment_digest"]), now, expires)

    def consume(self, lease: FreshTaskLease):
        now = time.time_ns()
        with ProcessFileLock(self.lock_path, timeout=30.0):
            c = self._connect(); c.execute("BEGIN IMMEDIATE")
            try:
                row = c.execute("SELECT * FROM tasks WHERE task_id=?", (lease.task_id,)).fetchone()
                if row is None: raise KeyError(lease.task_id)
                if row["state"] != "LEASED" or row["lease_id"] != lease.lease_id or row["consumer_id"] != lease.consumer_id: raise PermissionError("fresh task lease is invalid or already consumed")
                if int(row["expires_ns"]) < now: raise PermissionError("fresh task lease expired")
                doc = self._decrypt_row(row)
                cur = c.execute("UPDATE tasks SET state='CONSUMED',consumed_ns=? WHERE task_id=? AND state='LEASED'", (now, lease.task_id))
                if cur.rowcount != 1: raise PermissionError("fresh task consume race lost")
                c.execute("COMMIT"); return doc["task"]
            except Exception:
                if c.in_transaction: c.execute("ROLLBACK")
                raise
            finally: c.close()

    def reveal(self, task_id: str) -> dict:
        with self._connect() as c: row = c.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None: raise KeyError(task_id)
        if row["state"] not in {"CONSUMED", "CLOSED"}: raise PermissionError("task cannot be revealed before consumption")
        doc = self._decrypt_row(row); task, salt = doc["task"], doc["salt"]
        return {"task_id": task_id, "task": task, "salt": salt, "commitment_digest": row["commitment_digest"], "commitment_valid": self.commitment(task, salt) == row["commitment_digest"]}

    def close(self, task_id: str) -> None:
        with ProcessFileLock(self.lock_path, timeout=30.0):
            with self._connect() as c:
                c.execute("BEGIN IMMEDIATE"); cur = c.execute("UPDATE tasks SET state='CLOSED' WHERE task_id=? AND state='CONSUMED'", (task_id,))
                if cur.rowcount != 1: c.execute("ROLLBACK"); raise PermissionError("only consumed tasks can be closed")
                c.execute("COMMIT")

    def storage_boundary(self) -> dict:
        return {"database": str(self.path), "key_path": str(self.key_path), "encrypted_at_rest": True, "os_isolation_required_for_research_secrecy": True}
