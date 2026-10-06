from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

from kvcontinual.execution.durability.file_lock import ProcessFileLock
from minagi.egai.canonical import sha256_json
from .models import require_digest


@dataclass(frozen=True)
class AuthorityState:
    generation: int
    policy_generation: int
    policy_digest: str
    trusted_key_ids: tuple[str, ...]
    revoked_key_ids: tuple[str, ...] = ()
    previous_authority_state_digest: str | None = None
    created_ns: int = 0
    schema: str = "egai-rc14-authority-state-v1"

    def __post_init__(self) -> None:
        if self.generation < 1 or self.policy_generation < 1:
            raise ValueError("authority/policy generations must be positive")
        require_digest(self.policy_digest, field_name="policy_digest")
        if self.previous_authority_state_digest:
            require_digest(self.previous_authority_state_digest, field_name="previous_authority_state_digest")
        trusted = tuple(sorted(set(str(x) for x in self.trusted_key_ids)))
        revoked = tuple(sorted(set(str(x) for x in self.revoked_key_ids)))
        if not trusted:
            raise ValueError("authority state requires at least one trusted key")
        if set(trusted) & set(revoked):
            raise ValueError("a key cannot be both trusted and revoked")
        object.__setattr__(self, "trusted_key_ids", trusted)
        object.__setattr__(self, "revoked_key_ids", revoked)
        if not self.created_ns:
            object.__setattr__(self, "created_ns", time.time_ns())

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


class AuthorityStateRegistry:
    """Signed monotonic root-of-trust state for RC14 privileged transitions."""

    def __init__(self, root: str | Path, *, verifier: Any):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "authority_state.sqlite3"
        self.lock_path = self.root / "authority_state.lock"
        self.verifier = verifier
        with self._connect() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS authority_states(
              state_digest TEXT PRIMARY KEY,
              generation INTEGER NOT NULL UNIQUE,
              body_json TEXT NOT NULL,
              receipt_json TEXT NOT NULL,
              created_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS authority_control(
              singleton INTEGER PRIMARY KEY CHECK(singleton=1),
              current_digest TEXT,
              generation INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO authority_control(singleton,current_digest,generation) VALUES(1,NULL,0);
            """)

    def _connect(self):
        c = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=FULL")
        c.execute("PRAGMA busy_timeout=30000")
        return c

    @staticmethod
    def body(state: AuthorityState) -> dict[str, Any]:
        doc = asdict(state)
        doc["trusted_key_ids"] = list(state.trusted_key_ids)
        doc["revoked_key_ids"] = list(state.revoked_key_ids)
        return {"schema": "egai-rc14-authority-activation-v1", "authority_state": doc, "authority_state_digest": state.digest}

    def activate(self, state: AuthorityState, *, receipt: dict[str, Any]) -> str:
        body = self.body(state)
        if not self.verifier.verify(receipt, expected_body=body):
            raise PermissionError("invalid authority-state root signature")
        with ProcessFileLock(self.lock_path, timeout=30.0):
            c = self._connect(); c.execute("BEGIN IMMEDIATE")
            try:
                ctl = c.execute("SELECT * FROM authority_control WHERE singleton=1").fetchone()
                current = ctl["current_digest"]
                generation = int(ctl["generation"])
                if state.generation != generation + 1:
                    raise PermissionError("authority generation must advance exactly by one")
                if state.previous_authority_state_digest != current:
                    raise PermissionError("authority state predecessor mismatch")
                try:
                    c.execute(
                        "INSERT INTO authority_states VALUES(?,?,?,?,?)",
                        (state.digest, state.generation, json.dumps(asdict(state), sort_keys=True, separators=(",", ":")), json.dumps(receipt, sort_keys=True, separators=(",", ":")), time.time_ns()),
                    )
                except sqlite3.IntegrityError as exc:
                    raise RuntimeError("authority-state replay/collision") from exc
                c.execute("UPDATE authority_control SET current_digest=?,generation=? WHERE singleton=1", (state.digest, state.generation))
                c.execute("COMMIT")
                return state.digest
            except Exception:
                if c.in_transaction: c.execute("ROLLBACK")
                raise
            finally:
                c.close()

    def get(self, digest: str) -> AuthorityState:
        require_digest(digest, field_name="authority_state_digest")
        with self._connect() as c:
            row = c.execute("SELECT body_json FROM authority_states WHERE state_digest=?", (digest,)).fetchone()
        if row is None:
            raise KeyError(digest)
        state = AuthorityState(**json.loads(row[0]))
        if state.digest != digest:
            raise RuntimeError("stored authority-state digest mismatch")
        return state

    def current(self) -> AuthorityState | None:
        with self._connect() as c:
            row = c.execute("SELECT s.body_json FROM authority_control ctl JOIN authority_states s ON s.state_digest=ctl.current_digest WHERE ctl.singleton=1").fetchone()
        if row is None:
            return None
        return AuthorityState(**json.loads(row[0]))

    def assert_known(self, *, authority_state_digest: str, authority_generation: int, policy_generation: int, receipt: dict[str, Any] | None = None) -> AuthorityState:
        state = self.get(authority_state_digest)
        if state.generation != int(authority_generation) or state.policy_generation != int(policy_generation):
            raise PermissionError("transition authority/policy generation does not match authority state")
        if receipt is not None:
            key_id = str(receipt.get("key_id") or "")
            if key_id not in state.trusted_key_ids or key_id in state.revoked_key_ids:
                raise PermissionError("transition signing key is not trusted by bound authority state")
        return state

    def assert_current(self, *, authority_state_digest: str, authority_generation: int, policy_generation: int, receipt: dict[str, Any] | None = None) -> AuthorityState:
        require_digest(authority_state_digest, field_name="authority_state_digest")
        state = self.current()
        if state is None or state.digest != authority_state_digest:
            raise PermissionError("transition is not bound to current authority state")
        if state.generation != int(authority_generation) or state.policy_generation != int(policy_generation):
            raise PermissionError("transition authority/policy generation does not match authority state")
        if receipt is not None:
            key_id = str(receipt.get("key_id") or "")
            if key_id not in state.trusted_key_ids or key_id in state.revoked_key_ids:
                raise PermissionError("transition signing key is not trusted by current authority state")
        return state

    def verify(self) -> dict[str, Any]:
        with self._connect() as c:
            if str(c.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
                raise RuntimeError("authority-state sqlite integrity failure")
            rows = c.execute("SELECT * FROM authority_states ORDER BY generation").fetchall()
            ctl = dict(c.execute("SELECT * FROM authority_control WHERE singleton=1").fetchone())
        prev = None
        generation = 0
        for row in rows:
            state = AuthorityState(**json.loads(row["body_json"]))
            if state.digest != row["state_digest"]:
                raise RuntimeError("authority-state digest mismatch")
            if state.generation != generation + 1 or state.previous_authority_state_digest != prev:
                raise RuntimeError("authority-state chain discontinuity")
            receipt = json.loads(row["receipt_json"])
            if not self.verifier.verify(receipt, expected_body=self.body(state)):
                raise RuntimeError("authority-state signature invalid")
            prev, generation = state.digest, state.generation
        if ctl["current_digest"] != prev or int(ctl["generation"]) != generation:
            raise RuntimeError("authority-state control head mismatch")
        return {"states": len(rows), "current_digest": prev, "generation": generation}
