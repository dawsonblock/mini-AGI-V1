from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class QualificationLedgerEntry:
    sequence: int
    created_at: str
    previous_digest: str
    payload_digest: str
    payload: dict[str, Any]
    entry_digest: str

    @staticmethod
    def compute_digest(sequence: int, created_at: str, previous_digest: str, payload_digest: str) -> str:
        return _sha256_json({
            "sequence": int(sequence),
            "created_at": created_at,
            "previous_digest": previous_digest,
            "payload_digest": payload_digest,
        })


class QualificationLedger:
    """Append-only hash-chained evidence journal for oracle qualification runs.

    The ledger is intentionally simple JSONL so it remains inspectable without a
    database. It is not a distributed-consensus log; it detects accidental or
    unauthorized modification of local qualification evidence after the fact.
    """

    GENESIS = "sha256:" + "0" * 64

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def entries(self) -> list[QualificationLedgerEntry]:
        out: list[QualificationLedgerEntry] = []
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                out.append(QualificationLedgerEntry(**raw))
            except Exception as exc:
                raise ValueError(f"invalid qualification ledger record at line {line_no}") from exc
        return out

    def validate(self) -> str:
        prev = self.GENESIS
        expected_seq = 1
        for e in self.entries():
            if e.sequence != expected_seq:
                raise ValueError("qualification ledger sequence discontinuity")
            if e.previous_digest != prev:
                raise ValueError("qualification ledger hash-chain break")
            if _sha256_json(e.payload) != e.payload_digest:
                raise ValueError("qualification ledger payload digest mismatch")
            expected = QualificationLedgerEntry.compute_digest(
                e.sequence, e.created_at, e.previous_digest, e.payload_digest
            )
            if expected != e.entry_digest:
                raise ValueError("qualification ledger entry digest mismatch")
            prev = e.entry_digest
            expected_seq += 1
        return prev

    @property
    def head_digest(self) -> str:
        return self.validate()

    def append(self, payload: dict[str, Any]) -> QualificationLedgerEntry:
        entries = self.entries()
        prev = self.validate()
        seq = len(entries) + 1
        created_at = datetime.now(timezone.utc).isoformat()
        payload_digest = _sha256_json(payload)
        digest = QualificationLedgerEntry.compute_digest(seq, created_at, prev, payload_digest)
        entry = QualificationLedgerEntry(seq, created_at, prev, payload_digest, payload, digest)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), sort_keys=True, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return entry

    def evidence_digest(self) -> str:
        """Stable digest that binds the complete validated ledger history."""
        return _sha256_json({"head": self.head_digest, "entries": len(self.entries())})
