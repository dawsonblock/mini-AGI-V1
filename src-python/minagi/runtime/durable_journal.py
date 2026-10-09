"""v16.4.2 durable activation journal (UPGRADE_PLAN §3.5).

The journal is the crash-consistency substrate: every lifecycle
transition is appended and fsynced *before* the effect it describes,
and the active-version pointer is a separate atomically-replaced file.
Recovery replays the journal and reconciles it against the pointer —
never the other way around.

Records:

    {"activation_id": ..., "from": <state|null>, "to": <state>,
     "at": <unix seconds>, "detail": {...}, "record_digest": ...}

Completion records (to=ACTIVE with detail.kind="completion") are
signed by the runtime authority — evidence of what actually happened
is part of the durable record, not a best-effort afterthought.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer

JOURNAL_SCHEMA = "mini-agi-v16.4.2-activation-journal-v1"


class JournalError(RuntimeError):
    """Durable evidence could not be written — fail closed."""


@dataclass(frozen=True)
class JournalRecord:
    activation_id: str
    to_state: str
    from_state: str
    at: int
    detail: dict
    record_digest: str
    signer_key_id: str = ""
    signature_b64: str = ""

    def signed(self) -> bool:
        return bool(self.signer_key_id and self.signature_b64)


class DurableJournal:
    """Append-only, fsynced journal plus an atomically replaced
    active-version pointer."""

    def __init__(self, directory):
        self.root = Path(directory)
        self.journal_path = self.root / "activation_journal.jsonl"
        self.pointer_path = self.root / "active.json"

    # --- records ----------------------------------------------------
    def append(self, *, activation_id: str, from_state: str,
               to_state: str, at: int, detail: dict | None = None,
               signer: Ed25519Signer | None = None) -> JournalRecord:
        """Append one transition durably. Raises JournalError if the
        record cannot be persisted — callers treat that as 'the
        transition did not happen'."""
        body = {"schema": JOURNAL_SCHEMA,
                "activation_id": str(activation_id),
                "from_state": str(from_state) if from_state else "",
                "to_state": str(to_state), "at": int(at),
                "detail": dict(detail or {})}
        entry = dict(body)
        if signer is not None:
            env = signer.sign(body)
            entry["signer_key_id"] = env.key_id
            entry["signature_b64"] = env.signature_b64
        entry["record_digest"] = digest(body)
        line = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            raise JournalError(
                f"activation journal append failed: {exc}") from exc
        return JournalRecord(
            activation_id=str(activation_id),
            from_state=str(from_state) if from_state else "",
            to_state=str(to_state), at=int(at),
            detail=dict(detail or {}),
            record_digest=str(entry["record_digest"]),
            signer_key_id=str(entry.get("signer_key_id", "")),
            signature_b64=str(entry.get("signature_b64", "")))

    def records(self) -> list[JournalRecord]:
        if not self.journal_path.is_file():
            return []
        out = []
        for line in self.journal_path.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            out.append(JournalRecord(
                activation_id=str(e["activation_id"]),
                from_state=str(e.get("from_state", "")),
                to_state=str(e["to_state"]), at=int(e["at"]),
                detail=dict(e.get("detail") or {}),
                record_digest=str(e.get("record_digest", "")),
                signer_key_id=str(e.get("signer_key_id", "")),
                signature_b64=str(e.get("signature_b64", ""))))
        return out

    def history(self, activation_id: str) -> list[JournalRecord]:
        return [r for r in self.records()
                if r.activation_id == activation_id]

    # --- active-version pointer ------------------------------------
    def write_pointer(self, doc: dict) -> None:
        """Atomically replace the active-version pointer
        (tmp + fsync + os.replace): a crash mid-write leaves either the
        old or the new pointer, never a torn one."""
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.root), prefix=".active-",
                                   suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(doc, sort_keys=True))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.pointer_path)
            dir_fd = os.open(str(self.root), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except BaseException as exc:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise JournalError(
                f"active-version pointer write failed: {exc}") from exc

    def read_pointer(self) -> dict | None:
        if not self.pointer_path.is_file():
            return None
        try:
            doc = json.loads(self.pointer_path.read_text())
        except (OSError, ValueError) as exc:
            raise JournalError(
                f"active-version pointer unreadable: {exc}") from exc
        if not isinstance(doc, dict) or "activation_id" not in doc:
            raise JournalError("active-version pointer malformed")
        return doc
