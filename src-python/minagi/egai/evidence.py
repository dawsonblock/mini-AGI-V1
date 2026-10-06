from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import uuid

from .canonical import atomic_write_json, canonical_bytes, sha256_json
from .models import EvidenceClass, EvidenceOrigin, EvidenceRecord


GENESIS = "0" * 64
EVENT_SCHEMA = "mini-agi-egai-evidence-event-v2"
_INGEST_CAPABILITY = object()


class EvidenceLedger:
    """Append-only evidence ledger.

    The public API is intentionally read-only. Mutations require the private
    capability held by :class:`TrustedEvidenceIngestor`; learning components
    are not handed that capability.
    """

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.records = self.root / "records"
        self.payloads = self.root / "payloads"
        self.events_path = self.root / "events.jsonl"
        self.records.mkdir(parents=True, exist_ok=True)
        self.payloads.mkdir(parents=True, exist_ok=True)
        self.verify()

    def _events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        out: list[dict[str, Any]] = []
        prev = GENESIS
        for lineno, line in enumerate(self.events_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("schema") != EVENT_SCHEMA or event.get("prev") != prev:
                raise RuntimeError(f"evidence event chain/schema failure at line {lineno}")
            body = dict(event)
            got = str(body.pop("hash", ""))
            want = sha256_json(body).removeprefix("sha256:")
            if got != want:
                raise RuntimeError(f"evidence event hash mismatch at line {lineno}")
            out.append(event)
            prev = got
        return out

    @staticmethod
    def _record_doc(record: EvidenceRecord) -> dict[str, Any]:
        doc = asdict(record)
        doc["origin_class"] = record.origin_class.value
        doc["evidence_class"] = record.evidence_class.value
        doc["digest"] = record.digest
        return doc

    @staticmethod
    def _record_from_doc(doc: Mapping[str, Any]) -> EvidenceRecord:
        body = dict(doc)
        got = str(body.pop("digest", ""))
        body["origin_class"] = EvidenceOrigin(str(body["origin_class"]))
        body["evidence_class"] = EvidenceClass(str(body["evidence_class"]))
        record = EvidenceRecord(**body)
        if got != record.digest:
            raise RuntimeError("evidence record digest mismatch")
        return record

    def _append_trusted(self, record: EvidenceRecord, payload: Any, *, capability: object, actor: str) -> str:
        if capability is not _INGEST_CAPABILITY:
            raise PermissionError("evidence mutation requires trusted-ingestor capability")
        if sha256_json(payload) != record.payload_digest:
            raise ValueError("payload digest does not match evidence record")
        record_stem = record.digest.removeprefix("sha256:")
        payload_stem = record.payload_digest.removeprefix("sha256:")
        record_path = self.records / f"{record_stem}.json"
        payload_path = self.payloads / f"{payload_stem}.json"
        if record_path.exists():
            existing = self._record_from_doc(json.loads(record_path.read_text(encoding="utf-8")))
            if existing != record:
                raise RuntimeError("evidence digest collision")
            return record.digest

        if not payload_path.exists():
            atomic_write_json(payload_path, payload)
        elif sha256_json(json.loads(payload_path.read_text(encoding="utf-8"))) != record.payload_digest:
            raise RuntimeError("payload content-address collision")
        atomic_write_json(record_path, self._record_doc(record))

        events = self._events()
        event = {
            "schema": EVENT_SCHEMA,
            "seq": len(events) + 1,
            "ts": time.time(),
            "record_digest": record.digest,
            "payload_digest": record.payload_digest,
            "origin_class": record.origin_class.value,
            "evidence_class": record.evidence_class.value,
            "promotion_eligible": record.promotion_eligible,
            "actor": str(actor),
            "prev": events[-1]["hash"] if events else GENESIS,
        }
        event["hash"] = sha256_json(event).removeprefix("sha256:")
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.verify()
        return record.digest

    def load(self, digest: str) -> EvidenceRecord:
        stem = str(digest).removeprefix("sha256:")
        return self._record_from_doc(json.loads((self.records / f"{stem}.json").read_text(encoding="utf-8")))

    def load_payload(self, record_or_digest: EvidenceRecord | str) -> Any:
        record = record_or_digest if isinstance(record_or_digest, EvidenceRecord) else self.load(str(record_or_digest))
        stem = record.payload_digest.removeprefix("sha256:")
        value = json.loads((self.payloads / f"{stem}.json").read_text(encoding="utf-8"))
        if sha256_json(value) != record.payload_digest:
            raise RuntimeError("evidence payload digest mismatch")
        return value

    def list_digests(self) -> list[str]:
        return [str(e["record_digest"]) for e in self._events()]

    def verify(self) -> dict[str, Any]:
        events = self._events()
        referenced_records: set[str] = set()
        referenced_payloads: set[str] = set()
        for event in events:
            digest = str(event["record_digest"])
            record = self.load(digest)
            payload = self.load_payload(record)
            if sha256_json(payload) != record.payload_digest:
                raise RuntimeError("evidence payload verification failed")
            if bool(event.get("promotion_eligible")) != record.promotion_eligible:
                raise RuntimeError("evidence promotion eligibility mismatch")
            referenced_records.add(digest.removeprefix("sha256:"))
            referenced_payloads.add(record.payload_digest.removeprefix("sha256:"))
        extras = {p.stem for p in self.records.glob("*.json")} - referenced_records
        if extras:
            raise RuntimeError(f"unlogged evidence records: {sorted(extras)}")
        payload_extras = {p.stem for p in self.payloads.glob("*.json")} - referenced_payloads
        if payload_extras:
            raise RuntimeError(f"orphaned evidence payloads: {sorted(payload_extras)}")
        return {
            "ok": True,
            "records": len(events),
            "promotion_eligible": sum(bool(x.get("promotion_eligible")) for x in events),
            "head": events[-1]["hash"] if events else GENESIS,
        }


class TrustedEvidenceIngestor:
    """Narrow append capability for raw/derived evidence.

    This component may record model inference or simulation *as such*; it may
    not relabel simulation as observation. The type boundary is enforced by
    :class:`EvidenceRecord`.
    """

    def __init__(self, ledger: EvidenceLedger, *, ingestor_id: str):
        if not str(ingestor_id).strip():
            raise ValueError("ingestor_id is required")
        self.ledger = ledger
        self.ingestor_id = str(ingestor_id)

    def ingest(
        self,
        payload: Any,
        *,
        origin_class: EvidenceOrigin,
        production_identity_digest: str,
        provenance: Mapping[str, Any],
        observed_at: float | None = None,
        valid_from: float | None = None,
        valid_until: float | None = None,
        confidence: float = 1.0,
        parent_digests: Sequence[str] = (),
        source_locator: str = "",
        summary: str = "",
    ) -> EvidenceRecord:
        evidence_class = (
            EvidenceClass.SIMULATED if origin_class is EvidenceOrigin.SIMULATION
            else EvidenceClass.DERIVED if origin_class in {EvidenceOrigin.MODEL_INFERENCE, EvidenceOrigin.GROUNDED_REPLAY}
            else EvidenceClass.OBSERVED
        )
        record = EvidenceRecord(
            evidence_id="E-" + uuid.uuid4().hex,
            origin_class=origin_class,
            evidence_class=evidence_class,
            producer=self.ingestor_id,
            payload_digest=sha256_json(payload),
            provenance_digest=sha256_json(dict(provenance)),
            production_identity_digest=str(production_identity_digest),
            observed_at=float(time.time() if observed_at is None else observed_at),
            valid_from=None if valid_from is None else float(valid_from),
            valid_until=None if valid_until is None else float(valid_until),
            confidence=float(confidence),
            parent_digests=tuple(str(x) for x in parent_digests),
            source_locator=str(source_locator),
            summary=str(summary),
        )
        self.ledger._append_trusted(record, payload, capability=_INGEST_CAPABILITY, actor=self.ingestor_id)
        return record
