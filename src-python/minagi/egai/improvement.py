from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from .canonical import sha256_json
from .models import LearningLevel, PromotionVerdict


GENESIS = "0" * 64
EVENT_SCHEMA = "mini-agi-egai-improvement-event-v2"


@dataclass(frozen=True)
class ImprovementRecord:
    improvement_id: str
    proposal_digest: str
    candidate_digest: str
    qualification_digest: str
    promotion_decision_digest: str
    verdict: PromotionVerdict
    level: LearningLevel
    origin_evidence: tuple[str, ...]
    affected_components: tuple[str, ...]
    metrics: Mapping[str, float]
    production_identity_before: str
    production_identity_after: str = ""
    rollback_target: str = ""
    created_at: float = 0.0
    schema: str = "mini-agi-egai-improvement-record-v2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_evidence", tuple(self.origin_evidence))
        object.__setattr__(self, "affected_components", tuple(self.affected_components))
        object.__setattr__(self, "metrics", {str(k): float(v) for k, v in dict(self.metrics).items()})
        if not self.created_at:
            object.__setattr__(self, "created_at", float(time.time()))

    @property
    def digest(self) -> str:
        body = asdict(self)
        body["verdict"] = self.verdict.value
        body["level"] = int(self.level)
        return sha256_json(body)


@dataclass(frozen=True)
class FutureConsequence:
    improvement_digest: str
    observation_horizon: str
    forward_transfer: float
    retention_delta: float
    calibration_delta: float
    compute_delta: float
    capacity_delta: float
    rollback_triggered: bool = False
    notes: str = ""
    schema: str = "mini-agi-egai-future-consequence-v2"

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


class ImprovementLedger:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.verify()

    def _events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        prev = GENESIS
        for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("schema") != EVENT_SCHEMA or event.get("prev") != prev:
                raise RuntimeError(f"improvement ledger chain/schema failure at line {lineno}")
            body = dict(event)
            got = str(body.pop("hash", ""))
            want = sha256_json(body).removeprefix("sha256:")
            if got != want:
                raise RuntimeError(f"improvement ledger hash mismatch at line {lineno}")
            out.append(event)
            prev = got
        return out

    def _append(self, kind: str, payload: Mapping[str, Any]) -> str:
        events = self._events()
        event = {
            "schema": EVENT_SCHEMA,
            "seq": len(events) + 1,
            "ts": time.time(),
            "kind": str(kind),
            "payload": dict(payload),
            "prev": events[-1]["hash"] if events else GENESIS,
        }
        event["hash"] = sha256_json(event).removeprefix("sha256:")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.verify()
        return str(event["hash"])

    def append_improvement(self, record: ImprovementRecord) -> str:
        body = asdict(record)
        body["verdict"] = record.verdict.value
        body["level"] = int(record.level)
        body["digest"] = record.digest
        return self._append("decision", body)

    def append_consequence(self, consequence: FutureConsequence) -> str:
        body = asdict(consequence)
        body["digest"] = consequence.digest
        known = {str(e["payload"].get("digest")) for e in self._events() if e.get("kind") == "decision"}
        if consequence.improvement_digest not in known:
            raise ValueError("future consequence references unknown improvement")
        return self._append("future_consequence", body)

    def verify(self) -> dict[str, Any]:
        events = self._events()
        known: set[str] = set()
        for event in events:
            payload = dict(event.get("payload") or {})
            digest = str(payload.pop("digest", ""))
            if not digest or sha256_json(payload) != digest:
                raise RuntimeError("improvement payload digest mismatch")
            if event.get("kind") == "decision":
                known.add(digest)
            elif event.get("kind") == "future_consequence":
                ref = str(payload.get("improvement_digest", ""))
                if ref not in known:
                    raise RuntimeError("future consequence precedes or misses improvement decision")
            else:
                raise RuntimeError("unknown improvement ledger event kind")
        return {
            "ok": True,
            "events": len(events),
            "decisions": sum(x.get("kind") == "decision" for x in events),
            "future_consequences": sum(x.get("kind") == "future_consequence" for x in events),
            "head": events[-1]["hash"] if events else GENESIS,
        }

    def meta_learning_rows(self) -> list[dict[str, Any]]:
        events = self._events()
        decisions = {str(e["payload"]["digest"]): dict(e["payload"]) for e in events if e["kind"] == "decision"}
        rows: list[dict[str, Any]] = []
        for event in events:
            if event["kind"] != "future_consequence":
                continue
            c = dict(event["payload"])
            parent = decisions[str(c["improvement_digest"])]
            rows.append({"decision": parent, "future_consequence": c})
        return rows
