from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Mapping

from kvcontinual.execution.durability.file_lock import ProcessFileLock
from minagi.egai.canonical import sha256_json
from .metrics import ContinualMetrics, backward_transfer, forgetting, forward_transfer, interference_matrix

GENESIS = "0" * 64


@dataclass(frozen=True)
class ContinualExperimentMeasurement:
    variant: str
    old_task_history: Mapping[str, tuple[float, ...]]
    old_task_before: Mapping[str, float]
    old_task_after: Mapping[str, float]
    fresh_history_score: float
    fresh_frozen_baseline_score: float
    falsification_survival: float
    security_regressions: int = 0
    unauthorized_writes: int = 0
    memory_growth: float = 0.0
    compute_cost: float = 0.0
    latency_delta: float = 0.0
    reproducibility: float = 0.0

    def __post_init__(self) -> None:
        keys = set(self.old_task_before)
        if keys != set(self.old_task_after) or keys != set(self.old_task_history):
            raise ValueError("old-task history/before/after sets must match")
        vals = [self.fresh_history_score, self.fresh_frozen_baseline_score, self.falsification_survival,
                self.memory_growth, self.compute_cost, self.latency_delta, self.reproducibility]
        vals += [float(v) for v in self.old_task_before.values()] + [float(v) for v in self.old_task_after.values()]
        vals += [float(x) for row in self.old_task_history.values() for x in row]
        if any(not math.isfinite(float(v)) for v in vals):
            raise ValueError("experiment metrics must be finite")
        if not 0.0 <= float(self.falsification_survival) <= 1.0 or not 0.0 <= float(self.reproducibility) <= 1.0:
            raise ValueError("probability-like metrics must be in [0,1]")
        if self.security_regressions < 0 or self.unauthorized_writes < 0:
            raise ValueError("regression/write counts must be non-negative")


@dataclass(frozen=True)
class ContinualExperimentResultRC14:
    variant: str
    metrics: ContinualMetrics
    per_task_forgetting: Mapping[str, float]
    interference: Mapping[str, float]
    eligible_for_promotion_evidence: bool
    rejection_reasons: tuple[str, ...]
    schema: str = "egai-rc14-continual-experiment-result-v1"

    @property
    def digest(self) -> str:
        body = asdict(self)
        return sha256_json(body)


class ContinualExperimentHarnessRC14:
    """Computes comparable continual-learning outcomes; never promotes candidates."""

    can_promote = False
    can_activate = False

    def __init__(self, *, max_mean_forgetting: float = 0.05, max_worst_forgetting: float = 0.10,
                 min_fresh_task_gain: float = 0.0, min_falsification_survival: float = 1.0,
                 require_zero_security_regressions: bool = True, require_zero_unauthorized_writes: bool = True):
        self.max_mean_forgetting = float(max_mean_forgetting)
        self.max_worst_forgetting = float(max_worst_forgetting)
        self.min_fresh_task_gain = float(min_fresh_task_gain)
        self.min_falsification_survival = float(min_falsification_survival)
        self.require_zero_security_regressions = bool(require_zero_security_regressions)
        self.require_zero_unauthorized_writes = bool(require_zero_unauthorized_writes)

    def evaluate(self, measurement: ContinualExperimentMeasurement) -> ContinualExperimentResultRC14:
        per_forgetting = {
            task: forgetting(measurement.old_task_history[task], measurement.old_task_after[task])
            for task in sorted(measurement.old_task_after)
        }
        mean_forgetting = sum(per_forgetting.values()) / len(per_forgetting) if per_forgetting else 0.0
        worst = max(per_forgetting.values(), default=0.0)
        bwt_rows = [backward_transfer(measurement.old_task_after[k], measurement.old_task_before[k]) for k in sorted(measurement.old_task_before)]
        bwt = sum(bwt_rows) / len(bwt_rows) if bwt_rows else 0.0
        ft = forward_transfer(measurement.fresh_history_score, measurement.fresh_frozen_baseline_score)
        metrics = ContinualMetrics(
            fresh_task_gain=ft,
            mean_forgetting=mean_forgetting,
            worst_case_forgetting=worst,
            forward_transfer=ft,
            backward_transfer=bwt,
            falsification_survival=float(measurement.falsification_survival),
            security_regressions=int(measurement.security_regressions),
            unauthorized_writes=int(measurement.unauthorized_writes),
            memory_growth=float(measurement.memory_growth),
            compute_cost=float(measurement.compute_cost),
            latency_delta=float(measurement.latency_delta),
            reproducibility=float(measurement.reproducibility),
        )
        metrics.validate()
        reasons: list[str] = []
        if metrics.fresh_task_gain < self.min_fresh_task_gain: reasons.append("fresh_task_gain")
        if metrics.mean_forgetting > self.max_mean_forgetting: reasons.append("mean_forgetting")
        if metrics.worst_case_forgetting > self.max_worst_forgetting: reasons.append("worst_case_forgetting")
        if metrics.falsification_survival < self.min_falsification_survival: reasons.append("falsification_survival")
        if self.require_zero_security_regressions and metrics.security_regressions: reasons.append("security_regressions")
        if self.require_zero_unauthorized_writes and metrics.unauthorized_writes: reasons.append("unauthorized_writes")
        return ContinualExperimentResultRC14(
            variant=measurement.variant,
            metrics=metrics,
            per_task_forgetting=per_forgetting,
            interference=interference_matrix(measurement.old_task_before, measurement.old_task_after),
            eligible_for_promotion_evidence=not reasons,
            rejection_reasons=tuple(reasons),
        )

    @staticmethod
    def compare(results: Mapping[str, ContinualExperimentResultRC14], *, baseline: str) -> dict[str, dict[str, float | bool]]:
        if baseline not in results:
            raise KeyError("baseline result is required")
        b = results[baseline].metrics
        out: dict[str, dict[str, float | bool]] = {}
        for name in sorted(results):
            m = results[name].metrics
            out[name] = {
                "delta_fresh_task_gain": m.fresh_task_gain - b.fresh_task_gain,
                "delta_mean_forgetting": m.mean_forgetting - b.mean_forgetting,
                "delta_compute_cost": m.compute_cost - b.compute_cost,
                "eligible_for_promotion_evidence": results[name].eligible_for_promotion_evidence,
            }
        return out


class ContinualExperimentLedgerRC14:
    """Hash-chained immutable ledger for experiment measurements/results."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "experiments.sqlite3"
        self.lock_path = self.root / "experiments.lock"
        self._init_db()

    def _connect(self):
        c = sqlite3.connect(self.db_path, timeout=30.0)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=FULL")
        return c

    def _init_db(self) -> None:
        with self._connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS state(singleton INTEGER PRIMARY KEY CHECK(singleton=1), seq INTEGER NOT NULL, head TEXT NOT NULL)")
            c.execute("CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, measurement_digest TEXT NOT NULL, result_digest TEXT NOT NULL, payload_json TEXT NOT NULL, previous_digest TEXT NOT NULL, event_digest TEXT NOT NULL UNIQUE, committed_ns INTEGER NOT NULL)")
            c.execute("INSERT OR IGNORE INTO state(singleton,seq,head) VALUES(1,0,?)", (GENESIS,))
            c.commit()

    @staticmethod
    def _measurement_doc(m: ContinualExperimentMeasurement) -> dict:
        d = asdict(m)
        d["old_task_history"] = {k: list(v) for k, v in sorted(m.old_task_history.items())}
        return d

    @staticmethod
    def _result_doc(r: ContinualExperimentResultRC14) -> dict:
        return asdict(r)

    def append(self, measurement: ContinualExperimentMeasurement, result: ContinualExperimentResultRC14) -> str:
        if measurement.variant != result.variant:
            raise ValueError("measurement/result variant mismatch")
        md = sha256_json(self._measurement_doc(measurement))
        rd = result.digest
        payload = {"measurement": self._measurement_doc(measurement), "result": self._result_doc(result)}
        with ProcessFileLock(self.lock_path, timeout=30.0):
            with self._connect() as c:
                c.execute("BEGIN IMMEDIATE")
                state = c.execute("SELECT seq,head FROM state WHERE singleton=1").fetchone()
                seq = int(state["seq"]) + 1
                previous = str(state["head"])
                body = {"schema": "egai-rc14-continual-experiment-event-v1", "seq": seq,
                        "measurement_digest": md, "result_digest": rd, "previous_digest": previous}
                event_digest = sha256_json(body)
                c.execute("INSERT INTO events(seq,measurement_digest,result_digest,payload_json,previous_digest,event_digest,committed_ns) VALUES(?,?,?,?,?,?,?)",
                          (seq, md, rd, json.dumps(payload, sort_keys=True, separators=(",", ":")), previous, event_digest, time.time_ns()))
                c.execute("UPDATE state SET seq=?,head=? WHERE singleton=1", (seq, event_digest))
                c.commit()
        return event_digest

    def verify(self) -> dict:
        with self._connect() as c:
            if str(c.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
                raise RuntimeError("continual experiment sqlite integrity failure")
            rows = c.execute("SELECT * FROM events ORDER BY seq").fetchall()
            state = c.execute("SELECT seq,head FROM state WHERE singleton=1").fetchone()
        previous = GENESIS
        for seq, row in enumerate(rows, 1):
            if int(row["seq"]) != seq or str(row["previous_digest"]) != previous:
                raise RuntimeError("continual experiment chain discontinuity")
            payload = json.loads(str(row["payload_json"]))
            if sha256_json(payload["measurement"]) != str(row["measurement_digest"]):
                raise RuntimeError("continual experiment measurement digest mismatch")
            if sha256_json(payload["result"]) != str(row["result_digest"]):
                raise RuntimeError("continual experiment result digest mismatch")
            body = {"schema": "egai-rc14-continual-experiment-event-v1", "seq": seq,
                    "measurement_digest": str(row["measurement_digest"]), "result_digest": str(row["result_digest"]),
                    "previous_digest": previous}
            if sha256_json(body) != str(row["event_digest"]):
                raise RuntimeError("continual experiment event digest mismatch")
            previous = str(row["event_digest"])
        if int(state["seq"]) != len(rows) or str(state["head"]) != previous:
            raise RuntimeError("continual experiment head mismatch")
        return {"ok": True, "experiments": len(rows), "head": previous}
