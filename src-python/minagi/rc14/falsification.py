from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
import sqlite3
import time

from kvcontinual.execution.durability.file_lock import ProcessFileLock
from minagi.egai.canonical import sha256_json
from .models import require_digest


class FalsificationKind(str, Enum):
    COUNTEREXAMPLE = "counterexample"
    NEGATIVE_CONTROL = "negative_control"
    ADVERSARIAL = "adversarial"
    ABLATION = "ablation"
    OOD = "ood"
    RETENTION = "retention"
    SECURITY = "security"
    FRESH_HIDDEN = "fresh_hidden"


@dataclass(frozen=True)
class FalsificationCase:
    case_id: str
    kind: FalsificationKind
    payload_digest: str
    expected_property_digest: str
    fresh_task_lease_digest: str = ""

    def __post_init__(self) -> None:
        require_digest(self.payload_digest, field_name="payload_digest")
        require_digest(self.expected_property_digest, field_name="expected_property_digest")
        if self.fresh_task_lease_digest:
            require_digest(self.fresh_task_lease_digest, field_name="fresh_task_lease_digest")

    @property
    def digest(self) -> str:
        body = asdict(self)
        body["kind"] = self.kind.value
        return sha256_json(body)


@dataclass(frozen=True)
class FalsificationPlan:
    candidate_digest: str
    policy_digest: str
    cases: tuple[FalsificationCase, ...]
    generated_by: str
    min_pass_rate: float = 1.0
    preregistered: bool = True
    schema: str = "egai-rc14-falsification-plan-v2"

    def validate(self) -> None:
        require_digest(self.candidate_digest, field_name="candidate_digest")
        require_digest(self.policy_digest, field_name="policy_digest")
        if not self.preregistered:
            raise PermissionError("falsification plan must be preregistered")
        if not 0.0 <= float(self.min_pass_rate) <= 1.0:
            raise ValueError("min_pass_rate must be in [0,1]")
        if not self.cases:
            raise ValueError("falsification plan requires cases")
        kinds = {c.kind for c in self.cases}
        mandatory = {FalsificationKind.NEGATIVE_CONTROL, FalsificationKind.RETENTION, FalsificationKind.SECURITY}
        missing = mandatory - kinds
        if missing:
            raise ValueError(f"missing mandatory falsification classes: {sorted(x.value for x in missing)}")
        ids = [c.case_id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate falsification case IDs")

    @property
    def digest(self) -> str:
        self.validate()
        return sha256_json({
            "schema": self.schema,
            "candidate_digest": self.candidate_digest,
            "policy_digest": self.policy_digest,
            "cases": [c.digest for c in self.cases],
            "generated_by": self.generated_by,
            "min_pass_rate": float(self.min_pass_rate),
            "preregistered": self.preregistered,
        })


@dataclass(frozen=True)
class FalsificationOutcome:
    case_id: str
    case_digest: str
    passed: bool
    score: float
    evidence_digest: str

    def __post_init__(self) -> None:
        require_digest(self.case_digest, field_name="case_digest")
        require_digest(self.evidence_digest, field_name="evidence_digest")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be in [0,1]")


@dataclass(frozen=True)
class FalsificationRun:
    plan_digest: str
    candidate_digest: str
    outcomes: tuple[FalsificationOutcome, ...]
    evaluator_id: str
    environment_digest: str
    task_set_digest: str
    schema: str = "egai-rc14-falsification-run-v2"

    @property
    def digest(self) -> str:
        require_digest(self.plan_digest, field_name="plan_digest")
        require_digest(self.candidate_digest, field_name="candidate_digest")
        require_digest(self.environment_digest, field_name="environment_digest")
        require_digest(self.task_set_digest, field_name="task_set_digest")
        if not self.evaluator_id:
            raise ValueError("evaluator_id is required")
        return sha256_json({
            "schema": self.schema,
            "plan_digest": self.plan_digest,
            "candidate_digest": self.candidate_digest,
            "outcomes": [asdict(x) for x in self.outcomes],
            "evaluator_id": self.evaluator_id,
            "environment_digest": self.environment_digest,
            "task_set_digest": self.task_set_digest,
        })

    def validate_against(self, plan: FalsificationPlan) -> dict[str, float | bool]:
        plan.validate()
        if self.plan_digest != plan.digest or self.candidate_digest != plan.candidate_digest:
            raise PermissionError("falsification run is not bound to preregistered plan/candidate")
        expected = {c.case_id: c.digest for c in plan.cases}
        got = {o.case_id: o for o in self.outcomes}
        if set(expected) != set(got) or len(got) != len(self.outcomes):
            raise PermissionError("falsification outcomes do not exactly cover preregistered cases")
        for case_id, case_digest in expected.items():
            if got[case_id].case_digest != case_digest:
                raise PermissionError("falsification case substitution detected")
        pass_rate = sum(1 for o in self.outcomes if o.passed) / len(self.outcomes)
        by_kind = {kind: [got[c.case_id] for c in plan.cases if c.kind is kind] for kind in FalsificationKind}
        mandatory = (FalsificationKind.NEGATIVE_CONTROL, FalsificationKind.RETENTION, FalsificationKind.SECURITY)
        mandatory_ok = all(by_kind[kind] and all(o.passed for o in by_kind[kind]) for kind in mandatory)
        # Policy cannot be weakened by a candidate-provided lower plan threshold.
        effective_min_pass_rate = 1.0
        return {
            "pass_rate": pass_rate,
            "effective_min_pass_rate": effective_min_pass_rate,
            "mandatory_classes_passed": mandatory_ok,
            "passed": pass_rate >= effective_min_pass_rate and mandatory_ok,
        }


class PreregisteredPlanStore:
    """Immutable preregistration and run store.

    RC14.2 persists the exact evaluated run next to the sealed plan so qualification can
    be reconstructed later without trusting an in-memory caller object.
    """
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "falsification.sqlite3"
        self.lock_path = self.root / "falsification.lock"
        with self._connect() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=FULL")
            c.execute("CREATE TABLE IF NOT EXISTS plans(plan_digest TEXT PRIMARY KEY,candidate_digest TEXT NOT NULL,plan_json TEXT NOT NULL,sealed_ns INTEGER NOT NULL)")
            c.execute("CREATE TABLE IF NOT EXISTS runs(run_digest TEXT PRIMARY KEY,plan_digest TEXT NOT NULL,candidate_digest TEXT NOT NULL,run_json TEXT NOT NULL,recorded_ns INTEGER NOT NULL,FOREIGN KEY(plan_digest) REFERENCES plans(plan_digest))")
            c.execute("CREATE INDEX IF NOT EXISTS idx_runs_candidate ON runs(candidate_digest,recorded_ns)")

    def _connect(self):
        c = sqlite3.connect(self.db_path, timeout=30.0)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA busy_timeout=30000")
        c.execute("PRAGMA foreign_keys=ON")
        return c

    @staticmethod
    def _plan_doc(plan: FalsificationPlan) -> dict:
        return {
            "candidate_digest": plan.candidate_digest,
            "policy_digest": plan.policy_digest,
            "generated_by": plan.generated_by,
            "min_pass_rate": plan.min_pass_rate,
            "preregistered": plan.preregistered,
            "schema": plan.schema,
            "cases": [{**asdict(c), "kind": c.kind.value} for c in plan.cases],
        }

    @staticmethod
    def _run_doc(run: FalsificationRun) -> dict:
        return {
            "plan_digest": run.plan_digest,
            "candidate_digest": run.candidate_digest,
            "outcomes": [asdict(x) for x in run.outcomes],
            "evaluator_id": run.evaluator_id,
            "environment_digest": run.environment_digest,
            "task_set_digest": run.task_set_digest,
            "schema": run.schema,
        }

    def seal(self, plan: FalsificationPlan) -> str:
        plan.validate()
        doc = self._plan_doc(plan)
        with ProcessFileLock(self.lock_path, timeout=30.0):
            try:
                with self._connect() as c:
                    c.execute("INSERT INTO plans VALUES(?,?,?,?)", (plan.digest, plan.candidate_digest, json.dumps(doc, sort_keys=True, separators=(",", ":")), time.time_ns()))
                    c.commit()
            except sqlite3.IntegrityError:
                with self._connect() as c:
                    row = c.execute("SELECT plan_json FROM plans WHERE plan_digest=?", (plan.digest,)).fetchone()
                if row is None or json.loads(row[0]) != doc:
                    raise RuntimeError("falsification preregistration collision")
        return plan.digest

    def get(self, digest: str) -> FalsificationPlan:
        require_digest(digest, field_name="plan_digest")
        with self._connect() as c:
            row = c.execute("SELECT plan_json FROM plans WHERE plan_digest=?", (digest,)).fetchone()
        if row is None:
            raise KeyError(digest)
        doc = json.loads(row[0])
        cases = tuple(FalsificationCase(kind=FalsificationKind(c.pop("kind")), **c) for c in [dict(x) for x in doc.pop("cases")])
        plan = FalsificationPlan(cases=cases, **doc)
        if plan.digest != digest:
            raise RuntimeError("stored falsification plan digest mismatch")
        return plan

    def record_run(self, run: FalsificationRun) -> str:
        plan = self.get(run.plan_digest)
        run.validate_against(plan)
        doc = self._run_doc(run)
        with ProcessFileLock(self.lock_path, timeout=30.0):
            try:
                with self._connect() as c:
                    c.execute(
                        "INSERT INTO runs VALUES(?,?,?,?,?)",
                        (run.digest, run.plan_digest, run.candidate_digest, json.dumps(doc, sort_keys=True, separators=(",", ":")), time.time_ns()),
                    )
                    c.commit()
            except sqlite3.IntegrityError:
                with self._connect() as c:
                    row = c.execute("SELECT run_json FROM runs WHERE run_digest=?", (run.digest,)).fetchone()
                if row is None or json.loads(row[0]) != doc:
                    raise RuntimeError("falsification run collision")
        return run.digest

    def get_run(self, digest: str) -> FalsificationRun:
        require_digest(digest, field_name="falsification_run_digest")
        with self._connect() as c:
            row = c.execute("SELECT run_json FROM runs WHERE run_digest=?", (digest,)).fetchone()
        if row is None:
            raise KeyError(digest)
        doc = json.loads(row[0])
        outcomes = tuple(FalsificationOutcome(**x) for x in doc.pop("outcomes"))
        run = FalsificationRun(outcomes=outcomes, **doc)
        if run.digest != digest:
            raise RuntimeError("stored falsification run digest mismatch")
        plan = self.get(run.plan_digest)
        run.validate_against(plan)
        return run

    def verify(self) -> dict[str, int]:
        with self._connect() as c:
            if str(c.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
                raise RuntimeError("falsification sqlite integrity failure")
            plans = c.execute("SELECT plan_digest FROM plans ORDER BY sealed_ns").fetchall()
            runs = c.execute("SELECT run_digest FROM runs ORDER BY recorded_ns").fetchall()
        for row in plans:
            self.get(str(row[0]))
        for row in runs:
            self.get_run(str(row[0]))
        return {"plans": len(plans), "runs": len(runs)}
