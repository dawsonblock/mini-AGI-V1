from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import time
from typing import Any, Mapping, Sequence

from kvcontinual.execution.durability.file_lock import ProcessFileLock
from minagi.egai.canonical import sha256_json
from .experiments import ContinualExperimentMeasurement, ContinualExperimentHarnessRC14
from .models import require_digest

GENESIS = "0" * 64


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    p = Path(path)
    if not p.is_file() or p.is_symlink():
        raise ValueError("run artifact must be a regular non-symlink file")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


@dataclass(frozen=True)
class DatasetSplitManifestRC14:
    dataset_digest: str
    task_set_digest: str
    train_split_digest: str
    evaluation_split_digest: str
    hidden_split_commitment_digest: str
    preprocessing_digest: str
    task_schema_digest: str
    disjointness_proof_digest: str
    schema: str = "egai-rc14-dataset-split-manifest-v1"

    def __post_init__(self) -> None:
        for name in (
            "dataset_digest", "task_set_digest", "train_split_digest", "evaluation_split_digest",
            "hidden_split_commitment_digest", "preprocessing_digest", "task_schema_digest",
            "disjointness_proof_digest",
        ):
            require_digest(getattr(self, name), field_name=name)
        if len({self.train_split_digest, self.evaluation_split_digest, self.hidden_split_commitment_digest}) != 3:
            raise ValueError("train/evaluation/hidden splits must be cryptographically distinct")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class ExecutedRunReceiptRC14:
    plan_digest: str
    candidate_digest: str
    variant: str
    seed: int
    dataset_manifest_digest: str
    base_model_digest: str
    tokenizer_digest: str
    environment_digest: str
    runner_code_digest: str
    command_digest: str
    stdout_digest: str
    stderr_digest: str
    measurement_digest: str
    result_digest: str
    evaluator_id: str
    started_ns: int
    finished_ns: int
    exit_code: int
    receipt: Mapping[str, Any]
    schema: str = "egai-rc14-executed-run-receipt-v1"

    def __post_init__(self) -> None:
        for name in (
            "plan_digest", "candidate_digest", "dataset_manifest_digest", "base_model_digest",
            "tokenizer_digest", "environment_digest", "runner_code_digest", "command_digest",
            "stdout_digest", "stderr_digest", "measurement_digest", "result_digest",
        ):
            require_digest(getattr(self, name), field_name=name)
        if not self.variant or not self.evaluator_id or int(self.seed) < 0:
            raise ValueError("variant, evaluator_id and non-negative seed are required")
        if int(self.started_ns) <= 0 or int(self.finished_ns) < int(self.started_ns):
            raise ValueError("invalid execution time interval")
        if not isinstance(self.receipt, Mapping) or not isinstance(self.receipt.get("body"), Mapping):
            raise ValueError("signed external evaluator receipt required")
        object.__setattr__(self, "receipt", dict(self.receipt))

    @property
    def signed_body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_digest": self.plan_digest,
            "candidate_digest": self.candidate_digest,
            "variant": self.variant,
            "seed": int(self.seed),
            "dataset_manifest_digest": self.dataset_manifest_digest,
            "base_model_digest": self.base_model_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "environment_digest": self.environment_digest,
            "runner_code_digest": self.runner_code_digest,
            "command_digest": self.command_digest,
            "stdout_digest": self.stdout_digest,
            "stderr_digest": self.stderr_digest,
            "measurement_digest": self.measurement_digest,
            "result_digest": self.result_digest,
            "evaluator_id": self.evaluator_id,
            "started_ns": int(self.started_ns),
            "finished_ns": int(self.finished_ns),
            "exit_code": int(self.exit_code),
        }

    @property
    def digest(self) -> str:
        return sha256_json({**self.signed_body, "receipt": dict(self.receipt)})


@dataclass(frozen=True)
class ExecutedRunMatrixCertificateRC14:
    plan_digest: str
    candidate_digest: str
    dataset_manifest_digest: str
    run_receipt_digests: tuple[str, ...]
    evaluator_ids: tuple[str, ...]
    complete: bool
    all_successful: bool
    eligible: bool
    reasons: tuple[str, ...]
    schema: str = "egai-rc14-executed-run-matrix-certificate-v1"

    def __post_init__(self) -> None:
        for name in ("plan_digest", "candidate_digest", "dataset_manifest_digest"):
            require_digest(getattr(self, name), field_name=name)
        for d in self.run_receipt_digests:
            require_digest(d, field_name="run_receipt_digest")
        object.__setattr__(self, "run_receipt_digests", tuple(self.run_receipt_digests))
        object.__setattr__(self, "evaluator_ids", tuple(self.evaluator_ids))
        object.__setattr__(self, "reasons", tuple(self.reasons))

    @property
    def digest(self) -> str:
        d = asdict(self)
        d["run_receipt_digests"] = list(self.run_receipt_digests)
        d["evaluator_ids"] = list(self.evaluator_ids)
        d["reasons"] = list(self.reasons)
        return sha256_json(d)


class ExecutedRunStoreRC14:
    """Immutable ledger proving experiment cells came from signed external executions."""

    def __init__(self, root: str | os.PathLike, *, protocol, evaluator_verifiers: Mapping[str, Any]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "executed_runs.sqlite3"
        self.lock_path = self.root / "executed_runs.lock"
        self.protocol = protocol
        self.evaluator_verifiers = dict(evaluator_verifiers)
        self._init_db()

    def _connect(self):
        c = sqlite3.connect(self.db_path, timeout=30.0)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=FULL")
        return c

    def _init_db(self):
        with self._connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS datasets(digest TEXT PRIMARY KEY,payload_json TEXT NOT NULL)")
            c.execute("CREATE TABLE IF NOT EXISTS state(singleton INTEGER PRIMARY KEY CHECK(singleton=1),seq INTEGER NOT NULL,head TEXT NOT NULL)")
            c.execute("""CREATE TABLE IF NOT EXISTS runs(
                seq INTEGER PRIMARY KEY,plan_digest TEXT NOT NULL,variant TEXT NOT NULL,seed INTEGER NOT NULL,
                receipt_digest TEXT NOT NULL UNIQUE,dataset_manifest_digest TEXT NOT NULL,evaluator_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,previous_digest TEXT NOT NULL,event_digest TEXT NOT NULL UNIQUE,
                committed_ns INTEGER NOT NULL,UNIQUE(plan_digest,variant,seed))""")
            c.execute("INSERT OR IGNORE INTO state(singleton,seq,head) VALUES(1,0,?)", (GENESIS,))
            c.commit()

    def register_dataset(self, manifest: DatasetSplitManifestRC14) -> str:
        payload = json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":"))
        with ProcessFileLock(self.lock_path, timeout=30.0):
            with self._connect() as c:
                c.execute("BEGIN IMMEDIATE")
                row = c.execute("SELECT payload_json FROM datasets WHERE digest=?", (manifest.digest,)).fetchone()
                if row is not None and str(row["payload_json"]) != payload:
                    raise RuntimeError("dataset manifest digest collision")
                c.execute("INSERT OR IGNORE INTO datasets(digest,payload_json) VALUES(?,?)", (manifest.digest, payload))
                c.commit()
        return manifest.digest

    def get_dataset(self, digest: str) -> DatasetSplitManifestRC14:
        require_digest(digest, field_name="dataset_manifest_digest")
        with self._connect() as c:
            row = c.execute("SELECT payload_json FROM datasets WHERE digest=?", (digest,)).fetchone()
        if row is None:
            raise KeyError(digest)
        manifest = DatasetSplitManifestRC14(**json.loads(str(row["payload_json"])))
        if manifest.digest != digest:
            raise RuntimeError("stored dataset manifest digest mismatch")
        return manifest

    @staticmethod
    def measurement_digest(measurement: ContinualExperimentMeasurement) -> str:
        doc = asdict(measurement)
        doc["old_task_history"] = {k: list(v) for k, v in sorted(measurement.old_task_history.items())}
        return sha256_json(doc)

    def record(self, run: ExecutedRunReceiptRC14, measurement: ContinualExperimentMeasurement) -> str:
        plan = self.protocol.get_plan(run.plan_digest)
        dataset = self.get_dataset(run.dataset_manifest_digest)
        if run.candidate_digest != plan.candidate_digest:
            raise PermissionError("executed run candidate does not match sealed plan")
        if run.variant not in plan.variants or int(run.seed) not in plan.seeds:
            raise PermissionError("executed run lies outside preregistered matrix")
        if run.base_model_digest != plan.base_model_digest or run.tokenizer_digest != plan.tokenizer_digest:
            raise PermissionError("executed run model/tokenizer does not match sealed plan")
        if run.environment_digest != plan.environment_digest:
            raise PermissionError("executed run environment does not match sealed plan")
        if dataset.task_set_digest != plan.task_set_digest:
            raise PermissionError("dataset task set does not match sealed plan")
        if measurement.variant != run.variant or self.measurement_digest(measurement) != run.measurement_digest:
            raise PermissionError("executed run measurement binding mismatch")
        result = ContinualExperimentHarnessRC14().evaluate(measurement)
        if result.digest != run.result_digest:
            raise PermissionError("executed run result does not replay from measurement")
        if run.exit_code != 0:
            raise PermissionError("failed external execution cannot become experiment evidence")
        verifier = self.evaluator_verifiers.get(run.evaluator_id)
        if verifier is None or getattr(verifier, "key_id", run.evaluator_id) != run.evaluator_id:
            raise PermissionError("untrusted external execution evaluator")
        if not verifier.verify(dict(run.receipt), expected_body=run.signed_body):
            raise PermissionError("invalid external execution receipt")
        payload = {"run": asdict(run), "measurement": asdict(measurement)}
        payload["measurement"]["old_task_history"] = {k: list(v) for k, v in sorted(measurement.old_task_history.items())}
        with ProcessFileLock(self.lock_path, timeout=30.0):
            with self._connect() as c:
                c.execute("BEGIN IMMEDIATE")
                st = c.execute("SELECT seq,head FROM state WHERE singleton=1").fetchone()
                seq, prev = int(st["seq"]) + 1, str(st["head"])
                body = {"schema":"egai-rc14-executed-run-event-v1","seq":seq,"plan_digest":run.plan_digest,
                        "variant":run.variant,"seed":int(run.seed),"receipt_digest":run.digest,"previous_digest":prev}
                ev = sha256_json(body)
                try:
                    c.execute("INSERT INTO runs(seq,plan_digest,variant,seed,receipt_digest,dataset_manifest_digest,evaluator_id,payload_json,previous_digest,event_digest,committed_ns) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                              (seq,run.plan_digest,run.variant,int(run.seed),run.digest,run.dataset_manifest_digest,run.evaluator_id,json.dumps(payload,sort_keys=True,separators=(",",":")),prev,ev,time.time_ns()))
                except sqlite3.IntegrityError as exc:
                    raise PermissionError("duplicate executed experiment slot") from exc
                c.execute("UPDATE state SET seq=?,head=? WHERE singleton=1", (seq,ev))
                c.commit()
        return run.digest

    def certify(self, plan_digest: str) -> ExecutedRunMatrixCertificateRC14:
        plan = self.protocol.get_plan(plan_digest)
        with self._connect() as c:
            rows = list(c.execute("SELECT * FROM runs WHERE plan_digest=? ORDER BY variant,seed", (plan_digest,)))
        expected = {(v, int(s)) for v in plan.variants for s in plan.seeds}
        present = {(str(r["variant"]), int(r["seed"])) for r in rows}
        reasons: list[str] = []
        if present != expected:
            reasons.append("incomplete_executed_run_matrix")
        dataset_digests = {str(r["dataset_manifest_digest"]) for r in rows}
        if len(dataset_digests) > 1:
            reasons.append("mixed_dataset_manifests")
        all_success = True
        receipt_digests: list[str] = []
        evaluators: set[str] = set()
        for row in rows:
            payload = json.loads(str(row["payload_json"]))
            run_doc = dict(payload["run"])
            run = ExecutedRunReceiptRC14(**run_doc)
            mdoc = dict(payload["measurement"])
            mdoc["old_task_history"] = {k: tuple(v) for k, v in dict(mdoc["old_task_history"]).items()}
            measurement = ContinualExperimentMeasurement(**mdoc)
            verifier = self.evaluator_verifiers.get(run.evaluator_id)
            if verifier is None or not verifier.verify(run.receipt, expected_body=run.signed_body):
                raise RuntimeError("executed run signature verification failed")
            if run.digest != str(row["receipt_digest"]) or self.measurement_digest(measurement) != run.measurement_digest:
                raise RuntimeError("executed run payload digest mismatch")
            if ContinualExperimentHarnessRC14().evaluate(measurement).digest != run.result_digest:
                raise RuntimeError("executed run result replay mismatch")
            all_success = all_success and run.exit_code == 0
            receipt_digests.append(run.digest)
            evaluators.add(run.evaluator_id)
        complete = present == expected
        if not all_success:
            reasons.append("failed_external_execution")
        dataset_digest = next(iter(dataset_digests), "sha256:" + "0" * 64)
        return ExecutedRunMatrixCertificateRC14(plan.digest, plan.candidate_digest, dataset_digest,
                                                tuple(sorted(receipt_digests)), tuple(sorted(evaluators)),
                                                complete, all_success, complete and all_success and not reasons,
                                                tuple(reasons))

    def head_digest(self) -> str:
        with self._connect() as c:
            head = str(c.execute("SELECT head FROM state WHERE singleton=1").fetchone()["head"])
        return "sha256:" + head if head == GENESIS else require_digest(head, field_name="executed_run_head")

    def verify(self) -> dict[str, Any]:
        with self._connect() as c:
            if str(c.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
                raise RuntimeError("executed-run sqlite integrity failure")
            rows = list(c.execute("SELECT * FROM runs ORDER BY seq"))
            state = c.execute("SELECT seq,head FROM state WHERE singleton=1").fetchone()
        prev = GENESIS
        for i, row in enumerate(rows, 1):
            if int(row["seq"]) != i or str(row["previous_digest"]) != prev:
                raise RuntimeError("executed-run chain discontinuity")
            payload = json.loads(str(row["payload_json"]))
            run = ExecutedRunReceiptRC14(**dict(payload["run"]))
            verifier = self.evaluator_verifiers.get(run.evaluator_id)
            if verifier is None or not verifier.verify(run.receipt, expected_body=run.signed_body):
                raise RuntimeError("executed-run signature invalid")
            if run.digest != str(row["receipt_digest"]):
                raise RuntimeError("executed-run receipt digest mismatch")
            body = {"schema":"egai-rc14-executed-run-event-v1","seq":i,"plan_digest":run.plan_digest,
                    "variant":run.variant,"seed":int(run.seed),"receipt_digest":run.digest,"previous_digest":prev}
            ev = sha256_json(body)
            if ev != str(row["event_digest"]):
                raise RuntimeError("executed-run event digest mismatch")
            prev = ev
        if int(state["seq"]) != len(rows) or str(state["head"]) != prev:
            raise RuntimeError("executed-run head mismatch")
        return {"datasets": self._dataset_count(), "runs": len(rows), "head": prev}

    def _dataset_count(self) -> int:
        with self._connect() as c:
            return int(c.execute("SELECT COUNT(*) FROM datasets").fetchone()[0])


class SubprocessJSONExperimentRunnerRC14:
    """Runs an external evaluator without a shell and captures immutable output artifacts.

    The command must write a JSON ContinualExperimentMeasurement document to output_path.
    Signing remains external: this runner returns the exact receipt body to be signed by an evaluator key.
    """

    def __init__(self, *, runner_code_digest: str, timeout_seconds: int = 3600):
        self.runner_code_digest = require_digest(runner_code_digest, field_name="runner_code_digest")
        self.timeout_seconds = int(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def execute(self, argv: Sequence[str], *, output_path: str | os.PathLike, cwd: str | os.PathLike | None = None,
                env: Mapping[str, str] | None = None) -> tuple[ContinualExperimentMeasurement, dict[str, Any]]:
        if not argv or any(not isinstance(x, str) or not x for x in argv):
            raise ValueError("non-empty argv strings required")
        if any("\x00" in x for x in argv):
            raise ValueError("NUL in argv")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        started = time.time_ns()
        proc = subprocess.run(list(argv), cwd=None if cwd is None else str(cwd), env=None if env is None else {**os.environ, **dict(env)},
                              shell=False, capture_output=True, timeout=self.timeout_seconds, check=False)
        finished = time.time_ns()
        if not out.is_file() or out.is_symlink():
            raise RuntimeError("external evaluator did not produce a regular measurement JSON file")
        doc = json.loads(out.read_text(encoding="utf-8"))
        if "old_task_history" in doc:
            doc["old_task_history"] = {k: tuple(v) for k, v in dict(doc["old_task_history"]).items()}
        measurement = ContinualExperimentMeasurement(**doc)
        result = ContinualExperimentHarnessRC14().evaluate(measurement)
        body = {
            "runner_code_digest": self.runner_code_digest,
            "command_digest": sha256_json({"argv": list(argv)}),
            "stdout_digest": sha256_bytes(proc.stdout),
            "stderr_digest": sha256_bytes(proc.stderr),
            "measurement_digest": ExecutedRunStoreRC14.measurement_digest(measurement),
            "result_digest": result.digest,
            "started_ns": started,
            "finished_ns": finished,
            "exit_code": int(proc.returncode),
        }
        return measurement, body
