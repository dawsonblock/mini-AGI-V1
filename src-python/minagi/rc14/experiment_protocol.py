from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping

from kvcontinual.execution.durability.file_lock import ProcessFileLock
from minagi.egai.canonical import sha256_json
from .experiments import ContinualExperimentResultRC14
from .metrics import ContinualMetrics
from .models import require_digest

GENESIS = "0" * 64
DEFAULT_ABLATION_LADDER = (
    "frozen",
    "retrieval",
    "semantic_memory",
    "skills",
    "replay",
    "isolated_neural_memory",
)


@dataclass(frozen=True)
class PreregisteredContinualExperimentPlanRC14:
    experiment_id: str
    candidate_digest: str
    base_model_digest: str
    tokenizer_digest: str
    environment_digest: str
    task_set_digest: str
    hidden_task_commitment_digest: str
    variants: tuple[str, ...] = DEFAULT_ABLATION_LADDER
    seeds: tuple[int, ...] = (17, 29, 43)
    max_mean_forgetting: float = 0.05
    max_worst_forgetting: float = 0.10
    min_fresh_task_gain: float = 0.0
    min_incremental_gain_over_replay: float = 0.0
    min_falsification_survival: float = 1.0
    min_reproducibility: float = 0.90
    require_zero_security_regressions: bool = True
    require_zero_unauthorized_writes: bool = True
    schema: str = "egai-rc14-preregistered-continual-experiment-v1"

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        for name in ("candidate_digest", "base_model_digest", "tokenizer_digest", "environment_digest", "task_set_digest", "hidden_task_commitment_digest"):
            require_digest(getattr(self, name), field_name=name)
        if tuple(self.variants) != DEFAULT_ABLATION_LADDER:
            raise ValueError("RC14.7 requires the complete fixed ablation ladder")
        if len(set(self.seeds)) != len(self.seeds) or len(self.seeds) < 2 or any(int(s) < 0 for s in self.seeds):
            raise ValueError("at least two unique non-negative replicate seeds are required")
        for name in ("max_mean_forgetting", "max_worst_forgetting", "min_fresh_task_gain", "min_incremental_gain_over_replay", "min_falsification_survival", "min_reproducibility"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.max_mean_forgetting < 0 or self.max_worst_forgetting < 0:
            raise ValueError("forgetting thresholds must be non-negative")
        if not 0.0 <= self.min_falsification_survival <= 1.0 or not 0.0 <= self.min_reproducibility <= 1.0:
            raise ValueError("probability thresholds must be in [0,1]")
        object.__setattr__(self, "variants", tuple(str(x) for x in self.variants))
        object.__setattr__(self, "seeds", tuple(int(x) for x in self.seeds))

    @property
    def digest(self) -> str:
        d = asdict(self)
        d["variants"] = list(self.variants)
        d["seeds"] = list(self.seeds)
        return sha256_json(d)


@dataclass(frozen=True)
class ContinualExperimentReplicateRC14:
    plan_digest: str
    variant: str
    seed: int
    result_digest: str
    evaluator_id: str
    environment_digest: str
    task_set_digest: str
    schema: str = "egai-rc14-continual-experiment-replicate-v1"

    def __post_init__(self) -> None:
        for name in ("plan_digest", "result_digest", "environment_digest", "task_set_digest"):
            require_digest(getattr(self, name), field_name=name)
        if not self.variant or not self.evaluator_id or int(self.seed) < 0:
            raise ValueError("variant, evaluator_id and non-negative seed are required")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class ExperimentReplicateAttestationRC14:
    plan_digest: str; variant: str; seed: int; result_digest: str; evaluator_id: str; environment_digest: str; task_set_digest: str; observed_ns: int; receipt: Mapping[str, Any]
    schema: str = "egai-rc14-experiment-replicate-attestation-v1"
    def __post_init__(self):
        for n in ("plan_digest","result_digest","environment_digest","task_set_digest"): require_digest(getattr(self,n),field_name=n)
        if not self.variant or int(self.seed)<0 or not self.evaluator_id or int(self.observed_ns)<=0: raise ValueError("variant, non-negative seed, evaluator_id and observed_ns are required")
        if not isinstance(self.receipt,Mapping) or not isinstance(self.receipt.get("body"),Mapping): raise ValueError("signed experiment replicate receipt required")
        object.__setattr__(self,"receipt",dict(self.receipt))
    @property
    def signed_body(self):
        return {"schema":self.schema,"plan_digest":self.plan_digest,"variant":self.variant,"seed":int(self.seed),"result_digest":self.result_digest,"evaluator_id":self.evaluator_id,"environment_digest":self.environment_digest,"task_set_digest":self.task_set_digest,"observed_ns":int(self.observed_ns)}
    @property
    def digest(self): return sha256_json(asdict(self))


@dataclass(frozen=True)
class AblationQualificationCertificateRC14:
    plan_digest: str
    candidate_digest: str
    replicate_digests: tuple[str, ...]
    target_variant: str
    target_mean_fresh_gain: float
    replay_mean_fresh_gain: float
    incremental_gain_over_replay: float
    target_mean_forgetting: float
    target_worst_forgetting: float
    target_mean_reproducibility: float
    complete: bool
    eligible: bool
    reasons: tuple[str, ...]
    schema: str = "egai-rc14-ablation-qualification-certificate-v1"

    def __post_init__(self) -> None:
        require_digest(self.plan_digest, field_name="plan_digest")
        require_digest(self.candidate_digest, field_name="candidate_digest")
        for d in self.replicate_digests:
            require_digest(d, field_name="replicate_digest")
        vals = (self.target_mean_fresh_gain, self.replay_mean_fresh_gain, self.incremental_gain_over_replay,
                self.target_mean_forgetting, self.target_worst_forgetting, self.target_mean_reproducibility)
        if any(not math.isfinite(float(v)) for v in vals):
            raise ValueError("certificate metrics must be finite")
        object.__setattr__(self, "replicate_digests", tuple(str(x) for x in self.replicate_digests))
        object.__setattr__(self, "reasons", tuple(str(x) for x in self.reasons))

    @property
    def digest(self) -> str:
        d = asdict(self)
        d["replicate_digests"] = list(self.replicate_digests)
        d["reasons"] = list(self.reasons)
        return sha256_json(d)


def _metrics_from_doc(d: Mapping[str, object]) -> ContinualMetrics:
    m = ContinualMetrics(**d)
    m.validate()
    return m


def _result_from_doc(d: Mapping[str, object]) -> ContinualExperimentResultRC14:
    result = ContinualExperimentResultRC14(
        variant=str(d["variant"]),
        metrics=_metrics_from_doc(d["metrics"]),
        per_task_forgetting={str(k): float(v) for k, v in dict(d["per_task_forgetting"]).items()},
        interference={str(k): float(v) for k, v in dict(d["interference"]).items()},
        eligible_for_promotion_evidence=bool(d["eligible_for_promotion_evidence"]),
        rejection_reasons=tuple(str(x) for x in d["rejection_reasons"]),
        schema=str(d.get("schema", "egai-rc14-continual-experiment-result-v1")),
    )
    if result.digest != sha256_json({**dict(d), "metrics": dict(d["metrics"]) }):
        # The canonical dataclass digest is the authority; this check mainly catches malformed documents.
        if result.digest != sha256_json(asdict(result)):
            raise RuntimeError("result digest reconstruction failure")
    return result


class PreregisteredContinualExperimentStoreRC14:
    """Seals experiment plans before results and records one immutable result per plan/variant/seed."""

    def __init__(self, root: str | os.PathLike, *, replicate_verifiers: Mapping[str, Any] | None = None, strict_signed_replicates: bool = False):
        self.root = Path(root)
        self.replicate_verifiers = dict(replicate_verifiers or {})
        self.strict_signed_replicates = bool(strict_signed_replicates)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "protocol.sqlite3"
        self.lock_path = self.root / "protocol.lock"
        self._init_db()

    def _connect(self):
        c = sqlite3.connect(self.db_path, timeout=30.0)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=FULL")
        return c

    def _init_db(self) -> None:
        with self._connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS plans(plan_digest TEXT PRIMARY KEY, payload_json TEXT NOT NULL, sealed_ns INTEGER NOT NULL)")
            c.execute("CREATE TABLE IF NOT EXISTS state(singleton INTEGER PRIMARY KEY CHECK(singleton=1), seq INTEGER NOT NULL, head TEXT NOT NULL)")
            c.execute("""CREATE TABLE IF NOT EXISTS replicates(
                seq INTEGER PRIMARY KEY, plan_digest TEXT NOT NULL, variant TEXT NOT NULL, seed INTEGER NOT NULL,
                replicate_digest TEXT NOT NULL UNIQUE, result_digest TEXT NOT NULL, payload_json TEXT NOT NULL,
                previous_digest TEXT NOT NULL, event_digest TEXT NOT NULL UNIQUE, committed_ns INTEGER NOT NULL,
                UNIQUE(plan_digest,variant,seed), FOREIGN KEY(plan_digest) REFERENCES plans(plan_digest))""")
            c.execute("""CREATE TABLE IF NOT EXISTS replicate_attestations(replicate_digest TEXT PRIMARY KEY,attestation_digest TEXT NOT NULL UNIQUE,payload_json TEXT NOT NULL,FOREIGN KEY(replicate_digest) REFERENCES replicates(replicate_digest))""")
            c.execute("INSERT OR IGNORE INTO state(singleton,seq,head) VALUES(1,0,?)", (GENESIS,))
            c.commit()

    def seal(self, plan: PreregisteredContinualExperimentPlanRC14) -> str:
        payload = json.dumps(asdict(plan), sort_keys=True, separators=(",", ":"))
        with ProcessFileLock(self.lock_path, timeout=30.0):
            with self._connect() as c:
                c.execute("BEGIN IMMEDIATE")
                row = c.execute("SELECT payload_json FROM plans WHERE plan_digest=?", (plan.digest,)).fetchone()
                if row is not None and str(row["payload_json"]) != payload:
                    raise RuntimeError("experiment plan digest collision")
                c.execute("INSERT OR IGNORE INTO plans(plan_digest,payload_json,sealed_ns) VALUES(?,?,?)", (plan.digest, payload, time.time_ns()))
                c.commit()
        return plan.digest

    def get_plan(self, digest: str) -> PreregisteredContinualExperimentPlanRC14:
        require_digest(digest, field_name="plan_digest")
        with self._connect() as c:
            row = c.execute("SELECT payload_json FROM plans WHERE plan_digest=?", (digest,)).fetchone()
        if row is None:
            raise KeyError(digest)
        doc = json.loads(row["payload_json"])
        doc["variants"] = tuple(doc["variants"]); doc["seeds"] = tuple(doc["seeds"])
        plan = PreregisteredContinualExperimentPlanRC14(**doc)
        if plan.digest != digest:
            raise RuntimeError("stored experiment plan digest mismatch")
        return plan

    def record_result(self, plan_digest: str, *, variant: str, seed: int, result: ContinualExperimentResultRC14,
                      evaluator_id: str, environment_digest: str, task_set_digest: str, receipt: Mapping[str, Any] | None = None, observed_ns: int | None = None) -> str:
        plan = self.get_plan(plan_digest)
        if variant not in plan.variants or int(seed) not in plan.seeds:
            raise PermissionError("result is outside preregistered variant/seed matrix")
        if result.variant != variant:
            raise PermissionError("result variant does not match preregistered variant")
        if environment_digest != plan.environment_digest or task_set_digest != plan.task_set_digest:
            raise PermissionError("result environment/task set does not match preregistration")
        rep = ContinualExperimentReplicateRC14(plan_digest, variant, int(seed), result.digest, evaluator_id, environment_digest, task_set_digest)
        attestation = None
        if receipt is not None or self.strict_signed_replicates:
            if receipt is None: raise PermissionError("canonical experiment replicate requires signed evaluator receipt")
            verifier=self.replicate_verifiers.get(evaluator_id)
            if verifier is None: raise PermissionError("untrusted experiment evaluator")
            attestation=ExperimentReplicateAttestationRC14(plan_digest,variant,int(seed),result.digest,evaluator_id,environment_digest,task_set_digest,int(observed_ns or time.time_ns()),receipt)
            if getattr(verifier,"key_id",evaluator_id)!=evaluator_id: raise PermissionError("experiment evaluator_id does not match verifier key identity")
            if not verifier.verify(attestation.receipt,expected_body=attestation.signed_body): raise PermissionError("invalid experiment replicate signature")
        payload = {"replicate": asdict(rep), "result": asdict(result)}
        with ProcessFileLock(self.lock_path, timeout=30.0):
            with self._connect() as c:
                c.execute("BEGIN IMMEDIATE")
                state = c.execute("SELECT seq,head FROM state WHERE singleton=1").fetchone()
                seq = int(state["seq"]) + 1; previous = str(state["head"])
                body = {"schema": "egai-rc14-experiment-protocol-event-v1", "seq": seq, "replicate_digest": rep.digest,
                        "plan_digest": plan_digest, "variant": variant, "seed": int(seed), "result_digest": result.digest,
                        "previous_digest": previous}
                event_digest = sha256_json(body)
                try:
                    c.execute("INSERT INTO replicates(seq,plan_digest,variant,seed,replicate_digest,result_digest,payload_json,previous_digest,event_digest,committed_ns) VALUES(?,?,?,?,?,?,?,?,?,?)",
                              (seq, plan_digest, variant, int(seed), rep.digest, result.digest,
                               json.dumps(payload, sort_keys=True, separators=(",", ":")), previous, event_digest, time.time_ns()))
                except sqlite3.IntegrityError as exc:
                    raise PermissionError("duplicate preregistered replicate") from exc
                if attestation is not None:
                    c.execute("INSERT INTO replicate_attestations(replicate_digest,attestation_digest,payload_json) VALUES(?,?,?)",(rep.digest,attestation.digest,json.dumps(asdict(attestation),sort_keys=True,separators=(",",":"))))
                c.execute("UPDATE state SET seq=?,head=? WHERE singleton=1", (seq, event_digest)); c.commit()
        return rep.digest

    def evaluator_ids(self, plan_digest: str) -> set[str]:
        return {str(json.loads(row["payload_json"])["replicate"]["evaluator_id"]) for row in self._rows(plan_digest)}

    def _verify_replicate_attestation(self, replicate_digest: str) -> bool:
        with self._connect() as c: row=c.execute("SELECT payload_json FROM replicate_attestations WHERE replicate_digest=?",(replicate_digest,)).fetchone()
        if row is None:
            if self.strict_signed_replicates: raise RuntimeError("missing signed experiment replicate attestation")
            return False
        att=ExperimentReplicateAttestationRC14(**json.loads(row["payload_json"])); verifier=self.replicate_verifiers.get(att.evaluator_id)
        if verifier is None or not verifier.verify(att.receipt,expected_body=att.signed_body): raise RuntimeError("experiment replicate attestation signature invalid")
        return True

    def _rows(self, plan_digest: str):
        with self._connect() as c:
            return list(c.execute("SELECT * FROM replicates WHERE plan_digest=? ORDER BY variant,seed", (plan_digest,)))

    def certify(self, plan_digest: str) -> AblationQualificationCertificateRC14:
        plan = self.get_plan(plan_digest)
        rows = self._rows(plan_digest)
        expected = {(v, s) for v in plan.variants for s in plan.seeds}
        present = {(str(r["variant"]), int(r["seed"])) for r in rows}
        complete = present == expected
        reasons: list[str] = []
        if not complete:
            reasons.append("incomplete_preregistered_matrix")
        parsed: dict[str, list[ContinualExperimentResultRC14]] = {v: [] for v in plan.variants}
        digests: list[str] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            result = _result_from_doc(payload["result"])
            if result.digest != row["result_digest"]:
                raise RuntimeError("stored experiment result digest mismatch")
            rep = ContinualExperimentReplicateRC14(**payload["replicate"])
            if rep.digest != row["replicate_digest"]:
                raise RuntimeError("stored replicate digest mismatch")
            if self.strict_signed_replicates: self._verify_replicate_attestation(rep.digest)
            parsed[rep.variant].append(result); digests.append(rep.digest)
        target = parsed["isolated_neural_memory"]
        replay = parsed["replay"]
        def mean(rows, attr):
            return sum(float(getattr(x.metrics, attr)) for x in rows) / len(rows) if rows else 0.0
        target_gain = mean(target, "fresh_task_gain"); replay_gain = mean(replay, "fresh_task_gain")
        inc = target_gain - replay_gain
        mean_forgetting = mean(target, "mean_forgetting")
        worst = max((x.metrics.worst_case_forgetting for x in target), default=0.0)
        repro = mean(target, "reproducibility")
        if target_gain < plan.min_fresh_task_gain: reasons.append("fresh_task_gain")
        if inc < plan.min_incremental_gain_over_replay: reasons.append("incremental_gain_over_replay")
        if mean_forgetting > plan.max_mean_forgetting: reasons.append("mean_forgetting")
        if worst > plan.max_worst_forgetting: reasons.append("worst_case_forgetting")
        if repro < plan.min_reproducibility: reasons.append("reproducibility")
        for result in target:
            if not result.eligible_for_promotion_evidence: reasons.append("target_replicate_not_eligible"); break
            if result.metrics.falsification_survival < plan.min_falsification_survival: reasons.append("falsification_survival"); break
            if plan.require_zero_security_regressions and result.metrics.security_regressions: reasons.append("security_regressions"); break
            if plan.require_zero_unauthorized_writes and result.metrics.unauthorized_writes: reasons.append("unauthorized_writes"); break
        return AblationQualificationCertificateRC14(
            plan_digest=plan.digest, candidate_digest=plan.candidate_digest, replicate_digests=tuple(sorted(digests)),
            target_variant="isolated_neural_memory", target_mean_fresh_gain=target_gain, replay_mean_fresh_gain=replay_gain,
            incremental_gain_over_replay=inc, target_mean_forgetting=mean_forgetting, target_worst_forgetting=worst,
            target_mean_reproducibility=repro, complete=complete, eligible=complete and not reasons, reasons=tuple(dict.fromkeys(reasons)),
        )

    def contains_event_head(self, digest: str) -> bool:
        require_digest(digest, field_name="experiment_protocol_head_digest")
        if digest == "sha256:" + GENESIS:
            return True
        with self._connect() as c:
            return c.execute("SELECT 1 FROM replicates WHERE event_digest=?", (digest,)).fetchone() is not None

    def head_digest(self) -> str:
        with self._connect() as c:
            row = c.execute("SELECT head FROM state WHERE singleton=1").fetchone()
        head = str(row["head"])
        if head == GENESIS:
            return "sha256:" + head
        require_digest(head, field_name="experiment_protocol_head_digest")
        return head

    def verify(self) -> dict[str, int]:
        with self._connect() as c:
            if str(c.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
                raise RuntimeError("experiment protocol sqlite integrity failure")
            plans = list(c.execute("SELECT * FROM plans ORDER BY plan_digest")); rows = list(c.execute("SELECT * FROM replicates ORDER BY seq")); attestations=list(c.execute("SELECT * FROM replicate_attestations ORDER BY replicate_digest"))
            state = c.execute("SELECT seq,head FROM state WHERE singleton=1").fetchone()
        for p in plans:
            self.get_plan(str(p["plan_digest"]))
        head = GENESIS
        for i, row in enumerate(rows, 1):
            if int(row["seq"]) != i or row["previous_digest"] != head:
                raise RuntimeError("experiment protocol chain discontinuity")
            payload = json.loads(row["payload_json"])
            rep = ContinualExperimentReplicateRC14(**payload["replicate"])
            result = _result_from_doc(payload["result"])
            if rep.digest != row["replicate_digest"] or result.digest != row["result_digest"] or rep.result_digest != result.digest:
                raise RuntimeError("experiment protocol payload digest mismatch")
            if self.strict_signed_replicates: self._verify_replicate_attestation(rep.digest)
            body = {"schema": "egai-rc14-experiment-protocol-event-v1", "seq": i, "replicate_digest": rep.digest,
                    "plan_digest": rep.plan_digest, "variant": rep.variant, "seed": rep.seed, "result_digest": result.digest,
                    "previous_digest": head}
            expected = sha256_json(body)
            if expected != row["event_digest"]:
                raise RuntimeError("experiment protocol event digest mismatch")
            self.get_plan(rep.plan_digest)
            head = expected
        if int(state["seq"]) != len(rows) or str(state["head"]) != head:
            raise RuntimeError("experiment protocol head mismatch")
        if self.strict_signed_replicates and len(attestations)!=len(rows): raise RuntimeError("strict experiment protocol requires one signed attestation per replicate")
        result={"plans":len(plans),"replicates":len(rows)}
        if self.strict_signed_replicates: result["signed_replicates"]=len(attestations)
        return result
