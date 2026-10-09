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
from .experiment_protocol import PreregisteredContinualExperimentStoreRC14, _result_from_doc
from .models import require_digest

GENESIS = "0" * 64
REPRODUCTION_VARIANTS = ("replay", "isolated_neural_memory")


@dataclass(frozen=True)
class IndependentReproductionPlanRC14:
    reproduction_id: str
    source_plan_digest: str
    source_candidate_digest: str
    source_ablation_certificate_digest: str
    source_protocol_head_digest: str
    base_model_digest: str
    tokenizer_digest: str
    reproduction_environment_digest: str
    reproduction_task_set_digest: str
    hidden_task_commitment_digest: str
    seeds: tuple[int, ...] = (101, 211)
    min_distinct_reproducers: int = 2
    min_fresh_task_gain: float = 0.0
    min_incremental_gain_over_replay: float = 0.0
    max_mean_forgetting: float = 0.05
    max_worst_forgetting: float = 0.10
    min_reproducibility: float = 0.90
    min_falsification_survival: float = 1.0
    require_zero_security_regressions: bool = True
    require_zero_unauthorized_writes: bool = True
    variants: tuple[str, ...] = REPRODUCTION_VARIANTS
    schema: str = "egai-rc14-independent-reproduction-plan-v1"

    def __post_init__(self) -> None:
        if not self.reproduction_id:
            raise ValueError("reproduction_id is required")
        for name in (
            "source_plan_digest", "source_candidate_digest", "source_ablation_certificate_digest",
            "source_protocol_head_digest", "base_model_digest", "tokenizer_digest",
            "reproduction_environment_digest", "reproduction_task_set_digest", "hidden_task_commitment_digest",
        ):
            require_digest(getattr(self, name), field_name=name)
        if tuple(self.variants) != REPRODUCTION_VARIANTS:
            raise ValueError("independent reproduction requires the replay/isolated-neural-memory paired ladder")
        seeds = tuple(int(x) for x in self.seeds)
        if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(x < 0 for x in seeds):
            raise ValueError("at least two unique non-negative reproduction seeds are required")
        if int(self.min_distinct_reproducers) < 2 or int(self.min_distinct_reproducers) > len(seeds):
            raise ValueError("min_distinct_reproducers must be between 2 and number of seeds")
        for name in (
            "min_fresh_task_gain", "min_incremental_gain_over_replay", "max_mean_forgetting",
            "max_worst_forgetting", "min_reproducibility", "min_falsification_survival",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.max_mean_forgetting < 0 or self.max_worst_forgetting < 0:
            raise ValueError("forgetting thresholds must be non-negative")
        if not 0.0 <= self.min_reproducibility <= 1.0 or not 0.0 <= self.min_falsification_survival <= 1.0:
            raise ValueError("probability thresholds must be in [0,1]")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "variants", tuple(str(x) for x in self.variants))

    @property
    def digest(self) -> str:
        d = asdict(self); d["seeds"] = list(self.seeds); d["variants"] = list(self.variants)
        return sha256_json(d)


@dataclass(frozen=True)
class IndependentReproductionReplicateRC14:
    plan_digest: str
    variant: str
    seed: int
    result_digest: str
    reproducer_id: str
    environment_digest: str
    task_set_digest: str
    observed_ns: int
    receipt: Mapping[str, Any]
    schema: str = "egai-rc14-independent-reproduction-replicate-v1"

    def __post_init__(self) -> None:
        for name in ("plan_digest", "result_digest", "environment_digest", "task_set_digest"):
            require_digest(getattr(self, name), field_name=name)
        if self.variant not in REPRODUCTION_VARIANTS:
            raise ValueError("invalid reproduction variant")
        if int(self.seed) < 0 or not self.reproducer_id or int(self.observed_ns) <= 0:
            raise ValueError("seed, reproducer_id and observed_ns are required")
        if not isinstance(self.receipt, Mapping) or not isinstance(self.receipt.get("body"), Mapping):
            raise ValueError("signed reproduction receipt required")
        object.__setattr__(self, "receipt", dict(self.receipt))

    @property
    def signed_body(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "plan_digest": self.plan_digest, "variant": self.variant,
            "seed": int(self.seed), "result_digest": self.result_digest, "reproducer_id": self.reproducer_id,
            "environment_digest": self.environment_digest, "task_set_digest": self.task_set_digest,
            "observed_ns": int(self.observed_ns),
        }

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class IndependentReproductionCertificateRC14:
    plan_digest: str
    source_plan_digest: str
    candidate_digest: str
    source_ablation_certificate_digest: str
    replicate_digests: tuple[str, ...]
    reproducer_ids: tuple[str, ...]
    isolated_mean_fresh_gain: float
    replay_mean_fresh_gain: float
    incremental_gain_over_replay: float
    isolated_mean_forgetting: float
    isolated_worst_forgetting: float
    isolated_mean_reproducibility: float
    complete: bool
    eligible: bool
    reasons: tuple[str, ...]
    schema: str = "egai-rc14-independent-reproduction-certificate-v1"

    def __post_init__(self) -> None:
        for name in ("plan_digest", "source_plan_digest", "candidate_digest", "source_ablation_certificate_digest"):
            require_digest(getattr(self, name), field_name=name)
        for x in self.replicate_digests: require_digest(x, field_name="replicate_digest")
        values = (self.isolated_mean_fresh_gain, self.replay_mean_fresh_gain, self.incremental_gain_over_replay,
                  self.isolated_mean_forgetting, self.isolated_worst_forgetting, self.isolated_mean_reproducibility)
        if any(not math.isfinite(float(x)) for x in values): raise ValueError("certificate metrics must be finite")
        if len(set(self.reproducer_ids)) != len(self.reproducer_ids): raise ValueError("reproducer_ids must be unique")
        object.__setattr__(self,"replicate_digests",tuple(str(x) for x in self.replicate_digests))
        object.__setattr__(self,"reproducer_ids",tuple(str(x) for x in self.reproducer_ids))
        object.__setattr__(self,"reasons",tuple(str(x) for x in self.reasons))

    @property
    def digest(self) -> str:
        d=asdict(self); d["replicate_digests"]=list(self.replicate_digests); d["reproducer_ids"]=list(self.reproducer_ids); d["reasons"]=list(self.reasons)
        return sha256_json(d)


class IndependentReproductionStoreRC14:
    def __init__(self, root: str | os.PathLike, *, source_protocol: PreregisteredContinualExperimentStoreRC14,
                 reproducer_verifiers: Mapping[str, Any]):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.db_path=self.root/"independent_reproduction.sqlite3"; self.lock_path=self.root/"independent_reproduction.lock"
        self.source_protocol=source_protocol; self.reproducer_verifiers=dict(reproducer_verifiers); self._init_db()

    def _connect(self):
        c=sqlite3.connect(self.db_path,timeout=30.0); c.row_factory=sqlite3.Row; c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA synchronous=FULL"); return c

    def _init_db(self):
        with self._connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS plans(plan_digest TEXT PRIMARY KEY,payload_json TEXT NOT NULL,sealed_ns INTEGER NOT NULL)")
            c.execute("CREATE TABLE IF NOT EXISTS state(singleton INTEGER PRIMARY KEY CHECK(singleton=1),seq INTEGER NOT NULL,head TEXT NOT NULL)")
            c.execute("""CREATE TABLE IF NOT EXISTS replicates(seq INTEGER PRIMARY KEY,plan_digest TEXT NOT NULL,variant TEXT NOT NULL,seed INTEGER NOT NULL,replicate_digest TEXT NOT NULL UNIQUE,result_digest TEXT NOT NULL,reproducer_id TEXT NOT NULL,payload_json TEXT NOT NULL,previous_digest TEXT NOT NULL,event_digest TEXT NOT NULL UNIQUE,committed_ns INTEGER NOT NULL,UNIQUE(plan_digest,variant,seed),FOREIGN KEY(plan_digest) REFERENCES plans(plan_digest))""")
            c.execute("INSERT OR IGNORE INTO state(singleton,seq,head) VALUES(1,0,?)",(GENESIS,)); c.commit()

    def _source_evaluator_ids(self, plan_digest):
        ids=set()
        for row in self.source_protocol._rows(plan_digest):
            rid=str(json.loads(row["payload_json"])["replicate"].get("evaluator_id",""))
            if rid: ids.add(rid)
        return ids

    def seal(self, plan: IndependentReproductionPlanRC14) -> str:
        source=self.source_protocol.get_plan(plan.source_plan_digest); cert=self.source_protocol.certify(source.digest)
        if source.candidate_digest!=plan.source_candidate_digest: raise PermissionError("reproduction candidate does not match source experiment")
        if cert.digest!=plan.source_ablation_certificate_digest or not cert.eligible: raise PermissionError("source ablation certificate is not eligible or does not match")
        if not self.source_protocol.contains_event_head(plan.source_protocol_head_digest): raise PermissionError("reproduction plan references unknown source protocol head")
        if source.base_model_digest!=plan.base_model_digest or source.tokenizer_digest!=plan.tokenizer_digest: raise PermissionError("reproduction model/tokenizer does not match source experiment")
        if source.environment_digest==plan.reproduction_environment_digest: raise PermissionError("independent reproduction requires a different environment digest")
        if source.task_set_digest==plan.reproduction_task_set_digest: raise PermissionError("independent reproduction requires a different task-set digest")
        if plan.min_fresh_task_gain<source.min_fresh_task_gain or plan.min_incremental_gain_over_replay<source.min_incremental_gain_over_replay: raise PermissionError("reproduction gain threshold is weaker than source")
        if plan.max_mean_forgetting>source.max_mean_forgetting or plan.max_worst_forgetting>source.max_worst_forgetting: raise PermissionError("reproduction forgetting threshold is weaker than source")
        if plan.min_reproducibility<source.min_reproducibility: raise PermissionError("reproduction reproducibility threshold is weaker than source")
        if plan.min_falsification_survival<source.min_falsification_survival: raise PermissionError("reproduction falsification threshold is weaker than source")
        if source.require_zero_security_regressions and not plan.require_zero_security_regressions: raise PermissionError("reproduction cannot relax security-regression requirement")
        if source.require_zero_unauthorized_writes and not plan.require_zero_unauthorized_writes: raise PermissionError("reproduction cannot relax unauthorized-write requirement")
        payload=json.dumps(asdict(plan),sort_keys=True,separators=(",",":"))
        with ProcessFileLock(self.lock_path,timeout=30.0):
            with self._connect() as c:
                c.execute("BEGIN IMMEDIATE"); row=c.execute("SELECT payload_json FROM plans WHERE plan_digest=?",(plan.digest,)).fetchone()
                if row is not None and str(row["payload_json"])!=payload: raise RuntimeError("independent reproduction plan digest collision")
                c.execute("INSERT OR IGNORE INTO plans(plan_digest,payload_json,sealed_ns) VALUES(?,?,?)",(plan.digest,payload,time.time_ns())); c.commit()
        return plan.digest

    def get_plan(self,digest):
        require_digest(digest,field_name="reproduction_plan_digest")
        with self._connect() as c: row=c.execute("SELECT payload_json FROM plans WHERE plan_digest=?",(digest,)).fetchone()
        if row is None: raise KeyError(digest)
        doc=json.loads(row["payload_json"]); doc["seeds"]=tuple(doc["seeds"]); doc["variants"]=tuple(doc["variants"]); plan=IndependentReproductionPlanRC14(**doc)
        if plan.digest!=digest: raise RuntimeError("stored reproduction plan digest mismatch")
        return plan

    def record_result(self, plan_digest, *, variant, seed, result, reproducer_id, receipt, observed_ns=None):
        plan=self.get_plan(plan_digest)
        if variant not in plan.variants or int(seed) not in plan.seeds: raise PermissionError("reproduction result is outside preregistered matrix")
        if result.variant!=variant: raise PermissionError("reproduction result variant mismatch")
        if reproducer_id in self._source_evaluator_ids(plan.source_plan_digest): raise PermissionError("source evaluator identity cannot serve as independent reproducer")
        verifier=self.reproducer_verifiers.get(reproducer_id)
        if verifier is None: raise PermissionError("untrusted independent reproducer")
        rep=IndependentReproductionReplicateRC14(plan.digest,variant,int(seed),result.digest,reproducer_id,plan.reproduction_environment_digest,plan.reproduction_task_set_digest,int(observed_ns or time.time_ns()),receipt)
        if getattr(verifier,"key_id",reproducer_id)!=reproducer_id: raise PermissionError("reproducer identity does not match verifier key")
        if not verifier.verify(dict(rep.receipt),expected_body=rep.signed_body): raise PermissionError("invalid independent reproduction signature")
        payload={"replicate":asdict(rep),"result":asdict(result)}
        with ProcessFileLock(self.lock_path,timeout=30.0):
            with self._connect() as c:
                c.execute("BEGIN IMMEDIATE"); st=c.execute("SELECT seq,head FROM state WHERE singleton=1").fetchone(); seq=int(st["seq"])+1; prev=str(st["head"])
                body={"schema":"egai-rc14-independent-reproduction-event-v1","seq":seq,"plan_digest":plan.digest,"replicate_digest":rep.digest,"variant":variant,"seed":int(seed),"result_digest":result.digest,"reproducer_id":reproducer_id,"previous_digest":prev}; ev=sha256_json(body)
                try: c.execute("INSERT INTO replicates(seq,plan_digest,variant,seed,replicate_digest,result_digest,reproducer_id,payload_json,previous_digest,event_digest,committed_ns) VALUES(?,?,?,?,?,?,?,?,?,?,?)",(seq,plan.digest,variant,int(seed),rep.digest,result.digest,reproducer_id,json.dumps(payload,sort_keys=True,separators=(",",":")),prev,ev,time.time_ns()))
                except sqlite3.IntegrityError as exc: raise PermissionError("duplicate independent reproduction replicate") from exc
                c.execute("UPDATE state SET seq=?,head=? WHERE singleton=1",(seq,ev)); c.commit()
        return rep.digest

    def _rows(self,plan_digest):
        with self._connect() as c: return list(c.execute("SELECT * FROM replicates WHERE plan_digest=? ORDER BY variant,seed",(plan_digest,)))

    def certify(self,plan_digest):
        plan=self.get_plan(plan_digest); rows=self._rows(plan.digest); expected={(v,s) for v in plan.variants for s in plan.seeds}; present={(str(r["variant"]),int(r["seed"])) for r in rows}; complete=present==expected; reasons=[]
        if not complete: reasons.append("incomplete_reproduction_matrix")
        parsed={v:[] for v in plan.variants}; digests=[]; ids=set(); source_ids=self._source_evaluator_ids(plan.source_plan_digest)
        for row in rows:
            payload=json.loads(row["payload_json"]); rep=IndependentReproductionReplicateRC14(**dict(payload["replicate"])); result=_result_from_doc(payload["result"])
            verifier=self.reproducer_verifiers.get(rep.reproducer_id)
            if verifier is None or not verifier.verify(rep.receipt,expected_body=rep.signed_body): raise RuntimeError("independent reproduction signature verification failed")
            if rep.digest!=row["replicate_digest"] or result.digest!=row["result_digest"] or rep.result_digest!=result.digest: raise RuntimeError("independent reproduction payload digest mismatch")
            if rep.reproducer_id in source_ids: reasons.append("reproducer_not_independent_from_source")
            parsed[rep.variant].append(result); digests.append(rep.digest); ids.add(rep.reproducer_id)
        if len(ids)<plan.min_distinct_reproducers: reasons.append("insufficient_distinct_reproducers")
        isolated=parsed["isolated_neural_memory"]; replay=parsed["replay"]
        def mean(items, attr):
            return sum(float(getattr(x.metrics,attr)) for x in items)/len(items) if items else 0.0
        ig=mean(isolated,"fresh_task_gain"); rg=mean(replay,"fresh_task_gain"); inc=ig-rg; mf=mean(isolated,"mean_forgetting"); wf=max((x.metrics.worst_case_forgetting for x in isolated),default=0.0); repro=mean(isolated,"reproducibility")
        if ig<plan.min_fresh_task_gain: reasons.append("fresh_task_gain")
        if inc<plan.min_incremental_gain_over_replay: reasons.append("incremental_gain_over_replay")
        if mf>plan.max_mean_forgetting: reasons.append("mean_forgetting")
        if wf>plan.max_worst_forgetting: reasons.append("worst_case_forgetting")
        if repro<plan.min_reproducibility: reasons.append("reproducibility")
        for r in isolated:
            if not r.eligible_for_promotion_evidence: reasons.append("reproduction_target_not_eligible"); break
            if r.metrics.falsification_survival<plan.min_falsification_survival: reasons.append("falsification_survival"); break
            if plan.require_zero_security_regressions and r.metrics.security_regressions: reasons.append("security_regressions"); break
            if plan.require_zero_unauthorized_writes and r.metrics.unauthorized_writes: reasons.append("unauthorized_writes"); break
        return IndependentReproductionCertificateRC14(plan.digest,plan.source_plan_digest,plan.source_candidate_digest,plan.source_ablation_certificate_digest,tuple(sorted(digests)),tuple(sorted(ids)),ig,rg,inc,mf,wf,repro,complete,complete and not reasons,tuple(dict.fromkeys(reasons)))

    def contains_event_head(self,digest):
        require_digest(digest,field_name="reproduction_protocol_head_digest")
        if digest=="sha256:"+GENESIS: return True
        with self._connect() as c: return c.execute("SELECT 1 FROM replicates WHERE event_digest=?",(digest,)).fetchone() is not None

    def head_digest(self):
        with self._connect() as c: head=str(c.execute("SELECT head FROM state WHERE singleton=1").fetchone()["head"])
        return "sha256:"+head if head==GENESIS else require_digest(head,field_name="reproduction_protocol_head_digest")

    def verify(self):
        with self._connect() as c:
            if str(c.execute("PRAGMA integrity_check").fetchone()[0])!="ok": raise RuntimeError("independent reproduction sqlite integrity failure")
            plans=list(c.execute("SELECT * FROM plans ORDER BY plan_digest")); rows=list(c.execute("SELECT * FROM replicates ORDER BY seq")); state=c.execute("SELECT seq,head FROM state WHERE singleton=1").fetchone()
        for row in plans: self.seal(self.get_plan(str(row["plan_digest"])))
        head=GENESIS
        for i,row in enumerate(rows,1):
            if int(row["seq"])!=i or str(row["previous_digest"])!=head: raise RuntimeError("independent reproduction chain discontinuity")
            payload=json.loads(row["payload_json"]); rep=IndependentReproductionReplicateRC14(**dict(payload["replicate"])); result=_result_from_doc(payload["result"]); verifier=self.reproducer_verifiers.get(rep.reproducer_id)
            if verifier is None or not verifier.verify(rep.receipt,expected_body=rep.signed_body): raise RuntimeError("independent reproduction signature invalid")
            if rep.digest!=row["replicate_digest"] or result.digest!=row["result_digest"] or rep.result_digest!=result.digest: raise RuntimeError("independent reproduction stored payload mismatch")
            body={"schema":"egai-rc14-independent-reproduction-event-v1","seq":i,"plan_digest":rep.plan_digest,"replicate_digest":rep.digest,"variant":rep.variant,"seed":int(rep.seed),"result_digest":result.digest,"reproducer_id":rep.reproducer_id,"previous_digest":head}; ev=sha256_json(body)
            if ev!=row["event_digest"]: raise RuntimeError("independent reproduction event digest mismatch")
            head=ev
        if int(state["seq"])!=len(rows) or str(state["head"])!=head: raise RuntimeError("independent reproduction head mismatch")
        return {"plans":len(plans),"replicates":len(rows)}
