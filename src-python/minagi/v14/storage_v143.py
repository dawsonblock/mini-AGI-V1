from __future__ import annotations

import json
import time
from typing import Iterable

from egai.common.canonical import validate_digest
from .storage_v142 import GovernanceDBV142


class GovernanceDBV143(GovernanceDBV142):
    """Alpha3 governance DB with exact-content, one-use mutation capabilities."""
    def __init__(self,path):
        super().__init__(path)
        cols={r[1] for r in self.conn.execute("PRAGMA table_info(promotions)")}
        for name,decl in (
            ("build_digest","TEXT"),("evaluation_digest","TEXT"),("runtime_manifest_digest","TEXT"),
            ("artifact_root_digest","TEXT"),("status","TEXT NOT NULL DEFAULT 'ISSUED'"),("consumed_at","REAL"),
        ):
            if name not in cols:
                self.conn.execute(f"ALTER TABLE promotions ADD COLUMN {name} {decl}")
        qcols={r[1] for r in self.conn.execute("PRAGMA table_info(qualifications)")}
        for name,decl in (("raw_results_digest","TEXT"),("derived_metrics_digest","TEXT")):
            if name not in qcols:
                self.conn.execute(f"ALTER TABLE qualifications ADD COLUMN {name} {decl}")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS mutation_commitments(
          authorization_digest TEXT NOT NULL,
          mutation_scope TEXT NOT NULL,
          target_digest TEXT NOT NULL,
          consumed_at REAL,
          PRIMARY KEY(authorization_digest,mutation_scope,target_digest),
          FOREIGN KEY(authorization_digest) REFERENCES promotions(authorization_digest) ON DELETE RESTRICT
        );
        CREATE INDEX IF NOT EXISTS idx_mutation_commitments_open
          ON mutation_commitments(authorization_digest,mutation_scope,consumed_at);
        """)


    def record_qualification_v143(self, *, candidate_digest:str, evaluation_digest:str, qualification_digest:str, decision:str,
                                  qualifier_key_id:str, qualifier_generation:int, authority_generation:int, policy_generation:int,
                                  raw_results_digest:str, derived_metrics_digest:str, actor:str="qualification-authority"):
        candidate_digest,evaluation_digest,qualification_digest,raw_results_digest,derived_metrics_digest=map(validate_digest,
            (candidate_digest,evaluation_digest,qualification_digest,raw_results_digest,derived_metrics_digest))
        decision=str(decision).upper()
        if decision not in {"PROMOTE","REJECT"}: raise ValueError("invalid qualification decision")
        with self.transaction() as db:
            e=db.execute("SELECT candidate_digest,metrics_digest FROM evaluations WHERE evaluation_digest=?",(evaluation_digest,)).fetchone()
            if e is None or e["candidate_digest"]!=candidate_digest: raise PermissionError("qualification evaluation/candidate mismatch")
            if e["metrics_digest"]!=raw_results_digest: raise PermissionError("qualification raw-results/evaluation mismatch")
            db.execute("INSERT INTO qualifications(qualification_digest,candidate_digest,evaluation_digest,decision,qualifier_key_id,qualifier_generation,authority_generation,policy_generation,metrics_digest,created_at,raw_results_digest,derived_metrics_digest) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                       (qualification_digest,candidate_digest,evaluation_digest,decision,str(qualifier_key_id),int(qualifier_generation),int(authority_generation),int(policy_generation),derived_metrics_digest,time.time(),raw_results_digest,derived_metrics_digest))
            self._audit(db,event_type="candidate.qualified.v143",object_digest=qualification_digest,actor=actor,
                        payload={"candidate_digest":candidate_digest,"evaluation_digest":evaluation_digest,"decision":decision,
                                 "raw_results_digest":raw_results_digest,"derived_metrics_digest":derived_metrics_digest})

    def record_promotion_v143(self, *, candidate_digest:str, build_digest:str, evaluation_digest:str,
                              qualification_digest:str, authorization_digest:str, runtime_manifest_digest:str,
                              artifact_root_digest:str, authority_generation:int, policy_generation:int,
                              signer_key_id:str, mutation_commitments:Iterable[tuple[str,str]], actor:str="promotion-authority"):
        vals=[validate_digest(x) for x in (candidate_digest,build_digest,evaluation_digest,qualification_digest,
                                           authorization_digest,runtime_manifest_digest,artifact_root_digest)]
        candidate_digest,build_digest,evaluation_digest,qualification_digest,authorization_digest,runtime_manifest_digest,artifact_root_digest=vals
        commitments=tuple(sorted(set((str(s),validate_digest(t)) for s,t in mutation_commitments)))
        if not commitments: raise ValueError("mutation commitments required")
        with self.transaction() as db:
            q=db.execute("SELECT * FROM qualifications WHERE qualification_digest=?",(qualification_digest,)).fetchone()
            if q is None or q["candidate_digest"] != candidate_digest or q["evaluation_digest"] != evaluation_digest:
                raise PermissionError("promotion qualification chain mismatch")
            if q["decision"] != "PROMOTE": raise PermissionError("rejected qualification cannot be promoted")
            e=db.execute("SELECT build_digest FROM evaluations WHERE evaluation_digest=?",(evaluation_digest,)).fetchone()
            if e is None or e["build_digest"] != build_digest: raise PermissionError("promotion evaluation/build mismatch")
            scopes=sorted({s for s,_ in commitments})
            db.execute("INSERT INTO promotions(authorization_digest,candidate_digest,qualification_digest,authority_generation,policy_generation,signer_key_id,mutation_scopes_json,created_at,build_digest,evaluation_digest,runtime_manifest_digest,artifact_root_digest,status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (authorization_digest,candidate_digest,qualification_digest,int(authority_generation),int(policy_generation),str(signer_key_id),json.dumps(scopes),time.time(),build_digest,evaluation_digest,runtime_manifest_digest,artifact_root_digest,"ISSUED"))
            db.executemany("INSERT INTO mutation_commitments(authorization_digest,mutation_scope,target_digest) VALUES(?,?,?)",
                           [(authorization_digest,s,t) for s,t in commitments])
            self._audit(db,event_type="candidate.content_authorized",object_digest=authorization_digest,actor=actor,
                        payload={"candidate_digest":candidate_digest,"build_digest":build_digest,"evaluation_digest":evaluation_digest,
                                 "qualification_digest":qualification_digest,"runtime_manifest_digest":runtime_manifest_digest,
                                 "artifact_root_digest":artifact_root_digest,"mutation_commitments":commitments})

    def require_mutation(self, authorization_digest:str, *, mutation_scope:str, target_digest:str):
        authorization_digest=validate_digest(authorization_digest); target_digest=validate_digest(target_digest)
        row=self.conn.execute("SELECT p.*,m.consumed_at AS mutation_consumed_at FROM promotions p JOIN mutation_commitments m ON m.authorization_digest=p.authorization_digest WHERE p.authorization_digest=? AND m.mutation_scope=? AND m.target_digest=?",
                              (authorization_digest,str(mutation_scope),target_digest)).fetchone()
        if row is None: raise PermissionError("authorization does not bind this exact mutation")
        if row["mutation_consumed_at"] is not None: raise PermissionError("mutation authorization already consumed")
        return row

    def consume_mutation(self, authorization_digest:str, *, mutation_scope:str, target_digest:str, actor:str="persistence-gateway"):
        authorization_digest=validate_digest(authorization_digest); target_digest=validate_digest(target_digest)
        with self.transaction() as db:
            cur=db.execute("UPDATE mutation_commitments SET consumed_at=? WHERE authorization_digest=? AND mutation_scope=? AND target_digest=? AND consumed_at IS NULL",
                           (time.time(),authorization_digest,str(mutation_scope),target_digest))
            if cur.rowcount != 1: raise PermissionError("mutation authorization absent or already consumed")
            self._audit(db,event_type="mutation.consumed",object_digest=target_digest,actor=actor,
                        payload={"authorization_digest":authorization_digest,"mutation_scope":str(mutation_scope)})

    def activate_v143(self, *, candidate_digest:str, authorization_digest:str, runtime_manifest_digest:str,
                      activation_digest:str, actor:str="persistence-gateway"):
        candidate_digest,authorization_digest,runtime_manifest_digest,activation_digest=map(validate_digest,
            (candidate_digest,authorization_digest,runtime_manifest_digest,activation_digest))
        with self.transaction() as db:
            p=db.execute("SELECT * FROM promotions WHERE authorization_digest=?",(authorization_digest,)).fetchone()
            if p is None or p["candidate_digest"] != candidate_digest: raise PermissionError("activation authorization/candidate mismatch")
            if p["runtime_manifest_digest"] != runtime_manifest_digest: raise PermissionError("authorization not bound to runtime manifest")
            cur=db.execute("UPDATE mutation_commitments SET consumed_at=? WHERE authorization_digest=? AND mutation_scope='runtime.activate' AND target_digest=? AND consumed_at IS NULL",
                           (time.time(),authorization_digest,runtime_manifest_digest))
            if cur.rowcount != 1: raise PermissionError("runtime activation authorization absent or already consumed")
            head=db.execute("SELECT runtime_manifest_digest FROM runtime_head WHERE singleton=1").fetchone()
            previous=None if head is None else str(head["runtime_manifest_digest"])
            db.execute("INSERT INTO activations(activation_digest,candidate_digest,authorization_digest,runtime_manifest_digest,previous_runtime_manifest_digest,created_at) VALUES(?,?,?,?,?,?)",
                       (activation_digest,candidate_digest,authorization_digest,runtime_manifest_digest,previous,time.time()))
            db.execute("INSERT INTO runtime_head(singleton,runtime_manifest_digest,activation_digest,updated_at) VALUES(1,?,?,?) ON CONFLICT(singleton) DO UPDATE SET runtime_manifest_digest=excluded.runtime_manifest_digest,activation_digest=excluded.activation_digest,updated_at=excluded.updated_at",
                       (runtime_manifest_digest,activation_digest,time.time()))
            db.execute("UPDATE promotions SET status='CONSUMED',consumed_at=? WHERE authorization_digest=?",(time.time(),authorization_digest))
            self._audit(db,event_type="runtime.content_activated",object_digest=activation_digest,actor=actor,
                        payload={"candidate_digest":candidate_digest,"authorization_digest":authorization_digest,
                                 "runtime_manifest_digest":runtime_manifest_digest,"previous_runtime_manifest_digest":previous})
