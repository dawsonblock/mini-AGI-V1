from __future__ import annotations

from dataclasses import asdict, dataclass
import json,time
from pathlib import Path
from typing import Any,Mapping

from egai.common.canonical import digest,validate_digest
from .models import LearningProposalV14,LearningMechanism,PermanenceLevel
from .policy import GovernancePolicyV14
from .storage import ImmutableCAS
from .storage_v143 import GovernanceDBV143
from .evidence_v143 import EvidenceStrengthProofV143,EvidenceStrengthValidatorV143
from .evaluation_v143 import EvaluationCaseResultV143,SignedEvaluationBundleV143,EvaluationValidatorV143
from .qualification_v143 import QualificationRecordV143,QualificationValidatorV143
from .authority_v143 import MutationCommitmentV143,PromotionAuthorizationV143,PromotionAuthorizationValidatorV143,AuthorizationGateV143

RUNTIME_COMPONENT_SCHEMAS_V143={
 "production_identity_digest":"mini-agi-v14.1-alpha3-production-identity-v1",
 "belief_snapshot_digest":"mini-agi-v14.1-alpha3-belief-snapshot-v1",
 "skill_snapshot_digest":"mini-agi-v14.1-alpha3-skill-snapshot-v1",
 "adapter_set_digest":"mini-agi-v14.1-alpha3-adapter-set-v1",
 "routing_policy_digest":"mini-agi-v14.1-alpha3-routing-policy-v1",
 "execution_runtime_digest":"mini-agi-v14.1-alpha3-execution-runtime-v1",
}

MECHANISM_SCOPES={
 LearningMechanism.BELIEF_UPDATE:("belief.promote","runtime.activate"),
 LearningMechanism.SKILL:("skill.activate","runtime.activate"),
 LearningMechanism.ROUTING_POLICY:("routing.activate","runtime.activate"),
 LearningMechanism.COMPOSITION:("routing.activate","runtime.activate"),
 LearningMechanism.SPARSE_EPISODIC_ADAPTER:("adapter.activate","runtime.activate"),
 LearningMechanism.LORA:("adapter.activate","runtime.activate"),
 LearningMechanism.REPLAY_LORA:("adapter.activate","runtime.activate"),
 LearningMechanism.SELECTIVE_DECORRELATION_ADAPTER:("adapter.activate","runtime.activate"),
 LearningMechanism.ADAPTER_COMPOSITION:("adapter.activate","runtime.activate"),
 LearningMechanism.EPISODIC_STORE:("runtime.activate",),
 LearningMechanism.NONE:("runtime.activate",),
}

def derive_mutation_scopes(proposal:LearningProposalV14)->tuple[str,...]:
    scopes=MECHANISM_SCOPES.get(proposal.mechanism,("runtime.activate",))
    p=proposal.coordinates.permanence
    if "adapter.activate" in scopes and p < PermanenceLevel.L6_ISOLATED_NEURAL_MEMORY:
        raise PermissionError("adapter activation cannot be granted below L6")
    if "skill.activate" in scopes and p != PermanenceLevel.L4_REUSABLE_SKILL:
        raise PermissionError("skill activation requires L4 reusable-skill permanence")
    if "belief.promote" in scopes and p != PermanenceLevel.L3_SEMANTIC_BELIEF:
        raise PermissionError("belief promotion requires L3 semantic-belief permanence")
    return tuple(scopes)

@dataclass(frozen=True)
class RuntimeManifestV143:
    candidate_digest:str
    build_digest:str
    evaluation_digest:str
    qualification_digest:str
    artifact_root_digest:str
    production_identity_digest:str
    belief_snapshot_digest:str
    skill_snapshot_digest:str
    adapter_set_digest:str
    routing_policy_digest:str
    execution_runtime_digest:str
    authority_generation:int
    policy_generation:int
    created_at:float
    schema:str="mini-agi-v14.1-alpha3-runtime-manifest-v1"
    def __post_init__(self):
        for k,v in asdict(self).items():
            if k.endswith("_digest"): validate_digest(v)
    @property
    def digest(self): return digest(self)

class GovernedContinualRuntimeV143:
    """Canonical alpha3 runtime. It holds public validators and authority clients, never private signing keys."""
    def __init__(self,root:str|Path,*,policy:GovernancePolicyV14,evidence_client,evidence_validator:EvidenceStrengthValidatorV143,
                 evaluation_client,evaluation_validator:EvaluationValidatorV143,qualification_client,qualification_validator:QualificationValidatorV143,
                 promotion_client,promotion_validator:PromotionAuthorizationValidatorV143,audit_client=None,audit_verifier=None,trusted_audit_key_ids=(),audit_anchor=None,
                 cas=None,governance_db=None,production_mode:bool=False):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.policy=policy
        self.cas=cas or ImmutableCAS(self.root/"cas");self.db=governance_db or GovernanceDBV143(self.root/"governance.sqlite3")
        if production_mode:
            for name,client in (("evidence",evidence_client),("evaluation",evaluation_client),("qualification",qualification_client),("promotion",promotion_client)):
                if getattr(client,"is_local",True): raise RuntimeError(f"production mode requires out-of-process {name} authority client")
            if audit_client is not None and getattr(audit_client,"is_local",True): raise RuntimeError("production mode requires out-of-process audit authority client")
        self.production_mode=bool(production_mode)
        self.evidence_client=evidence_client;self.evidence_validator=evidence_validator
        self.evaluation_client=evaluation_client;self.evaluation_validator=evaluation_validator
        self.qualification_client=qualification_client;self.qualification_validator=qualification_validator
        self.promotion_client=promotion_client;self.promotion_validator=promotion_validator
        self.authorization_gate=AuthorizationGateV143(cas=self.cas,governance_db=self.db,validator=self.promotion_validator)
        self.audit_client=audit_client;self.audit_verifier=audit_verifier;self.trusted_audit=set(trusted_audit_key_ids);self.audit_anchor=audit_anchor

    def put_evidence(self,payload:Mapping[str,Any])->str:
        return self.cas.put_json({"schema":"mini-agi-v14.1-alpha3-evidence-record-v1","payload":dict(payload)})
    def put_supporting_receipt(self,kind:str,payload:Mapping[str,Any])->str:
        return self.cas.put_json({"schema":f"mini-agi-v14.1-alpha3-{kind}-receipt-v1","payload":dict(payload)})
    def put_runtime_component(self,field_name:str,payload:Mapping[str,Any])->str:
        return self.cas.put_json({"schema":RUNTIME_COMPONENT_SCHEMAS_V143[field_name],"payload":dict(payload)})

    def register_candidate(self,proposal:LearningProposalV14,*,evidence_proof:EvidenceStrengthProofV143,candidate_payload:Mapping[str,Any],actor="candidate-builder"):
        derived=self.evidence_validator.validate(evidence_proof,expected_evidence_digests=tuple(proposal.evidence_digests))
        if int(derived)!=int(proposal.coordinates.evidence_strength): raise PermissionError("proposal evidence strength is not authority-derived")
        self.policy.require(proposal); scopes=derive_mutation_scopes(proposal)
        epd=self.cas.put_json(asdict(evidence_proof));pd=self.cas.put_json(asdict(proposal))
        body={"schema":"mini-agi-v14.1-alpha3-candidate-manifest-v1","proposal_digest":pd,"evidence_proof_digest":epd,
              "production_identity_digest":proposal.production_identity_digest,"derived_mutation_scopes":scopes,"candidate_payload":dict(candidate_payload)}
        cd=self.cas.put_json(body);self.db.register_candidate(candidate_digest=cd,proposal_digest=pd,actor=actor);return cd

    def record_build(self,*,candidate_digest:str,build_payload:Mapping[str,Any],mutation_targets:Mapping[str,str]|None=None,actor="builder"):
        c=self._cas_body(candidate_digest); allowed=set(c.get("derived_mutation_scopes",()))-{"runtime.activate"}
        mt={str(k):validate_digest(str(v)) for k,v in dict(mutation_targets or {}).items()}
        if not set(mt)<=allowed: raise PermissionError("build requests mutation target outside policy-derived scopes")
        body={"schema":"mini-agi-v14.1-alpha3-build-manifest-v1","candidate_digest":candidate_digest,"mutation_targets":mt,"build_payload":dict(build_payload)}
        bd=self.cas.put_json(body);self.db.record_build(candidate_digest=candidate_digest,build_digest=bd,actor=actor);return bd

    def evaluate(self,*,candidate_digest:str,build_digest:str,case_results:tuple[EvaluationCaseResultV143,...])->SignedEvaluationBundleV143:
        if self.db.candidate(candidate_digest) is None or self.db.build(build_digest) is None: raise PermissionError("candidate/build absent")
        b=self.evaluation_client.issue(candidate_digest=candidate_digest,build_digest=build_digest,case_results=tuple(case_results))
        self.evaluation_validator.validate(b,candidate_digest=candidate_digest,build_digest=build_digest)
        ed=self.cas.put_json(asdict(b));
        if ed!=b.digest: raise RuntimeError("evaluation canonical digest mismatch")
        self.db.record_evaluation(candidate_digest=candidate_digest,build_digest=build_digest,evaluation_digest=b.digest,evaluator_key_id=b.signer_key_id,
                                  evaluator_generation=b.evaluator_generation,metrics_digest=b.raw_results_digest)
        return b

    def qualify(self,bundle:SignedEvaluationBundleV143)->QualificationRecordV143:
        row=self.db.evaluation(bundle.digest)
        if row is None: raise PermissionError("evaluation not authoritative")
        q=self.qualification_client.qualify(bundle,candidate_digest=bundle.candidate_digest,build_digest=bundle.build_digest)
        self.qualification_validator.validate(q,candidate_digest=q.candidate_digest,build_digest=q.build_digest,evaluation_digest=q.evaluation_digest)
        qd=self.cas.put_json(asdict(q));
        if qd!=q.digest: raise RuntimeError("qualification canonical digest mismatch")
        self.db.record_qualification_v143(candidate_digest=q.candidate_digest,evaluation_digest=q.evaluation_digest,qualification_digest=q.digest,decision=q.decision,
            qualifier_key_id=q.signer_key_id,qualifier_generation=q.qualifier_generation,authority_generation=q.authority_generation,
            policy_generation=q.policy_generation,raw_results_digest=q.raw_results_digest,derived_metrics_digest=q.derived_metrics_digest)
        return q

    def prepare_runtime_manifest(self,*,qualification:QualificationRecordV143,runtime_component_payloads:Mapping[str,Mapping[str,Any]]):
        required=set(RUNTIME_COMPONENT_SCHEMAS_V143)
        if set(runtime_component_payloads)!=required: raise ValueError("runtime component payload set mismatch")
        comps={n:self.put_runtime_component(n,runtime_component_payloads[n]) for n in sorted(required)}
        build=self._cas_body(qualification.build_digest); mutation_targets=dict(build.get("mutation_targets",{}))
        root=self.cas.put_json({"schema":"mini-agi-v14.1-alpha3-artifact-root-v1","candidate_digest":qualification.candidate_digest,
            "build_digest":qualification.build_digest,"evaluation_digest":qualification.evaluation_digest,"qualification_digest":qualification.digest,
            "runtime_components":comps,"mutation_targets":mutation_targets})
        m=RuntimeManifestV143(qualification.candidate_digest,qualification.build_digest,qualification.evaluation_digest,qualification.digest,root,
            comps["production_identity_digest"],comps["belief_snapshot_digest"],comps["skill_snapshot_digest"],comps["adapter_set_digest"],
            comps["routing_policy_digest"],comps["execution_runtime_digest"],self.policy.authority_generation,self.policy.policy_generation,time.time())
        md=self.cas.put_json(asdict(m));
        if md!=m.digest: raise RuntimeError("runtime manifest canonical digest mismatch")
        return m

    def authorize(self,qualification:QualificationRecordV143,manifest:RuntimeManifestV143)->PromotionAuthorizationV143:
        if qualification.decision!="PROMOTE": raise PermissionError("rejected qualification cannot authorize")
        c=self._cas_body(qualification.candidate_digest); allowed=tuple(c.get("derived_mutation_scopes",()))
        root=self._cas_body(manifest.artifact_root_digest); mt=dict(root.get("mutation_targets",{}))
        commits=[MutationCommitmentV143("runtime.activate",manifest.digest)]
        for scope,target in sorted(mt.items()):
            if scope not in allowed: raise PermissionError("artifact root contains non-policy mutation scope")
            commits.append(MutationCommitmentV143(scope,target))
        auth=self.promotion_client.authorize(qualification=qualification,runtime_manifest_digest=manifest.digest,
                                             artifact_root_digest=manifest.artifact_root_digest,mutation_commitments=tuple(commits))
        ad=self.cas.put_json(asdict(auth));
        if ad!=auth.digest: raise RuntimeError("promotion canonical digest mismatch")
        self.promotion_validator.validate(auth,candidate_digest=manifest.candidate_digest,build_digest=manifest.build_digest,evaluation_digest=manifest.evaluation_digest,
            qualification_digest=manifest.qualification_digest,runtime_manifest_digest=manifest.digest,artifact_root_digest=manifest.artifact_root_digest,
            required_scope="runtime.activate",target_digest=manifest.digest)
        return auth

    def activate(self,*,auth:PromotionAuthorizationV143,manifest:RuntimeManifestV143):
        self.promotion_validator.validate(auth,candidate_digest=manifest.candidate_digest,build_digest=manifest.build_digest,evaluation_digest=manifest.evaluation_digest,
            qualification_digest=manifest.qualification_digest,runtime_manifest_digest=manifest.digest,artifact_root_digest=manifest.artifact_root_digest,
            required_scope="runtime.activate",target_digest=manifest.digest)
        # ensure exact manifest and root still exist and match
        if self._cas_body(manifest.digest).get("artifact_root_digest")!=manifest.artifact_root_digest: raise PermissionError("runtime manifest/root mismatch")
        self._cas_body(manifest.artifact_root_digest)
        activation=self.cas.put_json({"schema":"mini-agi-v14.1-alpha3-activation-receipt-v1","candidate_digest":manifest.candidate_digest,
            "authorization_digest":auth.digest,"runtime_manifest_digest":manifest.digest})
        self.db.activate_v143(candidate_digest=manifest.candidate_digest,authorization_digest=auth.digest,runtime_manifest_digest=manifest.digest,activation_digest=activation)
        return activation

    def learning_cycle(self,*,proposal:LearningProposalV14,evidence_proof:EvidenceStrengthProofV143,candidate_payload:Mapping[str,Any],build_payload:Mapping[str,Any],
                       mutation_targets:Mapping[str,str]|None,case_results:tuple[EvaluationCaseResultV143,...],runtime_component_payloads:Mapping[str,Mapping[str,Any]]):
        c=self.register_candidate(proposal,evidence_proof=evidence_proof,candidate_payload=candidate_payload)
        b=self.record_build(candidate_digest=c,build_payload=build_payload,mutation_targets=mutation_targets)
        e=self.evaluate(candidate_digest=c,build_digest=b,case_results=case_results);q=self.qualify(e)
        out={"candidate_digest":c,"build_digest":b,"evaluation":e,"qualification":q}
        if q.decision!="PROMOTE": out["status"]="REJECTED";return out
        m=self.prepare_runtime_manifest(qualification=q,runtime_component_payloads=runtime_component_payloads)
        a=self.authorize(q,m);activation=self.activate(auth=a,manifest=m)
        out.update({"runtime_manifest":m,"authorization":a,"activation_digest":activation,"status":"ACTIVE"});return out

    def checkpoint_audit(self):
        if self.audit_client is None: raise RuntimeError("audit authority client not configured")
        return self.audit_client.checkpoint()
    def verify_audit(self):
        self.db.verify_audit_chain()
        if self.audit_anchor is None: raise RuntimeError("mandatory external audit anchor unavailable")
        anchor=self.audit_anchor.read_and_verify()
        row=self.db.conn.execute("SELECT * FROM audit_checkpoints WHERE checkpoint_digest=?",(anchor.checkpoint_digest,)).fetchone()
        if row is None or int(row["event_seq"])!=anchor.event_seq or row["event_digest"]!=anchor.event_digest: raise RuntimeError("external audit anchor not present in authoritative DB")
        if self.audit_verifier is not None and self.trusted_audit:self.db.verify_audit_checkpoints(verifier=self.audit_verifier,trusted_key_ids=self.trusted_audit)
        return True
    def _cas_body(self,d):
        raw=json.loads(self.cas.get_bytes(validate_digest(d)).decode("utf-8"));return raw.get("value",raw)
    def close(self): self.db.close()
