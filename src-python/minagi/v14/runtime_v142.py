from __future__ import annotations

from dataclasses import asdict, dataclass
import json, time
from pathlib import Path
from typing import Any, Mapping

from egai.common.canonical import digest, validate_digest
from .models import LearningProposalV14
from .policy import GovernancePolicyV14
from .storage_v142 import GovernanceDBV142, ImmutableCAS
from .qualification_v142 import (
    EvaluationAuthorityV142, EvaluationValidatorV142, QualificationAuthorityV142,
    QualificationPolicyV142, QualificationRecordV142, QualificationValidatorV142,
    SignedEvaluationBundleV142,
)
from .authority_v142 import (
    AuthorizationGateV142, PromotionAuthorityV142, PromotionAuthorizationV142,
    PromotionAuthorizationValidatorV142,
)
from .bitemporal import BiTemporalBeliefStore
from .skills_v141 import SkillLifecycleStore


RUNTIME_COMPONENT_SCHEMAS = {
    "production_identity_digest": "mini-agi-v14.1-alpha2-production-identity-v1",
    "belief_snapshot_digest": "mini-agi-v14.1-alpha2-belief-snapshot-v1",
    "skill_snapshot_digest": "mini-agi-v14.1-alpha2-skill-snapshot-v1",
    "adapter_set_digest": "mini-agi-v14.1-alpha2-adapter-set-v1",
    "routing_policy_digest": "mini-agi-v14.1-alpha2-routing-policy-v1",
    "execution_runtime_digest": "mini-agi-v14.1-alpha2-execution-runtime-v1",
}


@dataclass(frozen=True)
class RuntimeManifestV142:
    candidate_digest: str
    authorization_digest: str
    production_identity_digest: str
    belief_snapshot_digest: str
    skill_snapshot_digest: str
    adapter_set_digest: str
    routing_policy_digest: str
    execution_runtime_digest: str
    authority_generation: int
    policy_generation: int
    created_at: float
    schema: str = "mini-agi-v14.1-alpha2-runtime-manifest-v1"
    def __post_init__(self):
        for k,v in asdict(self).items():
            if k.endswith("_digest"): validate_digest(v)
    @property
    def digest(self): return digest(self)


class GovernedContinualRuntimeV142:
    """Canonical alpha2 authority-closed runtime.

    No caller supplies a qualification decision. Evaluations and qualifications
    are signed by distinct authorities, promotion is scope-bound, and runtime
    activation resolves every manifest dependency from CAS before committing.
    """

    def __init__(self, root: str | Path, *, policy: GovernancePolicyV14,
                 evaluation_signer, evaluation_verifier, trusted_evaluation_key_ids,
                 qualification_signer, qualification_verifier, trusted_qualification_key_ids,
                 promotion_signer, promotion_verifier, trusted_promotion_key_ids,
                 audit_signer=None, audit_verifier=None, trusted_audit_key_ids=(),
                 evaluator_id: str="evaluation-authority", evaluator_generation: int=1,
                 qualifier_id: str="qualification-authority", qualifier_generation: int=1,
                 qualification_policy: QualificationPolicyV142 | None=None):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.policy=policy
        self.cas=ImmutableCAS(self.root/"cas"); self.db=GovernanceDBV142(self.root/"governance.sqlite3")
        self.evaluator=EvaluationAuthorityV142(evaluator_id=evaluator_id,evaluator_generation=evaluator_generation,signer=evaluation_signer)
        self.evaluation_validator=EvaluationValidatorV142(verifier=evaluation_verifier,trusted_key_ids=trusted_evaluation_key_ids,evaluator_generation=evaluator_generation)
        self.qualifier=QualificationAuthorityV142(qualifier_id=qualifier_id,qualifier_generation=qualifier_generation,
            authority_generation=policy.authority_generation,policy_generation=policy.policy_generation,
            signer=qualification_signer,evaluation_validator=self.evaluation_validator,policy=qualification_policy)
        self.qualification_validator=QualificationValidatorV142(verifier=qualification_verifier,trusted_key_ids=trusted_qualification_key_ids,
            qualifier_generation=qualifier_generation,authority_generation=policy.authority_generation,policy_generation=policy.policy_generation)
        self.promotion=PromotionAuthorityV142(governance_db=self.db,signer=promotion_signer,
            qualification_validator=self.qualification_validator,authority_generation=policy.authority_generation,policy_generation=policy.policy_generation)
        self.promotion_validator=PromotionAuthorizationValidatorV142(verifier=promotion_verifier,trusted_key_ids=trusted_promotion_key_ids,
            authority_generation=policy.authority_generation,policy_generation=policy.policy_generation)
        self.authorization_gate=AuthorizationGateV142(cas=self.cas,governance_db=self.db,validator=self.promotion_validator)
        self.beliefs=BiTemporalBeliefStore(self.root/"beliefs.sqlite3",authorization_gate=self.authorization_gate)
        self.skills=SkillLifecycleStore(self.root/"skills.sqlite3",authorization_gate=self.authorization_gate)
        self.audit_signer=audit_signer; self.audit_verifier=audit_verifier; self.trusted_audit=set(trusted_audit_key_ids)

    def put_runtime_component(self, field_name: str, payload: Mapping[str,Any]) -> str:
        schema=RUNTIME_COMPONENT_SCHEMAS[field_name]
        return self.cas.put_json({"schema":schema,"payload":dict(payload)})

    def register_candidate(self, proposal: LearningProposalV14, *, candidate_payload: Mapping[str,Any], actor="candidate-builder") -> str:
        self.policy.require(proposal)
        pd=self.cas.put_json(asdict(proposal)); body={"schema":"mini-agi-v14.1-alpha2-candidate-manifest-v1","proposal_digest":pd,
             "production_identity_digest":proposal.production_identity_digest,"candidate_payload":dict(candidate_payload)}
        cd=self.cas.put_json(body); self.db.register_candidate(candidate_digest=cd,proposal_digest=pd,actor=actor); return cd

    def record_build(self, *, candidate_digest: str, build_payload: Mapping[str,Any], actor="builder") -> str:
        body={"schema":"mini-agi-v14.1-alpha2-build-manifest-v1","candidate_digest":candidate_digest,"build_payload":dict(build_payload)}
        bd=self.cas.put_json(body); self.db.record_build(candidate_digest=candidate_digest,build_digest=bd,actor=actor); return bd

    def evaluate(self, *, candidate_digest: str, build_digest: str, metrics_payload: Mapping[str,Any]) -> SignedEvaluationBundleV142:
        if self.db.candidate(candidate_digest) is None or self.db.build(build_digest) is None: raise PermissionError("candidate/build absent")
        bundle=self.evaluator.issue(candidate_digest=candidate_digest,build_digest=build_digest,metrics_payload=metrics_payload)
        stored=self.cas.put_json(asdict(bundle))
        if stored != bundle.digest: raise RuntimeError("evaluation canonical digest mismatch")
        self.db.record_evaluation(candidate_digest=candidate_digest,build_digest=build_digest,evaluation_digest=bundle.digest,
            evaluator_key_id=bundle.signer_key_id,evaluator_generation=bundle.evaluator_generation,metrics_digest=bundle.metrics_digest)
        return bundle

    def qualify(self, bundle: SignedEvaluationBundleV142) -> QualificationRecordV142:
        build_row=self.db.evaluation(bundle.digest)
        if build_row is None: raise PermissionError("evaluation not authoritative")
        rec=self.qualifier.qualify(bundle,candidate_digest=bundle.candidate_digest,build_digest=bundle.build_digest)
        stored=self.cas.put_json(asdict(rec))
        if stored != rec.digest: raise RuntimeError("qualification canonical digest mismatch")
        self.db.record_qualification(candidate_digest=rec.candidate_digest,evaluation_digest=rec.evaluation_digest,
            qualification_digest=rec.digest,decision=rec.decision,qualifier_key_id=rec.signer_key_id,
            qualifier_generation=rec.qualifier_generation,authority_generation=rec.authority_generation,
            policy_generation=rec.policy_generation,metrics_digest=rec.metrics_digest)
        return rec

    def _candidate_scopes(self, candidate_digest: str) -> tuple[str,...]:
        raw=json.loads(self.cas.get_bytes(candidate_digest).decode("utf-8")); body=raw.get("value", raw); payload=body.get("candidate_payload",{})
        supplied=payload.get("mutation_scopes")
        if supplied:
            scopes=tuple(sorted(set(str(x) for x in supplied)))
        else:
            kind=str(payload.get("kind","runtime")).lower()
            scopes={"skill":("skill.activate","runtime.activate"),"belief":("belief.promote","runtime.activate"),
                    "routing":("routing.activate","runtime.activate"),"adapter":("adapter.activate","runtime.activate")}.get(kind,("runtime.activate",))
        allowed={"runtime.activate","skill.activate","belief.promote","routing.activate","adapter.activate"}
        if any(x not in allowed for x in scopes): raise PermissionError("candidate requested unsupported mutation scope")
        return tuple(scopes)

    def authorize(self, qualification: QualificationRecordV142) -> PromotionAuthorizationV142:
        scopes=self._candidate_scopes(qualification.candidate_digest)
        auth=self.promotion.authorize(qualification=qualification,mutation_scopes=scopes)
        stored=self.cas.put_json(asdict(auth))
        if stored != auth.digest: raise RuntimeError("promotion canonical digest mismatch")
        return auth

    def _validate_component(self, field_name: str, value: str) -> None:
        validate_digest(value)
        try:
            raw=json.loads(self.cas.get_bytes(value).decode("utf-8")); body=raw.get("value", raw)
        except Exception as exc: raise PermissionError(f"runtime component missing/corrupt: {field_name}") from exc
        if body.get("schema") != RUNTIME_COMPONENT_SCHEMAS[field_name]:
            raise PermissionError(f"runtime component schema mismatch: {field_name}")

    def activate(self, *, auth: PromotionAuthorizationV142, production_identity_digest: str,
                 belief_snapshot_digest: str, skill_snapshot_digest: str, adapter_set_digest: str,
                 routing_policy_digest: str, execution_runtime_digest: str) -> RuntimeManifestV142:
        self.promotion_validator.validate(auth,candidate_digest=auth.candidate_digest,qualification_digest=auth.qualification_digest,required_scope="runtime.activate")
        for n,v in {"production_identity_digest":production_identity_digest,"belief_snapshot_digest":belief_snapshot_digest,
                    "skill_snapshot_digest":skill_snapshot_digest,"adapter_set_digest":adapter_set_digest,
                    "routing_policy_digest":routing_policy_digest,"execution_runtime_digest":execution_runtime_digest}.items():
            self._validate_component(n,v)
        manifest=RuntimeManifestV142(auth.candidate_digest,auth.digest,production_identity_digest,belief_snapshot_digest,
                    skill_snapshot_digest,adapter_set_digest,routing_policy_digest,execution_runtime_digest,
                    self.policy.authority_generation,self.policy.policy_generation,time.time())
        md=self.cas.put_json(asdict(manifest)); activation=self.cas.put_json({"schema":"mini-agi-v14.1-alpha2-activation-receipt-v1",
            "candidate_digest":auth.candidate_digest,"authorization_digest":auth.digest,"runtime_manifest_digest":md})
        self.db.activate(candidate_digest=auth.candidate_digest,authorization_digest=auth.digest,runtime_manifest_digest=md,activation_digest=activation)
        return manifest

    def learning_cycle(self, *, proposal: LearningProposalV14, candidate_payload: Mapping[str,Any],
                       build_payload: Mapping[str,Any], metrics_payload: Mapping[str,Any],
                       runtime_component_payloads: Mapping[str,Mapping[str,Any]]) -> dict[str,Any]:
        """Execute one complete governed candidate lifecycle.

        A rejected qualification terminates before authorization. A passing
        qualification is scope-authorized and activated only after every runtime
        dependency has been materialized as a typed CAS object.
        """
        candidate=self.register_candidate(proposal,candidate_payload=candidate_payload)
        build=self.record_build(candidate_digest=candidate,build_payload=build_payload)
        evaluation=self.evaluate(candidate_digest=candidate,build_digest=build,metrics_payload=metrics_payload)
        qualification=self.qualify(evaluation)
        result={"candidate_digest":candidate,"build_digest":build,"evaluation":evaluation,"qualification":qualification}
        if qualification.decision != "PROMOTE":
            result["status"]="REJECTED"
            return result
        auth=self.authorize(qualification)
        required=set(RUNTIME_COMPONENT_SCHEMAS)
        if set(runtime_component_payloads) != required:
            missing=sorted(required-set(runtime_component_payloads)); extra=sorted(set(runtime_component_payloads)-required)
            raise ValueError(f"runtime component payload set mismatch; missing={missing}, extra={extra}")
        components={name:self.put_runtime_component(name,runtime_component_payloads[name]) for name in sorted(required)}
        manifest=self.activate(auth=auth,**components)
        result.update({"authorization":auth,"runtime_manifest":manifest,"status":"ACTIVE"})
        return result

    def checkpoint_audit(self):
        if self.audit_signer is None: raise RuntimeError("audit signer not configured")
        return self.db.create_audit_checkpoint(signer=self.audit_signer,authority_generation=self.policy.authority_generation)

    def verify_audit(self) -> bool:
        self.db.verify_audit_chain()
        if self.audit_verifier is not None and self.trusted_audit:
            self.db.verify_audit_checkpoints(verifier=self.audit_verifier,trusted_key_ids=self.trusted_audit)
        return True

    def close(self):
        self.db.close()
