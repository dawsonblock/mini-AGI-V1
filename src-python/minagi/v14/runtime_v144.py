from __future__ import annotations

from dataclasses import asdict, dataclass
import json, time
from pathlib import Path
from typing import Any, Mapping

from egai.common.canonical import digest, validate_digest
from .models import EvidenceStrength, GovernanceCoordinates, LearningMechanism, LearningProposalV14, PermanenceLevel
from .runtime_v143 import GovernedContinualRuntimeV143, derive_mutation_scopes
from .storage_v144 import GovernanceDBV144
from .experience_v144 import ExperiencePipelineV144
from .persistence_v144 import ContentBoundPersistencePlannerV144
from .evidence_v144 import EvidenceStageReceiptV144, EvidenceStrengthProofV144
from .authority_v143 import PromotionAuthorizationV143


RUNTIME_COMPONENT_SCHEMAS_V144 = {
    "production_identity_digest": "mini-agi-v14.1-alpha4-production-identity-v1",
    "belief_snapshot_digest": "mini-agi-v14.1-alpha4-belief-snapshot-v1",
    "skill_snapshot_digest": "mini-agi-v14.1-alpha4-skill-snapshot-v1",
    "adapter_set_digest": "mini-agi-v14.1-alpha4-adapter-set-v1",
    "routing_policy_digest": "mini-agi-v14.1-alpha4-routing-policy-v1",
    "execution_runtime_digest": "mini-agi-v14.1-alpha4-execution-runtime-v1",
}


@dataclass(frozen=True)
class RuntimeManifestV144:
    candidate_digest: str
    build_digest: str
    evaluation_digest: str
    qualification_digest: str
    artifact_root_digest: str
    production_identity_digest: str
    belief_snapshot_digest: str
    skill_snapshot_digest: str
    adapter_set_digest: str
    routing_policy_digest: str
    execution_runtime_digest: str
    authority_generation: int
    policy_generation: int
    created_at: float
    schema: str = "mini-agi-v14.1-alpha4-runtime-manifest-v1"

    def __post_init__(self):
        for k, v in asdict(self).items():
            if k.endswith("_digest"):
                validate_digest(v)

    @property
    def digest(self):
        return digest(self)


class GovernedContinualRuntimeV144(GovernedContinualRuntimeV143):
    """Alpha4 evidence-closed runtime.

    Adds a canonical Experience -> Verification -> evidence-stage chain and
    atomically applies exact belief/skill mutations with runtime activation.
    Private authority keys remain outside the runtime; production mode rejects
    local authority clients for all newly introduced authorities as well.
    """

    def __init__(self, root: str | Path, *, verification_client, verification_validator,
                 evidence_stage_client, evidence_client, evidence_validator,
                 policy, evaluation_client, evaluation_validator, qualification_client,
                 qualification_validator, promotion_client, promotion_validator,
                 audit_client=None, audit_verifier=None, trusted_audit_key_ids=(), audit_anchor=None,
                 cas=None, governance_db=None, production_mode: bool = False):
        db = governance_db or GovernanceDBV144(Path(root) / "governance.sqlite3")
        super().__init__(root, policy=policy, evidence_client=evidence_client, evidence_validator=evidence_validator,
                         evaluation_client=evaluation_client, evaluation_validator=evaluation_validator,
                         qualification_client=qualification_client, qualification_validator=qualification_validator,
                         promotion_client=promotion_client, promotion_validator=promotion_validator,
                         audit_client=audit_client, audit_verifier=audit_verifier,
                         trusted_audit_key_ids=trusted_audit_key_ids, audit_anchor=audit_anchor,
                         cas=cas, governance_db=db, production_mode=False)
        if production_mode:
            for name, client in (
                ("verification", verification_client), ("evidence-stage", evidence_stage_client),
                ("evidence-strength", evidence_client), ("evaluation", evaluation_client),
                ("qualification", qualification_client), ("promotion", promotion_client),
            ):
                if getattr(client, "is_local", True):
                    raise RuntimeError(f"production mode requires out-of-process {name} authority client")
            if audit_client is not None and getattr(audit_client, "is_local", True):
                raise RuntimeError("production mode requires out-of-process audit authority client")
        self.production_mode = bool(production_mode)
        self.verification_client = verification_client
        self.verification_validator = verification_validator
        self.evidence_stage_client = evidence_stage_client
        self.experience_pipeline = ExperiencePipelineV144(
            cas=self.cas, governance_db=self.db, verification_client=verification_client,
            verification_validator=verification_validator,
        )
        self.persistence = ContentBoundPersistencePlannerV144(cas=self.cas, governance_db=self.db)

    def put_runtime_component(self, field_name: str, payload: Mapping[str, Any]) -> str:
        if field_name not in RUNTIME_COMPONENT_SCHEMAS_V144:
            raise KeyError(field_name)
        return self.cas.put_json({"schema": RUNTIME_COMPONENT_SCHEMAS_V144[field_name], "payload": dict(payload)})

    def ingest_verified_experience(self, **kwargs):
        return self.experience_pipeline.ingest(**kwargs)

    def issue_e2_evidence_proof(self, *, verified_record, verification_receipt) -> tuple[EvidenceStageReceiptV144, EvidenceStrengthProofV144]:
        stage = self.evidence_stage_client.issue(
            stage_strength=int(EvidenceStrength.E2_INDEPENDENTLY_VERIFIED),
            evidence_digests=(verified_record.evidence_root_digest,),
            parent_receipt_digests=(verification_receipt.digest,),
        )
        sd = self.cas.put_json(asdict(stage))
        if sd != stage.digest:
            raise RuntimeError("evidence-stage canonical digest mismatch")
        proof = self.evidence_client.issue(
            evidence_digests=(verified_record.evidence_root_digest,),
            highest_stage_receipt_digest=stage.digest,
        )
        pd = self.cas.put_json(asdict(proof))
        if pd != proof.digest:
            raise RuntimeError("evidence-strength proof canonical digest mismatch")
        self.evidence_validator.validate(proof, expected_evidence_digests=(verified_record.evidence_root_digest,))
        return stage, proof

    def prepare_runtime_manifest(self, *, qualification, runtime_component_payloads: Mapping[str, Mapping[str, Any]]):
        required = set(RUNTIME_COMPONENT_SCHEMAS_V144)
        if set(runtime_component_payloads) != required:
            raise ValueError("runtime component payload set mismatch")
        comps = {n: self.put_runtime_component(n, runtime_component_payloads[n]) for n in sorted(required)}
        build = self._cas_body(qualification.build_digest)
        mutation_targets = dict(build.get("mutation_targets", {}))
        # Snapshot components must explicitly include the exact learned-state mutation they activate.
        if "belief.promote" in mutation_targets:
            included = tuple(runtime_component_payloads["belief_snapshot_digest"].get("included_mutation_digests", ()))
            if mutation_targets["belief.promote"] not in included:
                raise PermissionError("belief snapshot does not include authorized belief mutation")
        if "skill.activate" in mutation_targets:
            included = tuple(runtime_component_payloads["skill_snapshot_digest"].get("included_mutation_digests", ()))
            if mutation_targets["skill.activate"] not in included:
                raise PermissionError("skill snapshot does not include authorized skill mutation")
        root = self.cas.put_json({
            "schema": "mini-agi-v14.1-alpha4-artifact-root-v1",
            "candidate_digest": qualification.candidate_digest,
            "build_digest": qualification.build_digest,
            "evaluation_digest": qualification.evaluation_digest,
            "qualification_digest": qualification.digest,
            "runtime_components": comps,
            "mutation_targets": mutation_targets,
        })
        m = RuntimeManifestV144(
            qualification.candidate_digest, qualification.build_digest, qualification.evaluation_digest,
            qualification.digest, root, comps["production_identity_digest"], comps["belief_snapshot_digest"],
            comps["skill_snapshot_digest"], comps["adapter_set_digest"], comps["routing_policy_digest"],
            comps["execution_runtime_digest"], self.policy.authority_generation, self.policy.policy_generation, time.time(),
        )
        md = self.cas.put_json(asdict(m))
        if md != m.digest:
            raise RuntimeError("runtime manifest canonical digest mismatch")
        return m

    def activate(self, *, auth: PromotionAuthorizationV143, manifest: RuntimeManifestV144):
        self.promotion_validator.validate(
            auth, candidate_digest=manifest.candidate_digest, build_digest=manifest.build_digest,
            evaluation_digest=manifest.evaluation_digest, qualification_digest=manifest.qualification_digest,
            runtime_manifest_digest=manifest.digest, artifact_root_digest=manifest.artifact_root_digest,
            required_scope="runtime.activate", target_digest=manifest.digest,
        )
        if self._cas_body(manifest.digest).get("artifact_root_digest") != manifest.artifact_root_digest:
            raise PermissionError("runtime manifest/root mismatch")
        root = self._cas_body(manifest.artifact_root_digest)
        mutation_targets = dict(root.get("mutation_targets", {}))
        mutations = []
        for scope, target in sorted(mutation_targets.items()):
            self.authorization_gate.require(auth.digest, mutation_scope=scope, target_digest=target, consume=False)
            mutations.append((scope, target, self._cas_body(target)))
        activation = self.cas.put_json({
            "schema": "mini-agi-v14.1-alpha4-atomic-activation-receipt-v1",
            "candidate_digest": manifest.candidate_digest,
            "authorization_digest": auth.digest,
            "runtime_manifest_digest": manifest.digest,
            "mutation_targets": sorted(mutation_targets.items()),
        })
        self.db.activate_with_mutations_v144(
            candidate_digest=manifest.candidate_digest, authorization_digest=auth.digest,
            runtime_manifest_digest=manifest.digest, activation_digest=activation, mutations=tuple(mutations),
        )
        return activation

    def full_belief_learning_cycle(self, *, prompt: str, attempted_output: str, repaired_output: str,
                                   expected_output: str, evidence_payload, production_identity_digest: str,
                                   rationale_digest: str, proposer_id: str, target: str,
                                   belief_subject: str, belief_predicate: str, belief_object,
                                   belief_valid_from: float, belief_confidence: float,
                                   case_results, runtime_component_payloads: Mapping[str, Mapping[str, Any]],
                                   expected_gain: float = 0.1, expected_interference: float = 0.0,
                                   score: float = 1.0, importance: float = 0.5):
        experience, verification, verified = self.ingest_verified_experience(
            prompt=prompt, attempted_output=attempted_output, repaired_output=repaired_output,
            expected_output=expected_output, evidence_payload=evidence_payload,
            production_identity_digest=production_identity_digest, score=score, importance=importance,
        )
        stage, proof = self.issue_e2_evidence_proof(verified_record=verified, verification_receipt=verification)
        proposal = LearningProposalV14.create(
            target=target, mechanism=LearningMechanism.BELIEF_UPDATE,
            coordinates=GovernanceCoordinates(PermanenceLevel.L3_SEMANTIC_BELIEF, EvidenceStrength.E2_INDEPENDENTLY_VERIFIED),
            evidence_digests=(verified.evidence_root_digest,), rationale_digest=rationale_digest,
            proposer_id=proposer_id, production_identity_digest=production_identity_digest,
            expected_gain=float(expected_gain), expected_interference=float(expected_interference),
        )
        candidate = self.register_candidate(proposal, evidence_proof=proof,
                                            candidate_payload={"verified_evidence_digest": verified.digest,
                                                               "experience_digest": experience.digest})
        belief = self.persistence.prepare_belief(
            subject=belief_subject, predicate=belief_predicate, object_value=belief_object,
            valid_from=belief_valid_from, evidence_digests=(verified.evidence_root_digest,),
            confidence=belief_confidence,
        )
        build = self.record_build(candidate_digest=candidate, build_payload={"kind": "belief-update"},
                                  mutation_targets={"belief.promote": belief.digest})
        evaluation = self.evaluate(candidate_digest=candidate, build_digest=build, case_results=tuple(case_results))
        qualification = self.qualify(evaluation)
        out = {
            "experience": experience, "verification": verification, "verified_evidence": verified,
            "evidence_stage": stage, "evidence_proof": proof, "proposal": proposal,
            "candidate_digest": candidate, "build_digest": build, "evaluation": evaluation,
            "qualification": qualification, "belief_revision": belief,
        }
        if qualification.decision != "PROMOTE":
            out["status"] = "REJECTED"
            return out
        payloads = {k: dict(v) for k, v in runtime_component_payloads.items()}
        payloads["belief_snapshot_digest"] = dict(payloads["belief_snapshot_digest"])
        inc = list(payloads["belief_snapshot_digest"].get("included_mutation_digests", ()))
        if belief.digest not in inc:
            inc.append(belief.digest)
        payloads["belief_snapshot_digest"]["included_mutation_digests"] = inc
        manifest = self.prepare_runtime_manifest(qualification=qualification, runtime_component_payloads=payloads)
        authorization = self.authorize(qualification, manifest)
        activation = self.activate(auth=authorization, manifest=manifest)
        out.update({"runtime_manifest": manifest, "authorization": authorization,
                    "activation_digest": activation, "status": "ACTIVE"})
        return out
