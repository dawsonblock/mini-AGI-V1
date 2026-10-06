from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .authority_artifacts import (
    AuthorityArtifactStore,
    EvidenceAttestationRC14,
    EvaluationBundleRC14,
    QualificationEvidenceRC14,
    QualificationRecordRC14,
    ContinualExperimentEvidenceRC14,
    IndependentReproductionEvidenceRC14,
)
from .falsification import FalsificationKind, PreregisteredPlanStore
from .models import LearningProposalRC14, PermanenceLevel
from .policy import GovernancePolicyRC14
from .neural_memory import (FrozenFoundationProofRC14, IsolatedNeuralMemoryManifestRC14, IsolatedNeuralMemoryQualificationRC14, IsolatedNeuralMemoryQualificationEngineRC14)
from .independent_reproduction import IndependentReproductionCertificateRC14


@dataclass
class QualificationEngineRC14:
    """Derive qualification facts from sealed evaluation evidence.

    The engine never accepts caller-supplied retention/security/evidence-strength flags.
    Those facts are reconstructed from the preregistered plan + immutable recorded run.
    Higher evidence levels are unlocked only by signed external attestations verified by
    scope-specific verifiers.
    """

    artifacts: AuthorityArtifactStore
    falsification: PreregisteredPlanStore
    policy: GovernancePolicyRC14
    attestation_verifiers: Mapping[str, Any]
    continual_experiment_protocol: Any | None = None
    independent_reproduction_store: Any | None = None
    executed_run_store: Any | None = None

    def _attestations(self, *, candidate_digest: str, evaluation_digest: str,
                      attestation_digests: Sequence[str]) -> dict[str, list[EvidenceAttestationRC14]]:
        out: dict[str, list[EvidenceAttestationRC14]] = {}
        seen_issuers: dict[str, set[str]] = {}
        for digest in attestation_digests:
            obj = self.artifacts.get(str(digest), expected_kind="evidence_attestation")
            if not isinstance(obj, EvidenceAttestationRC14):
                raise PermissionError("invalid evidence attestation artifact")
            if obj.candidate_digest != candidate_digest or obj.evaluation_digest != evaluation_digest:
                raise PermissionError("evidence attestation candidate/evaluation mismatch")
            verifier = self.attestation_verifiers.get(obj.scope)
            if verifier is None or not verifier.verify(obj.receipt, expected_body=obj.signed_body): raise PermissionError(f"untrusted evidence attestation for scope {obj.scope}")
            if getattr(verifier,"key_id",obj.issuer_id)!=obj.issuer_id: raise PermissionError("evidence attestation issuer_id does not match verifier key identity")
            issuer_set=seen_issuers.setdefault(obj.scope,set())
            if obj.issuer_id in issuer_set: raise PermissionError("duplicate evidence attestation issuer for same scope")
            issuer_set.add(obj.issuer_id)
            science={"independent_verification","replication","independent_reproduction"}
            if obj.scope in science:
                for other in science-{obj.scope}:
                    if obj.issuer_id in seen_issuers.get(other,set()): raise PermissionError("one signer cannot satisfy multiple scientific evidence scopes")
            if obj.scope=="independent_reproduction": self._verify_independent_reproduction_evidence(obj.evidence_digest,candidate_digest=candidate_digest)
            out.setdefault(obj.scope,[]).append(obj)
        return out

    def _verify_continual_experiment_evidence(self, digest: str, *, candidate_digest: str, proposal_digest: str) -> ContinualExperimentEvidenceRC14:
        if self.continual_experiment_protocol is None:
            raise PermissionError("L6 qualification requires configured preregistered experiment protocol store")
        obj = self.artifacts.get(digest, expected_kind="continual_experiment_evidence")
        if not isinstance(obj, ContinualExperimentEvidenceRC14) or not obj.qualified:
            raise PermissionError("continual-experiment evidence is not qualified")
        if obj.candidate_digest != candidate_digest or obj.proposal_digest != proposal_digest:
            raise PermissionError("continual-experiment evidence candidate/proposal mismatch")
        plan = self.continual_experiment_protocol.get_plan(obj.plan_digest)
        if plan.candidate_digest != candidate_digest:
            raise PermissionError("experiment plan candidate mismatch")
        certificate = self.continual_experiment_protocol.certify(plan.digest)
        if certificate.digest != obj.ablation_certificate_digest or not certificate.eligible:
            raise PermissionError("stored experiment evidence does not match replayed ablation certificate")
        if self.executed_run_store is not None:
            execution_certificate = self.executed_run_store.certify(plan.digest)
            if not execution_certificate.eligible:
                raise PermissionError("continual experiment evidence lacks complete signed external execution matrix")
        if not self.continual_experiment_protocol.contains_event_head(obj.experiment_protocol_head_digest):
            raise PermissionError("experiment evidence references unknown protocol head")
        manifest = IsolatedNeuralMemoryManifestRC14(**dict(obj.neural_memory_manifest_body))
        proof_body = dict(obj.frozen_foundation_proof_body)
        proof_body["allowed_trainable_parameters"] = tuple(proof_body["allowed_trainable_parameters"])
        proof_body["observed_changed_parameters"] = tuple(proof_body["observed_changed_parameters"])
        proof = FrozenFoundationProofRC14(**proof_body)
        qbody = dict(obj.neural_memory_qualification_body); qbody["reasons"] = tuple(qbody["reasons"])
        stored_qualification = IsolatedNeuralMemoryQualificationRC14(**qbody)
        derived = IsolatedNeuralMemoryQualificationEngineRC14().qualify(plan=plan, certificate=certificate, manifest=manifest, proof=proof)
        if manifest.digest != obj.neural_memory_manifest_digest or proof.digest != obj.frozen_foundation_proof_digest:
            raise PermissionError("experiment evidence embedded isolation artifacts do not match digests")
        if derived.digest != obj.neural_memory_qualification_digest or stored_qualification.digest != derived.digest or not derived.qualified:
            raise PermissionError("isolated-neural-memory qualification does not replay")
        return obj

    def _verify_independent_reproduction_evidence(self,digest: str,*,candidate_digest: str) -> IndependentReproductionEvidenceRC14:
        if self.independent_reproduction_store is None: raise PermissionError("independent reproduction evidence requires configured reproduction store")
        obj=self.artifacts.get(digest,expected_kind="independent_reproduction_evidence")
        if not isinstance(obj,IndependentReproductionEvidenceRC14) or not obj.qualified: raise PermissionError("independent reproduction evidence is not qualified")
        if obj.candidate_digest!=candidate_digest: raise PermissionError("independent reproduction evidence candidate mismatch")
        plan=self.independent_reproduction_store.get_plan(obj.reproduction_plan_digest)
        if plan.source_candidate_digest!=candidate_digest or plan.source_plan_digest!=obj.source_plan_digest: raise PermissionError("independent reproduction source binding mismatch")
        cert=self.independent_reproduction_store.certify(plan.digest)
        if cert.digest!=obj.reproduction_certificate_digest or not cert.eligible: raise PermissionError("independent reproduction evidence does not match replayed certificate")
        if not self.independent_reproduction_store.contains_event_head(obj.reproduction_protocol_head_digest): raise PermissionError("independent reproduction evidence references unknown protocol head")
        body=dict(obj.reproduction_certificate_body)
        for f in ("replicate_digests","reproducer_ids","reasons"): body[f]=tuple(body[f])
        if IndependentReproductionCertificateRC14(**body).digest!=cert.digest: raise PermissionError("embedded independent reproduction certificate does not replay")
        return obj

    @staticmethod
    def _kind_pass(plan, run, kind: FalsificationKind, *, require_present: bool = True) -> bool:
        case_ids = [case.case_id for case in plan.cases if case.kind is kind]
        if not case_ids:
            return not require_present
        by_id = {outcome.case_id: outcome for outcome in run.outcomes}
        return all(by_id[case_id].passed for case_id in case_ids)

    def build_record(self, proposal: LearningProposalRC14, *, candidate_digest: str,
                     evaluation_digest: str, attestation_digests: Sequence[str] = (),
                     continual_experiment_evidence_digest: str = "",
                     reasons: Sequence[str] = ()) -> QualificationRecordRC14:
        evaluation = self.artifacts.get(evaluation_digest, expected_kind="evaluation_bundle")
        if not isinstance(evaluation, EvaluationBundleRC14) or evaluation.candidate_digest != candidate_digest:
            raise PermissionError("evaluation bundle candidate mismatch")
        plan = self.falsification.get(evaluation.falsification_plan_digest)
        run = self.falsification.get_run(evaluation.falsification_run_digest)
        if plan.candidate_digest != candidate_digest or run.candidate_digest != candidate_digest:
            raise PermissionError("falsification evidence candidate mismatch")
        if run.environment_digest != evaluation.environment_digest:
            raise PermissionError("evaluation environment mismatch")
        result = run.validate_against(plan)

        retention = self._kind_pass(plan, run, FalsificationKind.RETENTION)
        negative = self._kind_pass(plan, run, FalsificationKind.NEGATIVE_CONTROL)
        security = self._kind_pass(plan, run, FalsificationKind.SECURITY)
        ood = self._kind_pass(plan, run, FalsificationKind.OOD)
        hidden = self._kind_pass(plan, run, FalsificationKind.FRESH_HIDDEN)
        has_ood = any(c.kind is FalsificationKind.OOD for c in plan.cases)
        has_hidden = any(c.kind is FalsificationKind.FRESH_HIDDEN for c in plan.cases)
        fresh_ood = bool(has_ood and has_hidden and ood and hidden)

        attestations = self._attestations(
            candidate_digest=candidate_digest,
            evaluation_digest=evaluation_digest,
            attestation_digests=attestation_digests,
        )
        independently_verified = bool(attestations.get("independent_verification"))
        replicated = independently_verified and bool(attestations.get("replication"))
        independently_reproduced = fresh_ood and replicated and bool(attestations.get("independent_reproduction"))
        operator_approved = bool(attestations.get("operator_approval"))
        internally_consistent = bool(result["passed"])

        experiment_evidence_digest = str(continual_experiment_evidence_digest or "")
        if proposal.coordinates.permanence == PermanenceLevel.L6_ISOLATED_NEURAL_MEMORY:
            if not experiment_evidence_digest:
                raise PermissionError("L6 qualification requires preregistered continual-experiment evidence")
            self._verify_continual_experiment_evidence(experiment_evidence_digest, candidate_digest=candidate_digest, proposal_digest=proposal.digest)
        elif experiment_evidence_digest:
            self._verify_continual_experiment_evidence(experiment_evidence_digest, candidate_digest=candidate_digest, proposal_digest=proposal.digest)

        evidence = QualificationEvidenceRC14(
            candidate_digest=candidate_digest,
            evaluation_digest=evaluation_digest,
            falsification_plan_digest=plan.digest,
            falsification_run_digest=run.digest,
            pass_rate=float(result["pass_rate"]),
            mandatory_classes_passed=bool(result["mandatory_classes_passed"]),
            retention_passed=retention,
            negative_controls_passed=negative,
            security_passed=security,
            fresh_ood_validated=fresh_ood,
            fresh_hidden_passed=hidden if has_hidden else False,
            evaluator_id=run.evaluator_id,
            environment_digest=run.environment_digest,
            task_set_digest=run.task_set_digest,
            attestation_digests=tuple(str(x) for x in attestation_digests),
            independently_verified=independently_verified,
            replicated=replicated,
            independently_reproduced=independently_reproduced,
            operator_approved=operator_approved,
            continual_experiment_evidence_digest=experiment_evidence_digest,
        )
        evidence_digest = self.artifacts.put(evidence)

        record = QualificationRecordRC14(
            proposal_digest=proposal.digest,
            candidate_digest=candidate_digest,
            evaluation_digest=evaluation_digest,
            policy_digest=self.policy.digest,
            passed=internally_consistent,
            qualification_evidence_digest=evidence_digest,
            internally_consistent=internally_consistent,
            independently_verified=independently_verified,
            replicated=replicated,
            fresh_ood_validated=fresh_ood and replicated,
            independently_reproduced=independently_reproduced,
            retention_passed=retention,
            negative_controls_passed=negative,
            security_passed=security,
            operator_approved=operator_approved,
            continual_experiment_evidence_digest=experiment_evidence_digest,
            reasons=tuple(str(x) for x in reasons),
        )
        # Policy evaluation here is intentionally separate from whether the evaluation itself
        # passed. High-permanence proposals can remain non-qualifying until attestations exist.
        return record

    def build_and_require(self, proposal: LearningProposalRC14, *, candidate_digest: str,
                          evaluation_digest: str, attestation_digests: Sequence[str] = (),
                          continual_experiment_evidence_digest: str = "",
                          reasons: Sequence[str] = ()) -> QualificationRecordRC14:
        record = self.build_record(
            proposal,
            candidate_digest=candidate_digest,
            evaluation_digest=evaluation_digest,
            attestation_digests=attestation_digests,
            continual_experiment_evidence_digest=continual_experiment_evidence_digest,
            reasons=reasons,
        )
        self.policy.require_verified(proposal, record)
        return record
