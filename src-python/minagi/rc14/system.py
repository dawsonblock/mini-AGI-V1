from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
from contextlib import contextmanager
from typing import Mapping, Any
import uuid
import time

from kvcontinual.execution.types import ReconstructionMode
from minagi.egai.canonical import sha256_json
from minagi.system import GovernedSystem
from .authority_artifacts import AuthorityArtifactStore
from .authority_state import AuthorityState, AuthorityStateRegistry
from .candidate_state import TransactionalCandidateStateStore
from .fresh_tasks import DurableFreshTaskAuthority
from .context_guard import ExecutionContextGuard
from .dream import DreamPolicyResearcher
from .falsification import PreregisteredPlanStore
from .replay_policy import AdaptiveReplayScheduler, ModelTimeClock
from .belief_revision import SemanticBeliefRevisionEngine
from .skill_induction import CrossTaskSkillInductionEngine
from .experiments import ContinualExperimentHarnessRC14, ContinualExperimentLedgerRC14
from .experiment_protocol import PreregisteredContinualExperimentStoreRC14
from .independent_reproduction import IndependentReproductionStoreRC14
from .executed_runs import ExecutedRunStoreRC14, ExecutedRunReceiptRC14
from .neural_memory import IsolatedNeuralMemoryQualificationEngineRC14
from .authority_artifacts import ContinualExperimentEvidenceRC14, IndependentReproductionEvidenceRC14
from .orchestrator import GovernedContinualOrchestratorRC14
from .qualification import QualificationEngineRC14
from .policy import GovernancePolicyRC14
from .state_epoch import StateEpochRegistryRC14, StateEpochRC14


class GovernedRC14System(GovernedSystem):
    """RC14.7 composition with preregistered empirical-learning evidence and witnessed serving closure.

    NOT WIRED — donor artifact. This class was written against an RC13-era
    ``GovernedSystem`` that provided a state-epoch registry
    (``self.state_epochs``), a runtime activation authority
    (``self.runtime_activation``) and ``verify_control_plane()``; that base
    was never imported into this tree. As written it cannot be constructed
    (the two verifier keyword arguments below are not accepted by
    ``minagi.system.GovernedSystem``), and nothing imports the class.
    Tracked as RC14-001 in docs/research/REMAINING_DEFECTS_V1641.md —
    decision needed: port the missing base, wire the class up, or retire
    it (the rest of the rc14 modules are wired and exercised by
    scripts/validation/import_probe.py and the donor-import tests)."""

    def __init__(self, root, backend, *, research_verifier, promotion_signer,
                 rc14_epoch_verifiers: Mapping[str, Any], rc14_candidate_verifiers: Mapping[str, Any],
                 rc14_authority_verifier: Any, policy: GovernancePolicyRC14 | None = None, budget=None,
                 runtime_transition_verifier=None, state_epoch_verifier=None, component_resolvers: Mapping[str, Any] | None = None,
                 fresh_task_authority: Any | None = None, qualification_attestation_verifiers: Mapping[str, Any] | None = None,
                 external_witness_verifiers: Mapping[str, Any] | None = None, runtime_attestation_verifiers: Mapping[str, Any] | None = None,
                 independent_reproduction_verifiers: Mapping[str, Any] | None = None, experiment_evaluator_verifiers: Mapping[str, Any] | None = None,
                 executed_run_evaluator_verifiers: Mapping[str, Any] | None = None):
        super().__init__(root, backend, research_verifier=research_verifier, promotion_signer=promotion_signer,
                         budget=budget, runtime_transition_verifier=runtime_transition_verifier, state_epoch_verifier=state_epoch_verifier)
        # Keep the inherited RC13 registry exactly where the superclass expects it.
        self.legacy_state_epochs = self.state_epochs
        self.policy = policy or GovernancePolicyRC14()
        self.component_resolvers = dict(component_resolvers or {})
        self.rc14_artifacts = AuthorityArtifactStore(Path(root) / "rc14_authority_artifacts")
        self.rc14_authority = AuthorityStateRegistry(Path(root) / "rc14_authority", verifier=rc14_authority_verifier)
        self.candidate_states = TransactionalCandidateStateStore(Path(root) / "rc14_candidates", verifiers=rc14_candidate_verifiers,
                                                                  artifacts=self.rc14_artifacts, authority_registry=self.rc14_authority)
        self.fresh_tasks = fresh_task_authority or DurableFreshTaskAuthority(Path(root) / "rc14_fresh_tasks")
        self.execution_context_guard = ExecutionContextGuard()
        self.rc14_state_epochs = StateEpochRegistryRC14(
            Path(root) / "state_epochs_rc14", verifiers=rc14_epoch_verifiers,
            artifacts=self.rc14_artifacts, authority_registry=self.rc14_authority,
            witness_verifiers=dict(external_witness_verifiers or {}),
            runtime_attestation_verifiers=dict(runtime_attestation_verifiers or {}),
            strict_serving_artifacts=True,
        )
        self.falsification_plans = PreregisteredPlanStore(Path(root) / "rc14_falsification")
        self.continual_experiment_protocol = PreregisteredContinualExperimentStoreRC14(Path(root)/"rc14_science"/"protocol",replicate_verifiers=dict(experiment_evaluator_verifiers or {}),strict_signed_replicates=True)
        self.executed_runs = ExecutedRunStoreRC14(Path(root)/"rc14_science"/"executed_runs", protocol=self.continual_experiment_protocol, evaluator_verifiers=dict(executed_run_evaluator_verifiers or experiment_evaluator_verifiers or {}))
        self.independent_reproduction = IndependentReproductionStoreRC14(Path(root)/"rc14_science"/"independent_reproduction",source_protocol=self.continual_experiment_protocol,reproducer_verifiers=dict(independent_reproduction_verifiers or {}))
        self.qualification_engine = QualificationEngineRC14(self.rc14_artifacts,self.falsification_plans,self.policy,dict(qualification_attestation_verifiers or {}),self.continual_experiment_protocol,self.independent_reproduction,self.executed_runs)
        self.dream = DreamPolicyResearcher()
        self.replay_clock = ModelTimeClock()
        self.replay_scheduler = AdaptiveReplayScheduler()
        self.belief_revision_engine = SemanticBeliefRevisionEngine()
        self.skill_induction_engine = CrossTaskSkillInductionEngine()
        self.continual_experiment_harness = ContinualExperimentHarnessRC14()
        self.continual_experiments = ContinualExperimentLedgerRC14(Path(root) / "rc14_science" / "experiments")
        self.isolated_neural_memory_qualifier = IsolatedNeuralMemoryQualificationEngineRC14()
        self.orchestrator = GovernedContinualOrchestratorRC14(self.candidate_states, self.rc14_artifacts, self.policy)

    def activate_rc14_authority_state(self, state: AuthorityState, *, receipt: dict[str, Any]) -> str:
        if state.policy_digest != self.policy.digest or state.policy_generation != self.policy.policy_generation:
            raise PermissionError("AuthorityState must bind the active RC14 governance policy")
        return self.rc14_authority.activate(state, receipt=receipt)

    def register_request_learning(self, proposal, *, candidate_artifact_digest: str, actor: str = "candidate-builder"):
        """Register learning produced by a live request under the exact pinned StateEpoch."""
        context = self.require_execution_context()
        if proposal.source_epoch_digest != context.epoch_digest or proposal.source_request_id != context.request_id:
            raise PermissionError("request-derived learning proposal is not bound to current ExecutionContext")
        return self.orchestrator.register(proposal, candidate_artifact_digest=candidate_artifact_digest, actor=actor)

    def prepare_rc14_state_epoch_for_candidate(self, candidate_digest: str, components: Mapping[str, str] | None = None) -> StateEpochRC14:
        """Prepare an epoch by deriving promotion closure from the signed candidate history."""
        snapshot = self.candidate_states.snapshot(candidate_digest)
        if snapshot.get("state") != "AUTHORIZED":
            raise PermissionError("candidate must be exactly AUTHORIZED before StateEpoch preparation")
        chain = self.candidate_states.stage_chain(candidate_digest)
        qd = chain.get("QUALIFIED")
        ad = chain.get("AUTHORIZED")
        if not qd or not ad:
            raise PermissionError("candidate history lacks qualification/authorization closure")
        auth = self.rc14_artifacts.get(ad, expected_kind="promotion_authorization")
        if auth.candidate_digest != candidate_digest or auth.qualification_digest != qd:
            raise PermissionError("candidate authorization chain mismatch")
        measured = dict(components or {})
        derived = {
            "qualification_bundle_digest": qd,
            "promotion_decision_digest": auth.promotion_decision_digest,
            "promotion_authorization_digest": ad,
        }
        for key, value in derived.items():
            if key in measured and measured[key] != value:
                raise PermissionError(f"caller-supplied {key} conflicts with signed candidate history")
            measured[key] = value
        return self.prepare_rc14_state_epoch(measured)

    def prepare_rc14_state_epoch(self, components: Mapping[str, str]) -> StateEpochRC14:
        runtime_state = self.runtime_activation.state(); runtime_digest = runtime_state.get("runtime_digest")
        if not runtime_digest: raise RuntimeError("cannot prepare RC14 StateEpoch without active governed runtime manifest")
        measured = dict(components)
        expected = {
            "execution_manifest_digest": str(runtime_digest),
            "governance_policy_digest": self.policy.digest,
            "evidence_root": sha256_json(self.evidence.list_digests()),
            "belief_snapshot_digest": self.beliefs.snapshot_digest,
            "skill_graph_root": self.skills.snapshot_digest,
        }
        for key, value in expected.items():
            supplied = measured.get(key)
            if supplied is not None and supplied != value: raise PermissionError(f"RC14 StateEpoch {key} does not match measured authoritative state")
            measured[key] = value
        promotion_keys = {"qualification_bundle_digest", "promotion_decision_digest", "promotion_authorization_digest"}
        missing_promotion = [key for key in promotion_keys if key not in measured]
        if missing_promotion:
            raise PermissionError("RC14 StateEpoch requires explicit promotion closure: " + ", ".join(sorted(missing_promotion)))
        # Prove the promoted candidate is actually at or beyond AUTHORIZED in the signed
        # candidate lifecycle before an epoch can even be prepared.
        auth = self.rc14_artifacts.get(str(measured["promotion_authorization_digest"]), expected_kind="promotion_authorization")
        snapshot = self.candidate_states.snapshot(auth.candidate_digest)
        authorized_or_later = {"AUTHORIZED", "EPOCH_PREPARED", "COMMITTED", "WITNESSED", "ATTESTED", "SERVABLE"}
        if snapshot.get("state") not in authorized_or_later:
            raise PermissionError("RC14 StateEpoch candidate has not reached signed AUTHORIZED state")
        for key in self.rc14_state_epochs.REQUIRED_COMPONENTS:
            if key in expected or key in promotion_keys:
                continue
            resolver = self.component_resolvers.get(key)
            if resolver is None:
                raise PermissionError(f"measured component resolver required for {key}; caller-supplied digest is not authoritative")
            value = str(resolver())
            supplied = measured.get(key)
            if supplied is not None and supplied != value:
                raise PermissionError(f"RC14 StateEpoch {key} does not match measured component resolver")
            measured[key] = value
        return self.rc14_state_epochs.prepare(measured)

    def _reconstruct_with_context(self, context, segment_ids, identity, mode, **kwargs):
        epoch = self.rc14_state_epochs.validate_context(context)
        runtime_state = self.runtime_activation.state()
        if runtime_state.get("runtime_digest") != epoch.execution_manifest_digest:
            mode = ReconstructionMode.EXACT
        with self.execution_context_guard.bind(context):
            return super().reconstruct(segment_ids, identity, mode, **kwargs)

    def require_execution_context(self, *, epoch_digest: str | None = None):
        """Fail closed for request-scoped operations that must not reread mutable heads."""
        context = self.execution_context_guard.require(epoch_digest=epoch_digest)
        self.rc14_state_epochs.validate_context(context)
        return context

    def current_execution_context(self):
        return self.execution_context_guard.current()

    @contextmanager
    def request_context(self, request_id: str | None = None, *, ttl_seconds: int = 3600):
        """Pin one SERVABLE StateEpoch for the complete logical request.

        This is the preferred RC14.7 integration boundary for tool use, retrieval,
        reconstruction and request-derived learning. Nested reconstruction reuses the
        already-bound context instead of acquiring a second lease.
        """
        if self.execution_context_guard.current() is not None:
            raise PermissionError("nested RC14 request_context is not allowed")
        rid = request_id or ("REQ-" + uuid.uuid4().hex)
        with self.rc14_state_epochs.pin(rid, ttl_seconds=ttl_seconds) as context:
            with self.execution_context_guard.bind(context):
                yield context

    def learning_source_epoch_digest(self) -> str:
        """Bind request-derived learning proposals to the exact serving epoch that produced them."""
        return self.require_execution_context().epoch_digest

    def reconstruct(self, segment_ids, identity, mode=ReconstructionMode.EXACT, *, execution_context=None, request_id: str | None = None, **kwargs):
        if self.rc14_state_epochs.current_serving_digest() is None:
            return super().reconstruct(segment_ids, identity, mode, **kwargs)
        bound = self.execution_context_guard.current()
        if execution_context is not None:
            if bound is not None and (bound.lease_id != execution_context.lease_id or bound.epoch_digest != execution_context.epoch_digest):
                raise PermissionError("explicit ExecutionContext conflicts with the active request context")
            return self._reconstruct_with_context(execution_context, segment_ids, identity, mode, **kwargs)
        if bound is not None:
            return self._reconstruct_with_context(bound, segment_ids, identity, mode, **kwargs)
        with self.request_context(request_id) as context:
            return self._reconstruct_with_context(context, segment_ids, identity, mode, **kwargs)


    def seal_continual_experiment_plan(self, plan):
        """Seal the complete RC14.7 source ablation matrix before any result is admitted."""
        return self.continual_experiment_protocol.seal(plan)

    def prepare_preregistered_experiment_result(self,plan_digest: str,*,variant: str,seed: int,measurement,evaluator_id: str,observed_ns: int|None=None):
        plan=self.continual_experiment_protocol.get_plan(plan_digest)
        if measurement.variant!=variant or variant not in plan.variants or int(seed) not in plan.seeds: raise PermissionError("measurement is outside the preregistered experiment slot")
        result=self.continual_experiment_harness.evaluate(measurement); observed=int(observed_ns or time.time_ns())
        body={"schema":"egai-rc14-experiment-replicate-attestation-v1","plan_digest":plan.digest,"variant":variant,"seed":int(seed),"result_digest":result.digest,"evaluator_id":evaluator_id,"environment_digest":plan.environment_digest,"task_set_digest":plan.task_set_digest,"observed_ns":observed}
        return result,observed,body

    def record_preregistered_experiment_result(self,plan_digest: str,*,variant: str,seed: int,measurement,evaluator_id: str,receipt: Mapping[str,Any],observed_ns: int,result=None):
        plan=self.continual_experiment_protocol.get_plan(plan_digest); derived=self.continual_experiment_harness.evaluate(measurement)
        if measurement.variant!=variant: raise PermissionError("measurement variant does not match preregistered slot")
        if result is not None and result.digest!=derived.digest: raise PermissionError("caller-provided experiment result does not replay from measurement")
        self.continual_experiments.append(measurement,derived)
        rd=self.continual_experiment_protocol.record_result(plan_digest,variant=variant,seed=seed,result=derived,evaluator_id=evaluator_id,environment_digest=plan.environment_digest,task_set_digest=plan.task_set_digest,receipt=receipt,observed_ns=observed_ns)
        return derived,rd

    def build_isolated_neural_memory_experiment_evidence(self, proposal, *, plan_digest: str, manifest, proof) -> str:
        """Create authority-store evidence only after the sealed full ablation matrix qualifies L6 memory."""
        plan = self.continual_experiment_protocol.get_plan(plan_digest)
        certificate = self.continual_experiment_protocol.certify(plan_digest)
        execution_certificate = self.executed_runs.certify(plan_digest)
        if not execution_certificate.eligible:
            raise PermissionError("L6 evidence requires a complete successful signed external execution matrix: " + "; ".join(execution_certificate.reasons))
        qualification = self.isolated_neural_memory_qualifier.qualify(
            plan=plan, certificate=certificate, manifest=manifest, proof=proof
        )
        if plan.candidate_digest != manifest.candidate_digest:
            raise PermissionError("experiment plan/candidate mismatch")
        candidate = self.rc14_artifacts.get(manifest.candidate_digest, expected_kind="candidate_manifest")
        if candidate.proposal_digest != proposal.digest:
            raise PermissionError("isolated-memory proposal does not match candidate manifest")
        evidence = ContinualExperimentEvidenceRC14(
            candidate_digest=manifest.candidate_digest, proposal_digest=proposal.digest, plan_digest=plan.digest,
            ablation_certificate_digest=certificate.digest, neural_memory_manifest_digest=manifest.digest,
            frozen_foundation_proof_digest=proof.digest, neural_memory_qualification_digest=qualification.digest,
            experiment_protocol_head_digest=self.continual_experiment_protocol.head_digest(),
            neural_memory_manifest_body=asdict(manifest), frozen_foundation_proof_body=asdict(proof),
            neural_memory_qualification_body=asdict(qualification), qualified=qualification.qualified,
        )
        if not evidence.qualified:
            raise PermissionError("isolated neural memory failed preregistered experiment/isolation qualification: " + "; ".join(qualification.reasons))
        return self.rc14_artifacts.put(evidence)

    def register_experiment_dataset(self, manifest):
        return self.executed_runs.register_dataset(manifest)

    def record_executed_experiment_run(self, run: ExecutedRunReceiptRC14, measurement):
        return self.executed_runs.record(run, measurement)

    def seal_independent_reproduction_plan(self,plan): return self.independent_reproduction.seal(plan)

    def prepare_independent_reproduction_result(self,plan_digest: str,*,variant: str,seed: int,measurement,reproducer_id: str,observed_ns: int|None=None):
        plan=self.independent_reproduction.get_plan(plan_digest)
        if measurement.variant!=variant or variant not in plan.variants or int(seed) not in plan.seeds: raise PermissionError("measurement is outside the independent reproduction slot")
        result=self.continual_experiment_harness.evaluate(measurement); observed=int(observed_ns or time.time_ns())
        body={"schema":"egai-rc14-independent-reproduction-replicate-v1","plan_digest":plan.digest,"variant":variant,"seed":int(seed),"result_digest":result.digest,"reproducer_id":reproducer_id,"environment_digest":plan.reproduction_environment_digest,"task_set_digest":plan.reproduction_task_set_digest,"observed_ns":observed}
        return result,observed,body

    def record_independent_reproduction_measurement(self,plan_digest: str,*,variant: str,seed: int,measurement,reproducer_id: str,receipt: Mapping[str,Any],observed_ns: int,result=None):
        derived=self.continual_experiment_harness.evaluate(measurement)
        if measurement.variant!=variant: raise PermissionError("reproduction measurement variant does not match preregistered slot")
        if result is not None and result.digest!=derived.digest: raise PermissionError("caller-provided reproduction result does not replay from measurement")
        rd=self.independent_reproduction.record_result(plan_digest,variant=variant,seed=seed,result=derived,reproducer_id=reproducer_id,receipt=receipt,observed_ns=observed_ns)
        return derived,rd

    def build_independent_reproduction_evidence(self,reproduction_plan_digest: str) -> str:
        plan=self.independent_reproduction.get_plan(reproduction_plan_digest); cert=self.independent_reproduction.certify(plan.digest)
        if not cert.eligible: raise PermissionError("independent reproduction did not qualify: "+"; ".join(cert.reasons))
        obj=IndependentReproductionEvidenceRC14(plan.source_candidate_digest,plan.source_plan_digest,plan.digest,cert.digest,self.independent_reproduction.head_digest(),asdict(cert),True)
        return self.rc14_artifacts.put(obj)

    def evaluate_continual_experiment(self, measurement, *, persist: bool = True):
        """Compute and optionally persist measured continual-learning evidence.

        The result is evidence only. It does not qualify, promote or activate a candidate.
        """
        result = self.continual_experiment_harness.evaluate(measurement)
        event_digest = self.continual_experiments.append(measurement, result) if persist else None
        return result, event_digest

    def verify_control_plane(self):
        result = super().verify_control_plane()
        # `state_epochs` above is truly the inherited RC13 registry; RC14 is separate.
        result["legacy_state_epochs"] = result.pop("state_epochs")
        result["rc14_authority"] = self.rc14_authority.verify()
        result["rc14_authority_artifacts"] = self.rc14_artifacts.verify()
        result["rc14_candidate_states"] = self.candidate_states.verify()
        result["rc14_state_epochs"] = self.rc14_state_epochs.verify()
        result["rc14_falsification"] = self.falsification_plans.verify()
        result["rc14_continual_experiments"] = self.continual_experiments.verify()
        result["rc14_continual_experiment_protocol"] = self.continual_experiment_protocol.verify()
        result["rc14_executed_runs"] = self.executed_runs.verify()
        result["rc14_independent_reproduction"] = self.independent_reproduction.verify()
        result["rc14_policy_digest"] = self.policy.digest
        return result

    def verify_rc14_deep(self):
        """Cross-plane verification beyond individual store integrity checks."""
        result = self.verify_control_plane()
        current_authority = self.rc14_authority.current()
        if current_authority is not None:
            if current_authority.policy_digest != self.policy.digest:
                raise RuntimeError("current authority state policy digest does not match active RC14 policy")
            if current_authority.policy_generation != self.policy.policy_generation:
                raise RuntimeError("current authority state policy generation mismatch")
        serving = self.rc14_state_epochs.current_serving_digest()
        if serving is not None:
            epoch, state = self.rc14_state_epochs.get(serving)
            if state.value != "SERVABLE":
                raise RuntimeError("serving StateEpoch is not SERVABLE")
            if epoch.governance_policy_digest != self.policy.digest:
                raise RuntimeError("serving StateEpoch is bound to a different governance policy")
            self.rc14_state_epochs._validate_promotion_closure(epoch.__dict__)
        result["rc14_cross_plane"] = {
            "authority_policy_bound": current_authority is None or current_authority.policy_digest == self.policy.digest,
            "serving_epoch": serving,
            "status": "verified",
        }
        return result
