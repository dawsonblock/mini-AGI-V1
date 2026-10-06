"""Canonical production API for governed continual serving.

v15 keeps the alpha5 authority trunk as the sole mutation authority and adds a
learning-to-serving layer: transfer-qualified trajectories compile to SkillIR,
sealed fresh tasks run paired A0/A1 evaluation, and adapter candidates are bound
to verified trajectory datasets, immutable output bytes, retention/security gates,
and the exact adapter-set root carried into governed QW3 serving.
"""
from minagi.v14.models import EvidenceStrength, GovernanceCoordinates, LearningMechanism, LearningProposalV14 as LearningProposal, PermanenceLevel
from minagi.v14.policy import GovernancePolicyV14 as GovernancePolicy, DEFAULT_REQUIRED_EVIDENCE
from minagi.v14.storage import ImmutableCAS
from minagi.v14.storage_v145 import GovernanceDBV145 as GovernanceDB
from minagi.v14.runtime_v145 import GovernedContinualRuntimeV145 as GovernedContinualRuntime
from minagi.v14.runtime_v144 import RuntimeManifestV144 as RuntimeManifest, RUNTIME_COMPONENT_SCHEMAS_V144 as RUNTIME_COMPONENT_SCHEMAS
from minagi.v14.verification import EpisodeVerificationAuthority, EpisodeVerificationReceipt, EpisodeVerificationValidator
from minagi.v14.evidence_v144 import EvidenceStageReceiptV144 as EvidenceStageReceipt, EvidenceStageAuthorityV144 as EvidenceStageAuthority, EvidenceStageValidatorV144 as EvidenceStageValidator, EvidenceStrengthProofV144 as EvidenceStrengthProof, EvidenceStrengthAuthorityV144 as EvidenceStrengthAuthority, EvidenceStrengthValidatorV144 as EvidenceStrengthValidator
from minagi.v14.provenance_v145 import FalsificationCaseCommitmentV145 as FalsificationCaseCommitment, EvaluationCaseResultV145 as EvaluationCaseResult, EvaluationProvenanceValidatorV145 as EvaluationProvenanceValidator, BoundEvaluationAuthorityV145 as EvaluationAuthority, BoundEvaluationValidatorV145 as EvaluationValidator
from minagi.v14.state_epoch_v145 import StateEpochV145 as StateEpoch, StateEpochLeaseV145 as StateEpochLease, EpochTransitionReceiptV145 as EpochTransitionReceipt, EpochTransitionAuthorityV145 as EpochTransitionAuthority, EpochTransitionValidatorV145 as EpochTransitionValidator, StateEpochRegistryV145 as StateEpochRegistry
from minagi.v14.trajectory_v145 import CandidateStatusV145 as SkillCandidateStatus, TrajectoryStepV145 as TrajectoryStep, VerifiedTrajectoryV145 as VerifiedTrajectory, LocalLessonV145 as LocalLesson, ContradictionV145 as Contradiction, SkillCandidateArtifactV145 as SkillCandidateArtifact, InductionReportV145 as InductionReport, TrajectorySkillInducerV145 as TrajectorySkillInducer, HeldOutTransferQualifierV145 as HeldOutTransferQualifier
from minagi.v14.experiment_v145 import FrozenArmRecordV145 as FrozenArmRecord, FrozenBaselineReportV145 as FrozenBaselineReport, FrozenBaselineGateV145 as FrozenBaselineGate
from minagi.v14.persistence_v144 import BeliefRevisionV144 as BeliefRevision, SkillRevisionV144 as SkillRevision, ContentBoundPersistencePlannerV144 as ContentBoundPersistencePlanner
from minagi.v14.fresh_tasks_v143 import FreshTaskConsumptionReceiptV143 as FreshTaskConsumptionReceipt, FreshTaskMetadataStoreV143 as FreshTaskMetadataStore, FreshTaskVaultAuthorityV143 as FreshTaskVaultAuthority
from minagi.v14.services_v144 import VerificationAuthorityClientV144 as VerificationAuthorityClient, EvidenceStageAuthorityClientV144 as EvidenceStageAuthorityClient, EvidenceStrengthAuthorityClientV144 as EvidenceStrengthAuthorityClient
from minagi.v14.services_v143 import EvaluationAuthorityClientV143 as EvaluationAuthorityClient, QualificationAuthorityClientV143 as QualificationAuthorityClient, PromotionAuthorityClientV143 as PromotionAuthorityClient
from minagi.v14.qualification_v142 import QualificationPolicyV142 as QualificationPolicy
from minagi.v14.qualification_v143 import QualificationAuthorityV143 as QualificationAuthority, QualificationValidatorV143 as QualificationValidator, QualificationRecordV143 as QualificationRecord
from minagi.v14.authority_v143 import PromotionAuthorityV143 as PromotionAuthority, PromotionAuthorizationValidatorV143 as PromotionAuthorizationValidator, PromotionAuthorizationV143 as PromotionAuthorization
from minagi.integration.qw3_state import ServedArtifactManifest, QW3RuntimeState, GovernedServingContract, RuntimeStateMismatch
from .skill_ir import SkillInstruction, SkillIR, SkillIRCompiler, SkillIRExecutor, SkillPolicyBundle, SkillPolicyRuntime
from .sealed_learning import SealedPairResult, SealedLearningExperiment, FreshTaskPairedEvaluator
from .learning_loop import QualifiedSkillPolicyCandidate, GovernedSkillLearningLoop


from .adapter_learning import (
    AdapterTrainingExample, VerifiedAdapterDataset, AdapterTrainingPlan, AdapterFile,
    AdapterArtifactManifest, AdapterPayloadBinding, AdapterSetBundle, AdapterArtifactBuilder, MlxLoraCandidateRunner,
    AdapterArmOutcome, SealedAdapterPairResult, SealedAdapterExperiment,
    FreshTaskAdapterEvaluator, QualifiedAdapterSetCandidate, GovernedAdapterLearningLoop,
)

from .native_adapter import (
    NativeAdapterFile, NativeAdapterTensor, NativeAdapterBundle,
    NativeQW3AdapterCompiler,
)


from .runtime_closure import (
    MeasuredRuntimeClosure, MeasuredGovernedServingContract, PhysicalArtifactPaths,
    sha256_file, sha256_runtime_path, tokenizer_identity_sha256_v155,
)
from .artifact_binding import (
    MaterializedCanonicalArtifact, materialize_canonical_artifact,
    build_physically_bound_served_manifest,
)

from .retrieval_policy import (
    RetrievalPolicy, EpisodicMemoryRecord, RetrievalHit, RetrievalPolicyRuntime,
)
from .agent_runtime import (
    AgentExecutionReceipt, AgentExecutionResult, GovernedAgentRuntime,
)

__all__ = [name for name in globals() if not name.startswith("_")]

from .neural_eval import (
    NeuralRuntimeEvidence, NeuralArmResult, QW3NeuralArm,
    SealedNeuralPair, SealedNeuralExperiment, SealedNeuralAdapterEvaluator,
)

# Recompute after v15.7 public imports.
__all__ = [name for name in globals() if not name.startswith("_")]

from .neural_campaign import (
    NeuralGateSpec, FreshTaskCommitmentRef, NeuralArmRunSpec, NeuralCampaignPlan,
    QW3LaunchSpec, NeuralCampaignResultBundle, PreregisteredNeuralCampaignRunner,
)

__all__ = [name for name in globals() if not name.startswith("_")]
