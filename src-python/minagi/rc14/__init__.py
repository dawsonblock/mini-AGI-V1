"""RC14 converged governed complementary continual-learning layer."""

from .models import (
    CandidateState, EpochState, EvidenceStrength, ExecutionContext, GovernanceCoordinates,
    LearningMechanism, LearningProposalRC14, PermanenceLevel,
)
from .policy import GovernancePolicyRC14, GovernanceRule
from .authority_state import AuthorityState, AuthorityStateRegistry
from .authority_artifacts import (AuthorityArtifactStore, CandidateManifestRC14, BuildManifestRC14, EvaluationBundleRC14, QualificationEvidenceRC14, EvidenceAttestationRC14, ExternalWitnessRC14, RuntimeAttestationRC14, ContinualExperimentEvidenceRC14, IndependentReproductionEvidenceRC14, QualificationRecordRC14, PromotionDecisionRC14, PromotionAuthorizationRC14)
from .candidate_state import TransactionalCandidateStateStore
from .state_epoch import StateEpochRC14, StateEpochRegistryRC14
from .falsification import (
    FalsificationCase, FalsificationKind, FalsificationOutcome, FalsificationPlan,
    FalsificationRun, PreregisteredPlanStore,
)
from .fresh_tasks import DurableFreshTaskAuthority, FreshTaskLease, HiddenTaskCommitment
from .fresh_task_service import FreshTaskAuthorityClient, FreshTaskAuthorityAdminClient, FreshTaskAuthorityServer
from .context_guard import ExecutionContextGuard
from .qualification import QualificationEngineRC14
from .dream import DreamCandidate, DreamPolicyResearcher, DreamWeights
from .replay_policy import AdaptiveReplayScheduler, ModelTimeClock, ReplayDecision, ReplayMemorySignalRC14, ReplayBatchRC14
from .metrics import ContinualMetrics, backward_transfer, forgetting, forward_transfer, interference_matrix
from .belief_revision import BeliefEvidenceSignal, BeliefRevisionProposalRC14, SemanticBeliefRevisionEngine
from .skill_induction import VerifiedSkillEpisodeRC14, HeldOutSkillEvaluationRC14, SkillInductionCandidateRC14, CrossTaskSkillInductionEngine
from .experiments import ContinualExperimentMeasurement, ContinualExperimentResultRC14, ContinualExperimentHarnessRC14, ContinualExperimentLedgerRC14
from .experiment_protocol import PreregisteredContinualExperimentPlanRC14, ContinualExperimentReplicateRC14, ExperimentReplicateAttestationRC14, AblationQualificationCertificateRC14, PreregisteredContinualExperimentStoreRC14, DEFAULT_ABLATION_LADDER
from .executed_runs import (DatasetSplitManifestRC14, ExecutedRunReceiptRC14, ExecutedRunMatrixCertificateRC14, ExecutedRunStoreRC14, SubprocessJSONExperimentRunnerRC14)
from .independent_reproduction import (IndependentReproductionPlanRC14, IndependentReproductionReplicateRC14, IndependentReproductionCertificateRC14, IndependentReproductionStoreRC14, REPRODUCTION_VARIANTS)
from .neural_memory import IsolatedNeuralMemoryManifestRC14, FrozenFoundationProofRC14, IsolatedNeuralMemoryQualificationRC14, IsolatedNeuralMemoryQualificationEngineRC14
from .orchestrator import GovernedContinualOrchestratorRC14
# GovernedRC14System (rc14/system.py) was retired: it was written against an
# RC13-era GovernedSystem that was never imported (state_epochs /
# runtime_activation / verify_control_plane exist nowhere in this tree), so it
# could not be constructed and nothing instantiated it. The module remains in
# git history and its donor archive identity in SOURCE_PROVENANCE.json — see
# RC14-001 in docs/research/REMAINING_DEFECTS_V1641.md.

__all__ = [name for name in globals() if not name.startswith("_")]
