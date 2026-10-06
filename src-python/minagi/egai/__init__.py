"""EGAI v2 research-governance layer for mini-AGI v12.

The package enforces a three-plane separation:

* cognitive plane: solve tasks using promoted state;
* learning plane: discover and stage proposals only;
* promotion plane: independently qualify and authorize persistence.

It sits above the v11 production authority/execution substrate and does not
replace its signed production transition, serving, or hardware qualification.
"""

from .models import (
    AuthorityPlane,
    Belief,
    BeliefStatus,
    EvidenceClass,
    EvidenceOrigin,
    EvidenceRecord,
    Hypothesis,
    HypothesisStatus,
    LearningAction,
    LearningLevel,
    LearningProposal,
    ProposalOption,
    PromotionVerdict,
    SkillManifest,
    TRANSFER_RINGS,
)
from .evidence import EvidenceLedger, TrustedEvidenceIngestor
from .proposals import LeastPermanentPlasticityProposer, PlasticityDecisionContext
from .replay import (
    ExperienceNode,
    GroundedReplayEngine,
    PrefixReplayView,
    ReplayResult,
    ReplayWorld,
    ReplayWorldCompiler,
    SimulationRecord,
    VisibleReplayNode,
    WorldKind,
)
from .qualification import (
    IndependentQualificationGate,
    PromotionConstraints,
    PromotionDecision,
    QualificationBundle,
    QualificationMetrics,
)
from .improvement import FutureConsequence, ImprovementLedger, ImprovementRecord
from .skills import SkillRepository
from .beliefs import BeliefRepository
from .authority import CognitivePlane, LearningPlane, PromotionAuthority, PromotionAuthorizationBody
from .experiment import (
    ExperimentCheckpoint,
    FrozenFoundationExperimentSpec,
    FrozenFoundationProtocol,
    compute_fte,
)

__all__ = [
    "AuthorityPlane", "Belief", "BeliefStatus", "EvidenceClass", "EvidenceOrigin", "EvidenceRecord",
    "Hypothesis", "HypothesisStatus",
    "LearningAction", "LearningLevel", "LearningProposal", "ProposalOption",
    "PromotionVerdict", "SkillManifest", "TRANSFER_RINGS", "EvidenceLedger",
    "TrustedEvidenceIngestor", "LeastPermanentPlasticityProposer",
    "PlasticityDecisionContext", "ExperienceNode", "GroundedReplayEngine",
    "PrefixReplayView", "ReplayResult", "ReplayWorld", "ReplayWorldCompiler",
    "SimulationRecord", "VisibleReplayNode", "WorldKind",
    "IndependentQualificationGate", "PromotionConstraints", "PromotionDecision",
    "QualificationBundle", "QualificationMetrics", "FutureConsequence",
    "ImprovementLedger", "ImprovementRecord", "SkillRepository", "CognitivePlane",
    "LearningPlane", "PromotionAuthority", "PromotionAuthorizationBody", "BeliefRepository",
    "ExperimentCheckpoint", "FrozenFoundationExperimentSpec", "FrozenFoundationProtocol",
    "compute_fte",
]
