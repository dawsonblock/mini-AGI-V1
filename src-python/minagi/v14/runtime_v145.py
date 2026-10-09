from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .runtime_v144 import GovernedContinualRuntimeV144, RuntimeManifestV144
from .storage_v145 import GovernanceDBV145
from .state_epoch_v145 import StateEpochRegistryV145
from .trajectory_v145 import TrajectorySkillInducerV145
from .experiment_v145 import FrozenBaselineGateV145


class GovernedContinualRuntimeV145(GovernedContinualRuntimeV144):
    """Alpha5 convergence layer.

    V144 still owns the proven atomic mutation transaction. V145 adds mechanically
    bound evaluation provenance (through its configured evaluator/validator),
    epistemic dependency invalidation, trajectory-grounded skill proposal helpers,
    frozen-foundation A0/A1 proof gates, and a post-activation StateEpoch serving
    lifecycle. No new authority path bypasses V144 activation.
    """

    def __init__(self, root: str | Path, *, epoch_transition_validator=None, **kwargs):
        root=Path(root)
        db=kwargs.pop("governance_db",None) or GovernanceDBV145(root/"governance.sqlite3")
        super().__init__(root,governance_db=db,**kwargs)
        self.db=db
        self.trajectory_inducer=TrajectorySkillInducerV145()
        self.frozen_baseline_gate=FrozenBaselineGateV145()
        self.state_epochs=None if epoch_transition_validator is None else StateEpochRegistryV145(
            governance_db=self.db,cas=self.cas,transition_validator=epoch_transition_validator)

    def register_learned_dependency(self, *, child_digest: str, parent_digests, relation="depends_on"):
        return self.db.add_dependencies_v145(child_digest=child_digest,parent_digests=tuple(parent_digests),relation=relation)

    def revoke_evidence(self, *, evidence_digest: str, reason: str, authority_digest: str | None=None):
        return self.db.invalidate_cascade_v145(root_digest=evidence_digest,reason=reason,authority_digest=authority_digest)

    def induce_skill_candidates(self, trajectories):
        report=self.trajectory_inducer.induce(tuple(trajectories))
        rd=self.cas.put_json(asdict(report))
        if rd != report.digest: raise RuntimeError("trajectory induction report digest mismatch")
        for c in report.candidates:
            cd=self.cas.put_json(asdict(c))
            if cd != c.digest: raise RuntimeError("skill candidate digest mismatch")
            self.db.add_dependencies_v145(child_digest=c.digest,parent_digests=c.supporting_evidence,relation="evidence_support")
            self.db.add_dependencies_v145(child_digest=c.digest,parent_digests=c.supporting_trajectory_digests,relation="trajectory_support")
        return report

    def evaluate_frozen_baseline(self, a0, a1):
        report=self.frozen_baseline_gate.evaluate(tuple(a0),tuple(a1))
        rd=self.cas.put_json(asdict(report))
        if rd != report.digest: raise RuntimeError("frozen-baseline report digest mismatch")
        return report

    def prepare_state_epoch(self, *, manifest: RuntimeManifestV144, authorization):
        if self.state_epochs is None: raise RuntimeError("StateEpoch transition validator not configured")
        return self.state_epochs.prepare(runtime_manifest_digest=manifest.digest,
                                         qualification_digest=manifest.qualification_digest,
                                         authorization_digest=authorization.digest)

    def activate_into_epoch(self, *, auth, manifest: RuntimeManifestV144, epoch):
        if self.state_epochs is None: raise RuntimeError("StateEpoch transition validator not configured")
        activation=self.activate(auth=auth,manifest=manifest)
        self.state_epochs.mark_local_commit(epoch=epoch,activation_digest=activation)
        self.db.add_dependencies_v145(child_digest=manifest.digest,
            parent_digests=(manifest.qualification_digest,manifest.artifact_root_digest),relation="runtime_closure")
        self.db.add_dependencies_v145(child_digest=epoch.digest,parent_digests=(manifest.digest,auth.digest),relation="serving_closure")
        return activation
