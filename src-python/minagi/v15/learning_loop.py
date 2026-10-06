from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

from egai.common.canonical import digest, validate_digest
from minagi.integration.qw3_state import ServedArtifactManifest
from .skill_ir import SkillIR, SkillIRCompiler, SkillPolicyBundle
from .sealed_learning import SealedLearningExperiment


@dataclass(frozen=True)
class QualifiedSkillPolicyCandidate:
    source_candidate_digest: str
    skill_ir_digest: str
    skill_policy_bundle_digest: str
    sealed_experiment_digest: str
    frozen_baseline_report_digest: str
    skill_policy_root: str
    decision: str
    schema: str = "mini-agi-v15.1-qualified-skill-policy-candidate-v1"

    def __post_init__(self) -> None:
        for d in (
            self.source_candidate_digest, self.skill_ir_digest, self.skill_policy_bundle_digest,
            self.sealed_experiment_digest, self.frozen_baseline_report_digest,
        ):
            validate_digest(d)
        if len(self.skill_policy_root) != 64 or any(c not in "0123456789abcdef" for c in self.skill_policy_root):
            raise ValueError("skill_policy_root must be lowercase sha256 hex")
        if self.decision not in {"PASS", "BLOCK"}:
            raise ValueError("invalid learning decision")

    @property
    def digest(self) -> str:
        return digest(self)


class GovernedSkillLearningLoop:
    """Proposal/build layer connecting trajectory learning to governed serving.

    This object has no promotion authority. It can compile a transfer-qualified
    skill, bind it to a sealed A0/A1 experiment, and prepare a new served-artifact
    manifest candidate. The existing qualification/promotion/StateEpoch chain
    remains the only path that can make that manifest SERVABLE.
    """

    def __init__(self, *, cas, compiler: SkillIRCompiler | None = None):
        self.cas = cas
        self.compiler = compiler or SkillIRCompiler()

    def prepare(self, *, candidate, experiment: SealedLearningExperiment,
                existing_skills: Iterable[SkillIR] = ()) -> tuple[QualifiedSkillPolicyCandidate, SkillPolicyBundle, SkillIR]:
        skill = self.compiler.compile(candidate)
        sd = self.cas.put_json(asdict(skill))
        if sd != skill.digest:
            raise RuntimeError("SkillIR CAS digest mismatch")
        bundle = SkillPolicyBundle.from_skills(tuple(existing_skills) + (skill,))
        bd = self.cas.put_json(asdict(bundle))
        if bd != bundle.digest:
            raise RuntimeError("skill policy bundle CAS digest mismatch")
        er = self.cas.put_json(asdict(experiment.baseline_report))
        if er != experiment.baseline_report.digest:
            raise RuntimeError("baseline report CAS digest mismatch")
        ed = self.cas.put_json(asdict(experiment))
        if ed != experiment.digest:
            raise RuntimeError("sealed experiment CAS digest mismatch")
        decision = experiment.baseline_report.decision
        q = QualifiedSkillPolicyCandidate(
            source_candidate_digest=candidate.digest,
            skill_ir_digest=skill.digest,
            skill_policy_bundle_digest=bundle.digest,
            sealed_experiment_digest=experiment.digest,
            frozen_baseline_report_digest=experiment.baseline_report.digest,
            skill_policy_root=bundle.root_hex,
            decision=decision,
        )
        qd = self.cas.put_json(asdict(q))
        if qd != q.digest:
            raise RuntimeError("qualified skill-policy candidate CAS digest mismatch")
        return q, bundle, skill

    @staticmethod
    def prepare_served_manifest(*, base: ServedArtifactManifest,
                                qualified: QualifiedSkillPolicyCandidate) -> ServedArtifactManifest:
        if qualified.decision != "PASS":
            raise PermissionError("blocked learning experiment cannot alter served state")
        return replace(base, skill_policy_root=qualified.skill_policy_root)
