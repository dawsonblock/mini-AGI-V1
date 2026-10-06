from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from pathlib import Path
from typing import Any, Mapping

from egai.common.canonical import digest, validate_digest

from .authority_v141 import PromotionAuthorityV141, PromotionAuthorizationV141, PromotionAuthorizationValidatorV141
from .models import LearningProposalV14
from .policy import GovernancePolicyV14
from .storage import GovernanceDB, ImmutableCAS


@dataclass(frozen=True)
class RuntimeManifestV141:
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
    schema: str = "mini-agi-v14.1-runtime-manifest-v1"

    def __post_init__(self) -> None:
        for name in (
            "candidate_digest", "authorization_digest", "production_identity_digest", "belief_snapshot_digest",
            "skill_snapshot_digest", "adapter_set_digest", "routing_policy_digest", "execution_runtime_digest"
        ):
            validate_digest(getattr(self, name))

    @property
    def digest(self) -> str:
        return digest(self)


class GovernedContinualRuntimeV141:
    """Canonical v14.1 artifact-chain coordinator.

    This is intentionally narrower than an autonomous learner. It makes the
    authoritative candidate -> build -> evaluation -> qualification -> signed
    promotion -> runtime activation path concrete and fail-closed. Builders,
    evaluators and repair providers remain external roles.
    """

    def __init__(self, root: str | Path, *, policy: GovernancePolicyV14,
                 promotion_signer, promotion_verifier, trusted_promotion_key_ids):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.cas = ImmutableCAS(self.root / "cas")
        self.db = GovernanceDB(self.root / "governance.sqlite3")
        self.promotion = PromotionAuthorityV141(
            governance_db=self.db, signer=promotion_signer,
            authority_generation=policy.authority_generation,
            policy_generation=policy.policy_generation,
        )
        self.promotion_validator = PromotionAuthorizationValidatorV141(
            verifier=promotion_verifier, trusted_key_ids=trusted_promotion_key_ids,
            authority_generation=policy.authority_generation,
            policy_generation=policy.policy_generation,
        )

    def register_candidate(self, proposal: LearningProposalV14, *, candidate_payload: Mapping[str, Any], actor: str = "candidate-builder") -> str:
        self.policy.require(proposal)
        proposal_digest = self.cas.put_json(asdict(proposal))
        body = {
            "schema": "mini-agi-v14.1-candidate-manifest-v1",
            "proposal_digest": proposal_digest,
            "production_identity_digest": proposal.production_identity_digest,
            "candidate_payload": dict(candidate_payload),
        }
        candidate_digest = self.cas.put_json(body)
        self.db.register_candidate(candidate_digest=candidate_digest, proposal_digest=proposal_digest, actor=actor)
        return candidate_digest

    def record_build(self, *, candidate_digest: str, build_payload: Mapping[str, Any], actor: str = "builder") -> str:
        validate_digest(candidate_digest)
        body = {"schema":"mini-agi-v14.1-build-manifest-v1","candidate_digest":candidate_digest,"build_payload":dict(build_payload)}
        build_digest = self.cas.put_json(body)
        self.db.record_build(candidate_digest=candidate_digest, build_digest=build_digest, actor=actor)
        return build_digest

    def record_evaluation(self, *, candidate_digest: str, build_digest: str, evaluation_payload: Mapping[str, Any], actor: str = "independent-evaluator") -> str:
        body = {"schema":"mini-agi-v14.1-evaluation-bundle-v1","candidate_digest":candidate_digest,"build_digest":build_digest,"evaluation_payload":dict(evaluation_payload)}
        evaluation_digest = self.cas.put_json(body)
        self.db.record_evaluation(candidate_digest=candidate_digest, build_digest=build_digest, evaluation_digest=evaluation_digest, actor=actor)
        return evaluation_digest

    def record_qualification(self, *, candidate_digest: str, evaluation_digest: str, decision: str,
                             metrics_payload: Mapping[str, Any], actor: str = "qualifier") -> str:
        body = {
            "schema":"mini-agi-v14.1-qualification-record-v1",
            "candidate_digest":candidate_digest,
            "evaluation_digest":evaluation_digest,
            "decision":str(decision).upper(),
            "authority_generation":self.policy.authority_generation,
            "policy_generation":self.policy.policy_generation,
            "metrics_payload":dict(metrics_payload),
        }
        qualification_digest = self.cas.put_json(body)
        self.db.record_qualification(
            candidate_digest=candidate_digest, evaluation_digest=evaluation_digest,
            qualification_digest=qualification_digest, decision=decision,
            authority_generation=self.policy.authority_generation,
            policy_generation=self.policy.policy_generation, actor=actor,
        )
        return qualification_digest

    def authorize(self, *, candidate_digest: str, qualification_digest: str) -> PromotionAuthorizationV141:
        auth = self.promotion.authorize(candidate_digest=candidate_digest, qualification_digest=qualification_digest)
        stored = self.cas.put_json(asdict(auth))
        if stored != auth.digest:
            raise RuntimeError("promotion authorization canonical digest mismatch")
        return auth

    def activate(self, *, auth: PromotionAuthorizationV141, production_identity_digest: str,
                 belief_snapshot_digest: str, skill_snapshot_digest: str, adapter_set_digest: str,
                 routing_policy_digest: str, execution_runtime_digest: str) -> RuntimeManifestV141:
        self.promotion_validator.validate(auth, candidate_digest=auth.candidate_digest, qualification_digest=auth.qualification_digest)
        manifest = RuntimeManifestV141(
            candidate_digest=auth.candidate_digest,
            authorization_digest=auth.digest,
            production_identity_digest=production_identity_digest,
            belief_snapshot_digest=belief_snapshot_digest,
            skill_snapshot_digest=skill_snapshot_digest,
            adapter_set_digest=adapter_set_digest,
            routing_policy_digest=routing_policy_digest,
            execution_runtime_digest=execution_runtime_digest,
            authority_generation=self.policy.authority_generation,
            policy_generation=self.policy.policy_generation,
            created_at=time.time(),
        )
        manifest_digest = self.cas.put_json(asdict(manifest))
        if manifest_digest != manifest.digest:
            raise RuntimeError("runtime manifest canonical digest mismatch")
        activation_body = {
            "schema":"mini-agi-v14.1-activation-receipt-v1",
            "candidate_digest":auth.candidate_digest,
            "authorization_digest":auth.digest,
            "runtime_manifest_digest":manifest_digest,
        }
        activation_digest = self.cas.put_json(activation_body)
        self.db.activate(
            candidate_digest=auth.candidate_digest, authorization_digest=auth.digest,
            runtime_manifest_digest=manifest_digest, activation_digest=activation_digest,
        )
        return manifest

    def close(self) -> None:
        self.db.close()
