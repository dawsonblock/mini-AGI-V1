from __future__ import annotations
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone

from egai.common.canonical import digest
from egai.common.crypto import SignedEnvelope
from egai.replay.policy import PrefixOnlyPolicyEvaluator, ReplayObjective
from egai.replay.vault import ReplayHistoryVault, SealedReplaySuite


@dataclass(frozen=True)
class ReplayPolicyGate:
    alpha: float = 0.05
    min_mean_gain: float = 0.01
    min_win_rate: float = 0.60
    max_loss_rate: float = 0.20
    max_cost_delta: float = 0.25
    min_holdout_episodes: int = 12
    require_positive_ci: bool = True
    bootstrap_samples: int = 2000

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        if not 0 <= self.min_win_rate <= 1 or not 0 <= self.max_loss_rate <= 1:
            raise ValueError("rates must be in [0,1]")
        if self.min_holdout_episodes < 1:
            raise ValueError("min_holdout_episodes must be positive")


@dataclass(frozen=True)
class ReplayPolicyPlan:
    plan_id: str
    version: str
    history_commitment_digest: str
    baseline_policy_digest: str
    candidate_policy_digest: str
    objective: dict
    gate: dict
    created_at: str
    registrar_id: str
    key_id: str = ""
    signature_b64: str = ""

    @property
    def digest(self):
        return digest(asdict(replace(self, key_id="", signature_b64="")))


@dataclass(frozen=True)
class ReplayPolicyReceipt:
    receipt_id: str
    plan_digest: str
    sealed_suite_digest: str
    baseline_policy_digest: str
    candidate_policy_digest: str
    raw_result_digest: str
    comparison: dict
    gates: dict
    qualified_offline: bool
    requires_live_canary: bool
    deployable: bool
    evaluator_id: str
    created_at: str
    key_id: str = ""
    signature_b64: str = ""

    @property
    def digest(self):
        return digest(asdict(replace(self, key_id="", signature_b64="")))


class ReplayPolicyPlanRegistrar:
    def __init__(self, registrar_id, signer):
        self.registrar_id = registrar_id
        self.signer = signer

    def register(
        self,
        plan_id,
        history_commitment_digest,
        baseline_policy_digest,
        candidate_policy_digest,
        objective: ReplayObjective,
        gate: ReplayPolicyGate,
    ):
        unsigned = ReplayPolicyPlan(
            plan_id,
            "1",
            history_commitment_digest,
            baseline_policy_digest,
            candidate_policy_digest,
            asdict(objective),
            asdict(gate),
            datetime.now(timezone.utc).isoformat(),
            self.registrar_id,
        )
        env = self.signer.sign(asdict(unsigned))
        return replace(unsigned, key_id=env.key_id, signature_b64=env.signature_b64)


class GroundedReplayPolicyAuthority:
    """Offline historical-policy qualification.

    Passing this gate never authorizes deployment. A separate live canary and the independent
    PromotionAuthority remain mandatory. This authority evaluates only encrypted, grounded
    historical replay episodes; model-predicted/counterfactual outcomes are not accepted.
    """

    def __init__(self, evaluator_id, signer, verifier, trust, artifact_store):
        self.evaluator_id = evaluator_id
        self.signer = signer
        self.verifier = verifier
        self.trust = trust
        self.artifacts = artifact_store

    def verify_plan(self, plan: ReplayPolicyPlan) -> bool:
        self.trust.require("benchmark", plan.key_id)
        return self.verifier.verify(
            asdict(replace(plan, key_id="", signature_b64="")),
            SignedEnvelope(plan.key_id, plan.signature_b64),
        )

    def evaluate(
        self,
        receipt_id,
        plan: ReplayPolicyPlan,
        sealed: SealedReplaySuite,
        vault_key: bytes,
        baseline_policy,
        candidate_policy,
    ):
        if not self.verify_plan(plan):
            raise PermissionError("invalid replay policy plan signature")
        if sealed.sealed_digest != plan.history_commitment_digest:
            raise PermissionError("replay-history commitment mismatch")
        episodes = ReplayHistoryVault.open(sealed, vault_key)
        gate = ReplayPolicyGate(**plan.gate)
        if len(episodes) < gate.min_holdout_episodes:
            raise ValueError("insufficient held-out replay episodes")
        evaluator = PrefixOnlyPolicyEvaluator(ReplayObjective(**plan.objective))
        comparison, baseline_results, candidate_results = evaluator.compare(
            episodes,
            baseline_policy,
            candidate_policy,
            gate.bootstrap_samples,
            gate.alpha,
            seed=0,
        )
        gates = {
            "mean_gain": comparison.mean_gain >= gate.min_mean_gain,
            "positive_ci": (comparison.ci_low > 0 if gate.require_positive_ci else True),
            "significance": comparison.sign_test_p <= gate.alpha,
            "win_rate": comparison.win_rate >= gate.min_win_rate,
            "loss_rate": comparison.loss_rate <= gate.max_loss_rate,
            "cost_delta": comparison.cost_delta <= gate.max_cost_delta,
            "evidence_closed": comparison.evidence_closed,
            "holdout_size": len(episodes) >= gate.min_holdout_episodes,
        }
        raw = {
            "baseline": [asdict(x) for x in baseline_results],
            "candidate": [asdict(x) for x in candidate_results],
        }
        raw_digest = self.artifacts.put_bytes(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode())
        qualified = all(gates.values())
        unsigned = ReplayPolicyReceipt(
            receipt_id,
            plan.digest,
            sealed.sealed_digest,
            plan.baseline_policy_digest,
            plan.candidate_policy_digest,
            raw_digest,
            asdict(comparison),
            gates,
            qualified,
            True,
            False,
            self.evaluator_id,
            datetime.now(timezone.utc).isoformat(),
        )
        env = self.signer.sign(asdict(unsigned))
        return replace(unsigned, key_id=env.key_id, signature_b64=env.signature_b64)

    def verify_receipt(self, receipt: ReplayPolicyReceipt) -> bool:
        self.trust.require("evaluator", receipt.key_id)
        return self.verifier.verify(
            asdict(replace(receipt, key_id="", signature_b64="")),
            SignedEnvelope(receipt.key_id, receipt.signature_b64),
        )
