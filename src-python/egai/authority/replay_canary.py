from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone

from egai.authority.replay_policy import ReplayPolicyReceipt
from egai.common.canonical import digest
from egai.common.crypto import SignedEnvelope


@dataclass(frozen=True)
class LiveCanaryGate:
    min_cases: int = 20
    min_mean_gain: float = 0.0
    max_loss_rate: float = 0.10
    require_security_clean: bool = True

    def __post_init__(self):
        if self.min_cases < 1:
            raise ValueError("min_cases must be positive")
        if not 0 <= self.max_loss_rate <= 1:
            raise ValueError("max_loss_rate must be in [0,1]")


@dataclass(frozen=True)
class LiveCanaryCase:
    case_id: str
    baseline_score: float
    candidate_score: float
    evidence_digest: str
    security_ok: bool = True

    def __post_init__(self):
        if not self.evidence_digest:
            raise ValueError("live canary cases require evidence_digest")
        for x in (self.baseline_score, self.candidate_score):
            if not 0.0 <= float(x) <= 1.0:
                raise ValueError("scores must be in [0,1]")


@dataclass(frozen=True)
class LiveCanaryReceipt:
    receipt_id: str
    offline_replay_receipt_digest: str
    candidate_policy_digest: str
    case_set_digest: str
    metrics: dict
    gates: dict
    passed: bool
    promotion_eligible: bool
    deployable: bool
    evaluator_id: str
    created_at: str
    key_id: str = ""
    signature_b64: str = ""

    @property
    def digest(self):
        return digest(asdict(replace(self, key_id="", signature_b64="")))


class ReplayLiveCanaryAuthority:
    """Evaluate a replay-qualified policy on real, evidence-backed canary outcomes.

    Passing this authority only makes the policy eligible for independent promotion. It does not
    activate or deploy the policy itself.
    """

    def __init__(self, evaluator_id, signer, verifier, trust):
        self.evaluator_id = evaluator_id
        self.signer = signer
        self.verifier = verifier
        self.trust = trust

    def verify_offline_receipt(self, receipt: ReplayPolicyReceipt) -> bool:
        self.trust.require("evaluator", receipt.key_id)
        return self.verifier.verify(
            asdict(replace(receipt, key_id="", signature_b64="")),
            SignedEnvelope(receipt.key_id, receipt.signature_b64),
        )

    def evaluate(self, receipt_id: str, offline: ReplayPolicyReceipt, cases, gate: LiveCanaryGate | None = None):
        gate = gate or LiveCanaryGate()
        if not self.verify_offline_receipt(offline):
            raise PermissionError("invalid offline replay receipt")
        if not offline.qualified_offline or not offline.requires_live_canary:
            raise PermissionError("policy is not eligible for live canary")
        xs = tuple(cases)
        if len(xs) < gate.min_cases:
            raise ValueError("insufficient live canary cases")
        gains = [float(c.candidate_score) - float(c.baseline_score) for c in xs]
        mean_gain = sum(gains) / len(gains)
        loss_rate = sum(g < 0 for g in gains) / len(gains)
        security_clean = all(c.security_ok for c in xs)
        gates = {
            "case_count": len(xs) >= gate.min_cases,
            "mean_gain": mean_gain >= gate.min_mean_gain,
            "loss_rate": loss_rate <= gate.max_loss_rate,
            "security_clean": security_clean if gate.require_security_clean else True,
            "evidence_closed": all(c.evidence_digest for c in xs),
        }
        metrics = {
            "n": len(xs),
            "mean_gain": mean_gain,
            "loss_rate": loss_rate,
            "candidate_mean": sum(float(c.candidate_score) for c in xs) / len(xs),
            "baseline_mean": sum(float(c.baseline_score) for c in xs) / len(xs),
        }
        passed = all(gates.values())
        case_set_digest = digest([asdict(c) for c in xs])
        unsigned = LiveCanaryReceipt(
            receipt_id,
            offline.digest,
            offline.candidate_policy_digest,
            case_set_digest,
            metrics,
            gates,
            passed,
            passed,
            False,
            self.evaluator_id,
            datetime.now(timezone.utc).isoformat(),
        )
        env = self.signer.sign(asdict(unsigned))
        return replace(unsigned, key_id=env.key_id, signature_b64=env.signature_b64)
