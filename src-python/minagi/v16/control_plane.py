from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Callable

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope


@dataclass(frozen=True)
class ConvergedQualificationPolicyV160:
    require_forward_transfer: bool = True
    require_plasticity_closure: bool = True
    require_runtime_closure2: bool = True
    require_dream_evidence_for_search_change: bool = True
    allow_foundation_update: bool = False
    schema: str = "mini-agi-v16-converged-policy-v1"

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class ConvergedEvidenceBundleV160:
    candidate_digest: str
    request_digest: str
    plasticity_report_digest: str
    forward_transfer_receipt_digest: str
    runtime_closure_digest: str
    dream_qualification_digest: str = ""
    search_policy_changed: bool = False
    foundation_changed: bool = False
    schema: str = "mini-agi-v16-converged-evidence-v1"

    def __post_init__(self) -> None:
        for name in (
            "candidate_digest",
            "request_digest",
            "plasticity_report_digest",
            "forward_transfer_receipt_digest",
            "runtime_closure_digest",
        ):
            validate_digest(getattr(self, name))
        if self.dream_qualification_digest:
            validate_digest(self.dream_qualification_digest)

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class ConvergedQualificationV160:
    candidate_digest: str
    evidence_bundle_digest: str
    policy_digest: str
    passed: bool
    gates: dict[str, bool]
    evaluator_id: str
    created_at: str
    key_id: str = ""
    signature_b64: str = ""
    schema: str = "mini-agi-v16-converged-qualification-v1"

    @property
    def digest(self) -> str:
        return digest(asdict(replace(self, key_id="", signature_b64="")))


class IndependentConvergedQualifierV160:
    """Independent release gate over evidence from all four planes.

    This class has no model/skill/plasticity execution methods.  It can only
    verify already-produced evidence and sign a qualification.  Runtime
    promotion remains a separate StateEpoch/promotion-authority action.
    """

    def __init__(
        self, *, evaluator_id: str, signer, verifier, trust,
        plasticity_verifier: Callable[[str], bool],
        forward_transfer_verifier: Callable[[str], bool],
        runtime_closure_verifier: Callable[[str], bool],
        dream_verifier: Callable[[str], bool] | None = None,
        policy: ConvergedQualificationPolicyV160 | None = None,
    ):
        self.evaluator_id = evaluator_id
        self.signer = signer
        self.verifier = verifier
        self.trust = trust
        self.plasticity_verifier = plasticity_verifier
        self.forward_transfer_verifier = forward_transfer_verifier
        self.runtime_closure_verifier = runtime_closure_verifier
        self.dream_verifier = dream_verifier
        self.policy = policy or ConvergedQualificationPolicyV160()

    def qualify(self, *, evidence: ConvergedEvidenceBundleV160) -> ConvergedQualificationV160:
        p = self.policy
        plasticity_ok = bool(self.plasticity_verifier(evidence.plasticity_report_digest))
        transfer_ok = bool(self.forward_transfer_verifier(evidence.forward_transfer_receipt_digest))
        runtime_ok = bool(self.runtime_closure_verifier(evidence.runtime_closure_digest))
        if evidence.search_policy_changed and p.require_dream_evidence_for_search_change:
            dream_ok = bool(
                evidence.dream_qualification_digest
                and self.dream_verifier is not None
                and self.dream_verifier(evidence.dream_qualification_digest)
            )
        else:
            dream_ok = True
        gates = {
            "plasticity_closure": (plasticity_ok if p.require_plasticity_closure else True),
            "forward_transfer": (transfer_ok if p.require_forward_transfer else True),
            "runtime_closure2": (runtime_ok if p.require_runtime_closure2 else True),
            "dream_search_qualification": dream_ok,
            "foundation_update_policy": (not evidence.foundation_changed) or p.allow_foundation_update,
        }
        unsigned = ConvergedQualificationV160(
            candidate_digest=evidence.candidate_digest,
            evidence_bundle_digest=evidence.digest,
            policy_digest=p.digest,
            passed=all(gates.values()),
            gates=gates,
            evaluator_id=self.evaluator_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        env = self.signer.sign(asdict(unsigned))
        return replace(unsigned, key_id=env.key_id, signature_b64=env.signature_b64)

    def verify(self, qualification: ConvergedQualificationV160) -> bool:
        self.trust.require("evaluator", qualification.key_id)
        unsigned = replace(qualification, key_id="", signature_b64="")
        return self.verifier.verify(asdict(unsigned), SignedEnvelope(qualification.key_id, qualification.signature_b64))
