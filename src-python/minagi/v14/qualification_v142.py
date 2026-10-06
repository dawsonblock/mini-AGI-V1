from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
from typing import Any, Mapping

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope


@dataclass(frozen=True)
class SignedEvaluationBundleV142:
    candidate_digest: str
    build_digest: str
    metrics_payload: Mapping[str, Any]
    evaluator_id: str
    evaluator_generation: int
    issued_at: float
    signer_key_id: str = ""
    signature_b64: str = ""
    schema: str = "mini-agi-v14.1-alpha2-evaluation-bundle-v1"

    def __post_init__(self) -> None:
        validate_digest(self.candidate_digest); validate_digest(self.build_digest)
        if not self.evaluator_id or self.evaluator_generation < 0:
            raise ValueError("evaluator identity/generation required")

    @property
    def metrics_digest(self) -> str:
        return digest(dict(self.metrics_payload))

    def unsigned(self):
        return replace(self, signer_key_id="", signature_b64="")

    @property
    def digest(self) -> str:
        return digest(self)


class EvaluationAuthorityV142:
    def __init__(self, *, evaluator_id: str, evaluator_generation: int, signer):
        self.evaluator_id = str(evaluator_id)
        self.evaluator_generation = int(evaluator_generation)
        self.signer = signer

    def issue(self, *, candidate_digest: str, build_digest: str, metrics_payload: Mapping[str, Any]) -> SignedEvaluationBundleV142:
        bundle = SignedEvaluationBundleV142(candidate_digest, build_digest, dict(metrics_payload), self.evaluator_id,
                                            self.evaluator_generation, time.time())
        env = self.signer.sign(asdict(bundle.unsigned()))
        return replace(bundle, signer_key_id=env.key_id, signature_b64=env.signature_b64)


class EvaluationValidatorV142:
    def __init__(self, *, verifier, trusted_key_ids, evaluator_generation: int):
        self.verifier = verifier; self.trusted = set(str(x) for x in trusted_key_ids); self.generation = int(evaluator_generation)

    def validate(self, bundle: SignedEvaluationBundleV142, *, candidate_digest: str, build_digest: str) -> bool:
        if bundle.signer_key_id not in self.trusted:
            raise PermissionError("untrusted evaluator key")
        if bundle.evaluator_generation != self.generation:
            raise PermissionError("stale evaluator generation")
        if bundle.candidate_digest != candidate_digest or bundle.build_digest != build_digest:
            raise PermissionError("evaluation binding mismatch")
        if not self.verifier.verify(asdict(bundle.unsigned()), SignedEnvelope(bundle.signer_key_id, bundle.signature_b64)):
            raise PermissionError("invalid evaluation signature")
        return True


@dataclass(frozen=True)
class QualificationPolicyV142:
    min_retention: float = 0.95
    min_security: float = 1.0
    max_forgetting: float = 0.02
    min_ood_delta: float = -0.01
    required_transfer_rings: tuple[str, ...] = ("R0", "R1", "R2")
    require_negative_controls: bool = True
    require_fresh_hidden_validation: bool = True
    require_falsification: bool = True

    def decide(self, metrics: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
        reasons: list[str] = []
        def num(name: str, default: float) -> float:
            try: return float(metrics.get(name, default))
            except Exception: return default
        if num("retention", -1.0) < self.min_retention: reasons.append("retention below threshold")
        if num("security", -1.0) < self.min_security: reasons.append("security below threshold")
        if num("forgetting", 1e9) > self.max_forgetting: reasons.append("forgetting above threshold")
        if num("ood_delta", -1e9) < self.min_ood_delta: reasons.append("OOD delta below threshold")
        rings = metrics.get("transfer_rings", ())
        if isinstance(rings, Mapping): passed = {str(k) for k,v in rings.items() if bool(v)}
        else: passed = {str(x) for x in rings}
        missing = [r for r in self.required_transfer_rings if r not in passed]
        if missing: reasons.append("missing transfer rings: " + ",".join(missing))
        if self.require_negative_controls and not bool(metrics.get("negative_controls_passed", False)):
            reasons.append("negative controls failed or absent")
        if self.require_fresh_hidden_validation and not bool(metrics.get("fresh_hidden_validated", False)):
            reasons.append("fresh hidden validation absent")
        if self.require_falsification and not bool(metrics.get("falsification_passed", False)):
            reasons.append("falsification suite failed or absent")
        return ("REJECT", tuple(reasons)) if reasons else ("PROMOTE", ())


@dataclass(frozen=True)
class QualificationRecordV142:
    candidate_digest: str
    evaluation_digest: str
    metrics_digest: str
    decision: str
    reasons: tuple[str, ...]
    qualifier_id: str
    qualifier_generation: int
    authority_generation: int
    policy_generation: int
    issued_at: float
    signer_key_id: str = ""
    signature_b64: str = ""
    schema: str = "mini-agi-v14.1-alpha2-qualification-record-v1"

    def __post_init__(self):
        validate_digest(self.candidate_digest); validate_digest(self.evaluation_digest); validate_digest(self.metrics_digest)
        if self.decision not in {"PROMOTE", "REJECT"}: raise ValueError("invalid qualification decision")
        if not self.qualifier_id: raise ValueError("qualifier_id required")

    def unsigned(self): return replace(self, signer_key_id="", signature_b64="")
    @property
    def digest(self): return digest(self)


class QualificationAuthorityV142:
    def __init__(self, *, qualifier_id: str, qualifier_generation: int, authority_generation: int,
                 policy_generation: int, signer, evaluation_validator: EvaluationValidatorV142,
                 policy: QualificationPolicyV142 | None = None):
        self.qualifier_id=str(qualifier_id); self.qualifier_generation=int(qualifier_generation)
        self.authority_generation=int(authority_generation); self.policy_generation=int(policy_generation)
        self.signer=signer; self.evaluation_validator=evaluation_validator; self.policy=policy or QualificationPolicyV142()

    def qualify(self, bundle: SignedEvaluationBundleV142, *, candidate_digest: str, build_digest: str) -> QualificationRecordV142:
        self.evaluation_validator.validate(bundle, candidate_digest=candidate_digest, build_digest=build_digest)
        decision, reasons = self.policy.decide(bundle.metrics_payload)
        rec = QualificationRecordV142(candidate_digest, bundle.digest, bundle.metrics_digest, decision, reasons,
                                      self.qualifier_id, self.qualifier_generation, self.authority_generation,
                                      self.policy_generation, time.time())
        env = self.signer.sign(asdict(rec.unsigned()))
        return replace(rec, signer_key_id=env.key_id, signature_b64=env.signature_b64)


class QualificationValidatorV142:
    def __init__(self, *, verifier, trusted_key_ids, qualifier_generation: int, authority_generation: int, policy_generation: int):
        self.verifier=verifier; self.trusted=set(str(x) for x in trusted_key_ids)
        self.qgen=int(qualifier_generation); self.agen=int(authority_generation); self.pgen=int(policy_generation)

    def validate(self, record: QualificationRecordV142, *, candidate_digest: str, evaluation_digest: str, metrics_digest: str) -> bool:
        if record.signer_key_id not in self.trusted: raise PermissionError("untrusted qualifier key")
        if record.qualifier_generation != self.qgen or record.authority_generation != self.agen or record.policy_generation != self.pgen:
            raise PermissionError("qualification generation mismatch")
        if record.candidate_digest != candidate_digest or record.evaluation_digest != evaluation_digest or record.metrics_digest != metrics_digest:
            raise PermissionError("qualification binding mismatch")
        if not self.verifier.verify(asdict(record.unsigned()), SignedEnvelope(record.signer_key_id, record.signature_b64)):
            raise PermissionError("invalid qualification signature")
        return True
