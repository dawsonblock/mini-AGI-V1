from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import time

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope
from .evaluation_v143 import SignedEvaluationBundleV143
from .fresh_tasks_v143 import FreshTaskConsumptionReceiptV143


def _cas_value(cas, d: str):
    raw = json.loads(cas.get_bytes(validate_digest(d)).decode("utf-8"))
    return raw.get("value", raw)


@dataclass(frozen=True)
class FalsificationCaseCommitmentV145:
    candidate_digest: str
    case_id: str
    category: str
    task_digest: str
    expected_property_digest: str
    preregistration_digest: str
    created_at: float
    schema: str = "mini-agi-v14.1-alpha5-falsification-case-commitment-v1"

    def __post_init__(self):
        for d in (self.candidate_digest, self.task_digest, self.expected_property_digest, self.preregistration_digest):
            validate_digest(d)
        if not self.case_id or not self.category:
            raise ValueError("case identity/category required")

    @property
    def digest(self):
        return digest(self)


@dataclass(frozen=True)
class EvaluationCaseResultV145:
    case_id: str
    category: str
    score: float
    passed: bool
    evidence_digest: str
    ring: str = ""
    provenance_kind: str = "ordinary"
    provenance_digest: str = ""
    task_digest: str = ""
    schema: str = "mini-agi-v14.1-alpha5-evaluation-case-v1"

    def __post_init__(self):
        validate_digest(self.evidence_digest)
        if self.provenance_digest:
            validate_digest(self.provenance_digest)
        if self.task_digest:
            validate_digest(self.task_digest)
        if self.provenance_kind not in {"ordinary", "fresh_task", "falsification"}:
            raise ValueError("invalid evaluation provenance kind")
        if not self.case_id or not self.category:
            raise ValueError("evaluation case identity/category required")

    @property
    def digest(self):
        return digest(self)


class EvaluationProvenanceValidatorV145:
    """Mechanically binds fresh-hidden/falsification labels to authority artifacts."""

    def __init__(self, *, cas, verifier, trusted_fresh_task_key_ids, fresh_task_authority_generation: int):
        self.cas = cas
        self.verifier = verifier
        self.trusted_fresh = {str(x) for x in trusted_fresh_task_key_ids}
        self.fresh_generation = int(fresh_task_authority_generation)

    def _fresh_receipt(self, d: str) -> FreshTaskConsumptionReceiptV143:
        body = _cas_value(self.cas, d)
        rec = FreshTaskConsumptionReceiptV143(**body)
        if rec.digest != d:
            raise PermissionError("fresh-task receipt digest mismatch")
        if rec.signer_key_id not in self.trusted_fresh:
            raise PermissionError("untrusted fresh-task authority key")
        if rec.authority_generation != self.fresh_generation:
            raise PermissionError("fresh-task authority generation mismatch")
        if not self.verifier.verify(asdict(rec.unsigned()), SignedEnvelope(rec.signer_key_id, rec.signature_b64)):
            raise PermissionError("invalid fresh-task consumption signature")
        return rec

    def validate_case(self, case: EvaluationCaseResultV145, *, candidate_digest: str) -> bool:
        validate_digest(candidate_digest)
        # Raw execution evidence must exist regardless of provenance kind.
        self.cas.get_bytes(case.evidence_digest)
        if case.category == "fresh_hidden":
            if case.provenance_kind != "fresh_task" or not case.provenance_digest:
                raise PermissionError("fresh_hidden case lacks fresh-task provenance")
            rec = self._fresh_receipt(case.provenance_digest)
            if case.task_digest and case.task_digest != rec.task_digest:
                raise PermissionError("fresh_hidden case/task digest mismatch")
        elif case.category in {"falsification", "negative_control"}:
            if case.provenance_kind != "falsification" or not case.provenance_digest:
                raise PermissionError(f"{case.category} case lacks preregistered falsification provenance")
            body = _cas_value(self.cas, case.provenance_digest)
            commitment = FalsificationCaseCommitmentV145(**body)
            if commitment.digest != case.provenance_digest:
                raise PermissionError("falsification commitment digest mismatch")
            if commitment.candidate_digest != candidate_digest or commitment.case_id != case.case_id:
                raise PermissionError("falsification commitment binding mismatch")
            if commitment.category != case.category:
                raise PermissionError("falsification category binding mismatch")
            if case.task_digest and commitment.task_digest != case.task_digest:
                raise PermissionError("falsification task binding mismatch")
        elif case.provenance_kind != "ordinary":
            raise PermissionError("ordinary evaluation category has unexpected privileged provenance")
        return True


class BoundEvaluationAuthorityV145:
    """Evaluation signer that refuses unbound privileged evaluation labels."""

    def __init__(self, *, evaluator_id: str, evaluator_generation: int, signer, cas,
                 provenance_validator: EvaluationProvenanceValidatorV145):
        self.evaluator_id = str(evaluator_id)
        self.evaluator_generation = int(evaluator_generation)
        self.signer = signer
        self.cas = cas
        self.provenance_validator = provenance_validator

    def issue(self, *, candidate_digest: str, build_digest: str,
              case_results: tuple[EvaluationCaseResultV145, ...]):
        validate_digest(candidate_digest); validate_digest(build_digest)
        cases = tuple(case_results)
        for case in cases:
            if not isinstance(case, EvaluationCaseResultV145):
                raise TypeError("alpha5 evaluator requires EvaluationCaseResultV145")
            self.provenance_validator.validate_case(case, candidate_digest=candidate_digest)
        raw = self.cas.put_json({
            "schema": "mini-agi-v14.1-alpha5-raw-evaluation-results-v1",
            "cases": [asdict(c) for c in cases],
        })
        bundle = SignedEvaluationBundleV143(
            candidate_digest, build_digest, cases, raw,
            self.evaluator_id, self.evaluator_generation, time.time(),
        )
        env = self.signer.sign(asdict(bundle.unsigned()))
        return replace(bundle, signer_key_id=env.key_id, signature_b64=env.signature_b64)


class BoundEvaluationValidatorV145:
    """Drop-in replacement for V143 validator, with provenance validation first."""

    def __init__(self, *, verifier, trusted_key_ids, evaluator_generation: int, cas,
                 provenance_validator: EvaluationProvenanceValidatorV145):
        self.verifier = verifier
        self.trusted = {str(x) for x in trusted_key_ids}
        self.generation = int(evaluator_generation)
        self.cas = cas
        self.provenance_validator = provenance_validator

    def validate(self, bundle: SignedEvaluationBundleV143, *, candidate_digest: str, build_digest: str) -> bool:
        if bundle.signer_key_id not in self.trusted:
            raise PermissionError("untrusted evaluator key")
        if bundle.evaluator_generation != self.generation:
            raise PermissionError("evaluator generation mismatch")
        if bundle.candidate_digest != candidate_digest or bundle.build_digest != build_digest:
            raise PermissionError("evaluation binding mismatch")
        for case in bundle.case_results:
            if not isinstance(case, EvaluationCaseResultV145):
                raise TypeError("alpha5 qualification requires provenance-bound evaluation cases")
            self.provenance_validator.validate_case(case, candidate_digest=candidate_digest)
        expected_raw = digest({
            "schema": "mini-agi-v14.1-alpha5-raw-evaluation-results-v1",
            "cases": [asdict(c) for c in bundle.case_results],
        })
        if expected_raw != bundle.raw_results_digest:
            raise PermissionError("raw evaluation digest mismatch")
        if not self.verifier.verify(asdict(bundle.unsigned()), SignedEnvelope(bundle.signer_key_id, bundle.signature_b64)):
            raise PermissionError("invalid evaluation signature")
        return True
