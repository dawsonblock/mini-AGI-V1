from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
from typing import Any

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope


@dataclass(frozen=True)
class EvaluationCaseResultV143:
    case_id: str
    category: str
    score: float
    passed: bool
    evidence_digest: str
    ring: str = ""
    schema: str = "mini-agi-v14.1-alpha3-evaluation-case-v1"
    def __post_init__(self):
        if not self.case_id or not self.category: raise ValueError("evaluation case identity/category required")
        validate_digest(self.evidence_digest)
        if self.ring and self.ring not in {f"R{i}" for i in range(7)}: raise ValueError("invalid transfer ring")
    @property
    def digest(self): return digest(self)


@dataclass(frozen=True)
class SignedEvaluationBundleV143:
    candidate_digest: str
    build_digest: str
    case_results: tuple[EvaluationCaseResultV143, ...]
    raw_results_digest: str
    evaluator_id: str
    evaluator_generation: int
    issued_at: float
    signer_key_id: str = ""
    signature_b64: str = ""
    schema: str = "mini-agi-v14.1-alpha3-signed-evaluation-bundle-v1"
    def __post_init__(self):
        validate_digest(self.candidate_digest); validate_digest(self.build_digest); validate_digest(self.raw_results_digest)
        if not self.case_results: raise ValueError("raw evaluation cases required")
        if not self.evaluator_id or self.evaluator_generation < 0: raise ValueError("invalid evaluator identity/generation")
    def unsigned(self): return replace(self, signer_key_id="", signature_b64="")
    @property
    def digest(self): return digest(self)


def derive_metrics(cases: tuple[EvaluationCaseResultV143,...]) -> dict[str,Any]:
    def scores(cat): return [float(c.score) for c in cases if c.category == cat]
    def passed(cat):
        xs=[c for c in cases if c.category == cat]
        return bool(xs) and all(bool(c.passed) for c in xs)
    ret=scores("retention"); sec=scores("security"); forget=scores("forgetting"); ood=scores("ood_delta")
    rings=sorted({c.ring for c in cases if c.category == "transfer" and c.ring and c.passed})
    return {
        "retention": min(ret) if ret else -1.0,
        "security": min(sec) if sec else -1.0,
        "forgetting": max(forget) if forget else 1e9,
        "ood_delta": min(ood) if ood else -1e9,
        "transfer_rings": rings,
        "negative_controls_passed": passed("negative_control"),
        "fresh_hidden_validated": passed("fresh_hidden"),
        "falsification_passed": passed("falsification"),
    }


class EvaluationAuthorityV143:
    def __init__(self, *, evaluator_id: str, evaluator_generation: int, signer, cas):
        self.evaluator_id=str(evaluator_id); self.evaluator_generation=int(evaluator_generation); self.signer=signer; self.cas=cas
    def issue(self, *, candidate_digest: str, build_digest: str, case_results: tuple[EvaluationCaseResultV143,...]):
        validate_digest(candidate_digest); validate_digest(build_digest)
        cases=tuple(case_results)
        for c in cases:
            try:self.cas.get_bytes(c.evidence_digest)
            except Exception as exc: raise PermissionError("evaluation case references missing raw evidence") from exc
        raw=self.cas.put_json({"schema":"mini-agi-v14.1-alpha3-raw-evaluation-results-v1","cases":[asdict(c) for c in cases]})
        bundle=SignedEvaluationBundleV143(candidate_digest,build_digest,cases,raw,self.evaluator_id,self.evaluator_generation,time.time())
        env=self.signer.sign(asdict(bundle.unsigned()))
        return replace(bundle,signer_key_id=env.key_id,signature_b64=env.signature_b64)


class EvaluationValidatorV143:
    def __init__(self, *, verifier, trusted_key_ids, evaluator_generation: int, cas):
        self.verifier=verifier; self.trusted=set(str(x) for x in trusted_key_ids); self.generation=int(evaluator_generation); self.cas=cas
    def validate(self,bundle:SignedEvaluationBundleV143,*,candidate_digest:str,build_digest:str)->bool:
        if bundle.signer_key_id not in self.trusted: raise PermissionError("untrusted evaluator key")
        if bundle.evaluator_generation != self.generation: raise PermissionError("evaluator generation mismatch")
        if bundle.candidate_digest != candidate_digest or bundle.build_digest != build_digest: raise PermissionError("evaluation binding mismatch")
        for c in bundle.case_results:
            try:self.cas.get_bytes(c.evidence_digest)
            except Exception as exc: raise PermissionError("raw evaluation evidence unavailable") from exc
        raw_expected=digest({"schema":"mini-agi-v14.1-alpha3-raw-evaluation-results-v1","cases":[asdict(c) for c in bundle.case_results]})
        if raw_expected != bundle.raw_results_digest: raise PermissionError("raw evaluation digest mismatch")
        if not self.verifier.verify(asdict(bundle.unsigned()),SignedEnvelope(bundle.signer_key_id,bundle.signature_b64)):
            raise PermissionError("invalid evaluation signature")
        return True
