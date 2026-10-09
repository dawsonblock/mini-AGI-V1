from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
import uuid

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope


@dataclass(frozen=True)
class EpisodeVerificationReceipt:
    receipt_id: str
    episode_id: str
    prompt_digest: str
    attempted_output_digest: str
    repaired_output_digest: str
    evidence_root_digest: str
    verifier_id: str
    score: float
    passed: bool
    importance: float
    issued_at: float
    expected_output_digest: str = ""
    scoring_policy_digest: str = ""
    pass_threshold: float = 1.0
    verifier_key_id: str = ""
    signature_b64: str = ""
    schema: str = "mini-agi-v14-episode-verification-v2"

    def __post_init__(self):
        if self.schema not in {"mini-agi-v14-episode-verification-v1","mini-agi-v14-episode-verification-v2"}:
            raise ValueError("unsupported verification receipt schema")
        for v in (self.prompt_digest,self.attempted_output_digest,self.repaired_output_digest,self.evidence_root_digest): validate_digest(v)
        if self.expected_output_digest: validate_digest(self.expected_output_digest)
        if self.scoring_policy_digest: validate_digest(self.scoring_policy_digest)
        if not self.episode_id or not self.verifier_id: raise ValueError("episode_id and verifier_id required")
        if not 0.0 <= float(self.score) <= 1.0: raise ValueError("score must be in [0,1]")
        if not 0.0 <= float(self.importance) <= 1.0: raise ValueError("importance must be in [0,1]")
        if not 0.0 <= float(self.pass_threshold) <= 1.0: raise ValueError("pass_threshold must be in [0,1]")
        if bool(self.passed) != (float(self.score) >= float(self.pass_threshold)):
            raise ValueError("passed must be derived from score/pass_threshold")

    @property
    def digest(self): return digest(self)
    def unsigned(self): return replace(self,verifier_key_id="",signature_b64="")


class EpisodeVerificationAuthority:
    def __init__(self, *, verifier_id: str, signer): self.verifier_id=str(verifier_id); self.signer=signer
    def issue(self, *, episode_id: str, prompt: str, attempted_output: str, repaired_output: str,
              evidence_root_digest: str, score: float, importance: float=.5,
              expected_output: str | None=None, scoring_policy_id: str="exact-repair-v1", pass_threshold: float=1.0) -> EpisodeVerificationReceipt:
        expected = repaired_output if expected_output is None else expected_output
        receipt=EpisodeVerificationReceipt(
            receipt_id="EVR-"+uuid.uuid4().hex,episode_id=str(episode_id),prompt_digest=digest({"prompt":str(prompt)}),
            attempted_output_digest=digest({"output":str(attempted_output)}),repaired_output_digest=digest({"output":str(repaired_output)}),
            evidence_root_digest=evidence_root_digest,verifier_id=self.verifier_id,score=float(score),
            passed=float(score)>=float(pass_threshold),importance=float(importance),issued_at=time.time(),
            expected_output_digest=digest({"expected_output":str(expected)}),
            scoring_policy_digest=digest({"scoring_policy_id":str(scoring_policy_id)}),pass_threshold=float(pass_threshold))
        env=self.signer.sign(asdict(receipt.unsigned())); return replace(receipt,verifier_key_id=env.key_id,signature_b64=env.signature_b64)


class EpisodeVerificationValidator:
    def __init__(self, *, verifier, trusted_key_ids): self.verifier=verifier; self.trusted=set(str(x) for x in trusted_key_ids)
    def validate(self, receipt: EpisodeVerificationReceipt, *, episode_id: str, prompt: str, attempted_output: str,
                 repaired_output: str | None=None, expected_output: str | None=None, scoring_policy_id: str | None=None) -> bool:
        if receipt.verifier_key_id not in self.trusted: raise PermissionError("untrusted verification authority")
        if not self.verifier.verify(asdict(receipt.unsigned()),SignedEnvelope(receipt.verifier_key_id,receipt.signature_b64)):
            raise PermissionError("invalid verification receipt signature")
        expected={"episode_id":str(episode_id),"prompt_digest":digest({"prompt":str(prompt)}),
                  "attempted_output_digest":digest({"output":str(attempted_output)})}
        for k,v in expected.items():
            if getattr(receipt,k)!=v: raise PermissionError(f"verification receipt binding mismatch: {k}")
        if repaired_output is not None and receipt.repaired_output_digest != digest({"output":str(repaired_output)}):
            raise PermissionError("verification receipt binding mismatch: repaired_output")
        if expected_output is not None and receipt.expected_output_digest != digest({"expected_output":str(expected_output)}):
            raise PermissionError("verification receipt binding mismatch: expected_output")
        if scoring_policy_id is not None and receipt.scoring_policy_digest != digest({"scoring_policy_id":str(scoring_policy_id)}):
            raise PermissionError("verification receipt binding mismatch: scoring_policy")
        if not receipt.passed or receipt.score < receipt.pass_threshold: raise PermissionError("episode verification did not pass")
        return True
