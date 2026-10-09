from __future__ import annotations

from dataclasses import asdict, dataclass
import time
import uuid

from egai.common.canonical import digest, validate_digest
from .verification import EpisodeVerificationValidator


@dataclass(frozen=True)
class ExperienceRecordV144:
    episode_id: str
    prompt_digest: str
    attempted_output_digest: str
    evidence_root_digest: str
    production_identity_digest: str
    observed_at: float
    schema: str = "mini-agi-v14.1-alpha4-experience-record-v1"

    def __post_init__(self):
        if not self.episode_id:
            raise ValueError("episode_id required")
        for d in (self.prompt_digest, self.attempted_output_digest, self.evidence_root_digest, self.production_identity_digest):
            validate_digest(d)

    @property
    def digest(self):
        return digest(self)


@dataclass(frozen=True)
class VerifiedEvidenceRecordV144:
    experience_digest: str
    evidence_root_digest: str
    verification_receipt_digest: str
    verified_at: float
    schema: str = "mini-agi-v14.1-alpha4-verified-evidence-record-v1"

    def __post_init__(self):
        for d in (self.experience_digest, self.evidence_root_digest, self.verification_receipt_digest):
            validate_digest(d)

    @property
    def digest(self):
        return digest(self)


class ExperiencePipelineV144:
    """Canonical experience -> signed verification -> verified-evidence path."""

    def __init__(self, *, cas, governance_db, verification_client, verification_validator: EpisodeVerificationValidator):
        self.cas = cas
        self.db = governance_db
        self.verification_client = verification_client
        self.verification_validator = verification_validator

    def ingest(self, *, prompt: str, attempted_output: str, repaired_output: str, expected_output: str,
               evidence_payload, production_identity_digest: str, score: float = 1.0, importance: float = 0.5,
               scoring_policy_id: str = "exact-repair-v1", pass_threshold: float = 1.0,
               episode_id: str | None = None):
        validate_digest(production_identity_digest)
        episode_id = episode_id or "EXP-" + uuid.uuid4().hex
        evidence_root = self.cas.put_json({"schema": "mini-agi-v14.1-alpha4-evidence-payload-v1", "payload": evidence_payload})
        prompt_digest = digest({"prompt": str(prompt)})
        attempt_digest = digest({"output": str(attempted_output)})
        exp = ExperienceRecordV144(episode_id, prompt_digest, attempt_digest, evidence_root,
                                   production_identity_digest, time.time())
        exp_digest = self.cas.put_json(asdict(exp))
        if exp_digest != exp.digest:
            raise RuntimeError("experience canonical digest mismatch")
        receipt = self.verification_client.issue(
            episode_id=episode_id, prompt=prompt, attempted_output=attempted_output,
            repaired_output=repaired_output, evidence_root_digest=evidence_root,
            score=float(score), importance=float(importance), expected_output=expected_output,
            scoring_policy_id=scoring_policy_id, pass_threshold=float(pass_threshold),
        )
        self.verification_validator.validate(
            receipt, episode_id=episode_id, prompt=prompt, attempted_output=attempted_output,
            repaired_output=repaired_output, expected_output=expected_output,
            scoring_policy_id=scoring_policy_id,
        )
        receipt_digest = self.cas.put_json(asdict(receipt))
        if receipt_digest != receipt.digest:
            raise RuntimeError("verification receipt canonical digest mismatch")
        verified = VerifiedEvidenceRecordV144(exp_digest, evidence_root, receipt_digest, time.time())
        verified_digest = self.cas.put_json(asdict(verified))
        if verified_digest != verified.digest:
            raise RuntimeError("verified-evidence canonical digest mismatch")
        self.db.record_verified_experience_v144(
            experience_digest=exp_digest, evidence_root_digest=evidence_root,
            verification_receipt_digest=receipt_digest, verified_evidence_digest=verified_digest,
        )
        return exp, receipt, verified
