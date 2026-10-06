from __future__ import annotations

from datetime import datetime, timezone

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .models import CanaryAttestation, PromotionManifest, QualificationRecord
from .signing import public_key_id, sign_manifest, verify_manifest


class PromotionDenied(RuntimeError):
    pass


class PromotionAuthority:
    """Independent authority: proposal code cannot bypass qualification or live canary."""

    def __init__(self, signing_key: Ed25519PrivateKey):
        self.signing_key = signing_key

    def issue(self, record: QualificationRecord, canary: CanaryAttestation, artifact_root_digest: str, generation: int) -> PromotionManifest:
        if not record.passed:
            raise PromotionDenied("qualification record did not pass")
        if not canary.passed:
            raise PromotionDenied("live canary did not pass")
        if canary.candidate_policy_id != record.candidate.policy_id:
            raise PromotionDenied("canary candidate does not match qualification candidate")
        if canary.baseline_policy_id != record.baseline_policy_id:
            raise PromotionDenied("canary baseline does not match qualification baseline")
        manifest = PromotionManifest(
            candidate=record.candidate,
            qualification_digest=record.evidence_digest,
            artifact_root_digest=artifact_root_digest,
            issued_at=datetime.now(timezone.utc).isoformat(),
            signer_key_id=public_key_id(self.signing_key.public_key()),
            signature_b64="",
            generation=int(generation),
        )
        return sign_manifest(manifest, self.signing_key)

    @staticmethod
    def verify(manifest: PromotionManifest, public_key: Ed25519PublicKey, expected_qualification_digest: str | None = None) -> bool:
        if expected_qualification_digest and manifest.qualification_digest != expected_qualification_digest:
            return False
        return verify_manifest(manifest, public_key)
