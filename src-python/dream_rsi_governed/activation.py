from __future__ import annotations

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .models import CanaryAttestation, PromotionManifest, QualificationRecord
from .promotion import PromotionAuthority


class ActivationDenied(RuntimeError):
    pass


def authorize_activation(
    manifest: PromotionManifest,
    qualification: QualificationRecord,
    canary: CanaryAttestation,
    signer_public_key: Ed25519PublicKey,
    observed_artifact_root_digest: str,
) -> bool:
    checks = {
        "qualification_passed": qualification.passed,
        "canary_passed": canary.passed,
        "candidate_matches_qualification": manifest.candidate == qualification.candidate,
        "canary_candidate_matches": canary.candidate_policy_id == manifest.candidate.policy_id,
        "baseline_matches": canary.baseline_policy_id == qualification.baseline_policy_id,
        "qualification_digest_matches": manifest.qualification_digest == qualification.evidence_digest,
        "artifact_digest_matches": manifest.artifact_root_digest == observed_artifact_root_digest,
        "signature_valid": PromotionAuthority.verify(manifest, signer_public_key, qualification.evidence_digest),
    }
    if not all(checks.values()):
        failed = [k for k,v in checks.items() if not v]
        raise ActivationDenied("activation denied: " + ", ".join(failed))
    return True
