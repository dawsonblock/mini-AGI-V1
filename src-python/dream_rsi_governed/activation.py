from __future__ import annotations

"""NOT THE LIVE CONTROL PLANE.

This module implements a competing activation step, which no live path reaches. It is retained
for provenance and for packaging coverage, but it is NOT the authority
that governs promotions.

The single live authority is `minagi.v161.authority`; the live promotion
path is `scripts/promote.py` -> `minagi.v161.authority`. The successor to
this module is `minagi.v161.trusted_launcher`.

See CONTROL_PLANE_MAP.md and CONTROL_PLANE_MANIFEST.json. Do not import
this module into a live path without recording that decision.
"""


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
