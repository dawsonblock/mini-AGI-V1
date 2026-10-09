from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from kvcontinual.execution.authority import Ed25519ReceiptSigner, Ed25519ReceiptVerifier
from kvcontinual.execution.cache.block import ArtifactQualification, ExecutionArtifact


@dataclass(frozen=True)
class ArtifactReceiptBody:
    source_content_digest: str
    execution_identity_digest: str
    model_weights_digest: str
    tokenizer_digest: str
    capture_manifest_digest: str
    artifact_digest: str
    algorithm: str
    seam_width: int
    numerical_tolerance: float
    backend_identity: str
    hardware_fingerprint: str
    kernel_build_digest: str
    oracle_result_digest: str
    policy_generation: str

    def validate(self) -> None:
        required = (
            self.source_content_digest, self.execution_identity_digest,
            self.model_weights_digest, self.tokenizer_digest,
            self.capture_manifest_digest, self.artifact_digest,
            self.backend_identity, self.hardware_fingerprint,
            self.kernel_build_digest, self.oracle_result_digest,
            self.policy_generation,
        )
        if not all(required):
            raise ValueError("artifact receipt has empty identity fields")
        if self.seam_width < 0 or not math.isfinite(self.numerical_tolerance) or self.numerical_tolerance < 0:
            raise ValueError("invalid seam/tolerance in artifact receipt")
        if not isinstance(self.algorithm, str) or not self.algorithm or len(self.algorithm) > 128:
            raise ValueError("invalid algorithm in artifact receipt")


def issue_artifact_receipt(signer: Ed25519ReceiptSigner, body: ArtifactReceiptBody) -> dict:
    body.validate()
    return signer.issue(asdict(body))


def verify_artifact_receipt(verifier: Ed25519ReceiptVerifier, receipt: dict, body: ArtifactReceiptBody) -> bool:
    body.validate()
    return verifier.verify(receipt, expected_body=asdict(body))


def qualification_from_signed_receipt(*, receipt: dict, body: ArtifactReceiptBody,
                                      max_logit_kl: float, min_top1_agreement: float,
                                      same_hidden_input_verified: bool = True,
                                      conv_boundary_complete: bool = True,
                                      attention_relocation_complete: bool = True,
                                      oracle_qualified: bool = True) -> ArtifactQualification:
    return ArtifactQualification(
        same_hidden_input_verified=same_hidden_input_verified,
        conv_boundary_complete=conv_boundary_complete,
        attention_relocation_complete=attention_relocation_complete,
        oracle_qualified=oracle_qualified,
        max_logit_kl=max_logit_kl,
        min_top1_agreement=min_top1_agreement,
        hardware_fingerprint=body.hardware_fingerprint,
        capture_manifest_digest=body.capture_manifest_digest,
        receipt_digest=str(receipt.get("body_sha256", "")),
        source_content_digest=body.source_content_digest,
        execution_identity_digest=body.execution_identity_digest,
        model_weights_digest=body.model_weights_digest,
        verifier_key_id=str(receipt.get("key_id", "")),
        signature_b64=str(receipt.get("signature_b64", "")),
    )


def body_for_artifact(artifact: ExecutionArtifact, *, artifact_digest: str,
                      backend_identity: str, hardware_fingerprint: str,
                      kernel_build_digest: str, oracle_result_digest: str,
                      policy_generation: str, numerical_tolerance: float,
                      algorithm: str = "HYPIC_SEAM8") -> ArtifactReceiptBody:
    model = artifact.identity.model
    return ArtifactReceiptBody(
        source_content_digest=artifact.source_content_digest,
        execution_identity_digest=artifact.identity.digest,
        model_weights_digest=model.base_weights,
        tokenizer_digest=model.tokenizer,
        capture_manifest_digest=artifact.capture_manifest_digest,
        artifact_digest=artifact_digest,
        algorithm=algorithm,
        seam_width=artifact.fixed_seam_width,
        numerical_tolerance=float(numerical_tolerance),
        backend_identity=backend_identity,
        hardware_fingerprint=hardware_fingerprint,
        kernel_build_digest=kernel_build_digest,
        oracle_result_digest=oracle_result_digest,
        policy_generation=policy_generation,
    )
