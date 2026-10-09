"""v16.4.2 admission grants (UPGRADE_PLAN §3.1).

A `MeasuredSnapshot` proves artifact *content*; it carries no
deployment authority. The artifact that authorizes a supervisor to
activate a candidate is the `AdmissionGrantV1` — a short-lived signed
object issued by the `admission` authority role after the full chain
(plan → qualification → promotion → manifest → measured bytes →
revocation) verifies.

    schema_version · grant_id · promotion_decision_digest
    qualification_digest · runtime_manifest_digest
    artifact_root_digest · backend_id · backend_binary_digest
    policy_epoch · revocation_epoch · issued_at · expires_at
    nonce · audience_runtime_identity · issuer_key_id · signature

Verification rules (all fail closed):

  * strict schema + Ed25519 signature by a currently authorized
    `admission` role identity;
  * `issued_at`/`expires_at` window encloses the verification time —
    a grant is deliberately short-lived;
  * `nonce` is a fresh 128-bit value (the supervisor journals
    consumed grant ids/nonces so a grant cannot be replayed for a
    second activation);
  * `audience_runtime_identity` must name the supervisor actually
    asking — a grant issued for another runtime is refused;
  * the digests a caller intends to serve (manifest, artifact root,
    backend) must equal the grant's bound digests — a grant authorizes
    exactly one measured artifact set on exactly one backend;
  * `revocation_epoch` must be at least the operative revocation
    epoch — a grant issued against stale revocation evidence is
    refused.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import Ed25519Signer, SignedEnvelope
from minagi.v161 import strict_schema
from minagi.v161.authority import as_utc

ADMISSION_GRANT_SCHEMA = "mini-agi-v16.4.2-admission-grant-v1"

#: Grants are deliberately short-lived: authorization to activate is
#: not a standing credential.
DEFAULT_GRANT_TTL_SECONDS = 300


class GrantRefused(PermissionError):
    """An admission grant is absent, invalid, expired, replayed, or
    bound to different artifacts than the ones being activated."""


@dataclass(frozen=True)
class AdmissionGrantV1:
    """The signed authorization to activate exactly one measured
    artifact set on exactly one backend for exactly one runtime."""
    grant_id: str
    promotion_decision_digest: str
    qualification_digest: str
    runtime_manifest_digest: str
    artifact_root_digest: str
    backend_id: str
    issued_at: int                   # unix seconds
    expires_at: int                  # unix seconds
    nonce: str                     # 128-bit hex
    audience_runtime_identity: str
    revocation_epoch: int = 0
    policy_epoch: int = 0
    backend_binary_digest: str = ""
    schema: str = ADMISSION_GRANT_SCHEMA

    def __post_init__(self):
        if not self.grant_id:
            raise ValueError("grant_id required")
        for name in ("promotion_decision_digest", "qualification_digest",
                     "runtime_manifest_digest", "artifact_root_digest"):
            validate_digest(getattr(self, name))
        if self.backend_binary_digest:
            validate_digest(self.backend_binary_digest)
        if not self.backend_id:
            raise ValueError("backend_id required")
        for name in ("issued_at", "expires_at", "policy_epoch",
                     "revocation_epoch"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{name} must be an integer")
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be after issued_at")
        if len(self.nonce) != 32 or any(
                c not in "0123456789abcdef" for c in self.nonce):
            raise ValueError("nonce must be 128-bit hex")
        if not self.audience_runtime_identity:
            raise ValueError("audience_runtime_identity required")

    @property
    def digest(self) -> str:
        return digest(self)

    def to_body(self) -> dict:
        return {"schema": self.schema, "grant_id": self.grant_id,
                "promotion_decision_digest": self.promotion_decision_digest,
                "qualification_digest": self.qualification_digest,
                "runtime_manifest_digest": self.runtime_manifest_digest,
                "artifact_root_digest": self.artifact_root_digest,
                "backend_id": self.backend_id,
                "backend_binary_digest": self.backend_binary_digest,
                "policy_epoch": self.policy_epoch,
                "revocation_epoch": self.revocation_epoch,
                "issued_at": self.issued_at,
                "expires_at": self.expires_at,
                "nonce": self.nonce,
                "audience_runtime_identity": self.audience_runtime_identity}

    def to_doc(self, *, signer: Ed25519Signer) -> dict:
        body = self.to_body()
        env = signer.sign(body)
        return {"value": body, "digest": digest(body),
                "signer_key_id": env.key_id,
                "signature_b64": env.signature_b64}

    @classmethod
    def from_value(cls, value: dict) -> "AdmissionGrantV1":
        return cls(
            grant_id=str(value["grant_id"]),
            promotion_decision_digest=str(
                value["promotion_decision_digest"]),
            qualification_digest=str(value["qualification_digest"]),
            runtime_manifest_digest=str(value["runtime_manifest_digest"]),
            artifact_root_digest=str(value["artifact_root_digest"]),
            backend_id=str(value["backend_id"]),
            backend_binary_digest=str(
                value.get("backend_binary_digest") or ""),
            policy_epoch=int(value.get("policy_epoch", 0)),
            revocation_epoch=int(value.get("revocation_epoch", 0)),
            issued_at=int(value["issued_at"]),
            expires_at=int(value["expires_at"]),
            nonce=str(value["nonce"]),
            audience_runtime_identity=str(
                value["audience_runtime_identity"]),
            schema=str(value["schema"]))


def issue_grant(signer: Ed25519Signer, *, decision_digest: str,
                qualification_digest: str, runtime_manifest_digest: str,
                artifact_root_digest: str, backend_id: str,
                audience_runtime_identity: str,
                now: datetime | None = None,
                ttl_seconds: int = DEFAULT_GRANT_TTL_SECONDS,
                revocation_epoch: int = 0, policy_epoch: int = 0,
                backend_binary_digest: str = "") -> dict:
    """Issue a signed, short-lived admission grant. The signer must be
    an `admission` authority — enforced on verification, and here by
    convention (callers hold only the keys they are entitled to)."""
    at = as_utc(now)
    grant = AdmissionGrantV1(
        grant_id=f"grant-{secrets.token_hex(8)}",
        promotion_decision_digest=decision_digest,
        qualification_digest=qualification_digest,
        runtime_manifest_digest=runtime_manifest_digest,
        artifact_root_digest=artifact_root_digest,
        backend_id=backend_id,
        backend_binary_digest=backend_binary_digest,
        policy_epoch=int(policy_epoch),
        revocation_epoch=int(revocation_epoch),
        issued_at=int(at.timestamp()),
        expires_at=int(at.timestamp()) + int(ttl_seconds),
        nonce=secrets.token_hex(16),
        audience_runtime_identity=audience_runtime_identity)
    return grant.to_doc(signer=signer)


def verify_grant(doc, registry, *, now: datetime | None = None,
                 audience_runtime_identity: str | None = None,
                 require_backend_id: str | None = None,
                 require_manifest_digest: str | None = None,
                 require_artifact_root_digest: str | None = None,
                 min_revocation_epoch: int = 0,
                 max_clock_skew_seconds: int = 300
                 ) -> AdmissionGrantV1:
    """Verify a signed admission grant end to end.

    Authenticity (signature + role), liveness (validity window +
    skew bound), scope (audience, backend, manifest, artifact-root
    digests), and freshness of the revocation evidence it was issued
    against — each failure is a refusal, never a warning."""
    if not isinstance(doc, dict) or "value" not in doc:
        raise GrantRefused(
            "admission grant: signed envelope required "
            "{value, digest, signer_key_id, signature_b64}")
    value = doc["value"]
    if not isinstance(value, dict):
        raise GrantRefused("admission grant: envelope value must be an "
                           "object")
    if doc.get("digest") != digest(value):
        raise GrantRefused("admission grant: envelope digest mismatch")
    kid = str(doc.get("signer_key_id", ""))
    if not kid:
        raise GrantRefused(
            "admission grant is unsigned — an unsigned grant is not "
            "authorization")
    at = as_utc(now)
    if not registry.is_authorized("admission", kid, now=at):
        raise GrantRefused(
            f"grant signer {kid!r} is not an authorized, currently "
            "valid admission authority")
    if not registry.verifier(now=at).verify(
            value, SignedEnvelope(kid, str(doc.get("signature_b64", "")))):
        raise GrantRefused("admission grant signature invalid")
    try:
        strict_schema.validate("admission_grant", value)
    except strict_schema.SchemaRefused as exc:
        raise GrantRefused(str(exc)) from exc
    grant = AdmissionGrantV1.from_value(value)

    ts = int(at.timestamp())
    if grant.issued_at > ts + int(max_clock_skew_seconds):
        raise GrantRefused("admission grant is future-dated")
    if ts > grant.expires_at:
        raise GrantRefused("admission grant expired — grants are "
                           "short-lived; request a fresh authorization")
    if audience_runtime_identity is not None and \
            grant.audience_runtime_identity != audience_runtime_identity:
        raise GrantRefused(
            f"grant is issued for runtime "
            f"{grant.audience_runtime_identity!r}, not this runtime "
            f"{audience_runtime_identity!r}")
    if require_backend_id is not None and \
            grant.backend_id != require_backend_id:
        raise GrantRefused(
            f"grant authorizes backend {grant.backend_id!r}, not "
            f"{require_backend_id!r}")
    if require_manifest_digest is not None and \
            grant.runtime_manifest_digest != require_manifest_digest:
        raise GrantRefused(
            "grant does not bind the runtime manifest being activated")
    if require_artifact_root_digest is not None and \
            grant.artifact_root_digest != require_artifact_root_digest:
        raise GrantRefused(
            "grant does not bind the measured artifact root being "
            "activated")
    if grant.revocation_epoch < int(min_revocation_epoch):
        raise GrantRefused(
            f"grant was issued against revocation epoch "
            f"{grant.revocation_epoch}; the operative epoch is "
            f"{min_revocation_epoch} — stale authorization refused")
    return grant
