"""v16.4.0 runtime admission control (Phase 3 / REPAIR-047..051).

Promotion is not complete until the runtime independently admits the
exact artifact and emits an activation receipt. The controller verifies
the whole authorization chain at load time (and again whenever active
artifacts change):

    campaign plan            (role: plan)
      -> qualification record (role: qualification; decision QUALIFIED;
                               binds campaign plan + evaluation bundle)
      -> promotion decision   (role: promotion; binds campaign, bundle,
                               qualification digest, exact per-seed
                               adapter digests and runtime-manifest
                               digests; unexpired; unrevoked)
      -> runtime manifest     (binds model / tokenizer / adapter bytes /
                               protocol; digest self-consistent)
      -> exact adapter bytes  measured on disk

Admission REFUSES (fail closed, never partially admits):

  * an unsigned, forged, expired, or revoked promotion decision
  * a decision signed by any role other than `promotion`
  * an adapter whose bytes differ from the signed manifest or from the
    decision-authorized digest (e.g. one altered byte)
  * a different base model or tokenizer than the manifest binds
  * a runtime backend not covered by the qualification
  * an incomplete chain (missing or digest-mismatched plan, bundle, or
    qualification record)
  * a replayed decision on the revocation list

A successful admission emits an ActivationReceipt — signed by the
`runtime` role when a runtime signer is supplied — binding the
decision, qualification, manifest, adapter digest, backend, and time.
Rollback re-admits a previously receipted release from its signed
artifacts and records the lineage; the research plane is not involved.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import Ed25519Signer, SignedEnvelope

from .authority import as_utc
from .peft_serving import AdapterClosureError, adapter_closure

ACTIVATION_RECEIPT_SCHEMA = "mini-agi-v16.4-activation-receipt-v1"
PROMOTION_DECISION_SCHEMA = "mini-agi-v16.5-promotion-decision-v1"

# Serving backends the qualification plane has actually qualified. A
# qualification record may narrow/extend this via `runtime_backends`;
# anything not covered is refused rather than assumed.
QUALIFIED_BACKENDS = ("hf-peft",)


class AdmissionRefused(PermissionError):
    """Runtime admission refused — the artifact must not be served."""


@dataclass(frozen=True)
class ActivationReceipt:
    """Evidence that the runtime independently admitted the exact
    artifact. Digest-bound; signed by the `runtime` role when the
    deployment holds a runtime signing key."""
    decision_digest: str
    qualification_record_digest: str
    runtime_manifest_digest: str
    adapter_digest: str
    backend: str
    admitted_at: str                 # ISO-8601 UTC
    rollback_of: str = ""            # decision digest rolled back from
    schema: str = ACTIVATION_RECEIPT_SCHEMA

    def __post_init__(self):
        for d in (self.decision_digest, self.qualification_record_digest,
                  self.runtime_manifest_digest, self.adapter_digest):
            validate_digest(d)
        if self.rollback_of:
            validate_digest(self.rollback_of)
        if not self.backend:
            raise ValueError("activation receipt requires a backend")
        if not self.admitted_at:
            raise ValueError("activation receipt requires admitted_at")

    @property
    def digest(self) -> str:
        return digest(self)

    def to_doc(self, *, signer: Ed25519Signer | None = None) -> dict:
        body = asdict(self)
        doc = {"value": body, "digest": digest(body)}
        if signer is not None:
            env = signer.sign(body)
            doc.update({"signer_key_id": env.key_id,
                        "signature_b64": env.signature_b64})
        return doc


def write_activation_receipt(receipt: ActivationReceipt, path, *,
                             signer: Ed25519Signer | None = None) -> dict:
    doc = receipt.to_doc(signer=signer)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return doc


def check_activation_receipt(doc, registry, *, now=None) -> list[str]:
    """Re-verify a persisted activation receipt (digest, fields, and —
    when signed — the runtime-role signature). Returns problems ([] =
    intact)."""
    problems: list[str] = []
    if not isinstance(doc, dict) or "value" not in doc:
        return ["activation receipt envelope missing"]
    value = doc["value"]
    if not isinstance(value, dict):
        return ["activation receipt value must be an object"]
    if doc.get("digest") != digest(value):
        problems.append("activation receipt digest mismatch")
    try:
        receipt = ActivationReceipt(**value)
    except (TypeError, ValueError) as exc:
        problems.append(f"activation receipt malformed: {exc}")
        return problems
    if receipt.digest != doc.get("digest"):
        problems.append("activation receipt digest is not self-consistent")
    if "signer_key_id" in doc:
        kid = str(doc.get("signer_key_id", ""))
        if not registry.is_authorized("runtime", kid, now=now):
            problems.append(
                "activation receipt signer is not an authorized, "
                "currently valid runtime authority")
        elif not registry.verifier(now=now).verify(
                value, SignedEnvelope(kid, str(doc.get("signature_b64", "")))):
            problems.append("activation receipt signature invalid")
    return problems


class RuntimeAdmissionController:
    """Admission control for the serving runtime. Holds the trust-root
    registry, an optional revocation list of promotion-decision
    digests, and an optional fixed verification time (tests)."""

    def __init__(self, registry, *, revoked_decision_digests=(),
                 now: datetime | None = None):
        self.registry = registry
        self.revoked = {str(d) for d in revoked_decision_digests}
        self._now = now

    def _at(self) -> datetime:
        return as_utc(self._now)

    def _verify_envelope(self, doc, role: str, what: str) -> dict:
        if not isinstance(doc, dict) or "value" not in doc:
            raise AdmissionRefused(
                f"{what}: signed envelope required "
                "{{value, digest, signer_key_id, signature_b64}} — a bare "
                "digest is not authorization")
        kid = str(doc.get("signer_key_id", ""))
        at = self._at()
        if not self.registry.is_authorized(role, kid, now=at):
            raise AdmissionRefused(
                f"{what}: signer {kid!r} is not an authorized, currently "
                f"valid {role} authority")
        value = doc["value"]
        if not isinstance(value, dict):
            raise AdmissionRefused(f"{what}: envelope value must be an object")
        if doc.get("digest") != digest(value):
            raise AdmissionRefused(f"{what}: envelope digest mismatch")
        if not self.registry.verifier(now=at).verify(
                value, SignedEnvelope(kid,
                                      str(doc.get("signature_b64", "")))):
            raise AdmissionRefused(f"{what}: signature invalid")
        return value

    def admit(self, *, decision_doc, qualification_doc, plan_doc,
              runtime_manifest, adapter_dir, seed: str,
              runtime_model_digest: str, runtime_tokenizer_digest: str,
              expected_backend: str = "hf-peft", rollback_of: str = "",
              ) -> ActivationReceipt:
        at = self._at()

        # --- promotion decision -------------------------------------
        decision = self._verify_envelope(decision_doc, "promotion",
                                         "promotion decision")
        decision_digest = str(decision_doc["digest"])
        if decision_digest in self.revoked:
            raise AdmissionRefused(
                "promotion decision revoked — replay refused")
        if decision.get("schema") != PROMOTION_DECISION_SCHEMA:
            raise AdmissionRefused(
                f"unrecognized promotion decision schema "
                f"{decision.get('schema')!r}")
        authorized_at = decision.get("authorized_at")
        expires_at = decision.get("expires_at")
        if not isinstance(authorized_at, int) \
                or not isinstance(expires_at, int):
            raise AdmissionRefused(
                "promotion decision requires integer "
                "authorized_at/expires_at — an undated decision cannot "
                "be admitted")
        ts = int(at.timestamp())
        if ts < authorized_at:
            raise AdmissionRefused("promotion decision not yet valid")
        if ts > expires_at:
            raise AdmissionRefused("promotion decision expired")

        # --- qualification record -----------------------------------
        qual = self._verify_envelope(qualification_doc, "qualification",
                                     "qualification record")
        if qualification_doc.get("digest") != \
                decision.get("qualification_record_digest"):
            raise AdmissionRefused(
                "qualification record digest does not match the decision "
                "binding (substituted or stale qualification)")
        if qual.get("decision") != "QUALIFIED":
            raise AdmissionRefused(
                f"qualification decision {qual.get('decision')!r} — only "
                "QUALIFIED evidence may be admitted")
        if qual.get("campaign_plan_digest") != \
                decision.get("campaign_plan_digest"):
            raise AdmissionRefused(
                "qualification record binds a different campaign plan")
        if decision.get("evaluation_bundle_digest") \
                and qual.get("evaluation_bundle_digest") \
                and qual["evaluation_bundle_digest"] != \
                decision["evaluation_bundle_digest"]:
            raise AdmissionRefused(
                "evaluation bundle digest mismatch between decision and "
                "qualification")

        # --- campaign plan ------------------------------------------
        plan = self._verify_envelope(plan_doc, "plan", "campaign plan")
        if plan_doc.get("digest") != decision.get("campaign_plan_digest"):
            raise AdmissionRefused(
                "campaign plan digest does not match the decision binding")

        # --- runtime manifest ---------------------------------------
        if not isinstance(runtime_manifest, dict) \
                or "digest" not in runtime_manifest:
            raise AdmissionRefused("runtime manifest required")
        body = {k: v for k, v in runtime_manifest.items() if k != "digest"}
        if digest(body) != runtime_manifest["digest"]:
            raise AdmissionRefused("runtime manifest digest mismatch")
        if runtime_manifest.get("campaign_digest") != \
                decision.get("campaign_plan_digest"):
            raise AdmissionRefused(
                "runtime manifest binds a different campaign")
        if runtime_manifest.get("qualification_record_digest") != \
                qualification_doc.get("digest"):
            raise AdmissionRefused(
                "runtime manifest binds a different qualification record")
        authorized_manifest = (decision.get("runtime_manifest_digests")
                               or {}).get(seed)
        if authorized_manifest is None:
            raise AdmissionRefused(
                f"promotion decision does not authorize seed {seed!r}")
        if runtime_manifest["digest"] != authorized_manifest:
            raise AdmissionRefused(
                "runtime manifest digest is not covered by the decision")

        # --- backend coverage ---------------------------------------
        manifest_backend = str(runtime_manifest.get("serving_stack", ""))
        if manifest_backend != expected_backend:
            raise AdmissionRefused(
                f"requested backend {expected_backend!r} != manifest "
                f"backend {manifest_backend!r}")
        covered = qual.get("runtime_backends") or list(QUALIFIED_BACKENDS)
        if expected_backend not in covered:
            raise AdmissionRefused(
                f"runtime backend {expected_backend!r} is not covered by "
                f"the qualification ({sorted(covered)})")

        # --- exact adapter bytes ------------------------------------
        try:
            closure = adapter_closure(adapter_dir)
        except AdapterClosureError as exc:
            raise AdmissionRefused(
                f"adapter closure failed: {exc}") from exc
        if closure.adapter_digest != runtime_manifest.get("adapter_digest"):
            raise AdmissionRefused(
                "adapter bytes differ from the signed manifest")
        authorized_adapter = (decision.get("adapter_artifact_digests")
                              or {}).get(seed)
        if closure.adapter_digest != authorized_adapter:
            raise AdmissionRefused(
                "adapter bytes differ from the decision-authorized digest")

        # --- base model / tokenizer identity ------------------------
        if str(runtime_model_digest) != str(
                runtime_manifest.get("model_digest")):
            raise AdmissionRefused(
                "base model digest differs from the manifest")
        if str(runtime_tokenizer_digest) != str(
                runtime_manifest.get("tokenizer_digest")):
            raise AdmissionRefused(
                "tokenizer digest differs from the manifest")

        # --- protocol link (when the signed plan declares one) ------
        plan_protocol = plan.get("experiment_protocol_digest")
        manifest_protocol = runtime_manifest.get("protocol_digest")
        if plan_protocol and manifest_protocol \
                and plan_protocol != manifest_protocol:
            raise AdmissionRefused(
                "runtime manifest protocol differs from the signed plan")

        return ActivationReceipt(
            decision_digest=decision_digest,
            qualification_record_digest=str(qualification_doc["digest"]),
            runtime_manifest_digest=str(runtime_manifest["digest"]),
            adapter_digest=closure.adapter_digest,
            backend=expected_backend,
            admitted_at=at.isoformat(),
            rollback_of=rollback_of)

    def rollback(self, *, current_decision_digest: str,
                 **previous_release) -> ActivationReceipt:
        """Re-admit a previously receipted release from its signed
        artifacts, recording the lineage. The research plane is not
        consulted; the previous release must still pass full admission
        (tampered rollback targets are refused)."""
        validate_digest(str(current_decision_digest))
        return self.admit(rollback_of=str(current_decision_digest),
                          **previous_release)
