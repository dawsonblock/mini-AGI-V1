"""v16.4.1 runtime admission control (Phase 3 / REPAIR-047..051 + the
v16.4.1 security closure).

Promotion is not complete until the runtime independently admits the
exact artifact and emits an activation receipt. The controller verifies
the whole authorization chain at load time (and again whenever active
artifacts change):

    campaign plan            (role: plan; strict schema; protocol bound)
      -> qualification record (role: qualification; decision QUALIFIED;
                               strict schema; binds campaign plan +
                               evaluation bundle)
      -> promotion decision   (role: promotion; binds campaign, bundle,
                               qualification digest, exact per-seed
                               adapter digests and runtime-manifest
                               digests; unexpired; unrevoked)
      -> runtime manifest     (binds model / tokenizer / adapter bytes /
                               protocol / qualification; digest
                               self-consistent; strict schema)
      -> exact adapter bytes  measured on disk (symlinks and special
                               files refused; authorized file listing
                               enforced when the manifest carries one)

The tokenizer identity uses the plan's convention: the tokenizer-named
files inside the model snapshot root, re-measured from whichever
physical root is supplied (`artifact_closure.tokenizer_artifact_digest`).
An explicit tokenizer path is an optional cross-check, and the serving
backend loads the tokenizer from the verified model artifact itself —
there is no separately substitutable tokenizer path in the served set.

Admission REFUSES (fail closed, never partially admits):

  * an unsigned, forged, expired, or revoked promotion decision
  * a decision signed by any role other than `promotion`
  * an unknown schema version, an absent required digest, an
    unsupported backend identifier, or a stale revocation list
  * an adapter whose bytes differ from the signed manifest or from the
    decision-authorized digest (e.g. one altered byte, or a symlink
    added to the approved directory)
  * a different base model or tokenizer than the manifest binds
  * a runtime backend not covered by the qualification
  * an incomplete chain (missing or digest-mismatched plan, bundle, or
    qualification record)
  * a replayed decision on the revocation list

A successful admission emits an ActivationReceipt. The admission-only
receipt (this module) is digest-bound and signed by the `runtime` role;
only the TrustedRuntimeLauncher (`trusted_launcher.py`) produces the
PRODUCTION receipt, which additionally binds the artifact digests
actually measured at load, the executing backend, and a replay
nonce — and is emitted only after the backend loaded successfully.
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

from . import strict_schema
from .artifact_closure import (ArtifactClosureError, close_tree,
                               tokenizer_artifact_digest)
from .authority import as_utc
from .peft_serving import AdapterClosureError, adapter_closure

ACTIVATION_RECEIPT_SCHEMA = "mini-agi-v16.4.1-activation-receipt-v2"
PROMOTION_DECISION_SCHEMA = "mini-agi-v16.5-promotion-decision-v1"
REVOCATION_LIST_SCHEMA = "mini-agi-v16.4.1-revocation-list-v1"

# Serving backends the qualification plane has actually qualified
# (strict_schema.SUPPORTED_BACKENDS). v16.4.2: a qualification record
# must declare backend coverage explicitly — there is no default.


class AdmissionRefused(PermissionError):
    """Runtime admission refused — the artifact must not be served."""


@dataclass(frozen=True)
class RevocationList:
    """Signed-or-frozen list of revoked promotion-decision digests with
    a generation time. Admission may enforce a maximum age: a stale
    list is refused (fail closed) instead of being treated as empty."""
    digests: tuple[str, ...]
    generated_at: int
    schema: str = REVOCATION_LIST_SCHEMA

    def __post_init__(self):
        for d in self.digests:
            validate_digest(str(d))
        if isinstance(self.generated_at, bool) \
                or not isinstance(self.generated_at, int):
            raise ValueError("revocation list requires integer generated_at")

    @property
    def digest(self) -> str:
        return digest(self)

    def contains(self, decision_digest: str) -> bool:
        return str(decision_digest) in {str(d) for d in self.digests}

    @classmethod
    def load(cls, path) -> "RevocationList":
        doc = json.loads(Path(path).read_text())
        if isinstance(doc, dict):
            if doc.get("schema") != REVOCATION_LIST_SCHEMA:
                raise ValueError(
                    f"unrecognized revocation list schema "
                    f"{doc.get('schema')!r}")
            return cls(digests=tuple(str(d) for d in doc.get("digests", ())),
                       generated_at=doc.get("generated_at"))
        if isinstance(doc, list):  # legacy bare list: no freshness data
            return cls(digests=tuple(str(d) for d in doc), generated_at=0)
        raise ValueError("revocation list must be an object or a list")


@dataclass(frozen=True)
class ActivationReceipt:
    """Evidence that the runtime independently admitted the exact
    artifact.

    Admission-only receipts (RuntimeAdmissionController) bind the
    decision/qualification/manifest/adapter digests. Production
    receipts (TrustedRuntimeLauncher) additionally bind
    `candidate_digest`, the digests actually measured at load
    (`loaded_artifact_digests`), and a replay `activation_nonce`; both
    carry the runtime-role signature, and the verification helper
    refuses unsigned receipts."""
    decision_digest: str
    qualification_record_digest: str
    runtime_manifest_digest: str
    adapter_digest: str
    backend: str
    admitted_at: str                 # ISO-8601 UTC
    candidate_digest: str = ""       # exact candidate identity
    loaded_artifact_digests: tuple[tuple[str, str], ...] = ()
    activation_nonce: str = ""       # replay protection (128-bit hex)
    rollback_of: str = ""            # decision digest rolled back from
    schema: str = ACTIVATION_RECEIPT_SCHEMA

    def __post_init__(self):
        for d in (self.decision_digest, self.qualification_record_digest,
                  self.runtime_manifest_digest, self.adapter_digest):
            validate_digest(d)
        if self.candidate_digest:
            validate_digest(self.candidate_digest)
        if self.rollback_of:
            validate_digest(self.rollback_of)
        if not self.backend:
            raise ValueError("activation receipt requires a backend")
        if not self.admitted_at:
            raise ValueError("activation receipt requires admitted_at")
        for name, d in self.loaded_artifact_digests:
            if not name:
                raise ValueError("loaded artifact name required")
            validate_digest(d)
        if self.activation_nonce and (
                len(self.activation_nonce) != 32
                or any(c not in "0123456789abcdef"
                       for c in self.activation_nonce)):
            raise ValueError("activation_nonce must be 128-bit hex")

    # --- plan vocabulary aliases -------------------------------------
    @property
    def promotion_digest(self) -> str:
        return self.decision_digest

    @property
    def qualification_digest(self) -> str:
        return self.qualification_record_digest

    @property
    def backend_id(self) -> str:
        return self.backend

    @property
    def activated_at(self) -> str:
        return self.admitted_at

    @property
    def digest(self) -> str:
        return digest(self)

    @property
    def production(self) -> bool:
        """True only for receipts that bind the bytes measured at load."""
        return bool(self.loaded_artifact_digests and self.activation_nonce
                    and self.candidate_digest)

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
    if signer is None:
        raise ValueError(
            "an unsigned activation receipt is not authorization evidence; "
            "a protected runtime identity must sign it")
    doc = receipt.to_doc(signer=signer)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return doc


def check_activation_receipt(doc, registry, *, now=None,
                             require_signature: bool = True,
                             require_production: bool = False) -> list[str]:
    """Re-verify a persisted activation receipt: digest self-consistency,
    strict schema, and the runtime-role signature.

    v16.4.1: an unsigned receipt is a problem, not a pass — signature
    validity is required by default, and `require_production` demands
    the load-measured artifact digests, the replay nonce, and a
    production schema version. Signature validity alone is still not
    sufficient for production: the signer must have measured and loaded
    the artifacts, which only a production receipt records."""
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
    try:
        strict_schema.validate("activation_receipt", value,
                               require_production=require_production)
    except strict_schema.SchemaRefused as exc:
        problems.append(str(exc))
    if "signer_key_id" not in doc:
        if require_signature:
            problems.append(
                "activation receipt is unsigned — an unsigned receipt is "
                "not authorization evidence")
    else:
        kid = str(doc.get("signer_key_id", ""))
        if not registry.is_authorized("runtime", kid, now=now):
            problems.append(
                "activation receipt signer is not an authorized, "
                "currently valid runtime authority")
        elif not registry.verifier(now=now).verify(
                value, SignedEnvelope(kid, str(doc.get("signature_b64", "")))):
            problems.append("activation receipt signature invalid")
    if require_production and not receipt.production:
        problems.append(
            "activation receipt does not bind the artifacts measured at "
            "load (admission-only receipt cannot authorize serving)")
    return problems


class RuntimeAdmissionController:
    """Admission control for the serving runtime. Holds the trust-root
    registry, revocation evidence, and an optional fixed verification
    time (tests).

    v16.4.2: the production path authenticates revocation evidence — a
    `RevocationSnapshotV2` signed by the dedicated `revocation` role —
    instead of a bare digest list. Legacy unsigned `RevocationList`
    objects remain accepted for research-side admission checks but are
    refused by the supervised production launch path."""

    def __init__(self, registry, *, revoked_decision_digests=(),
                 revocation_list: RevocationList | None = None,
                 revocation_snapshot=None,
                 max_revocation_age_seconds: int | None = None,
                 now: datetime | None = None):
        self.registry = registry
        self.revoked = {str(d) for d in revoked_decision_digests}
        self.revocation_list = revocation_list
        self.revocation_snapshot = revocation_snapshot
        self.max_revocation_age_seconds = max_revocation_age_seconds
        self._now = now

    def _at(self) -> datetime:
        return as_utc(self._now)

    def _check_revocation_freshness(self) -> None:
        if self.max_revocation_age_seconds is None:
            return
        if self.revocation_snapshot is not None:
            return  # V2 snapshots carry their own issued_at/valid_until
        if self.revocation_list is None:
            raise AdmissionRefused(
                "revocation freshness required but no revocation evidence "
                "was supplied — refusing rather than assuming nothing is "
                "revoked")
        age = int(self._at().timestamp()) - int(
            self.revocation_list.generated_at)
        if age > int(self.max_revocation_age_seconds):
            raise AdmissionRefused(
                f"revocation list is stale ({age}s > "
                f"{self.max_revocation_age_seconds}s) — refresh it before "
                "admitting")

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

    def _schema(self, kind: str, value: dict, what: str) -> None:
        try:
            strict_schema.validate(kind, value)
        except strict_schema.SchemaRefused as exc:
            raise AdmissionRefused(str(exc)) from exc

    def _measure(self, what: str, path, resolve_symlinks: bool) -> str:
        try:
            return close_tree(path, resolve_symlinks=resolve_symlinks).digest
        except ArtifactClosureError as exc:
            raise AdmissionRefused(f"{what} closure failed: {exc}") from exc

    def admit(self, *, decision_doc, qualification_doc, plan_doc,
              runtime_manifest, adapter_dir, seed: str,
              runtime_model_digest: str | None = None,
              runtime_tokenizer_digest: str | None = None,
              runtime_model_path=None, runtime_tokenizer_path=None,
              expected_backend: str = "hf-peft", rollback_of: str = "",
              ) -> ActivationReceipt:
        at = self._at()
        self._check_revocation_freshness()

        # --- promotion decision -------------------------------------
        decision = self._verify_envelope(decision_doc, "promotion",
                                         "promotion decision")
        decision_digest = str(decision_doc["digest"])
        if decision_digest in self.revoked or (
                self.revocation_list is not None
                and self.revocation_list.contains(decision_digest)) or (
                self.revocation_snapshot is not None
                and self.revocation_snapshot.contains(decision_digest)):
            raise AdmissionRefused(
                "promotion decision revoked — replay refused")
        if self.revocation_snapshot is not None and \
                self.revocation_snapshot.revokes_key(
                    str(decision_doc.get("signer_key_id", ""))):
            raise AdmissionRefused(
                "the key that signed this promotion decision has been "
                "revoked — evidence signed by it no longer authorizes")
        self._schema("promotion_decision", decision, "promotion decision")
        if decision.get("schema") != PROMOTION_DECISION_SCHEMA:
            raise AdmissionRefused(
                f"unrecognized promotion decision schema "
                f"{decision.get('schema')!r}")
        authorized_at = decision.get("authorized_at")
        expires_at = decision.get("expires_at")
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
        self._schema("qualification_record", qual, "qualification record")
        if qual.get("campaign_plan_digest") != \
                decision.get("campaign_plan_digest"):
            raise AdmissionRefused(
                "qualification record binds a different campaign plan")
        qual_bundle = qual.get("evaluation_bundle_digest")
        if not qual_bundle:
            raise AdmissionRefused(
                "qualification record does not bind an evaluation bundle "
                "digest — admission cannot rely on an unbound evaluation")
        if qual_bundle != decision.get("evaluation_bundle_digest"):
            raise AdmissionRefused(
                "evaluation bundle digest mismatch between decision and "
                "qualification")

        # --- campaign plan ------------------------------------------
        plan = self._verify_envelope(plan_doc, "plan", "campaign plan")
        if plan_doc.get("digest") != decision.get("campaign_plan_digest"):
            raise AdmissionRefused(
                "campaign plan digest does not match the decision binding")
        self._schema("campaign_plan", plan, "campaign plan")

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
        # v16.4.2: coverage must be an explicit, signed declaration.
        # A qualification that omits runtime_backends covers NOTHING —
        # there is no default (qualifying hf-peft never transitively
        # qualifies qwen-native-cuda or qwen-native-metal).
        manifest_backend = str(runtime_manifest.get("serving_stack", ""))
        if manifest_backend != expected_backend:
            raise AdmissionRefused(
                f"requested backend {expected_backend!r} != manifest "
                f"backend {manifest_backend!r}")
        covered = qual.get("runtime_backends")
        if not covered or not isinstance(covered, (list, tuple)):
            raise AdmissionRefused(
                "qualification record omits runtime backend coverage — "
                "admission requires a nonempty, explicitly signed "
                "runtime_backends declaration (UPGRADE_PLAN §3.3)")
        if expected_backend not in covered:
            raise AdmissionRefused(
                f"runtime backend {expected_backend!r} is not covered by "
                f"the qualification ({sorted(covered)})")
        self._schema("runtime_manifest", runtime_manifest,
                     "runtime manifest")

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
        listed = runtime_manifest.get("adapter_files")
        if listed is not None:
            from .artifact_closure import expected_from_manifest, \
                verify_entries
            try:
                verify_entries(closure.entries,
                               expected_from_manifest(listed),
                               what="adapter")
            except ArtifactClosureError as exc:
                raise AdmissionRefused(
                    f"adapter closure failed: {exc}") from exc

        # --- base model / tokenizer identity ------------------------
        model_digest = self._resolve_identity(
            "base model", runtime_model_digest, runtime_model_path)
        if model_digest != str(runtime_manifest.get("model_digest")):
            raise AdmissionRefused(
                "base model digest differs from the manifest")
        # The signed plan binds the tokenizer as the tokenizer-named
        # files inside the model snapshot root
        # (`artifact_closure.tokenizer_artifact_digest` — the same
        # convention `physical_identity_digests` records in the plan).
        # Measure that convention from whichever physical root is
        # supplied: the model snapshot and/or an explicit tokenizer
        # path (a cross-check). A claimed digest is never trusted when
        # a root is available.
        tokenizer_digest = self._resolve_tokenizer_identity(
            runtime_tokenizer_digest, runtime_tokenizer_path,
            runtime_model_path)
        if tokenizer_digest != str(runtime_manifest.get("tokenizer_digest")):
            raise AdmissionRefused(
                "tokenizer digest differs from the manifest")

        # --- protocol link (mandatory) ------------------------------
        plan_protocol = plan.get("experiment_protocol_digest")
        manifest_protocol = runtime_manifest.get("protocol_digest")
        if not plan_protocol:
            raise AdmissionRefused(
                "signed plan does not bind an experiment protocol digest")
        if not manifest_protocol:
            raise AdmissionRefused(
                "runtime manifest does not bind the plan's protocol digest")
        if plan_protocol != manifest_protocol:
            raise AdmissionRefused(
                "runtime manifest protocol differs from the signed plan")

        return ActivationReceipt(
            decision_digest=decision_digest,
            qualification_record_digest=str(qualification_doc["digest"]),
            runtime_manifest_digest=str(runtime_manifest["digest"]),
            adapter_digest=closure.adapter_digest,
            backend=expected_backend,
            admitted_at=at.isoformat(),
            candidate_digest=closure.adapter_digest,
            rollback_of=rollback_of)

    def _resolve_identity(self, what: str, claimed: str | None,
                          path) -> str:
        """Physically measure an artifact identity. When a path is
        supplied it wins — a caller-provided digest that disagrees with
        the measured bytes is refused rather than trusted."""
        if path is None:
            if not claimed:
                raise AdmissionRefused(
                    f"{what}: a physical path or a measured digest is "
                    "required (caller-provided digests alone are refused "
                    "by the trusted launcher)")
            try:
                validate_digest(str(claimed))
            except ValueError as exc:
                raise AdmissionRefused(
                    f"{what}: malformed digest {claimed!r}") from exc
            return str(claimed)
        measured = self._measure(what, path, resolve_symlinks=True)
        if claimed and str(claimed) != measured:
            raise AdmissionRefused(
                f"{what}: caller-provided digest disagrees with the "
                "physically measured bytes")
        return measured

    def _resolve_tokenizer_identity(self, claimed: str | None,
                                    tokenizer_path,
                                    model_path) -> str:
        """Measure the plan's tokenizer convention (the tokenizer-named
        files inside a snapshot root) from the supplied physical roots.
        The model snapshot and an explicit tokenizer path must agree
        when both are supplied; a claimed digest is only a cross-check
        and is refused when it disagrees with measured bytes."""
        measured: list[tuple[str, str]] = []
        for what, root in (("tokenizer path", tokenizer_path),
                           ("model snapshot", model_path)):
            if root is None:
                continue
            try:
                measured.append((what, tokenizer_artifact_digest(
                    root, resolve_symlinks=True)))
            except ArtifactClosureError as exc:
                raise AdmissionRefused(
                    f"{what} tokenizer closure failed: {exc}") from exc
        if not measured:
            return self._resolve_identity("tokenizer", claimed, None)
        first_what, first = measured[0]
        for what, other in measured[1:]:
            if other != first:
                raise AdmissionRefused(
                    f"tokenizer artifact differs between the {first_what} "
                    f"and the {what}")
        if claimed and str(claimed) != first:
            raise AdmissionRefused(
                "tokenizer: caller-provided digest disagrees with the "
                "physically measured bytes")
        return first

    def rollback(self, *, current_decision_digest: str,
                 **previous_release) -> ActivationReceipt:
        """Re-admit a previously receipted release from its signed
        artifacts, recording the lineage. The research plane is not
        consulted; the previous release must still pass full admission
        (tampered rollback targets are refused)."""
        validate_digest(str(current_decision_digest))
        return self.admit(rollback_of=str(current_decision_digest),
                          **previous_release)
