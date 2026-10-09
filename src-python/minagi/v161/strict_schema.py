"""v16.4.1 strict, versioned schemas for authority-bearing artifacts.

The v16.4.0 admission path validated only the promotion decision's
schema string; the campaign plan and the qualification record were
accepted whatever their schema said, several digest bindings were
compared "when present" on both sides (so omitting one side silently
disabled the check), and the protocol link was optional. A document
with an unknown version, a missing digest, or an unsupported backend
identifier could therefore pass as authority.

Every authority-bearing artifact now passes through exactly one
validator here, and unknown versions are refused rather than
tolerated:

    campaign_plan          the signed preregistration
    qualification_record   the qualification authority's gate decision
    promotion_decision     the promotion authority's authorization
    runtime_manifest       the exact artifact/config binding
    activation_receipt     the runtime's load-time evidence

Validators check structure, schema version, digest well-formedness,
timestamp typing, and backend identifiers. Signature verification and
authorization (roles, validity windows, revocation) remain the job of
`AuthorityRegistry` / the admission controller — schema validation
never substitutes for a signature check, and vice versa.
"""
from __future__ import annotations

from typing import Mapping

from egai.common.canonical import DIGEST_RE

# Backends the qualification plane has actually qualified. A runtime
# manifest declaring anything else is refused; a qualification record
# may narrow this set (never widen it beyond the supported set).
SUPPORTED_BACKENDS = ("hf-peft",)

CAMPAIGN_PLAN_SCHEMAS = frozenset({
    "mini-agi-v16.1-colab-campaign-plan-v1",
    "mini-agi-v16.2-colab-campaign-plan-v1",
    "mini-agi-v16.3-colab-campaign-plan-v1",
    "mini-agi-v16.4-colab-campaign-plan-v1",
    "mini-agi-v16.5-colab-campaign-plan-v1",
    "mini-agi-v16.6-colab-campaign-plan-v1",
})

QUALIFICATION_RECORD_SCHEMAS = frozenset({
    "mini-agi-v16.2-campaign1-qualification-v1",
    "mini-agi-v16.2-qualification-record-v1",
    "mini-agi-v16.5-qualification-record-v1",
})

QUALIFICATION_DECISIONS = frozenset({
    "QUALIFIED", "REFUSED", "REFUSE", "INVALID_EVIDENCE", "INCOMPLETE",
})

PROMOTION_DECISION_SCHEMAS = frozenset({
    "mini-agi-v16.5-promotion-decision-v1",
})

RUNTIME_MANIFEST_SCHEMAS = frozenset({
    "mini-agi-v16.5-peft-runtime-manifest-v1",
    "mini-agi-v16.5-peft-runtime-manifest-v2",
})

# Production manifests must carry the explicit authorized file listing.
RUNTIME_MANIFEST_PRODUCTION_SCHEMAS = frozenset({
    "mini-agi-v16.5-peft-runtime-manifest-v2",
})

ACTIVATION_RECEIPT_SCHEMAS = frozenset({
    "mini-agi-v16.4-activation-receipt-v1",
    "mini-agi-v16.4.1-activation-receipt-v2",
})

ACTIVATION_RECEIPT_PRODUCTION_SCHEMAS = frozenset({
    "mini-agi-v16.4.1-activation-receipt-v2",
})

# v16.4.2 — authority closure: authenticated revocation snapshots and
# admission grants.
REVOCATION_SNAPSHOT_SCHEMAS = frozenset({
    "mini-agi-v16.4.2-revocation-snapshot-v2",
})

ADMISSION_GRANT_SCHEMAS = frozenset({
    "mini-agi-v16.4.2-admission-grant-v1",
})

KNOWN_SCHEMAS: Mapping[str, frozenset[str]] = {
    "campaign_plan": CAMPAIGN_PLAN_SCHEMAS,
    "qualification_record": QUALIFICATION_RECORD_SCHEMAS,
    "promotion_decision": PROMOTION_DECISION_SCHEMAS,
    "runtime_manifest": RUNTIME_MANIFEST_SCHEMAS,
    "activation_receipt": ACTIVATION_RECEIPT_SCHEMAS,
    "revocation_snapshot": REVOCATION_SNAPSHOT_SCHEMAS,
    "admission_grant": ADMISSION_GRANT_SCHEMAS,
}


class SchemaRefused(PermissionError):
    """Artifact fails strict schema validation — refuse it."""


def _object(value, what: str) -> dict:
    if not isinstance(value, dict):
        raise SchemaRefused(f"{what} must be a JSON object")
    return value


def _known_schema(value: dict, kind: str, what: str) -> str:
    schema = value.get("schema")
    known = KNOWN_SCHEMAS[kind]
    if not isinstance(schema, str) or schema not in known:
        raise SchemaRefused(
            f"{what}: unrecognized schema {schema!r} (known: "
            f"{sorted(known)})")
    return schema


def _digest(value: dict, field: str, what: str) -> str:
    got = value.get(field)
    if not isinstance(got, str) or not DIGEST_RE.fullmatch(got):
        raise SchemaRefused(
            f"{what}: required digest field {field!r} is absent or "
            f"malformed ({got!r})")
    return got


def _digest_optional(value: dict, field: str, what: str) -> str | None:
    got = value.get(field)
    if got is None or got == "":
        return None
    if not isinstance(got, str) or not DIGEST_RE.fullmatch(got):
        raise SchemaRefused(
            f"{what}: field {field!r} is present but malformed ({got!r})")
    return got


def _text(value: dict, field: str, what: str) -> str:
    got = value.get(field)
    if not isinstance(got, str) or not got:
        raise SchemaRefused(f"{what}: required text field {field!r} missing")
    return got


def _int(value: dict, field: str, what: str) -> int:
    got = value.get(field)
    if isinstance(got, bool) or not isinstance(got, int):
        raise SchemaRefused(
            f"{what}: required integer field {field!r} missing or "
            f"malformed ({got!r})")
    return got


def validate_backend(backend: str, what: str) -> str:
    if backend not in SUPPORTED_BACKENDS:
        raise SchemaRefused(
            f"{what}: unsupported backend identifier {backend!r} "
            f"(supported: {list(SUPPORTED_BACKENDS)})")
    return backend


def validate_campaign_plan(value: dict) -> dict:
    what = "campaign plan"
    value = _object(value, what)
    _known_schema(value, "campaign_plan", what)
    _text(value, "campaign_id", what)
    _digest(value, "experiment_protocol_digest", what)
    return value


def validate_qualification_record(value: dict) -> dict:
    what = "qualification record"
    value = _object(value, what)
    _known_schema(value, "qualification_record", what)
    _text(value, "campaign_id", what)
    decision = _text(value, "decision", what)
    if decision not in QUALIFICATION_DECISIONS:
        raise SchemaRefused(
            f"{what}: unknown decision {decision!r}")
    _digest(value, "campaign_plan_digest", what)
    _digest_optional(value, "evaluation_bundle_digest", what)
    backends = value.get("runtime_backends")
    if backends is not None:
        if not isinstance(backends, (list, tuple)) or not backends:
            raise SchemaRefused(
                f"{what}: runtime_backends must be a non-empty list when "
                "present")
        for b in backends:
            if not isinstance(b, str) or b not in SUPPORTED_BACKENDS:
                raise SchemaRefused(
                    f"{what}: runtime_backends declares unsupported "
                    f"backend {b!r} (supported: {list(SUPPORTED_BACKENDS)})")
    return value


def validate_promotion_decision(value: dict) -> dict:
    what = "promotion decision"
    value = _object(value, what)
    _known_schema(value, "promotion_decision", what)
    _text(value, "campaign_id", what)
    _digest(value, "campaign_plan_digest", what)
    _digest(value, "qualification_record_digest", what)
    _digest(value, "evaluation_bundle_digest", what)
    _text(value, "adapter", what)
    seeds = value.get("adapter_artifact_digests")
    if not isinstance(seeds, dict) or not seeds:
        raise SchemaRefused(
            f"{what}: adapter_artifact_digests must bind at least one seed")
    for seed, d in seeds.items():
        if not isinstance(seed, str) or not seed:
            raise SchemaRefused(f"{what}: malformed seed key {seed!r}")
        if not isinstance(d, str) or not DIGEST_RE.fullmatch(d):
            raise SchemaRefused(
                f"{what}: seed {seed!r} carries a malformed adapter digest")
    manifests = value.get("runtime_manifest_digests")
    if not isinstance(manifests, dict) or set(manifests) != set(seeds):
        raise SchemaRefused(
            f"{what}: runtime_manifest_digests must cover exactly the "
            f"authorized seeds {sorted(seeds)}")
    for seed, d in manifests.items():
        if not isinstance(d, str) or not DIGEST_RE.fullmatch(d):
            raise SchemaRefused(
                f"{what}: seed {seed!r} carries a malformed manifest digest")
    authorized_at = _int(value, "authorized_at", what)
    expires_at = _int(value, "expires_at", what)
    if expires_at <= authorized_at:
        raise SchemaRefused(
            f"{what}: expires_at must be after authorized_at")
    return value


def validate_runtime_manifest(value: dict, *,
                              require_production: bool = False) -> dict:
    what = "runtime manifest"
    value = _object(value, what)
    schema = _known_schema(value, "runtime_manifest", what)
    if require_production and schema not in RUNTIME_MANIFEST_PRODUCTION_SCHEMAS:
        raise SchemaRefused(
            f"{what}: schema {schema!r} does not carry the explicit "
            "authorized artifact listing required for production "
            "admission")
    _text(value, "model_id", what)
    _text(value, "model_revision", what)
    _digest(value, "model_digest", what)
    _digest(value, "tokenizer_digest", what)
    _digest(value, "adapter_digest", what)
    _digest(value, "protocol_digest", what)
    _digest(value, "campaign_digest", what)
    _digest(value, "qualification_record_digest", what)
    validate_backend(str(value.get("serving_stack", "")), what)
    lora = value.get("lora")
    if not isinstance(lora, dict):
        raise SchemaRefused(f"{what}: lora footprint missing")
    if not isinstance(lora.get("rank"), int) or lora["rank"] <= 0:
        raise SchemaRefused(f"{what}: lora rank must be a positive integer")
    targets = lora.get("target_modules")
    if not isinstance(targets, (list, tuple)) or not targets:
        raise SchemaRefused(f"{what}: lora target_modules must be non-empty")
    files = value.get("adapter_files")
    if files is not None:
        if not isinstance(files, (list, tuple)) or not files:
            raise SchemaRefused(
                f"{what}: adapter_files must be a non-empty listing")
        seen = set()
        for doc in files:
            if not isinstance(doc, dict):
                raise SchemaRefused(
                    f"{what}: adapter_files entries must be objects")
            rel = doc.get("path")
            if not isinstance(rel, str) or not rel or rel.startswith("/") \
                    or ".." in rel.split("/"):
                raise SchemaRefused(
                    f"{what}: unsafe artifact listing path {rel!r}")
            if rel in seen:
                raise SchemaRefused(
                    f"{what}: duplicate artifact listing entry {rel!r}")
            seen.add(rel)
            if not isinstance(doc.get("size"), int) or doc["size"] < 0:
                raise SchemaRefused(
                    f"{what}: artifact listing entry {rel!r} needs an "
                    "integer size")
            d = doc.get("sha256")
            if not isinstance(d, str) or not DIGEST_RE.fullmatch(d):
                raise SchemaRefused(
                    f"{what}: artifact listing entry {rel!r} needs a "
                    "content digest")
    if require_production and files is None:
        raise SchemaRefused(
            f"{what}: production admission requires the explicit "
            "adapter_files listing")
    return value


def validate_activation_receipt(value: dict, *,
                                require_production: bool = False) -> dict:
    what = "activation receipt"
    value = _object(value, what)
    schema = _known_schema(value, "activation_receipt", what)
    _digest(value, "decision_digest", what)
    _digest(value, "qualification_record_digest", what)
    _digest(value, "runtime_manifest_digest", what)
    _digest(value, "adapter_digest", what)
    validate_backend(str(value.get("backend", "")), what)
    _text(value, "admitted_at", what)
    if require_production:
        if schema not in ACTIVATION_RECEIPT_PRODUCTION_SCHEMAS:
            raise SchemaRefused(
                f"{what}: schema {schema!r} is not a production "
                "activation receipt")
        _digest(value, "candidate_digest", what)
        nonce = value.get("activation_nonce")
        if not isinstance(nonce, str) or len(nonce) != 32 \
                or any(c not in "0123456789abcdef" for c in nonce):
            raise SchemaRefused(
                f"{what}: production receipt requires a 128-bit hex "
                "activation_nonce")
        loaded = value.get("loaded_artifact_digests")
        if not isinstance(loaded, (list, tuple)) or not loaded:
            raise SchemaRefused(
                f"{what}: production receipt requires the artifact "
                "digests actually measured at load")
        for pair in loaded:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise SchemaRefused(
                    f"{what}: loaded_artifact_digests entries must be "
                    "(name, digest) pairs")
            name, d = pair
            if not isinstance(name, str) or not name:
                raise SchemaRefused(
                    f"{what}: loaded artifact name must be a non-empty "
                    "string")
            if not isinstance(d, str) or not DIGEST_RE.fullmatch(d):
                raise SchemaRefused(
                    f"{what}: loaded artifact {name!r} carries a malformed "
                    "digest")
    return value


def validate_revocation_snapshot(value: dict) -> dict:
    """v16.4.2 revocation snapshot — the authenticated, monotonic
    replacement for the bare RevocationList."""
    what = "revocation snapshot"
    value = _object(value, what)
    _known_schema(value, "revocation_snapshot", what)
    epoch = _int(value, "epoch", what)
    if epoch < 0:
        raise SchemaRefused(f"{what}: epoch must be >= 0")
    issued = _int(value, "issued_at", what)
    until = _int(value, "valid_until", what)
    if until <= issued:
        raise SchemaRefused(
            f"{what}: valid_until must be after issued_at")
    digests = value.get("revoked_decision_digests")
    if not isinstance(digests, (list, tuple)):
        raise SchemaRefused(
            f"{what}: revoked_decision_digests must be a list")
    for d in digests:
        if not isinstance(d, str) or not DIGEST_RE.fullmatch(d):
            raise SchemaRefused(
                f"{what}: malformed revoked decision digest {d!r}")
    kids = value.get("revoked_key_ids")
    if not isinstance(kids, (list, tuple)):
        raise SchemaRefused(
            f"{what}: revoked_key_ids must be a list")
    for k in kids:
        if not isinstance(k, str) or not k:
            raise SchemaRefused(
                f"{what}: malformed revoked key id {k!r}")
    _digest_optional(value, "previous_snapshot_digest", what)
    return value


def validate_admission_grant(value: dict) -> dict:
    """v16.4.2 admission grant — the short-lived authorization the
    admission service issues to the serving supervisor."""
    what = "admission grant"
    value = _object(value, what)
    _known_schema(value, "admission_grant", what)
    _text(value, "grant_id", what)
    _digest(value, "promotion_decision_digest", what)
    _digest(value, "qualification_digest", what)
    _digest(value, "runtime_manifest_digest", what)
    _digest(value, "artifact_root_digest", what)
    _text(value, "backend_id", what)
    _digest_optional(value, "backend_binary_digest", what)
    _int(value, "policy_epoch", what)
    _int(value, "revocation_epoch", what)
    issued = _int(value, "issued_at", what)
    expires = _int(value, "expires_at", what)
    if expires <= issued:
        raise SchemaRefused(f"{what}: expires_at must be after issued_at")
    nonce = value.get("nonce")
    if not isinstance(nonce, str) or len(nonce) != 32 \
            or any(c not in "0123456789abcdef" for c in nonce):
        raise SchemaRefused(
            f"{what}: nonce must be a 128-bit hex string")
    _text(value, "audience_runtime_identity", what)
    return value


def validate(kind: str, value, *, require_production: bool = False) -> dict:
    """Validate one authority-bearing artifact by kind."""
    if kind == "campaign_plan":
        return validate_campaign_plan(value)
    if kind == "qualification_record":
        return validate_qualification_record(value)
    if kind == "promotion_decision":
        return validate_promotion_decision(value)
    if kind == "runtime_manifest":
        return validate_runtime_manifest(
            value, require_production=require_production)
    if kind == "activation_receipt":
        return validate_activation_receipt(
            value, require_production=require_production)
    if kind == "revocation_snapshot":
        return validate_revocation_snapshot(value)
    if kind == "admission_grant":
        return validate_admission_grant(value)
    raise SchemaRefused(f"unknown artifact kind {kind!r}")
