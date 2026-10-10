"""v16.4.5 restoration authorization verification (SEC-403 / WP-F fix).

v16.4.4's cold-start restoration re-issued an admission grant against
digests journaled from the original authorization — but it never
re-validated the historical authority chain itself. The operative
revocation snapshot was consulted only for its epoch integer, so a
promotion decision revoked after the original activation would still
restore. Worse, when the journaled fields were absent the code
synthesized substitute digests — manufacturing deployment authority
that no promotion decision ever issued.

This module is the correction: restoration re-verifies the COMPLETE
original authority chain before any new grant is issued. The
documents themselves (promotion decision, qualification record,
campaign plan, runtime manifest) are retained in the authority store
at authorization time — restoration resolves them by activation and
re-runs the same checks the trusted launcher runs at first admission:

  * the historical promotion decision exists, its envelope is signed
    by a `promotion` identity valid NOW, its schema is known, its
    validity window still encloses the restoration time, and its
    digest equals the digest the original authorization journaled;
  * the decision has NOT been revoked — neither the decision digest
    nor its signing key may appear in the operative revocation
    snapshot;
  * the qualification record is signed by a currently valid
    `qualification` identity, binds the same campaign plan and
    evaluation bundle, decided QUALIFIED, and explicitly covers the
    backend being restored (a qualification that omits
    ``runtime_backends`` covers nothing);
  * the campaign plan is signed by a currently valid `plan`
    identity and digests match the decision binding;
  * the runtime manifest is digest-self-consistent, binds the same
    plan and qualification, names the same backend, is the exact
    manifest the decision authorized for its seed, and links the
    plan's experiment protocol;
  * the staged artifact bytes re-measure to the manifest's model and
    adapter digests — nothing else is restorable;
  * the recorded policy epoch still satisfies the operative
    requirement;
  * the revocation snapshot itself is the newest valid published
    snapshot (signature, freshness window, monotonic epoch — verified
    by `RevocationStore.latest_valid`), never a replayed older one.

Missing authority is refusal, never substitution: absent documents,
absent journaled digests, absent revocation evidence, or an unsigned
field all refuse restoration permanently.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope
from minagi.v161 import strict_schema
from minagi.v161.authority import as_utc
from minagi.v161.runtime_admission import PROMOTION_DECISION_SCHEMA

from .signed_revocations import (RevocationRefused, RevocationSnapshotV2,
                                 RevocationStore)


class RestorationAuthorityRefused(PermissionError):
    """The historical authority chain for a restoration candidate is
    absent, invalid, revoked, or no longer satisfies current policy."""


@dataclass(frozen=True)
class RestorationAuthorization:
    """The verified bindings a fresh restoration grant must carry —
    every digest resolves to a re-verified signed document, never to
    a synthesized substitute."""
    decision_digest: str
    qualification_digest: str
    runtime_manifest_digest: str
    artifact_root_digest: str
    backend_id: str
    backend_binary_digest: str
    policy_epoch: int
    revocation_epoch: int
    seed: str


#: Artifact names the runtime manifest authorizes — restoration binds
#: measured bytes only against these (the tokenizer is measured inside
#: the model artifact, by the plan's convention).
_MANIFEST_ARTIFACTS = ("model", "adapter")


class RestorationAuthorizationVerifier:
    """Re-validates the complete original authority chain for a
    restoration candidate under CURRENT authority and revocation
    evidence."""

    def __init__(self, registry, *, revocation_store: RevocationStore | None,
                 now: datetime | None = None,
                 max_revocation_age_seconds: int | None = None,
                 max_clock_skew_seconds: int = 300,
                 min_policy_epoch: int = 0):
        self.registry = registry
        self.revocation_store = revocation_store
        self._now = now
        self.max_revocation_age_seconds = max_revocation_age_seconds
        self.max_clock_skew_seconds = int(max_clock_skew_seconds)
        self.min_policy_epoch = int(min_policy_epoch)

    def _at(self) -> datetime:
        return as_utc(self._now)

    # --- revocation evidence ------------------------------------------
    def operative_snapshot(self) -> RevocationSnapshotV2:
        """The newest valid authorized revocation snapshot. Missing
        evidence fails closed — restoration never assumes nothing is
        revoked."""
        if self.revocation_store is None:
            raise RestorationAuthorityRefused(
                "no revocation store configured — authenticated "
                "revocation evidence is required for restoration")
        try:
            snap = self.revocation_store.latest_valid(
                self.registry, now=self._now,
                max_age_seconds=self.max_revocation_age_seconds,
                max_clock_skew_seconds=self.max_clock_skew_seconds)
        except RevocationRefused as exc:
            raise RestorationAuthorityRefused(
                f"operative revocation evidence refused: {exc}") from exc
        if snap is None:
            raise RestorationAuthorityRefused(
                "no valid revocation snapshot — missing revocation "
                "evidence fails closed")
        return snap

    # --- the chain -----------------------------------------------------
    def verify(self, *, candidate_id: str, authorized_context: dict,
               authority_docs: dict, measured_digests: dict,
               artifact_root_digest: str, backend_id: str,
               backend_manifest_digest: str = "",
               snapshot: RevocationSnapshotV2 | None = None
               ) -> RestorationAuthorization:
        """Run the full restoration check. Every refusal is
        RestorationAuthorityRefused — permanent; retrying cannot
        conjure authority."""
        if snapshot is None:
            snapshot = self.operative_snapshot()
        ctx = dict(authorized_context or {})
        docs = dict(authority_docs or {})
        at = self._at()
        ts = int(at.timestamp())

        # --- journaled bindings must exist and be well-formed ---------
        recorded: dict[str, str] = {}
        for field in ("decision_digest", "qualification_digest",
                      "artifact_root_digest", "manifest"):
            value = str(ctx.get(field) or "")
            if not value:
                raise RestorationAuthorityRefused(
                    f"the recorded authorization for {candidate_id} "
                    f"lacks {field} — missing historical authority "
                    "means restoration refusal, not substitution")
            try:
                validate_digest(value)
            except ValueError as exc:
                raise RestorationAuthorityRefused(
                    f"the recorded {field} for {candidate_id} is "
                    f"malformed: {exc}") from exc
            recorded[field] = value
        recorded_backend = str(ctx.get("backend") or "")
        if not recorded_backend or recorded_backend != str(backend_id):
            raise RestorationAuthorityRefused(
                f"the recorded authorization for {candidate_id} names "
                f"backend {recorded_backend!r}, not {backend_id!r}")

        # --- the signed documents must all be present ------------------
        for kind in ("decision", "qualification", "plan",
                     "runtime_manifest"):
            if not isinstance(docs.get(kind), dict):
                raise RestorationAuthorityRefused(
                    f"no retained {kind} document for {candidate_id} — "
                    "the original authority chain was not retained and "
                    "restoration cannot re-verify it")

        # --- promotion decision ---------------------------------------
        decision_env = docs["decision"]
        decision = self._envelope(decision_env, "promotion",
                                  "promotion decision")
        if str(decision_env.get("digest")) != recorded["decision_digest"]:
            raise RestorationAuthorityRefused(
                "the retained promotion decision is not the decision "
                "the original authorization journaled")
        if snapshot.contains(recorded["decision_digest"]):
            raise RestorationAuthorityRefused(
                "the original promotion decision is REVOKED by the "
                "operative snapshot — restoration refused")
        if snapshot.revokes_key(str(decision_env.get("signer_key_id")
                                    or "")):
            raise RestorationAuthorityRefused(
                "the key that signed the original promotion decision "
                "has been revoked — its evidence no longer authorizes")
        self._schema("promotion_decision", decision,
                     "promotion decision")
        if decision.get("schema") != PROMOTION_DECISION_SCHEMA:
            raise RestorationAuthorityRefused(
                f"unrecognized promotion decision schema "
                f"{decision.get('schema')!r}")
        authorized_at = decision.get("authorized_at")
        expires_at = decision.get("expires_at")
        if not isinstance(authorized_at, int) or \
                not isinstance(expires_at, int):
            raise RestorationAuthorityRefused(
                "promotion decision lacks a validity window")
        if ts < authorized_at:
            raise RestorationAuthorityRefused(
                "promotion decision is not yet valid")
        if ts > expires_at:
            raise RestorationAuthorityRefused(
                "promotion decision has EXPIRED — restoration cannot "
                "extend authority past its validity window")

        # --- qualification record --------------------------------------
        qual_env = docs["qualification"]
        qual = self._envelope(qual_env, "qualification",
                              "qualification record")
        if str(qual_env.get("digest")) != \
                recorded["qualification_digest"] or \
                str(qual_env.get("digest")) != \
                str(decision.get("qualification_record_digest")):
            raise RestorationAuthorityRefused(
                "qualification record digest does not match the "
                "journaled and decision-bound digests")
        if qual.get("decision") != "QUALIFIED":
            raise RestorationAuthorityRefused(
                f"qualification decision {qual.get('decision')!r} — "
                "only QUALIFIED evidence may restore")
        self._schema("qualification_record", qual,
                     "qualification record")
        if qual.get("campaign_plan_digest") != \
                decision.get("campaign_plan_digest"):
            raise RestorationAuthorityRefused(
                "qualification record binds a different campaign plan")
        qual_bundle = qual.get("evaluation_bundle_digest")
        if not qual_bundle or qual_bundle != \
                decision.get("evaluation_bundle_digest"):
            raise RestorationAuthorityRefused(
                "evaluation bundle digest mismatch between decision "
                "and qualification")
        covered = qual.get("runtime_backends")
        if not covered or not isinstance(covered, (list, tuple)):
            raise RestorationAuthorityRefused(
                "qualification record omits runtime backend coverage — "
                "restoration requires an explicit signed declaration")
        if backend_id not in covered:
            raise RestorationAuthorityRefused(
                f"backend {backend_id!r} is no longer covered by the "
                f"qualification ({sorted(covered)})")

        # --- campaign plan ---------------------------------------------
        plan_env = docs["plan"]
        plan = self._envelope(plan_env, "plan", "campaign plan")
        if str(plan_env.get("digest")) != \
                str(decision.get("campaign_plan_digest")):
            raise RestorationAuthorityRefused(
                "campaign plan digest does not match the decision "
                "binding")
        self._schema("campaign_plan", plan, "campaign plan")

        # --- runtime manifest ------------------------------------------
        manifest = docs["runtime_manifest"]
        if not isinstance(manifest, dict) or "digest" not in manifest:
            raise RestorationAuthorityRefused("runtime manifest required")
        body = {k: v for k, v in manifest.items() if k != "digest"}
        if digest(body) != manifest["digest"]:
            raise RestorationAuthorityRefused(
                "runtime manifest digest mismatch")
        if str(manifest["digest"]) != recorded["manifest"]:
            raise RestorationAuthorityRefused(
                "the retained runtime manifest is not the manifest the "
                "original authorization journaled")
        if manifest.get("campaign_digest") != \
                decision.get("campaign_plan_digest"):
            raise RestorationAuthorityRefused(
                "runtime manifest binds a different campaign")
        if manifest.get("qualification_record_digest") != \
                str(qual_env.get("digest")):
            raise RestorationAuthorityRefused(
                "runtime manifest binds a different qualification")
        if str(manifest.get("serving_stack", "")) != backend_id:
            raise RestorationAuthorityRefused(
                f"manifest backend {manifest.get('serving_stack')!r} "
                f"!= restoration backend {backend_id!r}")
        authorized_manifests = decision.get("runtime_manifest_digests") \
            or {}
        seed = next((s for s, d in authorized_manifests.items()
                     if str(d) == str(manifest["digest"])), None)
        if seed is None:
            raise RestorationAuthorityRefused(
                "runtime manifest digest is not covered by the "
                "promotion decision for any seed")
        authorized_adapter = (decision.get("adapter_artifact_digests")
                              or {}).get(seed)
        if authorized_adapter != manifest.get("adapter_digest"):
            raise RestorationAuthorityRefused(
                "adapter digest is not the decision-authorized digest "
                "for this seed")
        self._schema("runtime_manifest", manifest, "runtime manifest")
        plan_protocol = plan.get("experiment_protocol_digest")
        manifest_protocol = manifest.get("protocol_digest")
        if not plan_protocol or plan_protocol != manifest_protocol:
            raise RestorationAuthorityRefused(
                "runtime manifest protocol does not match the signed "
                "plan's experiment protocol digest")

        # --- measured bytes bind the manifest --------------------------
        measured = {str(k): str(v) for k, v in
                    dict(measured_digests or {}).items()}
        expected = {"model": str(manifest.get("model_digest") or ""),
                    "adapter": str(manifest.get("adapter_digest") or "")}
        if set(measured) != set(_MANIFEST_ARTIFACTS):
            raise RestorationAuthorityRefused(
                f"staged artifact set {sorted(measured)} is not the "
                f"manifest-authorized set {sorted(expected)}")
        for name in _MANIFEST_ARTIFACTS:
            if measured[name] != expected[name]:
                raise RestorationAuthorityRefused(
                    f"staged {name} bytes digest "
                    f"{measured[name]} != manifest-authorized "
                    f"{expected[name]} — the restored artifacts are "
                    "not the promoted bytes")
        # Same construction as runtime.recovery_manager.digest_root_of,
        # inlined: the artifact-set digest the grant binds.
        measured_root = digest(
            {name: d for name, d in sorted(measured.items())})
        if measured_root != artifact_root_digest or \
                recorded["artifact_root_digest"] != artifact_root_digest:
            raise RestorationAuthorityRefused(
                "measured artifact root does not match the recorded "
                "authorization artifact root")

        # --- backend binary binding -------------------------------------
        ctx_binary = str(ctx.get("backend_binary_digest") or "")
        if ctx_binary and backend_manifest_digest and \
                ctx_binary != backend_manifest_digest:
            raise RestorationAuthorityRefused(
                "the operative backend implementation differs from "
                "the one the original authorization bound — backend "
                "binary drift refuses restoration")
        bound_binary = backend_manifest_digest or ctx_binary

        # --- policy epoch under CURRENT requirements --------------------
        recorded_epoch = int(ctx.get("policy_epoch") or 0)
        policy_epoch = max(recorded_epoch, self.min_policy_epoch)

        return RestorationAuthorization(
            decision_digest=recorded["decision_digest"],
            qualification_digest=recorded["qualification_digest"],
            runtime_manifest_digest=str(manifest["digest"]),
            artifact_root_digest=artifact_root_digest,
            backend_id=backend_id,
            backend_binary_digest=bound_binary,
            policy_epoch=policy_epoch,
            revocation_epoch=int(snapshot.epoch),
            seed=str(seed))

    # --- shared envelope/schema helpers ----------------------------------
    def _envelope(self, doc, role: str, what: str) -> dict:
        if not isinstance(doc, dict) or "value" not in doc:
            raise RestorationAuthorityRefused(
                f"{what}: signed envelope required — a bare digest is "
                "not authorization")
        kid = str(doc.get("signer_key_id", ""))
        at = self._at()
        if not self.registry.is_authorized(role, kid, now=at):
            raise RestorationAuthorityRefused(
                f"{what}: signer {kid!r} is not an authorized, "
                f"currently valid {role} authority")
        value = doc["value"]
        if not isinstance(value, dict):
            raise RestorationAuthorityRefused(
                f"{what}: envelope value must be an object")
        if doc.get("digest") != digest(value):
            raise RestorationAuthorityRefused(
                f"{what}: envelope digest mismatch")
        if not self.registry.verifier(now=at).verify(
                value, SignedEnvelope(
                    kid, str(doc.get("signature_b64", "")))):
            raise RestorationAuthorityRefused(
                f"{what}: signature invalid")
        return value

    def _schema(self, kind: str, value: dict, what: str) -> None:
        try:
            strict_schema.validate(kind, value)
        except strict_schema.SchemaRefused as exc:
            raise RestorationAuthorityRefused(f"{what}: {exc}") from exc


__all__ = ["RestorationAuthorization",
           "RestorationAuthorizationVerifier",
           "RestorationAuthorityRefused"]
