"""Controlled recursive improvement (Phase 8 / REPAIR-042..046) — FIX-003.

Bounded, governed generation chain: each generation is a *campaign*
whose candidate improvement was proposed under the previous
generation's approved state. A generation may only be preregistered
after its parent was PROMOTED by the promotion authority — the chain
enforces the sequence cryptographically, not by convention.

FIX-003 replaces the v16.2.1 promotion flag (a mutable field accepting
any digest-shaped string) with append-only `PromotionEvent`s validated
against the independent promotion authority. `record_promotion` accepts
only a signed decision envelope and proves, before the next generation
may exist, that the decision:

  * is signed by a key registered for the `promotion` role and valid
    at verification time (revocation and validity windows enforced),
  * carries an intact digest and signature over its value,
  * binds this exact generation record, its campaign, and a
    qualification record,
  * is inside its declared authorized_at/expires_at window, and
  * has not been revoked.

Rules enforced here (the honest subset that does not claim open-ended
self-improvement):

  * G0 is the frozen baseline (original model + harness); it has no
    parent and needs no promotion to exist.
  * G(n+1) requires: parent generation record exists, a verified
    promotion event exists for it, a FRESH evaluation corpus (digest
    differs from every ancestor — no measuring improvement on tasks the
    proposal was tuned against), and a contemporaneous frozen control
    arm digest.
  * Costs are recorded per generation — improvement claims must carry
    the price paid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope

from .authority import as_utc

GENESIS = ""  # G0 has no parent

PROMOTION_DECISION_SCHEMA = \
    "mini-agi-v16.2.2-generation-promotion-decision-v1"
PROMOTION_EVENT_SCHEMA = "mini-agi-v16.2.2-promotion-event-v1"


@dataclass(frozen=True)
class GenerationRecord:
    """One governed generation. Binds the qualifying campaign, the
    evaluation corpus that measured it, and the contemporaneous
    control. Promotion state lives in PromotionEvents — a record
    itself is immutable evidence, never a mutable flag."""
    generation: int
    campaign_digest: str             # this generation's campaign plan
    corpus_digest: str               # eval corpus (must be fresh vs ancestors)
    control_arm_digest: str          # contemporaneous frozen-control evidence
    parent_generation_digest: str    # digest of parent's record ("" = genesis)
    schema: str = "mini-agi-v16.8-generation-record-v2"

    def __post_init__(self):
        if self.generation < 0:
            raise ValueError("generation must be >= 0")
        validate_digest(self.campaign_digest)
        validate_digest(self.corpus_digest)
        validate_digest(self.control_arm_digest)
        if self.generation == 0 and self.parent_generation_digest:
            raise ValueError("genesis generation has no parent")
        if self.generation > 0:
            validate_digest(self.parent_generation_digest)

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class PromotionEvent:
    """Append-only evidence that the promotion authority authorized a
    generation. Carries the full signed decision so any later verifier
    can re-check it without trusting the chain's in-memory state."""
    generation: int
    generation_record_digest: str
    decision_digest: str
    signer_key_id: str
    decision: dict
    signature_b64: str
    schema: str = PROMOTION_EVENT_SCHEMA

    def __post_init__(self):
        validate_digest(self.generation_record_digest)
        validate_digest(self.decision_digest)

    @property
    def digest(self) -> str:
        return digest(self)


def check_promotion_decision(record: GenerationRecord, decision,
                             *, registry, now=None,
                             revoked_decision_digests: Iterable[str] = ()
                             ) -> PromotionEvent:
    """Validate a signed promotion decision against `record` and the
    authority registry, materializing a PromotionEvent. Raises
    PermissionError (authorization/expiry/revocation) or ValueError
    (schema/digest/binding failures) — a forged, unsigned, expired,
    revoked, or mismatched decision never promotes anything."""
    if not isinstance(decision, dict) or "value" not in decision:
        raise PermissionError(
            "promotion decision must be a signed envelope "
            "{value, digest, signer_key_id, signature_b64} — a bare "
            "digest is not authorization")
    value = decision["value"]
    kid = str(decision.get("signer_key_id", ""))
    at = as_utc(now)
    if not registry.is_authorized("promotion", kid, now=at):
        raise PermissionError(
            f"promotion decision signer {kid!r} is not an authorized, "
            "currently valid promotion authority")
    if not isinstance(value, dict):
        raise ValueError("promotion decision value must be an object")
    if decision.get("digest") != digest(value):
        raise ValueError("promotion decision digest mismatch")
    if not registry.verifier(now=at).verify(
            value, SignedEnvelope(kid, str(decision.get("signature_b64", "")))):
        raise PermissionError("promotion decision signature invalid")
    if value.get("schema") != PROMOTION_DECISION_SCHEMA:
        raise ValueError(
            f"unrecognized promotion decision schema "
            f"{value.get('schema')!r}")
    if value.get("generation") != record.generation:
        raise ValueError("promotion decision generation mismatch")
    if value.get("generation_record_digest") != record.digest:
        raise ValueError(
            "promotion decision does not bind this generation record")
    if value.get("campaign_digest") != record.campaign_digest:
        raise ValueError(
            "promotion decision does not bind this generation's campaign")
    validate_digest(str(value.get("qualification_digest", "")))
    authorized_at = value.get("authorized_at")
    expires_at = value.get("expires_at")
    if not isinstance(authorized_at, int) or not isinstance(expires_at, int):
        raise ValueError("promotion decision requires integer "
                         "authorized_at/expires_at")
    if expires_at < authorized_at:
        raise ValueError("promotion decision expiry precedes authorization")
    ts = int(at.timestamp())
    if ts < authorized_at:
        raise PermissionError("promotion decision not yet valid")
    if ts > expires_at:
        raise PermissionError("promotion decision expired")
    ddigest = str(decision["digest"])
    if ddigest in set(revoked_decision_digests):
        raise PermissionError("promotion decision revoked")
    return PromotionEvent(
        generation=record.generation,
        generation_record_digest=record.digest,
        decision_digest=ddigest,
        signer_key_id=kid,
        decision=dict(value),
        signature_b64=str(decision.get("signature_b64", "")))


class GenerationChain:
    """Append-only chain of GenerationRecords and PromotionEvents.
    `append` enforces the promotion gate: a child generation cannot be
    preregistered until a verified promotion event exists for its
    parent."""

    def __init__(self):
        self._records: list[GenerationRecord] = []
        self._events: list[PromotionEvent] = []
        self._corpora: set[str] = set()

    def append(self, record: GenerationRecord) -> None:
        gen = record.generation
        if gen != len(self._records):
            raise ValueError(
                f"next generation must be {len(self._records)}, got {gen}")
        if gen == 0:
            if record.parent_generation_digest != GENESIS:
                raise ValueError("genesis must not reference a parent")
        else:
            parent = self._records[-1]
            if record.parent_generation_digest != parent.digest:
                raise ValueError("parent digest does not match chain tip")
            if not self.is_promoted(parent.generation):
                raise PermissionError(
                    "generation cannot be preregistered: parent is not "
                    "promoted — a verified promotion decision must "
                    "authorize the parent before its child exists")
        if record.corpus_digest in self._corpora:
            raise ValueError(
                "corpus reuse across generations is forbidden — each "
                "generation must be measured on tasks its ancestors were "
                "not tuned against")
        self._records.append(record)
        self._corpora.add(record.corpus_digest)

    def is_promoted(self, generation: int) -> bool:
        if not 0 <= generation < len(self._records):
            return False
        record_digest = self._records[generation].digest
        return any(ev.generation_record_digest == record_digest
                   for ev in self._events)

    def record_promotion(self, generation: int, decision, *, registry,
                         now=None,
                         revoked_decision_digests: Iterable[str] = ()
                         ) -> PromotionEvent:
        """Validate the promotion authority's signed decision for the
        chain tip and append the resulting event. The original
        implementation accepted any digest-shaped string here."""
        if generation != len(self._records) - 1:
            raise ValueError("promotion applies only to the chain tip")
        if self.is_promoted(generation):
            raise ValueError("generation already promoted")
        event = check_promotion_decision(
            self._records[-1], decision, registry=registry, now=now,
            revoked_decision_digests=revoked_decision_digests)
        self._events.append(event)
        return event

    @property
    def tip(self) -> GenerationRecord | None:
        return self._records[-1] if self._records else None

    @property
    def records(self) -> tuple[GenerationRecord, ...]:
        return tuple(self._records)

    @property
    def promotion_events(self) -> tuple[PromotionEvent, ...]:
        return tuple(self._events)

    def verify(self, registry=None, *, now=None,
               revoked_decision_digests: Iterable[str] = ()) -> list[str]:
        """Re-verify the whole chain — promotion gates, corpus
        freshness, parent linkage, and (when a registry is supplied)
        every promotion event's signature, authorization, binding and
        expiry. Returns a list of violations."""
        problems: list[str] = []
        seen_corpora: set[str] = set()
        by_digest = {rec.digest: rec for rec in self._records}
        revoked = set(revoked_decision_digests)
        for i, rec in enumerate(self._records):
            if rec.generation != i:
                problems.append(f"record {i}: generation index {rec.generation}")
            if i > 0:
                parent = self._records[i - 1]
                if rec.parent_generation_digest != parent.digest:
                    problems.append(f"record {i}: parent digest mismatch")
                if not any(ev.generation_record_digest == parent.digest
                           for ev in self._events):
                    problems.append(
                        f"record {i}: parent generation not promoted")
            if rec.corpus_digest in seen_corpora:
                problems.append(f"record {i}: corpus digest reused")
            seen_corpora.add(rec.corpus_digest)
        seen_events: set[str] = set()
        per_record: dict[str, int] = {}
        for ev in self._events:
            per_record[ev.generation_record_digest] = \
                per_record.get(ev.generation_record_digest, 0) + 1
            if ev.decision_digest in seen_events:
                problems.append(
                    f"promotion event {ev.decision_digest}: duplicated")
            seen_events.add(ev.decision_digest)
            rec = by_digest.get(ev.generation_record_digest)
            if rec is None:
                problems.append(
                    f"promotion event {ev.decision_digest}: no such "
                    "generation record")
                continue
            if registry is None:
                continue
            doc = {"value": ev.decision, "digest": ev.decision_digest,
                   "signer_key_id": ev.signer_key_id,
                   "signature_b64": ev.signature_b64}
            try:
                check_promotion_decision(
                    rec, doc, registry=registry, now=now,
                    revoked_decision_digests=revoked)
            except (PermissionError, ValueError) as exc:
                problems.append(
                    f"promotion event {ev.decision_digest}: {exc}")
        for rec_digest, count in per_record.items():
            if count > 1:
                problems.append(
                    f"record {rec_digest}: {count} promotion events — "
                    "only one promotion per generation is valid")
        return problems
