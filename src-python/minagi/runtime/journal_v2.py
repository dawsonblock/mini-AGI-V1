"""v16.4.3 verified journal reading + V1 migration (HARDENING_PLAN WP4 /
SEC-205).

v16.4.2 parsed persisted journal lines into records without verifying
digests, signatures, or transition legality — recovery then trusted that
history. The v16.4.3 authority store hash-chains every event;
`verify_event_log` adds the remaining layers:

  * canonical digests — payload digest recomputed from stored detail,
    event digest recomputed over the signed body;
  * hash-chain continuity — every `previous_digest` equals the previous
    event's digest (truncation or splicing breaks the chain);
  * state-transition grammar — each activation's event subsequence must
    follow the activation protocol (with the explicit exceptions for
    rollback's ACTIVE→ACTIVE control events);
  * signature + role binding — authority-bearing event kinds must carry
    a valid signature from the role authorized to make them;
  * checkpoint anchoring — the service anchors the latest accepted
    {sequence, event_digest} under `meta['anchor']`, so a rewritten or
    truncated tail is detected on next boot.

Compatibility (the spec's migration clause): the V1 JSONL journal wrote
more than one ACTIVE event per commit and wrote rollback as
ABORTED→ACTIVE. `migrate_v1_journal` validates the V1 records
(digest + signature where present), refuses ambiguous/contradictory
histories, and replays them into the authority store as `migrated`
events ending in a signed migration checkpoint. V1 events are never
rewritten to look valid — they are imported as evidence of what the
legacy log recorded.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer, SignedEnvelope
from minagi.v161.authority import as_utc

from .activation_state import ActivationState, check_transition
from .authority_store import AuthorityStore, StoreCorrupt

#: Event kinds whose events must be signed by the `runtime` role.
SIGNED_EVENT_KINDS = frozenset({
    "activation_completion", "rollback_completion", "quarantined",
    "routing_observed", "deployment_restored", "deployment_unavailable",
    "restoration_completed"})

#: Control event kinds that legitimately record a same-state marker
#: (observations and deployment-level markers are not activation
#: transitions).
_SAME_STATE_KINDS = frozenset({
    "rollback_intent", "rollback_pointer", "rollback_completion",
    "activation_completion", "migrated", "routing_observed",
    "deployment_restored", "deployment_unavailable",
    "restoration_completed", "unload_deferred",
    "backend_terminated"})

#: Activation ids that are not per-activation journal chains at all —
#: migration/deployment-level pseudo records.
_PSEUDO_IDS = frozenset({"__migration__", "__deployment__"})


class JournalRefused(StoreCorrupt):
    """The durable event log failed verification — recovery cannot
    proceed on untrusted history."""


def _verify_signature(e: dict, registry, *, now: datetime | None) -> None:
    kid = str(e.get("signer_key_id") or "")
    sig = str(e.get("signature_b64") or "")
    body = {"sequence": e["sequence"], "activation_id": e["activation_id"],
            "event_type": e["event_type"], "from_state": e["from_state"],
            "to_state": e["to_state"], "at": e["at"],
            "payload_digest": e["payload_digest"],
            "previous_digest": e["previous_digest"]}
    if not kid or not sig:
        raise JournalRefused(
            f"event {e['sequence']} ({e['event_type']}) is unsigned — "
            "authority-bearing events require a signature")
    if not registry.is_authorized("runtime", kid, now=now):
        raise JournalRefused(
            f"event {e['sequence']} signer {kid!r} is not an authorized "
            "runtime authority")
    if not registry.verifier(now=now).verify(
            body, SignedEnvelope(kid, sig)):
        raise JournalRefused(
            f"event {e['sequence']} signature invalid")


def _verify_grammar(events: list[dict]) -> None:
    """Each activation's transition subsequence must satisfy the
    activation protocol. Control events (rollback, completion) legitimately
    stay ACTIVE->ACTIVE; migrated V1 records are checked leniently."""
    last_to: dict[str, str] = {}
    for e in events:
        aid = e["activation_id"]
        if aid in _PSEUDO_IDS or \
                e["event_type"] == "migration_checkpoint":
            continue  # migration/deployment pseudo-records are not
            # activation chains
        prev = last_to.get(aid, "")
        frm, to, kind = e["from_state"], e["to_state"], e["event_type"]
        if kind == "migrated":
            last_to[aid] = to
            continue
        if not prev:
            if frm != "" or to != "REQUESTED":
                raise JournalRefused(
                    f"activation {aid}: first event must be "
                    f"'' -> REQUESTED, got {frm!r} -> {to!r}")
            last_to[aid] = to
            continue
        if frm != prev:
            raise JournalRefused(
                f"activation {aid}: event {e['sequence']} claims "
                f"{frm!r} but the recorded state was {prev!r}")
        if kind in _SAME_STATE_KINDS and to == prev:
            last_to[aid] = to
            continue
        try:
            check_transition(ActivationState(prev), ActivationState(to))
        except Exception as exc:
            raise JournalRefused(
                f"activation {aid}: illegal transition {prev} -> {to} "
                f"at event {e['sequence']}: {exc}") from exc
        last_to[aid] = to


def verify_event_log(store: AuthorityStore, registry, *,
                     now: datetime | None = None) -> list[dict]:
    """The single validated reading interface for the durable event log:
    hash chain + payload digests (store.verify_chain), then signature/role
    and transition-grammar validation. Raises JournalRefused/StoreCorrupt."""
    events = store.verify_chain()
    for e in events:
        if e["event_type"] in SIGNED_EVENT_KINDS:
            _verify_signature(e, registry, now=now)
    _verify_grammar(events)
    anchor = store.get_meta("anchor")
    if anchor:
        try:
            a = json.loads(anchor)
            seq, dgst = int(a["sequence"]), str(a["event_digest"])
            row = [e for e in events if e["sequence"] == seq]
            if not row or row[0]["event_digest"] != dgst:
                raise JournalRefused(
                    "the anchored checkpoint {sequence, event_digest} no "
                    "longer matches the durable log — the tail was "
                    "rewritten or truncated")
        except (ValueError, KeyError, TypeError) as exc:
            raise JournalRefused(
                f"stored anchor checkpoint is malformed: {exc}") from exc
    return events


def anchor_checkpoint(store: AuthorityStore, signer: Ed25519Signer, *,
                      now: datetime | None = None) -> dict:
    """Anchor the newest verified event as a signed checkpoint — the
    independently-checkable bound on tail tampering."""
    events = store.verify_chain()
    if not events:
        raise JournalRefused("no events to anchor")
    last = events[-1]
    body = {"schema": "mini-agi-v16.4.3-journal-anchor-v1",
            "sequence": int(last["sequence"]),
            "event_digest": str(last["event_digest"]),
            "anchored_at": int(as_utc(now).timestamp())}
    env = signer.sign(body)
    store.set_meta("anchor", json.dumps(
        dict(body, signer_key_id=env.key_id,
             signature_b64=env.signature_b64), sort_keys=True))
    return dict(body, signer_key_id=env.key_id,
                signature_b64=env.signature_b64)


# --- V1 -> V2 migration -------------------------------------------------

def _v1_event_type(rec: dict) -> str:
    to_state = str(rec.get("to_state", ""))
    kind = str((rec.get("detail") or {}).get("kind", ""))
    if kind == "activation_intent":
        return "commit_intent"
    if kind == "activation_completion":
        return "activation_completion"
    if kind == "rollback":
        return "migrated"          # legacy ABORTED->ACTIVE rollback
    if to_state == "REQUESTED":
        return "requested"
    if to_state == "AUTHORIZED":
        return "authorized"
    if to_state == "QUARANTINED":
        return "quarantined"
    return "transition"


def migrate_v1_journal(journal_dir, store: AuthorityStore, *,
                       registry, migration_signer: Ed25519Signer,
                       now: datetime | None = None) -> dict:
    """Validate a legacy V1 JSONL journal and import it into the
    authority store. Refuses empty, digested-mismatch, or contradictory
    histories; ends with a signed migration checkpoint."""
    src = Path(journal_dir) / "activation_journal.jsonl"
    if not src.is_file():
        return {"migrated": 0, "note": "no V1 journal"}
    if store.events():
        raise JournalRefused(
            "authority store is not empty — refusing to mix a V1 import "
            "with existing V2 history")
    lines = [ln for ln in src.read_text().splitlines() if ln.strip()]
    if not lines:
        return {"migrated": 0, "note": "empty V1 journal"}
    src_digest = digest("\n".join(lines))
    records, seen_terminal = [], {}
    for i, ln in enumerate(lines):
        try:
            rec = json.loads(ln)
        except ValueError as exc:
            raise JournalRefused(
                f"V1 record {i} is not JSON — quarantined: {exc}") from exc
        body = {k: rec[k] for k in
                ("schema", "activation_id", "from_state", "to_state",
                 "at", "detail") if k in rec}
        if rec.get("record_digest") != digest(body):
            raise JournalRefused(
                f"V1 record {i} digest mismatch — record was modified; "
                "the legacy journal cannot be trusted")
        kid = str(rec.get("signer_key_id") or "")
        sig = str(rec.get("signature_b64") or "")
        if kid and sig:
            if not registry.is_authorized("runtime", kid, now=now) or \
                    not registry.verifier(now=now).verify(
                        body, SignedEnvelope(kid, sig)):
                raise JournalRefused(
                    f"V1 record {i} signature invalid")
        aid = str(rec.get("activation_id", ""))
        to_state = str(rec.get("to_state", ""))
        if seen_terminal.get(aid) and to_state not in ("ACTIVE",):
            raise JournalRefused(
                f"V1 record {i}: activation {aid} continues after a "
                "terminal record — contradictory history quarantined")
        if to_state in ("ABORTED", "QUARANTINED"):
            seen_terminal[aid] = True
        records.append(rec)
    at = int(as_utc(now).timestamp())
    for rec in records:
        # The migration authority signs each imported event — it attests
        # that the V1 record was verified at migration time. The original
        # signature stays in detail as evidence; V2 bodies cannot reuse
        # V1 signatures (they bind different fields).
        store.append_event(
            activation_id=str(rec.get("activation_id", "")),
            event_type=_v1_event_type(rec),
            from_state=str(rec.get("from_state", "")),
            to_state=str(rec.get("to_state", "")),
            at=int(rec.get("at", 0)),
            detail=dict(rec.get("detail") or {},
                        migrated_from="v1",
                        v1_record_digest=str(rec.get("record_digest", "")),
                        v1_signer_key_id=str(rec.get("signer_key_id", ""))),
            signer=migration_signer)
    checkpoint_body = {
        "schema": "mini-agi-v16.4.3-journal-migration-v1",
        "source_journal_digest": src_digest,
        "records_imported": len(records),
        "first_v2_sequence": 1,
        "last_v2_sequence": store.last_event_sequence(),
        "migrated_at": at}
    env = migration_signer.sign(checkpoint_body)
    checkpoint = dict(checkpoint_body, signer_key_id=env.key_id,
                      signature_b64=env.signature_b64)
    store.append_event(
        activation_id="__migration__", event_type="migration_checkpoint",
        from_state="", to_state="MIGRATED", at=at,
        detail={"checkpoint_digest": digest(checkpoint_body)},
        signer=migration_signer)
    store.set_meta("migration_checkpoint",
                   json.dumps(checkpoint, sort_keys=True))
    anchor_checkpoint(store, migration_signer, now=now)
    return {"migrated": len(records),
            "checkpoint_digest": digest(checkpoint_body)}
