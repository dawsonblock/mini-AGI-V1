"""Authenticated adapter-registry state (V4-28).

An adapter *manifest* signature protects one manifest — it says nothing
about the registry's mutable safety state. The v2 registry file stored
``revoked``/``suspended``/``health`` as unsigned JSON: deleting a
revocation from the file silently restored access, and loading an
older still-valid snapshot could silently undo a newer revocation.

This module provides the two mechanisms that close that gap:

* **Signed, monotonically sequenced state.** ``seal_state`` binds the
  registry body to a monotonic ``seq`` under an authority signature;
  ``open_state`` refuses a state whose signature does not verify
  (tampering — including a deleted revocation — fails closed) and a
  state whose ``seq`` is below the caller's ``min_seq`` (a stale
  snapshot cannot roll safety state backwards).
* **Authoritative revocation log.** An append-only JSONL of signed
  revocation entries, kept under the revocation authority's custody.
  Loading a registry merges the log's verified revocations over the
  file's own list, so removing a revocation from a registry snapshot —
  signed or not — never restores access while the log survives.

The router (``verified_routing.py``) consumes these; nothing here knows
about routing.

v4.0.3 (Phase 2) completes the mechanism:

* **``RegistryEventLog``** — a hash-chained, signed state-transition
  ledger (ADAPTER_REGISTERED/QUALIFIED/REVOKED/REQUALIFIED/SUSPENDED/
  RESTORED). A revocation-only log cannot distinguish "still revoked"
  from "requalified under a new generation": the event log binds every
  transition to the adapter's qualification generation, the previous
  event's digest and a monotonic sequence, so a revoke cannot be
  silently dropped and a REQUALIFIED adapter activates only under its
  NEW qualification generation.
* **``RegistryStateAuthority``** — a durable, separately-stored signed
  head recording the newest accepted registry generation
  (``seq`` + ``state_digest`` + the event log's head digest). A
  registry file must satisfy the authority's head exactly; deleting
  the authority fails closed, and a stale signed snapshot cannot pass
  because the head binds the newest seq. The authority lives under a
  different path than the registry it authenticates — it is the
  caller's job to hold it under separate custody; a second mutable
  file next to the registry is not the assurance boundary.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from egai.common.canonical import digest, validate_digest  # noqa: F401
from egai.common.crypto import Ed25519Signer, Ed25519Verifier, SignedEnvelope

SCHEMA_REGISTRY_V3 = "mini-agi-v4.0-adapter-registry-v3"
SCHEMA_REVOCATION = "mini-agi-v4.0-revocation-entry-v1"
SCHEMA_EVENT = "mini-agi-v4.0-registry-event-v1"
SCHEMA_HEAD = "mini-agi-v4.0-registry-head-v1"

GENESIS_DIGEST = "sha256:" + "0" * 64

EVENT_TYPES = (
    "ADAPTER_REGISTERED",
    "ADAPTER_QUALIFIED",
    "ADAPTER_REVOKED",
    "ADAPTER_REQUALIFIED",
    "ADAPTER_SUSPENDED",
    "ADAPTER_RESTORED",
)

#: Transitions that leave the adapter routable.
_ACTIVATING = ("ADAPTER_REGISTERED", "ADAPTER_QUALIFIED",
               "ADAPTER_REQUALIFIED", "ADAPTER_RESTORED")
#: Transitions that remove the adapter from service.
_DEACTIVATING = ("ADAPTER_REVOKED", "ADAPTER_SUSPENDED")


def seal_state(state: dict, seq: int,
               signer: Ed25519Signer | None = None) -> dict:
    """Serialize registry state as a schema-v3 document.

    ``state`` carries the mutable safety fields (adapters, revoked,
    suspended, health). With ``signer`` the document is an authenticated
    authorization snapshot; without one it is a claim about state —
    loadable only in non-strict mode."""
    doc = {"schema": SCHEMA_REGISTRY_V3, "seq": int(seq),
           "state": dict(state)}
    if signer is not None:
        env = signer.sign(doc)
        doc["signer_key_id"] = env.key_id
        doc["signature_b64"] = env.signature_b64
    return doc


def open_state(doc: dict,
               verifier: Ed25519Verifier | None = None, *,
               min_seq: int = 0) -> dict:
    """Open a schema-v3 registry document.

    Returns ``{"state", "seq", "authenticated"}``. With a verifier, a
    missing or invalid signature raises — tampered state fails closed.
    ``min_seq`` is the caller's high-water mark: a snapshot older than
    it cannot silently undo newer safety actions."""
    if doc.get("schema") != SCHEMA_REGISTRY_V3:
        raise ValueError(f"registry schema must be {SCHEMA_REGISTRY_V3}")
    seq = int(doc.get("seq", -1))
    signed = bool(doc.get("signer_key_id"))
    if verifier is not None:
        body = {k: v for k, v in doc.items()
                if k in ("schema", "seq", "state")}
        if not signed or not verifier.verify(
                body, SignedEnvelope(str(doc.get("signer_key_id")),
                                     str(doc.get("signature_b64", "")))):
            raise ValueError(
                "registry state signature invalid — refusing tampered "
                "or unauthenticated state")
    if seq < int(min_seq):
        raise ValueError(
            f"registry state seq {seq} predates required minimum "
            f"{min_seq} — a stale snapshot cannot roll back newer "
            "revocations")
    state = doc.get("state")
    if not isinstance(state, dict):
        raise ValueError("registry state body missing")
    return {"state": state, "seq": seq, "authenticated": signed}


class RevocationLog:
    """Append-only signed revocation ledger.

    Each entry binds ``(adapter_id, seq)`` under the revocation
    authority's signature. ``load`` returns every adapter whose entry
    verifies — entries that fail verification are reported, never
    trusted. Unioned over a loaded registry's own revocation list, the
    log makes revocation deletion ineffective.
    """

    def __init__(self, path):
        self.path = Path(path)

    def append(self, signer: Ed25519Signer, adapter_id: str,
               seq: int) -> dict:
        body = {"schema": SCHEMA_REVOCATION, "adapter_id": str(adapter_id),
                "seq": int(seq),
                "revoked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
                "signer_key_id": signer.key_id}
        env = signer.sign(body)
        entry = {**body, "signature_b64": env.signature_b64}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry, sort_keys=True,
                                separators=(",", ":")) + "\n")
        return entry

    def load(self, verifier: Ed25519Verifier | None = None
             ) -> tuple[set[str], list[str], int]:
        """(revoked adapter ids, integrity failures, max verified seq)."""
        revoked: set[str] = set()
        failures: list[str] = []
        max_seq = -1
        if not self.path.is_file():
            return revoked, failures, max_seq
        for i, line in enumerate(self.path.read_text().splitlines()):
            if not line.strip():
                continue
            where = f"revocation_log[{i}]"
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                failures.append(f"{where}: unparseable")
                continue
            if e.get("schema") != SCHEMA_REVOCATION:
                failures.append(f"{where}: bad schema")
                continue
            body = {k: v for k, v in e.items() if k != "signature_b64"}
            kid, sig = str(e.get("signer_key_id", "")), \
                str(e.get("signature_b64", ""))
            if verifier is not None:
                if not kid or not sig or not verifier.verify(
                        body, SignedEnvelope(kid, sig)):
                    failures.append(f"{where}: signature invalid")
                    continue
            elif not kid or not sig:
                failures.append(f"{where}: unsigned entry")
                continue
            aid = e.get("adapter_id")
            if not aid:
                failures.append(f"{where}: adapter_id missing")
                continue
            revoked.add(str(aid))
            max_seq = max(max_seq, int(e.get("seq", -1)))
        return revoked, failures, max_seq


class RegistryEventLog:
    """Append-only, hash-chained, signed registry state-transition log.

    Each entry records ``schema``, ``seq`` (0-based, consecutive),
    ``at``, ``event`` (one of EVENT_TYPES), ``adapter_id``,
    ``qualification_generation``, ``artifact_digest``,
    ``model_revision``, ``evidence_digest``, ``prev_event_digest``,
    ``signer_key_id`` and ``signature_b64`` — the signature covers
    everything except ``signature_b64`` itself and ``event_digest``.
    ``event_digest`` chains the log: every entry binds its predecessor,
    so a removed or reordered event breaks the chain and ``verify``
    refuses to fold the log into effective state.
    """

    SCHEMA = SCHEMA_EVENT

    def __init__(self, path: str | Path):
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def _entries(self) -> list[dict]:
        if not self._path.exists():
            return []
        out = []
        for ln in self._path.read_text().splitlines():
            if ln.strip():
                out.append(json.loads(ln))
        return out

    def last_entry(self) -> dict | None:
        entries = self._entries()
        return entries[-1] if entries else None

    def last_digest(self) -> str:
        last = self.last_entry()
        return last.get("event_digest", "") if last else GENESIS_DIGEST

    def append(self, signer: Ed25519Signer, event_type: str,
               adapter_id: str, *, qualification_generation: int = 0,
               artifact_digest: str = "", model_revision: str = "",
               evidence_digest: str = "", at: int | None = None) -> dict:
        """Append a signed event; seq/prev are derived from the tail."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown registry event {event_type!r}")
        last = self.last_entry()
        entry = {
            "schema": SCHEMA_EVENT,
            "seq": int(last["seq"]) + 1 if last else 0,
            "at": int(time.time()) if at is None else int(at),
            "event": event_type,
            "adapter_id": adapter_id,
            "qualification_generation": int(qualification_generation),
            "artifact_digest": artifact_digest,
            "model_revision": model_revision,
            "evidence_digest": evidence_digest,
            "prev_event_digest": (last["event_digest"] if last
                                  else GENESIS_DIGEST),
            "signer_key_id": signer.key_id,
        }
        entry["signature_b64"] = signer.sign(entry).signature_b64
        entry["event_digest"] = digest(entry)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def verify(self, verifier: Ed25519Verifier | None,
               authority=None,
               authorized_roles: tuple = ("promotion", "revocation",
                                          "admission")) -> tuple:
        """Structural + cryptographic check of the whole chain.

        Returns ``(entries, failures)``. Any failure means the log
        must NOT be folded into effective state (fail closed): a
        dropped tail event is a dropped revocation.
        """
        failures = []
        entries = self._entries()
        prev_digest = GENESIS_DIGEST
        for i, e in enumerate(entries):
            where = f"event[{i}]"
            if not isinstance(e, dict) or e.get("schema") != SCHEMA_EVENT:
                failures.append(f"{where}: not a {SCHEMA_EVENT} entry")
                continue
            if e.get("event") not in EVENT_TYPES:
                failures.append(f"{where}: unknown event "
                                f"{e.get('event')!r}")
            if int(e.get("seq", -1)) != i:
                failures.append(f"{where}: seq {e.get('seq')} "
                                f"(expected {i} — gap or rollback)")
            if e.get("prev_event_digest") != prev_digest:
                failures.append(f"{where}: prev_event_digest mismatch "
                                f"(chain break)")
            body = {k: v for k, v in e.items()
                    if k not in ("event_digest",)}
            if digest(body) != e.get("event_digest"):
                failures.append(f"{where}: event_digest does not "
                                f"recompute")
            if verifier is not None:
                kid, sig = e.get("signer_key_id"), e.get("signature_b64")
                if not kid or not sig or not verifier.verify(
                        {k: v for k, v in e.items()
                         if k not in ("signature_b64", "event_digest")},
                        SignedEnvelope(kid, sig)):
                    failures.append(f"{where}: signature invalid")
                elif authority is not None:
                    rec = authority.record_for(kid)
                    if (rec is None
                            or rec.role not in authorized_roles
                            or not authority.is_authorized(
                                rec.role, kid)):
                        failures.append(
                            f"{where}: signer not authorized for "
                            f"{authorized_roles}")
            elif not e.get("signer_key_id") or not e.get(
                    "signature_b64"):
                failures.append(f"{where}: unsigned entry")
            prev_digest = e.get("event_digest", "")
        return entries, failures

    def effective_state(self, verifier: Ed25519Verifier | None,
                        authority=None) -> tuple:
        """Fold the verified chain into per-adapter state.

        Returns ``({adapter_id: {"status", "generation"}},
        failures)``. Transition legality is enforced:
        REVOKED/SUSPENDED require a live adapter; RESTORED only lifts a
        suspension (reinstatement after revocation requires a new
        REQUALIFIED generation); every qualification event must carry
        a generation strictly greater than the adapter's current one.
        """
        entries, failures = self.verify(verifier, authority)
        if failures:
            return {}, failures
        state = {}
        for e in entries:
            aid = e["adapter_id"]
            cur = state.get(aid, {"status": "unknown", "generation": 0})
            ev, gen = e["event"], int(e.get("qualification_generation", 0))
            if ev in ("ADAPTER_REGISTERED", "ADAPTER_QUALIFIED"):
                if gen <= cur["generation"] and cur["generation"]:
                    failures.append(
                        f"seq {e['seq']}: {ev} for {aid} at outdated "
                        f"generation {gen} (current "
                        f"{cur['generation']})")
                    continue
                state[aid] = {"status": "active", "generation": gen}
            elif ev == "ADAPTER_REQUALIFIED":
                if cur["status"] == "unknown":
                    failures.append(
                        f"seq {e['seq']}: REQUALIFIED {aid} with no "
                        f"prior registration")
                    continue
                if gen <= cur["generation"]:
                    failures.append(
                        f"seq {e['seq']}: REQUALIFIED {aid} at "
                        f"outdated generation {gen} (current "
                        f"{cur['generation']})")
                    continue
                state[aid] = {"status": "active", "generation": gen}
            elif ev == "ADAPTER_REVOKED":
                if cur["status"] == "unknown":
                    failures.append(
                        f"seq {e['seq']}: REVOKED {aid} with no prior "
                        f"registration")
                    continue
                if cur["status"] == "revoked":
                    failures.append(
                        f"seq {e['seq']}: duplicate REVOKED for {aid}")
                    continue
                cur["status"] = "revoked"
                state[aid] = cur
            elif ev == "ADAPTER_SUSPENDED":
                if cur["status"] != "active":
                    failures.append(
                        f"seq {e['seq']}: SUSPENDED {aid} while "
                        f"{cur['status']}")
                    continue
                cur["status"] = "suspended"
                state[aid] = cur
            elif ev == "ADAPTER_RESTORED":
                if cur["status"] != "suspended":
                    failures.append(
                        f"seq {e['seq']}: RESTORED {aid} while "
                        f"{cur['status']} — reinstatement from "
                        f"revocation requires ADAPTER_REQUALIFIED")
                    continue
                cur["status"] = "active"
                state[aid] = cur
        return state, failures


class RegistryStateAuthority:
    """Durable signed head for the registry's newest accepted state.

    Stored separately from the registry it authenticates. The head
    binds ``seq``, ``state_digest`` and the event log's head digest
    with a registry signer's signature; ``record_head`` is monotonic —
    a head for seq N will never be overwritten by seq <= N, so an
    operator cannot attest an older snapshot over the newest one.
    """

    SCHEMA = SCHEMA_HEAD

    def __init__(self, path: str | Path):
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def head(self, verifier: Ed25519Verifier | None = None):
        """Return ``(doc, reason)`` — ``(None, reason)`` on failure."""
        if not self._path.exists():
            return None, "registry state authority unavailable"
        try:
            doc = json.loads(self._path.read_text())
        except Exception as e:
            return None, f"authority head unreadable: {e}"
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA_HEAD:
            return None, "authority head schema unrecognized"
        if verifier is not None:
            kid, sig = doc.get("signer_key_id"), doc.get("signature_b64")
            if not kid or not sig:
                return None, "authority head unsigned"
            body = {k: v for k, v in doc.items()
                    if k != "signature_b64"}
            if not verifier.verify(body, SignedEnvelope(kid, sig)):
                return None, "authority head signature invalid"
        return doc, ""

    def record_head(self, signer: Ed25519Signer, *, seq: int,
                    state_digest: str,
                    event_head_digest: str = GENESIS_DIGEST,
                    at: int | None = None) -> dict:
        """Attest the newest accepted state. Monotonic in ``seq``."""
        prev = None
        if self._path.exists():
            try:
                prev = json.loads(self._path.read_text())
            except Exception:
                prev = None
        if prev is not None and int(seq) <= int(prev.get("seq", -1)):
            raise ValueError(
                f"state authority head is monotonic: cannot attest "
                f"seq {seq} over recorded {prev.get('seq')}")
        doc = {
            "schema": SCHEMA_HEAD,
            "seq": int(seq),
            "state_digest": state_digest,
            "event_head_digest": event_head_digest,
            "at": int(time.time()) if at is None else int(at),
            "signer_key_id": signer.key_id,
        }
        doc["signature_b64"] = signer.sign(doc).signature_b64
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        tmp.replace(self._path)
        return doc

    def check(self, seq: int, state_digest: str, *,
              verifier: Ed25519Verifier,
              event_head_digest: str | None = None) -> str:
        """Return "" iff the attested head matches this exact state."""
        doc, reason = self.head(verifier)
        if doc is None:
            return reason
        if int(doc.get("seq", -1)) != int(seq):
            return (f"registry seq {seq} does not match authority head "
                    f"{doc.get('seq')} — stale or unattested snapshot")
        if doc.get("state_digest") != state_digest:
            return "registry state digest does not match authority head"
        recorded_ev = doc.get("event_head_digest", GENESIS_DIGEST)
        if (event_head_digest or GENESIS_DIGEST) != recorded_ev:
            return ("registry event log head does not match authority "
                    "head — truncated, absent or replaced ledger")
        return ""
