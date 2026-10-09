"""v16.5 authority separation — role keys, trust root, authority ledger.

The campaign chain distinguishes six signing roles:

    plan                preregisters immutable experiments
    execution_witness   attests worker executions (receipts)
    evaluation          attests observed predictions and derived metrics
    qualification       applies frozen acceptance criteria
    promotion           authorizes deployment of qualified artifacts
    runtime             admits artifacts at load time and attests the
                        activation receipt (v16.4.0)
    revocation          signs monotonic revocation snapshots (v16.4.2)
    admission           issues short-lived admission grants binding
                        measured artifacts to a verified authorization
                        chain (v16.4.2)

Each role has its own Ed25519 identity, registered in a trust root
(TRUST_ROOT.json). The registry fail-closed rejects:

  * a key_id assigned to more than one role (same-key collapse —
    cryptographic separation with one key is no separation),
  * signatures from key_ids not registered as active for that role,
  * revoked keys,
  * keys outside their not_before/not_after validity window (FIX-005:
    the window is enforced at verification time; an unset bound is
    open, a malformed bound is rejected).

The authority ledger (AUTHORITY_LEDGER.jsonl) is append-only: every
record carries its body digest, the previous record's digest, and a
signature from the role authorized for that record kind. Preregistration
must be the first record — evidence cannot precede a trusted plan.

Deployment note: this module enforces the *cryptographic* role
separation. Genuine operational separation additionally requires the
private keys to live under different OS identities / processes / hosts
(the Colab harness provisions all roles in one step as a development
convenience — that is not the assurance boundary). See
scripts/authority_bootstrap.py.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from egai.common.canonical import digest
from egai.common.crypto import (Ed25519Signer, Ed25519Verifier,
                                SignedEnvelope)

AUTHORITY_ROLES = ("plan", "execution_witness", "evaluation",
                   "qualification", "promotion", "runtime",
                   "revocation", "admission")

# ledger record kind -> the only role permitted to sign it
RECORD_KIND_ROLE = {
    "experiment_preregistration": "plan",
    "evaluation_bundle": "evaluation",
    "qualification_record": "qualification",
    "promotion_decision": "promotion",
    "activation_receipt": "runtime",
    "revocation_snapshot": "revocation",
    "admission_grant": "admission",
}

TRUST_ROOT_SCHEMA = "mini-agi-v16.5-authority-trust-root-v1"
LEDGER_SCHEMA = "mini-agi-v16.5-authority-ledger-v1"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(now: datetime | None) -> datetime:
    """Normalize a verification time. None -> current UTC. Naive
    datetimes are rejected (canonical objects forbid them)."""
    if now is None:
        return utc_now()
    if now.tzinfo is None:
        raise ValueError("naive datetime forbidden for authority checks")
    return now.astimezone(timezone.utc)


def _parse_bound(value: str, field: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"authority {field} is not an ISO-8601 timestamp: "
            f"{value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class AuthorityRecord:
    role: str
    key_id: str
    public_key_b64: str
    status: str = "active"          # active | revoked
    not_before: str = ""
    not_after: str = ""

    def __post_init__(self):
        if self.role not in AUTHORITY_ROLES:
            raise ValueError(f"unknown authority role: {self.role}")
        if not self.key_id or not self.public_key_b64:
            raise ValueError("authority key_id/public_key required")
        if self.status not in ("active", "revoked"):
            raise ValueError("status must be active|revoked")

    def public_bytes(self) -> bytes:
        return base64.b64decode(self.public_key_b64)

    def invalid_reason(self, now: datetime) -> str:
        """'' if this identity may sign at `now`, else why it may not.
        Revocation and validity windows are enforced here — the original
        registry stored not_before/not_after but never checked them."""
        if self.status != "active":
            return f"key status is {self.status}"
        not_before = _parse_bound(self.not_before, "not_before")
        not_after = _parse_bound(self.not_after, "not_after")
        if not_before and now < not_before:
            return f"key not valid before {self.not_before}"
        if not_after and now > not_after:
            return f"key expired at {self.not_after}"
        return ""


class AuthorityRegistry:
    """Role -> authorized signing identities, from a trust root."""

    def __init__(self, records: Iterable[AuthorityRecord]):
        self._by_key: dict[str, AuthorityRecord] = {}
        for r in records:
            if r.key_id in self._by_key:
                raise ValueError(
                    f"key_id {r.key_id} assigned to multiple roles "
                    "(same-key collapse is not separation)")
            self._by_key[r.key_id] = r

    @classmethod
    def load(cls, trust_root_path) -> "AuthorityRegistry":
        doc = json.loads(Path(trust_root_path).read_text())
        if doc.get("schema") != TRUST_ROOT_SCHEMA:
            raise ValueError("unrecognized trust-root schema")
        return cls(AuthorityRecord(**a) for a in doc["authorities"])

    def record_for(self, key_id: str) -> AuthorityRecord | None:
        return self._by_key.get(key_id)

    def is_authorized(self, role: str, key_id: str, *,
                      now: datetime | None = None) -> bool:
        r = self._by_key.get(key_id)
        if r is None or r.role != role:
            return False
        return not r.invalid_reason(as_utc(now))

    def verifier(self, *, now: datetime | None = None) -> Ed25519Verifier:
        """Verifier registering only identities valid at `now` — an
        expired or not-yet-valid key must not verify anything."""
        at = as_utc(now)
        v = Ed25519Verifier()
        for r in self._by_key.values():
            if not r.invalid_reason(at):
                v.register(r.key_id, r.public_bytes())
        return v

    def assert_authorized(self, role: str, key_id: str, *,
                          now: datetime | None = None):
        at = as_utc(now)
        rec = self._by_key.get(key_id)
        if rec is None:
            why = "unregistered signing identity"
        elif rec.role != role:
            why = f"key registered for role {rec.role}, not {role}"
        else:
            why = rec.invalid_reason(at)
        if why:
            raise PermissionError(
                f"{role} signature from {key_id}: {why}")


def provision_role(keys_dir, role: str) -> Ed25519Signer:
    """Load (or generate) a role's private signing key."""
    keys_dir = Path(keys_dir)
    keys_dir.mkdir(parents=True, exist_ok=True)
    key_path = keys_dir / f"{role}.pem"
    if key_path.is_file():
        # key_id is derived from the public key so a reloaded key always
        # identifies as the identity originally registered in the trust
        # root — never let a label alias a different identity.
        return Ed25519Signer.from_private_bytes(key_path.read_bytes())
    signer = Ed25519Signer.generate()
    key_path.write_bytes(signer.private_bytes())
    key_path.chmod(0o600)
    return signer


def write_trust_root(keys_dir, path) -> AuthorityRegistry:
    """Materialize role keys and emit TRUST_ROOT.json (pubkeys only)."""
    records = []
    for role in AUTHORITY_ROLES:
        s = provision_role(keys_dir, role)
        records.append(AuthorityRecord(
            role=role, key_id=s.key_id,
            public_key_b64=base64.b64encode(s.public_bytes()).decode()))
    doc = {"schema": TRUST_ROOT_SCHEMA,
           "authorities": [
               {"role": r.role, "key_id": r.key_id,
                "public_key_b64": r.public_key_b64,
                "status": r.status,
                "not_before": r.not_before,
                "not_after": r.not_after} for r in records]}
    Path(path).write_text(json.dumps(doc, indent=2, sort_keys=True))
    return AuthorityRegistry(records)


class AuthorityLedger:
    """Append-only chain of signed authority records."""

    def __init__(self, path):
        self.path = Path(path)

    def _records(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines()
                if line.strip()]

    def append(self, signer: Ed25519Signer, kind: str,
               body: dict) -> dict:
        if kind not in RECORD_KIND_ROLE:
            raise ValueError(f"unknown ledger record kind: {kind}")
        records = self._records()
        prev = records[-1]["record_digest"] if records else \
            "sha256:" + "0" * 64
        entry = {"seq": len(records), "schema": LEDGER_SCHEMA,
                 "kind": kind, "prev_record_digest": prev,
                 "body_digest": digest(body)}
        env = signer.sign(entry)
        entry.update({"signer_key_id": env.key_id,
                      "signature_b64": env.signature_b64})
        entry["record_digest"] = digest(entry)
        with self.path.open("a") as f:
            f.write(json.dumps(entry, sort_keys=True,
                               separators=(",", ":")) + "\n")
        return entry

    def verify(self, registry: AuthorityRegistry,
               verifier: Ed25519Verifier | None = None, *,
               now: datetime | None = None) -> list[str]:
        """Return a list of integrity failures ([] = chain intact).

        Authorization is checked at `now` (default: current UTC) — a
        record signed by a key that has since expired, or by a key not
        yet valid, is a failure."""
        verifier = verifier or registry.verifier(now=now)
        failures = []
        prev = "sha256:" + "0" * 64
        for i, rec in enumerate(self._records()):
            where = f"ledger[{i}]"
            if int(rec.get("seq", -1)) != i:
                failures.append(f"{where}: seq gap")
            if rec.get("prev_record_digest") != prev:
                failures.append(f"{where}: broken prev_record_digest chain")
            if rec.get("schema") != LEDGER_SCHEMA:
                failures.append(f"{where}: bad schema")
            if rec.get("record_digest") != digest(
                    {k: v for k, v in rec.items()
                     if k != "record_digest"}):
                failures.append(f"{where}: record_digest mismatch")
            kind = rec.get("kind")
            role = RECORD_KIND_ROLE.get(kind)
            if role is None:
                failures.append(f"{where}: unknown kind {kind!r}")
            else:
                kid = rec.get("signer_key_id", "")
                if not registry.is_authorized(role, kid, now=now):
                    failures.append(
                        f"{where}: signer not authorized for {role}")
                body = {k: v for k, v in rec.items()
                        if k not in ("signer_key_id", "signature_b64",
                                     "record_digest")}
                if not verifier.verify(
                        body, SignedEnvelope(
                            kid, rec.get("signature_b64", ""))):
                    failures.append(f"{where}: signature invalid")
            prev = rec.get("record_digest", prev)
        if self._records() and self._records()[0].get("kind") != \
                "experiment_preregistration":
            failures.append("ledger[0]: first record must be "
                            "experiment_preregistration")
        return failures
