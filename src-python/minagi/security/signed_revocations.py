"""v16.4.2 authenticated revocation snapshots (UPGRADE_PLAN §3.4).

The v16.4.1 `RevocationList` is a bare list of digests with a
generation timestamp: unsigned, non-monotonic, and unable to reject a
sufficiently future-dated list. `RevocationSnapshotV2` replaces it for
the production path:

    schema                     strict-schema versioned
    epoch                      monotonically increasing; the store
                               refuses a regression
    issued_at / valid_until    the snapshot's own validity window
    revoked_decision_digests   promotion decisions withdrawn
    revoked_key_ids            compromised or retired authority keys
    previous_snapshot_digest   chain to the superseded snapshot
    issuer_key_id + signature  the `revocation` authority role — a
                               dedicated role, not promotion/runtime

Verification rules (all fail closed):

  * the signature verifies and the signer is a currently authorized
    `revocation` role identity;
  * the snapshot must not be older than the configured freshness limit
    (age checked against `issued_at`);
  * `issued_at` beyond the configured clock-skew allowance is refused —
    a snapshot "from 2100" is evidence of forgery, not freshness;
  * `valid_until` already elapsed is refused;
  * epochs cannot move backward — the durable store tracks the newest
    published epoch and rejects a replayed older one;
  * missing revocation evidence fails closed for production
    activation (enforced by the caller — nothing here silently
    substitutes an empty list).

The durable store (`RevocationStore`) publishes snapshots atomically
(write + fsync + rename) and, when asked for the operative snapshot,
selects the *newest valid* authorized snapshot rather than an
arbitrarily selected older one.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import Ed25519Signer, SignedEnvelope
from minagi.v161 import strict_schema
from minagi.v161.authority import as_utc

REVOCATION_SNAPSHOT_SCHEMA = "mini-agi-v16.4.2-revocation-snapshot-v2"

#: Default tolerance for a snapshot whose issued_at is ahead of the
#: verifier's clock (seconds). Generous enough for NTP jitter, tight
#: enough to refuse "from 2100" forgeries.
DEFAULT_CLOCK_SKEW_SECONDS = 300

ZERO_DIGEST = "sha256:" + "0" * 64


class RevocationRefused(PermissionError):
    """Revocation evidence is absent, invalid, or not operative."""


@dataclass(frozen=True)
class RevocationSnapshotV2:
    """A signed, epoch-ordered revocation snapshot."""
    epoch: int
    issued_at: int                     # unix seconds
    valid_until: int                   # unix seconds
    revoked_decision_digests: tuple[str, ...] = ()
    revoked_key_ids: tuple[str, ...] = ()
    previous_snapshot_digest: str = ""
    schema: str = REVOCATION_SNAPSHOT_SCHEMA

    def __post_init__(self):
        for name in ("epoch", "issued_at", "valid_until"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{name} must be an integer")
        if self.epoch < 0:
            raise ValueError("epoch must be >= 0")
        if self.valid_until <= self.issued_at:
            raise ValueError("valid_until must be after issued_at")
        for d in self.revoked_decision_digests:
            validate_digest(str(d))
        for k in self.revoked_key_ids:
            if not k:
                raise ValueError("revoked key ids must be non-empty")
        if self.previous_snapshot_digest:
            validate_digest(self.previous_snapshot_digest)

    @property
    def digest(self) -> str:
        return digest(self)

    def contains(self, decision_digest: str) -> bool:
        return str(decision_digest) in {
            str(d) for d in self.revoked_decision_digests}

    def revokes_key(self, key_id: str) -> bool:
        return str(key_id) in {str(k) for k in self.revoked_key_ids}

    def to_doc(self, *, signer: Ed25519Signer) -> dict:
        body = {"schema": self.schema, "epoch": self.epoch,
                "issued_at": self.issued_at,
                "valid_until": self.valid_until,
                "revoked_decision_digests":
                    list(self.revoked_decision_digests),
                "revoked_key_ids": list(self.revoked_key_ids),
                "previous_snapshot_digest": self.previous_snapshot_digest}
        env = signer.sign(body)
        return {"value": body, "digest": digest(body),
                "signer_key_id": env.key_id,
                "signature_b64": env.signature_b64}

    @classmethod
    def from_value(cls, value: dict) -> "RevocationSnapshotV2":
        return cls(
            epoch=int(value["epoch"]),
            issued_at=int(value["issued_at"]),
            valid_until=int(value["valid_until"]),
            revoked_decision_digests=tuple(
                str(d) for d in value.get("revoked_decision_digests", ())),
            revoked_key_ids=tuple(
                str(k) for k in value.get("revoked_key_ids", ())),
            previous_snapshot_digest=str(
                value.get("previous_snapshot_digest") or ""),
            schema=str(value["schema"]))


def _envelope_fields(doc) -> tuple[dict, str]:
    """Split a signed snapshot document into (value, signer_key_id),
    checking the envelope is self-consistent before cryptography."""
    if not isinstance(doc, dict) or "value" not in doc:
        raise RevocationRefused(
            "revocation snapshot: signed envelope required "
            "{value, digest, signer_key_id, signature_b64} — a bare "
            "digest list is not authenticated revocation evidence")
    value = doc["value"]
    if not isinstance(value, dict):
        raise RevocationRefused(
            "revocation snapshot: envelope value must be an object")
    if doc.get("digest") != digest(value):
        raise RevocationRefused(
            "revocation snapshot: envelope digest mismatch")
    if "signer_key_id" not in doc:
        raise RevocationRefused(
            "revocation snapshot is unsigned — unsigned revocation "
            "evidence cannot constrain production authority")
    return value, str(doc.get("signer_key_id", ""))


def verify_snapshot(doc, registry, *, now: datetime | None = None,
                    max_age_seconds: int | None = None,
                    max_clock_skew_seconds: int = DEFAULT_CLOCK_SKEW_SECONDS,
                    min_epoch: int = 0) -> RevocationSnapshotV2:
    """Verify a signed revocation snapshot end to end and return it.

    Order matters: structure and signature first (is this authentic?),
    then temporal checks (is it operative now?), then monotonicity
    (is it the newest evidence we should believe?)."""
    value, kid = _envelope_fields(doc)
    at = as_utc(now)
    ts = int(at.timestamp())

    if not registry.is_authorized("revocation", kid, now=at):
        raise RevocationRefused(
            f"revocation snapshot signer {kid!r} is not an authorized, "
            "currently valid revocation authority")
    if not registry.verifier(now=at).verify(
            value, SignedEnvelope(kid, str(doc.get("signature_b64", "")))):
        raise RevocationRefused("revocation snapshot signature invalid")

    try:
        strict_schema.validate("revocation_snapshot", value)
    except strict_schema.SchemaRefused as exc:
        raise RevocationRefused(str(exc)) from exc
    snap = RevocationSnapshotV2.from_value(value)

    if snap.issued_at > ts + int(max_clock_skew_seconds):
        raise RevocationRefused(
            f"revocation snapshot is future-dated ({snap.issued_at} > "
            f"{ts}+{max_clock_skew_seconds}s) — refused, not treated as "
            "fresh")
    if ts > snap.valid_until:
        raise RevocationRefused(
            "revocation snapshot has passed its valid_until")
    if max_age_seconds is not None and \
            ts - snap.issued_at > int(max_age_seconds):
        raise RevocationRefused(
            f"revocation snapshot is stale ({ts - snap.issued_at}s > "
            f"{max_age_seconds}s)")
    if snap.epoch < int(min_epoch):
        raise RevocationRefused(
            f"revocation epoch {snap.epoch} predates the operative epoch "
            f"{min_epoch} — replayed older evidence is refused")
    return snap


class RevocationStore:
    """Durable, atomic, epoch-monotonic store of signed snapshots.

    Files live under `<dir>/revocations/epoch-<N>.json`; the operative
    snapshot is the newest valid one, not whichever file a caller hands
    in. `publish` refuses an epoch regression so an older snapshot can
    never displace newer evidence."""

    def __init__(self, directory):
        self.root = Path(directory)
        self.snapshots_dir = self.root / "revocations"

    def _snapshot_path(self, epoch: int) -> Path:
        return self.snapshots_dir / f"epoch-{epoch:020d}.json"

    def epochs(self) -> list[int]:
        if not self.snapshots_dir.is_dir():
            return []
        out = []
        for p in self.snapshots_dir.glob("epoch-*.json"):
            try:
                out.append(int(p.stem.split("-", 1)[1]))
            except (ValueError, IndexError):
                continue
        return sorted(out)

    def publish(self, doc) -> Path:
        """Durably store a verified-or-unverified signed snapshot
        (verification happens on read; storage is transport). An epoch
        that would replace a higher published epoch is refused."""
        _, _kid = _envelope_fields(doc)
        value = doc["value"]
        try:
            epoch = int(value["epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RevocationRefused(
                f"snapshot carries no integer epoch: {exc}") from exc
        existing = self.epochs()
        if existing and epoch < existing[-1]:
            raise RevocationRefused(
                f"snapshot epoch {epoch} regresses below the newest "
                f"published epoch {existing[-1]} — replay refused")
        if epoch in existing:
            raise RevocationRefused(
                f"snapshot epoch {epoch} is already published — epochs "
                "are write-once")
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        target = self._snapshot_path(epoch)
        fd, tmp = tempfile.mkstemp(
            dir=str(self.snapshots_dir), prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(doc, sort_keys=True))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
            dir_fd = os.open(str(self.snapshots_dir), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return target

    def load(self, epoch: int):
        p = self._snapshot_path(epoch)
        if not p.is_file():
            return None
        return json.loads(p.read_text())

    def latest_valid(self, registry, *, now: datetime | None = None,
                     max_age_seconds: int | None = None,
                     max_clock_skew_seconds:
                     int = DEFAULT_CLOCK_SKEW_SECONDS,
                     require: bool = True) -> RevocationSnapshotV2 | None:
        """The newest valid authorized snapshot.

        Iterates published epochs newest-first and returns the first
        that fully verifies. Invalid snapshots are never silently
        selected: a newest-but-invalid snapshot is reported, and the
        search does not fall through to an older one — admission must
        use the newest authorized evidence, not an arbitrarily selected
        older one (a newer malformed file may be an attacker's attempt
        to hide fresh revocations)."""
        epochs = self.epochs()
        if not epochs:
            if require:
                raise RevocationRefused(
                    "no revocation snapshots published — missing "
                    "revocation evidence fails closed")
            return None
        newest = epochs[-1]
        doc = self.load(newest)
        return verify_snapshot(
            doc, registry, now=now, max_age_seconds=max_age_seconds,
            max_clock_skew_seconds=max_clock_skew_seconds,
            min_epoch=0)
