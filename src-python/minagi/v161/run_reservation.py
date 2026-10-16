"""Atomic run reservation and publication (V4-26).

The pre-v4.0.2 fit-runner lifecycle had three evidence-integrity
defects:

* The staging directory name was derived from the run id alone
  (``.staging-{run_id}``), so two concurrent attempts at the same run
  wrote into one directory and destroyed each other's evidence — and
  an existing staging dir was deleted outright (``shutil.rmtree``).
* Failure cleanup removed an existing *final* directory
  (``shutil.rmtree(final_dir)`` then ``os.replace``), so a failed
  attempt could delete another attempt's committed run.
* Publication used ``os.replace(staging, final_dir)``, which replaces
  an existing empty committed directory, and the existence check was
  check-then-create — a race, not a reservation.

This module replaces that lifecycle:

* **Atomic, exclusive claim.** ``.claims/<run_id>/`` is created with
  ``os.mkdir`` — exactly one attempt wins; the loser keeps its own
  attempt directory with a CLAIM_LOST record and never touches the
  canonical name.
* **Unique, ownership-bound staging.** Every attempt works in
  ``.staging/<run_id>.<attempt_id>`` — a directory name embedding both
  the run id and a fresh attempt id, created exclusively. No attempt
  ever removes or reuses another attempt's staging.
* **Durable commit records.** Publication writes RUN_COMMIT.json into
  staging first, then ``os.rename``s the attempt directory onto the
  canonical name — only while holding the claim and only when the
  canonical name is still absent. A committed run is never replaced:
  publish refuses if the destination appeared in the meantime.
* **Failure records that cannot delete other work.** ``fail`` records
  RUN_COMMIT{status: FAILED} inside the attempt's own directory and
  renames it to ``<run_id>.FAILED-<attempt_id>`` next to the canonical
  run, then releases the claim so a later attempt may still claim the
  name. A failed attempt never deletes or replaces anything it does
  not own.

Filesystem notes: the claim and the staging create both use atomic
``os.mkdir`` (EEXIST = contested). ``os.rename`` of a directory onto a
non-empty directory already fails on POSIX; onto an *empty* directory
it succeeds — which is why publish re-checks ``final.exists()`` and
requires the claim to still name this attempt before renaming. The
claim file makes the reservation auditable after the fact.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from egai.common.canonical import digest

CLAIMS_DIRNAME = ".claims"
STAGING_DIRNAME = ".staging"
CLAIM_FILE = "CLAIM.json"
RUN_COMMIT = "RUN_COMMIT.json"

SCHEMA_CLAIM = "mini-agi-v4.0-run-claim-v1"


class ReservationError(Exception):
    """Base class for reservation failures."""


class DestinationCommitted(ReservationError):
    """The canonical run directory already exists — committed runs are
    immutable, so the attempt is refused before any work begins."""


class ClaimHeld(ReservationError):
    """Another attempt currently holds the claim for this run id."""


def new_attempt_id() -> str:
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class RunReservation:
    """One attempt's exclusive claim on a run destination.

    Obtain via :func:`reserve`. The attempt writes into
    ``self.staging`` and finishes with exactly one of:

    * :meth:`publish` — the canonical ``<run_root>/<run_id>`` is
      created atomically from this attempt's staging (caller must have
      already written RUN_COMMIT.json into staging).
    * :meth:`fail` — the staging is renamed to a durable
      ``<run_id>.FAILED-<attempt_id>`` sibling (after the caller writes
      RUN_COMMIT.json) and the claim is released.
    * :meth:`abandon` — evidence is preserved in place under
      ``.staging/`` and the claim released (for pre-commit aborts that
      still want the partial directory kept for forensics).

    A reservation whose claim was lost (``owns_claim == False``) must
    not publish; its staging already carries a CLAIM_LOST record.
    """

    def __init__(self, run_root: Path, run_id: str, attempt_id: str,
                 staging: Path, claim_dir: Path, owns_claim: bool):
        self.run_root = Path(run_root)
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.staging = Path(staging)
        self.claim_dir = Path(claim_dir)
        self.final_dir = self.run_root / run_id
        self.owns_claim = owns_claim
        self._closed = False

    # -- internals ----------------------------------------------------

    def _claim_names_us(self) -> bool:
        try:
            doc = json.loads((self.claim_dir / CLAIM_FILE).read_text())
        except Exception:
            return False
        return doc.get("attempt_id") == self.attempt_id

    def _release_claim(self) -> None:
        """Drop our claim so the canonical name stays available.

        Only removes the claim when the claim record names THIS
        attempt — a racing claim that replaced ours is untouched."""
        if not self.owns_claim:
            return
        if self._claim_names_us():
            try:
                (self.claim_dir / CLAIM_FILE).unlink()
                self.claim_dir.rmdir()
            except OSError:
                pass
        self.owns_claim = False

    def _preserved_failure_dir(self) -> Path:
        """A durable, never-replacing sibling name for this attempt."""
        base = self.run_root / f"{self.run_id}.FAILED-{self.attempt_id}"
        path, i = base, 0
        while path.exists():
            i += 1
            path = self.run_root / f"{self.run_id}.FAILED-" \
                                   f"{self.attempt_id}-{i}"
        return path

    # -- terminal transitions -----------------------------------------

    def publish(self) -> Path:
        """Atomically publish staging as the canonical run directory.

        Refuses (raises) rather than ever replacing a committed run.
        Returns the final directory path."""
        if self._closed:
            raise ReservationError("reservation already closed")
        if not self.owns_claim or not self._claim_names_us():
            raise ReservationError(
                f"cannot publish {self.run_id}: this attempt does not "
                "hold the claim")
        if self.final_dir.exists():
            raise ReservationError(
                f"cannot publish {self.run_id}: destination already "
                f"exists — committed runs are never replaced "
                f"({self.final_dir})")
        if not (self.staging / RUN_COMMIT).is_file():
            raise ReservationError(
                "cannot publish without RUN_COMMIT.json — a run must "
                "commit its evidence inventory before publication")
        os.rename(self.staging, self.final_dir)
        self._closed = True
        # keep the claim dir as the durable record of who claimed it
        return self.final_dir

    def fail(self) -> Path:
        """Preserve this attempt's evidence as a FAILED sibling record.

        The staging dir (which the caller has already written
        RUN_COMMIT.json{status: FAILED} into) is renamed next to the
        canonical name. The claim is released: a failed attempt must
        not burn the canonical name nor touch an existing run."""
        if self._closed:
            raise ReservationError("reservation already closed")
        target = self._preserved_failure_dir()
        os.rename(self.staging, target)
        self._release_claim()
        self._closed = True
        return target

    def abandon(self) -> Path:
        """Release the claim, keep the staging dir in place under
        ``.staging/`` (uncommitted partial evidence)."""
        self._release_claim()
        self._closed = True
        return self.staging


def reserve(run_root: str | Path, run_id: str,
            attempt_id: str | None = None,
            config_digest: str = "") -> RunReservation:
    """Reserve a run destination and return the attempt's reservation.

    Raises DestinationCommitted when a committed run already owns the
    canonical name. When another attempt holds the claim, returns a
    reservation with ``owns_claim=False`` whose staging records
    CLAIM_LOST — the attempt may keep forensic evidence but cannot
    publish."""
    run_root = Path(run_root)
    attempt_id = attempt_id or new_attempt_id()
    claims = run_root / CLAIMS_DIRNAME
    staging_root = run_root / STAGING_DIRNAME
    final_dir = run_root / run_id

    claims.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    if final_dir.exists():
        raise DestinationCommitted(
            f"run destination already exists, refusing to overwrite: "
            f"{final_dir} — a committed run is never modified")

    # Unique ownership-bound staging. A same-name staging directory is
    # never deleted or reused — a collision (same attempt id, same run)
    # is itself an integrity event and aborts the attempt.
    staging = staging_root / f"{run_id}.{attempt_id}"
    staging.mkdir()  # FileExistsError -> refuse; never rmtree
    (staging / "ATTEMPT.json").write_text(json.dumps({
        "schema": SCHEMA_CLAIM, "run_id": run_id,
        "attempt_id": attempt_id, "config_digest": config_digest,
        "created_utc": _utc_now()}, indent=2, sort_keys=True) + "\n")

    claim_dir = claims / run_id
    owns = False
    try:
        claim_dir.mkdir()  # atomic claim — EEXIST means contested
        owns = True
    except FileExistsError:
        owns = False

    res = RunReservation(run_root, run_id, attempt_id, staging,
                         claim_dir, owns)
    if owns:
        (claim_dir / CLAIM_FILE).write_text(json.dumps({
            "schema": SCHEMA_CLAIM, "run_id": run_id,
            "attempt_id": attempt_id, "config_digest": config_digest,
            "staging": staging.name,
            "created_utc": _utc_now()}, indent=2, sort_keys=True) + "\n")
    else:
        holder = {}
        try:
            holder = json.loads((claim_dir / CLAIM_FILE).read_text())
        except Exception:
            pass
        (staging / "CLAIM_LOST.json").write_text(json.dumps({
            "schema": SCHEMA_CLAIM, "run_id": run_id,
            "attempt_id": attempt_id, "status": "CLAIM_LOST",
            "held_by": holder.get("attempt_id"),
            "created_utc": _utc_now()}, indent=2, sort_keys=True) + "\n")
    return res


def reservation_digest(res: RunReservation) -> str:
    """Canonical digest binding run/attempt identity for receipts."""
    return digest({"schema": SCHEMA_CLAIM, "run_id": res.run_id,
                   "attempt_id": res.attempt_id})
