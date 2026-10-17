"""v16.4.2 immutable runtime snapshot (REPAIR-105, TOCTOU closure +
UPGRADE_PLAN §3.1).

Verification alone is not enough: an artifact can be verified and then
swapped before the serving backend opens it (time-of-check /
time-of-use). The launcher therefore never hands the backend a
mutable path. It stages a `MeasuredSnapshot`:

  * `stage_snapshot()` copies every authorized artifact exactly once,
    hashing the bytes as they are written (`copy_closure`), then
    compares the staged digest against the digest the signed runtime
    manifest authorized. Bytes that changed after verification cannot
    make the staged copy match; bytes that match are the bytes that
    were staged.
  * the staged tree is re-measured (`verify_snapshot`) immediately
    before load and the files are frozen read-only (`freeze_tree`).
  * `MeasuredSnapshot` can only be constructed by this module (a
    module-private sentinel), and the serving backend accepts nothing
    else — there is no public path that loads a mutable artifact.

A `MeasuredSnapshot` proves *content*, not *authorization* (v16.4.2):
possessing one does not imply that a promotion authority authorized
serving — that evidence is the `AdmissionGrantV1` issued by the
admission service (`minagi.security.admission_grants`) and verified
independently by the serving supervisor (`minagi.runtime.supervisor`).

A failed stage/verify removes the snapshot directory and raises
`SnapshotError`; nothing is served.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping

from egai.common.canonical import validate_digest

from .artifact_closure import (ArtifactClosureError, TreeClosure,
                               copy_closure, close_tree, freeze_tree)

_SENTINEL = object()

SNAPSHOT_SCHEMA = "mini-agi-v16.4.2-measured-snapshot-v1"


class SnapshotError(PermissionError):
    """The measured snapshot could not be staged or verified."""


class MeasuredSnapshot:
    """A staged, verified, frozen copy of the exact authorized
    artifacts. Constructible only by `stage_snapshot`.

    Measurement is not authorization: this object proves content, not
    that serving was authorized — the `AdmissionGrantV1` binds it to a
    verified promotion chain (UPGRADE_PLAN §3.1)."""

    __slots__ = ("_root", "_digests", "_manifest_digest", "_closures")

    def __init__(self, *, root: Path, digests: Mapping[str, str],
                 manifest_digest: str, closures: Mapping[str, TreeClosure],
                 _sentinel=None):
        if _sentinel is not _SENTINEL:
            raise SnapshotError(
                "MeasuredSnapshot can only be produced by stage_snapshot — "
                "a raw path is not a measured snapshot")
        validate_digest(manifest_digest)
        for name, d in digests.items():
            validate_digest(d)
        self._root = Path(root)
        self._digests = dict(digests)
        self._manifest_digest = str(manifest_digest)
        self._closures = dict(closures)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest_digest(self) -> str:
        return self._manifest_digest

    @property
    def artifact_digests(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._digests.items()))

    @property
    def schema(self) -> str:
        return SNAPSHOT_SCHEMA

    def digest_of(self, name: str) -> str:
        return self._digests[name]

    def path(self, name: str) -> Path:
        if name not in self._digests:
            raise SnapshotError(f"snapshot does not contain artifact {name!r}")
        return self._root / name

    def closure(self, name: str) -> TreeClosure:
        return self._closures[name]

    def __reduce__(self):
        """MeasuredSnapshot does not pickle (v16.4.5 SEC-401): the
        worker IPC is typed JSON, and a snapshot crosses it as a
        descriptor (``snapshot_descriptor``) that the worker re-
        measures through ``snapshot_from_descriptor``. Pickle would
        restore measurement evidence without measuring."""
        raise SnapshotError(
            "MeasuredSnapshot cannot be pickled — transport it as a "
            "descriptor and re-measure in the receiving process")

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"MeasuredSnapshot(root={str(self._root)!r}, "
                f"artifacts={sorted(self._digests)}, "
                f"manifest={self._manifest_digest[:20]}...)")


def stage_snapshot(dest, artifacts: Mapping[str, object], *,
                   expected_digests: Mapping[str, str],
                   resolve_symlinks: Mapping[str, bool] | None = None,
                   manifest_digest: str) -> MeasuredSnapshot:
    """Copy `artifacts` (name -> path) into `dest` and verify the staged
    bytes against `expected_digests` (name -> authorized digest)."""
    dest = Path(dest)
    if dest.exists():
        raise SnapshotError(f"snapshot destination already exists: {dest}")
    missing = set(expected_digests) - set(artifacts)
    if missing:
        raise SnapshotError(
            f"authorized artifacts not supplied for staging: {sorted(missing)}")
    extra = set(artifacts) - set(expected_digests)
    if extra:
        raise SnapshotError(
            "artifacts not authorized by the manifest cannot be staged: "
            f"{sorted(extra)}")
    policy = dict(resolve_symlinks or {})
    closures: dict[str, TreeClosure] = {}
    try:
        dest.mkdir(parents=True)
        for name, src in artifacts.items():
            staged = dest / name
            closure = copy_closure(
                src, staged, resolve_symlinks=bool(policy.get(name, False)))
            authorized = str(expected_digests[name])
            if closure.digest != authorized:
                raise SnapshotError(
                    f"staged {name} digest {closure.digest} != authorized "
                    f"{authorized} — artifact changed between verification "
                    "and staging")
            closures[name] = closure
        freeze_tree(dest)
        snapshot = MeasuredSnapshot(
            root=dest, digests=expected_digests, manifest_digest=manifest_digest,
            closures=closures, _sentinel=_SENTINEL)
        verify_snapshot(snapshot)
        return snapshot
    except (ArtifactClosureError, SnapshotError, OSError, KeyError,
            ValueError) as exc:
        _discard(dest)
        if isinstance(exc, SnapshotError):
            raise
        raise SnapshotError(f"snapshot staging failed: {exc}") from exc


def verify_snapshot(snapshot: MeasuredSnapshot) -> None:
    """Re-measure every staged artifact from disk and compare against the
    authorized digests. Called immediately before load."""
    problems: list[str] = []
    for name, authorized in snapshot.artifact_digests:
        path = snapshot.root / name
        try:
            actual = close_tree(path).digest
        except ArtifactClosureError as exc:
            problems.append(f"{name}: {exc}")
            continue
        if actual != authorized:
            problems.append(
                f"{name}: staged digest {actual} != authorized {authorized}")
    if problems:
        raise SnapshotError(
            "staged snapshot no longer matches the authorized artifacts: "
            + "; ".join(problems))


def snapshot_descriptor(snapshot: MeasuredSnapshot) -> dict:
    """The JSON-safe description of a measured snapshot for worker
    IPC: root + authorized digests + manifest digest. It is a
    reference to already-staged immutable bytes, not a serialised
    object — the receiver must re-measure before use."""
    if not isinstance(snapshot, MeasuredSnapshot):
        raise SnapshotError(
            "snapshot_descriptor requires a MeasuredSnapshot")
    return {"root": str(snapshot.root),
            "digests": dict(snapshot.artifact_digests),
            "manifest_digest": snapshot.manifest_digest}


def snapshot_from_descriptor(desc) -> MeasuredSnapshot:
    """Reconstruct a MeasuredSnapshot inside the worker from its
    descriptor: re-measure every artifact from disk and refuse when
    bytes no longer match the authorized digests. The descriptor is
    trusted only as a locator — proof still comes from measurement."""
    if not isinstance(desc, dict):
        raise SnapshotError("snapshot descriptor must be an object")
    root = Path(str(desc.get("root") or ""))
    raw_digests = desc.get("digests")
    if not isinstance(raw_digests, dict) or not raw_digests:
        raise SnapshotError(
            "snapshot descriptor must name a non-empty digest map")
    if len(raw_digests) > 64:
        raise SnapshotError("snapshot descriptor names too many artifacts")
    digests: dict[str, str] = {}
    for name, d in raw_digests.items():
        name = str(name)
        if not name or name != Path(name).name:
            raise SnapshotError(
                f"artifact name {name!r} is not a plain child of the "
                "snapshot root")
        try:
            validate_digest(str(d))
        except ValueError as exc:
            raise SnapshotError(
                f"artifact digest for {name!r} malformed: {exc}") from exc
        digests[name] = str(d)
    try:
        manifest_digest = str(desc["manifest_digest"])
        validate_digest(manifest_digest)
    except (KeyError, ValueError) as exc:
        raise SnapshotError(
            f"snapshot descriptor manifest digest invalid: {exc}") from exc
    if not root.is_dir():
        raise SnapshotError(
            f"snapshot root {root} is absent — the staged artifacts do "
            "not exist in this process")
    closures: dict[str, TreeClosure] = {}
    problems: list[str] = []
    for name, authorized in digests.items():
        try:
            closure = close_tree(root / name)
        except ArtifactClosureError as exc:
            problems.append(f"{name}: {exc}")
            continue
        if closure.digest != authorized:
            problems.append(
                f"{name}: measured digest {closure.digest} != "
                f"authorized {authorized}")
            continue
        closures[name] = closure
    if problems:
        raise SnapshotError(
            "snapshot descriptor does not match on-disk bytes: "
            + "; ".join(problems))
    return MeasuredSnapshot(
        root=root, digests=digests, manifest_digest=manifest_digest,
        closures=closures, _sentinel=_SENTINEL)


def _discard(path: Path) -> None:
    if not path.exists():
        return
    try:
        for p in sorted(path.rglob("*"), reverse=True):
            try:
                p.chmod(0o700)
            except OSError:
                pass
        path.chmod(0o700)
        shutil.rmtree(path)
    except OSError:
        pass
