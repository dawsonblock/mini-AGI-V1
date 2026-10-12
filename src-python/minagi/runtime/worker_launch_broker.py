"""v16.4.8 restricted worker-launch broker (WP8) — plan Step 11.

A small, narrowly authorized component standing between the trusted
supervisor and the process-creation primitives. It accepts ONLY a
:class:`LaunchWorkerRequest` — a worker id, approval digests, and a
nonce — and turns it into ONE managed systemd unit running the
pinned NsJail binary with a deterministically generated
configuration.

The broker never takes from a caller:

  * an executable path or an argv vector — the jail's exec_bin is
    built here from the pinned interpreter and the fixed worker
    module (``worker``) or probe module (``probe``);
  * an environment list — the jail's envar set is allowlisted and
    secret-screened here;
  * a mount list — binds are computed from the approved runtime
    closure, digest-resolved artifacts, and the worker scratch;
  * a property list — unit properties come from the frozen profile.

The broker holds no signing material and cannot admit, promote, or
sign anything. Its privilege is limited to creating and destroying
worker units and workspaces under ``worker_state_root``.

Containment is kernel-owned: systemd places the unit in its own
cgroup v2 subtree; ``systemd-run --pipe --wait --collect`` gives a
bidirectional channel to the unit's stdin/stdout and propagates its
exit; teardown is ``systemctl stop`` verified by
``cgroup.populated == 0`` — never /proc scanning claimed as proof.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .isolation_profile import (IsolationProfile, MountSpec,
                                ProfileRefused, resolve_mounts)
from .nsjail_config import config_digest, render_config


class BrokerError(RuntimeError):
    """A launch or lifecycle operation failed in execution."""


class BrokerRefused(PermissionError):
    """The request itself is refused — malformed, replayed, or
    outside the approved boundary. Fail closed."""


#: envar names the jail may carry — the same screening rule as the
#: supervisor-side environment: nothing credential-shaped crosses.
_ALLOWED_ENV = frozenset({
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "HOME",
    "PYTHONPATH", "TMPDIR", "TMP", "TEMP",
    "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM", "MKL_NUM_THREADS",
    "MINAGI_WORKER_UNIT", "MINAGI_WORKER_ROLE",
    "MINAGI_PROBE_SCRATCH", "MINAGI_PROBE_ARGS"})

_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

# Credential-shaped names that are nevertheless standard, benign
# performance/verbosity knobs — no secret material crosses.
_BENIGN_MARKER_NAMES = frozenset({
    "TOKENIZERS_PARALLELISM", "TORCH_LOGS", "TRANSFORMERS_VERBOSITY",
    "OPENBLAS_NUM_THREADS"})

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
# Opaque broker-generated identifiers ONLY (SEC-903): "w-" + 32 hex.
# Caller-shaped names — ".", "..", path separators, encoded traversal —
# can never name a unit or a workspace directory.
_ID_RE = re.compile(r"^w-[0-9a-f]{32}$")


def new_worker_id() -> str:
    """The ONLY source of worker ids — generated here, never taken
    from a request, model, or API caller."""
    return f"w-{secrets.token_hex(16)}"


class ContainmentState(str, Enum):
    """The ONE shared containment decision (SEC-902). Every consumer —
    launch acceptance, status, terminate, executor reports — reads
    this, so no path can certify emptiness the others wouldn't."""
    RUNNING = "running"                    # kernel: populated 1
    EMPTY = "empty"                        # kernel: populated 0
    REMOVED_VERIFIED = "removed_verified"  # identity-bound teardown
    UNKNOWN = "unknown"                    # evidence absent/invalid


@dataclass(frozen=True)
class LaunchWorkerRequest:
    """The whole authority a launch request carries — digests of
    already-approved material plus a replay nonce. No executable, no
    argv, no env, no mounts, no shell."""
    worker_id: str
    signed_backend_manifest_digest: str
    approved_artifact_digests: tuple = ()
    approved_isolation_profile_digest: str = ""
    execution_budget_digest: str = ""
    request_nonce: str = ""
    role: str = "worker"                # "worker" | "probe"
    authorization: dict = field(default_factory=dict)  # SEC-905


@dataclass(frozen=True)
class AuthorizedLaunchRecord:
    """v16.4.10 (SEC-905) — the typed launch authorization the broker
    INDEPENDENTLY verifies. It binds one activation to the exact
    backend manifest, runtime manifest, artifact root, isolation
    profile and execution budget that were approved upstream, plus
    freshness (expiry, policy/revocation epochs) and a single-use
    nonce. The signed envelope is an Ed25519 `runtime`-role record —
    verification needs only trusted public keys and the read-only
    approved-record resolver; the broker holds no signing keys."""
    activation_id: str
    backend_manifest_digest: str
    runtime_manifest_digest: str
    artifact_root_digest: str
    isolation_profile_digest: str
    execution_budget_digest: str
    policy_epoch: int
    revocation_epoch: int
    issued_at: int
    expires_at: int
    nonce: str
    signer_key_id: str = ""

    def to_body(self) -> dict:
        return {
            "schema": "minagi-authorized-launch-v1",
            "activation_id": self.activation_id,
            "backend_manifest_digest": self.backend_manifest_digest,
            "runtime_manifest_digest": self.runtime_manifest_digest,
            "artifact_root_digest": self.artifact_root_digest,
            "isolation_profile_digest": self.isolation_profile_digest,
            "execution_budget_digest": self.execution_budget_digest,
            "policy_epoch": self.policy_epoch,
            "revocation_epoch": self.revocation_epoch,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "nonce": self.nonce}


def issue_launch_authorization(signer, *, record: AuthorizedLaunchRecord
                               ) -> dict:
    """Sign an AuthorizedLaunchRecord — the supervisor-side helper.
    `signer` must be a `runtime`-role Ed25519Signer registered in the
    trust root the broker will verify against."""
    from egai.common.canonical import digest
    body = record.to_body()
    env = signer.sign(body)
    return {"value": body, "digest": digest(body),
            "signer_key_id": env.key_id,
            "signature_b64": env.signature_b64}


class MappingRecordResolver:
    """Read-only approved-record resolver (SEC-905): an explicit
    digest->record map built by the supervisor from records it has
    ALREADY verified (backend manifest, runtime manifest, artifact
    root, isolation profile, execution budget). The broker asks only
    "does an approved record exist for this digest" — invented
    digests resolve to nothing and are refused."""

    def __init__(self, approved: dict, *, revocation_epoch_floor: int = 0):
        self._approved = dict(approved)
        self._revocation_floor = int(revocation_epoch_floor)

    def __call__(self, digest_value: str) -> bool:
        return str(digest_value) in self._approved

    def revocation_epoch_floor(self) -> int:
        return self._revocation_floor


def verify_launch_authorization(doc, registry, resolver, *,
                                req: LaunchWorkerRequest,
                                now=None) -> AuthorizedLaunchRecord:
    """Independently verify a worker launch's authority (SEC-905).

    Checks, each a refusal — never a warning:

    * signed-envelope shape and value digest,
    * signer is an authorized, currently-valid `runtime` authority,
    * the envelope signature verifies,
    * the authorization has not expired (short lifetime),
    * the record binds THIS request's backend manifest, profile,
      budget and artifact digests — a valid record for a different
      activation cannot substitute,
    * every referenced digest resolves to an APPROVED record through
      the read-only resolver (invented digests have no record),
    * policy/revocation epochs satisfy the resolver's current floor.
    """
    from egai.common.canonical import digest
    from egai.common.crypto import SignedEnvelope
    if not isinstance(doc, dict) or not isinstance(
            doc.get("value"), dict):
        raise BrokerRefused(
            "launch authorization: signed envelope required")
    value = doc["value"]
    if doc.get("digest") != digest(value):
        raise BrokerRefused(
            "launch authorization: envelope digest mismatch")
    kid = str(doc.get("signer_key_id", ""))
    if not kid:
        raise BrokerRefused("launch authorization is unsigned")
    if not registry.is_authorized("runtime", kid, now=now):
        raise BrokerRefused(
            f"launch-authority signer {kid!r} is not an authorized, "
            "currently-valid runtime authority")
    if not registry.verifier(now=now).verify(
            value, SignedEnvelope(
                kid, str(doc.get("signature_b64", "")))):
        raise BrokerRefused("launch authorization: bad signature")
    rec = AuthorizedLaunchRecord(
        activation_id=str(value.get("activation_id", "")),
        backend_manifest_digest=str(
            value.get("backend_manifest_digest", "")),
        runtime_manifest_digest=str(
            value.get("runtime_manifest_digest", "")),
        artifact_root_digest=str(
            value.get("artifact_root_digest", "")),
        isolation_profile_digest=str(
            value.get("isolation_profile_digest", "")),
        execution_budget_digest=str(
            value.get("execution_budget_digest", "")),
        policy_epoch=int(value.get("policy_epoch", 0)),
        revocation_epoch=int(value.get("revocation_epoch", 0)),
        issued_at=int(value.get("issued_at", 0)),
        expires_at=int(value.get("expires_at", 0)),
        nonce=str(value.get("nonce", "")), signer_key_id=kid)
    if now is not None and rec.expires_at <= int(now.timestamp()):
        raise BrokerRefused("launch authorization expired")
    # the authorization must bind THIS launch's authority fields —
    # a record for another activation/profile/budget is not reusable
    bound = (
        ("backend_manifest_digest",
         req.signed_backend_manifest_digest),
        ("isolation_profile_digest",
         req.approved_isolation_profile_digest),
        ("execution_budget_digest", req.execution_budget_digest))
    for field_name, request_value in bound:
        if getattr(rec, field_name) != str(request_value):
            raise BrokerRefused(
                f"launch authorization {field_name} does not bind "
                "this request")
    # every referenced digest must resolve to an approved record —
    # invented digests have no record to resolve to
    for name, d in (("backend_manifest_digest",
                     rec.backend_manifest_digest),
                    ("runtime_manifest_digest",
                     rec.runtime_manifest_digest),
                    ("artifact_root_digest", rec.artifact_root_digest),
                    ("isolation_profile_digest",
                     rec.isolation_profile_digest),
                    ("execution_budget_digest",
                     rec.execution_budget_digest)):
        if not _DIGEST_RE.match(d):
            raise BrokerRefused(
                f"launch authorization {name} is not sha256:<hex>")
        if not resolver(d):
            raise BrokerRefused(
                f"launch authorization {name} resolves to no "
                "approved record")
    floor = resolver.revocation_epoch_floor() \
        if hasattr(resolver, "revocation_epoch_floor") else 0
    if rec.revocation_epoch < int(floor):
        raise BrokerRefused(
            "launch authorization predates the current revocation "
            "epoch — the promotion it relies on may be revoked")
    return rec


@dataclass
class LaunchRecord:
    """The broker's identity for one launched containment unit."""
    worker_id: str
    unit_name: str
    proc: object                       # systemd-run client Popen
    workspace: Path
    scratch: Path
    overflow_dir: Path
    overflow_fd: int
    config_path: Path
    cgroup_path: Path | None
    invocation_id: str = ""            # identity-bound teardown token
    unit_token: str = ""
    role: str = "worker"
    digests: dict = field(default_factory=dict)
    started_at: float = 0.0
    user_manager: bool = False
    stderr_log: object = None           # unit diagnostic log path
    stderr_fh: object = None            # open handle, broker-owned


class SystemdClient:
    """Exact, argv-vector systemd operations — never a shell, never
    string interpolation into a command line."""

    def __init__(self, *, user_manager: bool = False,
                 systemd_run: str | None = None,
                 systemctl: str | None = None):
        self.user_manager = bool(user_manager)
        self.systemd_run = (systemd_run
                            or shutil.which("systemd-run"))
        self.systemctl = systemctl or shutil.which("systemctl")

    @property
    def available(self) -> bool:
        return bool(self.systemd_run and self.systemctl)

    def _ctl(self, *args, timeout: float = 20.0):
        argv = [self.systemctl, *(["--user"] if self.user_manager
                                  else []), *args]
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout)

    def run_service(self, unit_name: str, properties, argv,
                    *, cwd: str | None = None, stderr=None):
        """Spawn the transient unit with a piped protocol channel:
        ``--pipe`` wires the unit's stdin/stdout/stderr to this
        process's pipes; ``--wait`` propagates exit status;
        ``--collect`` GCs the unit record when it ends."""
        cmd = [self.systemd_run,
               *(["--user"] if self.user_manager else []),
               "--pipe", "--wait", "--collect",
               "--unit", unit_name]
        for prop in properties:
            cmd += ["-p", str(prop)]
        cmd += ["--", *[str(a) for a in argv]]
        # stderr goes to a caller-supplied sink (a bounded log file)
        # — an undrained PIPE can fill and stall the unit lifecycle.
        return subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr if stderr is not None else subprocess.PIPE,
            cwd=cwd, start_new_session=True)

    def show(self, unit_name: str, prop: str) -> str:
        try:
            out = self._ctl("show", unit_name,
                            f"--property={prop}", "--value")
        except (OSError, subprocess.SubprocessError):
            return ""
        return (out.stdout or "").strip()

    def is_active(self, unit_name: str) -> bool:
        try:
            out = self._ctl("is-active", unit_name)
        except (OSError, subprocess.SubprocessError):
            return False
        return (out.stdout or "").strip() == "active"

    def stop(self, unit_name: str):
        return self._ctl("stop", unit_name)

    def kill(self, unit_name: str, sig: str = "SIGKILL"):
        return self._ctl("kill", unit_name, f"--signal={sig}",
                         "--kill-whom=all")

    def cgroup_path(self, unit_name: str) -> Path | None:
        """Resolve the unit's cgroup via its ControlGroup property —
        works for system and user managers without guessing slice
        layouts."""
        rel = self.show(unit_name, "ControlGroup")
        if not rel:
            return None
        path = Path("/sys/fs/cgroup") / rel.lstrip("/")
        return path if path.is_dir() else None


def cgroup_populated(cgroup_path) -> bool | None:
    """The kernel's own occupancy flag (SEC-901, strict):
    True = populated 1, False = populated 0, None = unknown.

    cgroup v2 defines ``populated`` as a recursive indicator with
    exactly two valid values. ANY malformed evidence — missing file,
    malformed line, duplicate field, unexpected value — is Unknown,
    never empty. ``cgroup.procs`` is NOT a fallback: a removed cgroup
    is proven by identity-bound unit teardown, not by an absent or
    stale procs listing."""
    try:
        lines = (Path(cgroup_path) / "cgroup.events").read_text(
            ).splitlines()
    except (OSError, UnicodeError):
        return None
    fields = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            return None
        key, value = parts
        if key in fields:
            return None
        fields[key] = value
    value = fields.get("populated")
    if value == "1":
        return True
    if value == "0":
        return False
    return None


def evaluate_containment(*, populated, cgroup_gone: bool,
                         unit_active: bool, unit_same: bool) -> \
        ContainmentState:
    """The one decision rule (SEC-902):

    populated 1                          -> RUNNING  (occupied — always
                                           wins over service state)
    populated 0                          -> EMPTY
    cgroup dir gone AND unit inactive
      AND same unit invocation           -> REMOVED_VERIFIED
    anything else                        -> UNKNOWN

    A systemd "inactive" or an exited client NEVER overrides positive
    occupancy — they are service evidence, not kernel evidence.
    ``unit_same`` is the InvocationID match binding the teardown to
    THIS worker's unit, so a reused unit name cannot satisfy an old
    worker's teardown check."""
    if populated is True:
        return ContainmentState.RUNNING
    if populated is False:
        return ContainmentState.EMPTY
    if cgroup_gone and not unit_active and unit_same:
        return ContainmentState.REMOVED_VERIFIED
    return ContainmentState.UNKNOWN


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def runtime_closure_paths() -> tuple:
    """The read-only closure the jailed interpreter needs: the
    interpreter itself, its shared library dirs, stdlib and every
    live sys.path entry — the same computation the launcher's ro
    binds used, now owned by the broker."""
    import sys
    import sysconfig
    binds: list = []

    def add(p):
        if not p:
            return
        p = str(p)
        if not p.startswith("/"):
            try:
                p = str(Path(p).resolve())
            except OSError:
                return
        try:
            if not Path(p).exists():
                return
        except OSError:
            return
        if p not in binds:
            binds.append(p)

    add(sys.executable)
    for d in ("/usr/lib", "/lib", "/lib64", "/usr/local/lib",
              "/usr/lib64", sysconfig.get_paths().get("stdlib", ""),
              sysconfig.get_paths().get("platstdlib", "")):
        add(d)
    for d in sys.path:
        add(d)
    return tuple(binds)


class WorkerLaunchBroker:
    """The privileged launch surface — narrow, digest-bound, and
    replay-guarded.

    worker_state_root   root for worker workspaces (e.g.
                        /var/lib/minagi/workers)
    approved_profiles   {profile_digest: IsolationProfile} — only
                        registered profiles launch (SEC-703)
    artifact_resolver   callable(digest) -> path, for approved
                        read-only model artifacts
    nsjail_path         the pinned NsJail binary
    nsjail_digest       REQUIRED: sha256 the binary must measure —
                        the broker never runs an unmeasured binary
    worker_uid/gid      the separate worker OS identity the jail
                        maps to (outside_id in the user namespace)
    worker_python       the interpreter exec_bin inside the jail
    python_path_entries dirs bound ro + exported as PYTHONPATH —
                        derived from the supervisor's own runtime
    denied_paths        authority material that may never be mounted
    user_manager        use ``systemd-run --user`` (rootless) instead
                        of the system manager
    systemd             injectable SystemdClient (tests) — when
                        provided, no binaries are probed
    nonce_dir           durable replay-marker directory
    authority_registry  AuthorityRegistry holding trusted public
                        keys — the broker verifies `runtime`-role
                        launch authorizations against it (SEC-905)
    record_resolver     read-only callable(digest) -> bool plus
                        revocation_epoch_floor() — digests must
                        resolve to APPROVED records, not just parse
    require_authorization
                        when True, a worker launch without a fully
                        verified AuthorizedLaunchRecord is refused.
                        Production wiring sets both authority inputs
                        and this flag; tests/probes may run without.
    """

    #: fixed in-jail argvs — the only executables this broker runs
    WORKER_ARGV_MODULE = ("-m", "minagi.runtime.worker_backend")
    PROBE_ARGV_MODULE = ("-m", "minagi.runtime.isolation_preflight",
                         "--probe")

    def __init__(self, *, worker_state_root,
                 approved_profiles: dict | None = None,
                 artifact_resolver=None,
                 nsjail_path: str = "/usr/local/bin/nsjail",
                 nsjail_digest: str | None = None,
                 worker_uid: int = 65534, worker_gid: int = 65534,
                 worker_python: str | None = None,
                 python_path_entries: tuple = (),
                 denied_paths: tuple = (),
                 user_manager: bool = False,
                 systemd: SystemdClient | None = None,
                 unit_prefix: str = "miniagi-worker",
                 nonce_dir=None,
                 authority_registry=None,
                 record_resolver=None,
                 require_authorization: bool | None = None):
        import sys
        self.worker_state_root = Path(worker_state_root)
        self.approved_profiles = dict(approved_profiles or {})
        self.artifact_resolver = artifact_resolver
        self.nsjail_path = str(nsjail_path)
        self.worker_uid = int(worker_uid)
        self.worker_gid = int(worker_gid)
        self.worker_python = str(worker_python or sys.executable)
        self.python_path_entries = tuple(
            str(Path(p).resolve()) for p in python_path_entries if p)
        self.denied_paths = tuple(str(p) for p in denied_paths)
        self.user_manager = bool(user_manager)
        self.systemd = systemd or SystemdClient(
            user_manager=user_manager)
        self.unit_prefix = str(unit_prefix)
        self.nonce_dir = (Path(nonce_dir) if nonce_dir
                          else self.worker_state_root / ".nonces")
        self._records: dict[str, LaunchRecord] = {}
        self._workspaces: dict[str, tuple] = {}
        # SEC-905 launch authority: a configured registry+resolver
        # makes verified authorization mandatory unless explicitly
        # disabled for probe-only test brokers.
        self.authority_registry = authority_registry
        self.record_resolver = record_resolver
        if require_authorization is None:
            require_authorization = authority_registry is not None
        self.require_authorization = bool(require_authorization)
        if self.require_authorization and (
                authority_registry is None or record_resolver is None):
            raise BrokerRefused(
                "require_authorization needs an authority_registry "
                "AND a record_resolver — verification without trusted "
                "keys and approved records is impossible")
        self._nsjail_digest = None
        if nsjail_digest is not None:
            self._nsjail_digest = str(nsjail_digest)
            measured = self._measure_nsjail()
            if measured != self._nsjail_digest:
                raise BrokerRefused(
                    f"nsjail binary {self.nsjail_path} measures "
                    f"{measured} — expected {self._nsjail_digest}: the "
                    "broker runs only the approved binary")
        elif systemd is None:
            raise BrokerRefused(
                "nsjail_digest is required — the broker never runs an "
                "unmeasured sandbox binary")

    # --- qualification ---------------------------------------------------
    def _measure_nsjail(self) -> str:
        try:
            return _sha256_file(self.nsjail_path)
        except OSError as exc:
            raise BrokerRefused(
                f"nsjail binary {self.nsjail_path} unreadable: {exc}") \
                from exc

    def host_report(self) -> dict:
        """What this broker can actually enforce on this host."""
        checks = {
            "platform_linux": os.uname().sysname == "Linux",
            "systemd_client": self.systemd.available,
            "cgroup_v2": Path(
                "/sys/fs/cgroup/cgroup.controllers").is_file(),
            "nsjail_binary": Path(self.nsjail_path).is_file(),
            "nsjail_digest_pinned": self._nsjail_digest is not None,
            "user_manager": self.user_manager,
        }
        checks["ok"] = all((
            checks["platform_linux"], checks["systemd_client"],
            checks["cgroup_v2"], checks["nsjail_binary"],
            checks["nsjail_digest_pinned"]))
        return checks

    # --- request validation ----------------------------------------------
    def _validate_request(self, req: LaunchWorkerRequest) -> None:
        if not isinstance(req, LaunchWorkerRequest):
            raise BrokerRefused("launch requires a LaunchWorkerRequest")
        if not _ID_RE.match(str(req.worker_id)):
            raise BrokerRefused(
                f"worker_id {req.worker_id!r} is not a safe unit "
                "identifier")
        if req.role not in ("worker", "probe"):
            raise BrokerRefused(f"role {req.role!r} is not launchable")
        if not req.request_nonce:
            raise BrokerRefused("request_nonce is required")
        for d in req.approved_artifact_digests:
            if not _DIGEST_RE.match(str(d)):
                raise BrokerRefused(
                    f"artifact digest {d!r} is not sha256:<hex>")
        if req.approved_isolation_profile_digest:
            if not _DIGEST_RE.match(
                    str(req.approved_isolation_profile_digest)):
                raise BrokerRefused(
                    "isolation profile digest is not sha256:<hex>")
        # The broker enforces the COMPLETE authority contract itself —
        # a worker launch must carry well-formed digests for the signed
        # backend manifest AND the approved execution budget. A probe
        # carries no model authority, so these may be empty there but
        # are still format-checked when present.
        for name in ("signed_backend_manifest_digest",
                     "execution_budget_digest"):
            value = str(getattr(req, name) or "")
            if req.role == "worker" and not _DIGEST_RE.match(value):
                raise BrokerRefused(
                    f"{name} {value!r} is not sha256:<hex> — a worker "
                    "launch must be bound to verified authority "
                    "records, not placeholders")
            if req.role == "probe" and value                     and not _DIGEST_RE.match(value):
                raise BrokerRefused(
                    f"{name} {value!r} is not sha256:<hex>")
        # SEC-903: a worker id is broker-generated state, never a
        # caller input — reusing a live id or a live workspace dir is
        # an attack on workspace isolation, not a collision.
        if str(req.worker_id) in self._records:
            raise BrokerRefused(
                f"worker_id {req.worker_id!r} is already in use")
        # SEC-905: a worker launch carries a signed, currently-valid,
        # record-resolved authorization; a probe carries none (it runs
        # fixed probe code — no model authority to bind).
        if req.role == "worker" and self.require_authorization:
            from datetime import datetime, timezone
            rec = verify_launch_authorization(
                req.authorization, self.authority_registry,
                self.record_resolver, req=req,
                now=datetime.now(timezone.utc))
            # The authorization's nonce IS the reserved nonce — the
            # replay check below is the single-use mechanism for the
            # authorization itself.
            if rec.nonce != str(req.request_nonce):
                raise BrokerRefused(
                    "authorization nonce does not match the launch "
                    "request nonce")
        self._nonce_reserve(req.request_nonce)

    def _nonce_reserve(self, nonce: str) -> None:
        """Single-use reservation as ONE kernel-atomic operation:
        O_CREAT|O_EXCL either creates the marker (fresh nonce, caller
        wins) or fails EEXIST (replay). A check-then-write pair could
        admit two concurrent launches on the same nonce."""
        try:
            self.nonce_dir.mkdir(mode=0o700, parents=True,
                                 exist_ok=True)
            marker = (self.nonce_dir
                      / f"{hashlib.sha256(nonce.encode()).hexdigest()}")
            fd = os.open(str(marker),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                os.write(fd, str(time.time()).encode())
            finally:
                os.close(fd)
        except FileExistsError:
            raise BrokerRefused(
                "request_nonce already used — replay refused") from None
        except OSError as exc:
            raise BrokerRefused(
                f"nonce store unwritable — replay cannot be "
                f"prevented: {exc}") from exc

    # --- launch ------------------------------------------------------------
    def _exec_argv(self, role: str) -> list:
        module = (self.PROBE_ARGV_MODULE if role == "probe"
                  else self.WORKER_ARGV_MODULE)
        return [self.worker_python, *module]

    def _filter_env(self, env: dict) -> dict:
        out = {}
        for k, v in dict(env or {}).items():
            key = str(k)
            if key not in _ALLOWED_ENV:
                continue
            if any(m in key.upper() for m in _SECRET_MARKERS) \
                    and key.upper() not in _BENIGN_MARKER_NAMES:
                raise BrokerRefused(
                    f"envar {key!r} is credential-shaped — refused")
            out[key] = str(v)
        return out

    def _resolve_artifacts(self, digests) -> tuple:
        paths = []
        for d in digests or ():
            if self.artifact_resolver is None:
                raise BrokerRefused(
                    "artifact digest supplied but no artifact "
                    "resolver is configured")
            p = self.artifact_resolver(d)
            if p is None:
                raise BrokerRefused(
                    f"artifact digest {d} is not admitted")
            paths.append(str(p))
        return tuple(paths)

    def create_workspace(self, worker_id: str) -> tuple:
        """The worker's private tree under the broker-owned root:
        ``<root>/<worker_id>/scratch`` writable by the worker
        identity; nothing else under root is visible to it.

        SEC-903: ids are broker-generated ``w-<hex>`` only, the
        workspace dir is created descriptor-relative to a pinned root
        handle with O_EXCL (no reuse, no symlink games), and the
        resolved path must land beneath the resolved root."""
        wid = str(worker_id)
        if not _ID_RE.match(wid):
            raise BrokerRefused(f"worker_id {wid!r} unsafe")
        if wid in self._records:
            raise BrokerRefused(f"worker_id {wid!r} already in use")
        existing = self._workspaces.get(wid)
        if existing is not None:
            # this broker created it for this worker (prepare →
            # launch is one logical creation) — each caller gets its
            # own descriptor; prior callers may have closed theirs
            ws0, sc0, ov0, _ = existing
            fd = os.open(
                str(ov0), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0))
            return ws0, sc0, ov0, fd
        try:
            self.worker_state_root.mkdir(
                mode=0o711, parents=True, exist_ok=True)
            # 0711, not 0700: the userns-mapped worker uid must be able
            # to TRAVERSE (o+x) the tree to reach its bind-mounted
            # scratch — a denied traversal makes nsjail's mount of the
            # scratch source fail with EACCES. Listing stays denied.
            try:
                os.chmod(self.worker_state_root, 0o711)
            except OSError:
                pass
            root_real = Path(os.path.realpath(self.worker_state_root))
            root_fd = os.open(
                str(root_real),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0))
            try:
                # O_EXCL: an existing entry — live workspace, stale
                # dir, or an attacker-planted link — is a refusal,
                # never an adopt-and-continue.
                os.mkdir(wid, mode=0o711, dir_fd=root_fd)
            except FileExistsError:
                os.close(root_fd)
                raise BrokerRefused(
                    f"workspace for {wid!r} already exists — worker "
                    "ids are single-use") from None
            ws = root_real / wid
            ws_fd = os.open(
                str(wid),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd)
            os.close(root_fd)
            try:
                os.mkdir("scratch", mode=0o750, dir_fd=ws_fd)
                scratch_fd = os.open(
                    "scratch",
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0), dir_fd=ws_fd)
                try:
                    os.mkdir("overflow", mode=0o750, dir_fd=scratch_fd)
                finally:
                    os.close(scratch_fd)
            finally:
                os.close(ws_fd)
            # resolved-path proof: the created tree must sit directly
            # beneath the pinned root — anything else means a link
            # or race walked the tree somewhere else.
            if Path(os.path.realpath(ws)) != root_real / wid:
                raise BrokerRefused(
                    f"workspace {wid!r} did not resolve under the "
                    "pinned worker-state root")
            scratch = ws / "scratch"
            overflow = scratch / "overflow"
            try:
                os.chown(scratch, self.worker_uid, os.getgid())
                os.chown(overflow, self.worker_uid, os.getgid())
            except OSError:
                pass  # userns-only path: ids map, host chown best effort
            overflow_fd = os.open(
                str(overflow),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise BrokerError(
                f"worker workspace under {self.worker_state_root} "
                f"failed: {exc}") from exc
        self._workspaces[wid] = (ws, scratch, overflow, overflow_fd)
        return ws, scratch, overflow, overflow_fd

    def launch(self, req: LaunchWorkerRequest, *,
               env: dict | None = None,
               extra_ro_paths: tuple = (),
               unit_token: str | None = None) -> LaunchRecord:
        """Validate -> resolve -> generate -> run -> verify the unit
        is actually created and contained."""
        self._validate_request(req)
        profile = self.approved_profiles.get(
            str(req.approved_isolation_profile_digest)) \
            if req.approved_isolation_profile_digest else None
        if profile is None:
            if req.role == "worker" \
                    and not req.approved_isolation_profile_digest:
                # SEC-905: NEVER substitute an implicit profile for a
                # worker — production launches must name the approved
                # profile digest they were authorized under.
                raise BrokerRefused(
                    "a worker launch must carry the approved "
                    "isolation-profile digest it was authorized under")
            if req.approved_isolation_profile_digest:
                raise BrokerRefused(
                    "isolation profile digest is not a registered "
                    "approved profile — the frozen profile is the only "
                    "boundary a production worker may run under")
            if len(self.approved_profiles) == 1:
                profile = next(iter(self.approved_profiles.values()))
        if profile is None:
            raise BrokerRefused(
                "no approved isolation profile resolved")
        artifact_paths = self._resolve_artifacts(
            req.approved_artifact_digests)
        ws, scratch, overflow, overflow_fd = self.create_workspace(
            req.worker_id)
        token = unit_token or (
            f"{os.getpid()}-{secrets.token_hex(8)}")
        env_jail = self._filter_env(env)
        env_jail["MINAGI_WORKER_UNIT"] = token
        env_jail["MINAGI_WORKER_ROLE"] = req.role
        env_jail.setdefault("TMPDIR", "/work")
        env_jail.setdefault("TMP", "/work")
        env_jail.setdefault("TEMP", "/work")
        if self.python_path_entries:
            env_jail["PYTHONPATH"] = os.pathsep.join(
                self.python_path_entries)
        env_jail.setdefault("HOME", "/work")

        mounts = resolve_mounts(
            profile,
            runtime_paths=tuple(runtime_closure_paths())
            + self.python_path_entries,
            artifact_paths=artifact_paths,
            scratch_src=str(scratch), scratch_dst="/work",
            denied_paths=self.denied_paths)
        # extra ro paths arrive only from the trusted executor (the
        # admitted snapshot dir) — still screened through _allowed.
        extra = []
        for p in extra_ro_paths:
            rp = str(Path(p).resolve())
            base = rp.rstrip("/")
            for deny in self.denied_paths:
                dr = str(Path(str(deny)).resolve())
                if rp == dr or rp.startswith(dr + "/") \
                        or dr.startswith(base + "/"):
                    raise BrokerRefused(
                        f"extra ro path {rp!r} overlaps denied "
                        f"{dr!r} — refused")
            extra.append(MountSpec(src=rp, dst=rp, rw=False))
        mounts = tuple(mounts) + tuple(extra)

        exec_argv = self._exec_argv(req.role)
        config_text = render_config(
            profile, exec_argv=exec_argv, env=env_jail, mounts=mounts,
            inside_uid=self.worker_uid, inside_gid=self.worker_gid,
            outside_uid=self.worker_uid, outside_gid=self.worker_gid,
            cwd="/work", name=f"{self.unit_prefix}-{req.worker_id}")
        cfg_path = ws / "nsjail.cfg"
        try:
            cfg_path.write_text(config_text)
            os.chmod(cfg_path, 0o400)
        except OSError as exc:
            try:
                os.close(overflow_fd)
            except OSError:
                pass
            raise BrokerError(
                f"nsjail config unwritable: {exc}") from exc
        cfg_digest = config_digest(config_text)

        unit_name = f"{self.unit_prefix}-{req.worker_id}.service"
        properties = list(profile.unit_properties)
        properties.append(f"TasksMax={int(profile.unit_tasks_max)}")
        if profile.unit_memory_max:
            properties.append(
                f"MemoryMax={int(profile.unit_memory_max)}")
        if profile.memory_mb:
            properties.append(
                f"MemoryMax={int(profile.memory_mb) << 20}")
        properties.append(
            f"SyslogIdentifier={unit_name[:-len('.service')]}")
        stderr_log = ws / "unit-stderr.log"
        stderr_fh = open(stderr_log, "wb")
        proc = self.systemd.run_service(
            unit_name, properties,
            [self.nsjail_path, "--config", str(cfg_path)],
            cwd=str(scratch), stderr=stderr_fh)
        record = LaunchRecord(
            worker_id=req.worker_id, unit_name=unit_name, proc=proc,
            workspace=ws, scratch=scratch, overflow_dir=overflow,
            overflow_fd=overflow_fd, config_path=cfg_path,
            stderr_log=stderr_log, stderr_fh=stderr_fh,
            cgroup_path=None, unit_token=token, role=req.role,
            digests={
                "policy": str(req.approved_isolation_profile_digest),
                "config": cfg_digest,
                "nsjail_binary": self._nsjail_digest or "",
                "artifacts": tuple(req.approved_artifact_digests),
                "backend_manifest":
                    str(req.signed_backend_manifest_digest),
                "execution_budget": str(req.execution_budget_digest)},
            started_at=time.time(), user_manager=self.user_manager)
        self._records[req.worker_id] = record
        # The unit must actually exist and be placed — a spawned client
        # without a live unit is not containment.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            record.cgroup_path = self.systemd.cgroup_path(unit_name)
            if record.cgroup_path is not None:
                break
            if proc.poll() is not None:
                break
            time.sleep(0.05)
        if record.cgroup_path is not None:
            # Identity binding (SEC-902): record the unit's
            # InvocationID so a later teardown check can prove it is
            # looking at THIS invocation, not a reused unit name.
            record.invocation_id = self.systemd.show(
                unit_name, "InvocationID")
        if record.cgroup_path is None:
            # No verified kernel containment — the launch FAILS. A
            # live client process is not evidence of confinement.
            detail = self._abort_launch(record)
            raise BrokerError(
                f"worker unit {unit_name} never reached a live cgroup "
                f"— launch refused, not degraded: {detail}")
        return record

    def _abort_launch(self, rec: LaunchRecord) -> str:
        """Tear down a launch that never proved containment: stop the
        unit, reap the client, close every handle, drop the workspace,
        and leave nothing running or registered."""
        try:
            self.systemd.stop(rec.unit_name)
        except Exception:  # noqa: BLE001
            pass
        try:
            rec.proc.wait(timeout=5.0)
        except Exception:  # noqa: BLE001
            for act in (lambda: self.systemd.kill(
                            rec.unit_name, "SIGKILL"),
                        rec.proc.kill):
                try:
                    act()
                except Exception:  # noqa: BLE001
                    pass
            try:
                rec.proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass
        tail = ""
        try:
            if rec.stderr_fh is not None:
                rec.stderr_fh.close()
                rec.stderr_fh = None
            if rec.stderr_log is not None \
                    and Path(rec.stderr_log).is_file():
                tail = Path(rec.stderr_log).read_bytes(
                    )[-400:].decode(errors="replace")
        except OSError:
            pass
        for stream in (rec.proc.stdin, rec.proc.stdout,
                       getattr(rec.proc, "stderr", None)):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        if rec.overflow_fd is not None:
            try:
                os.close(rec.overflow_fd)
            except OSError:
                pass
            rec.overflow_fd = None
        self._records.pop(rec.worker_id, None)
        self._workspaces.pop(rec.worker_id, None)
        shutil.rmtree(rec.workspace, ignore_errors=True)
        return tail or "no unit diagnostics captured"

    # --- lifecycle ---------------------------------------------------------
    def _containment(self, rec: LaunchRecord) -> tuple:
        """Gather the kernel+service evidence and apply the ONE shared
        decision rule (SEC-902). Returns (state, populated)."""
        if rec.cgroup_path is not None \
                and not Path(rec.cgroup_path).is_dir():
            rec.cgroup_path = None  # collected by systemd
        if rec.cgroup_path is None:
            rec.cgroup_path = self.systemd.cgroup_path(rec.unit_name)
        populated = (cgroup_populated(rec.cgroup_path)
                     if rec.cgroup_path else None)
        unit_active = self.systemd.is_active(rec.unit_name)
        # identity binding: the unit currently answering must be the
        # invocation this record launched — a reused name is not proof
        if rec.cgroup_path is None and not unit_active:
            unit_same = True   # the recorded invocation is over
        else:
            current = self.systemd.show(rec.unit_name, "InvocationID")
            unit_same = bool(rec.invocation_id) \
                and current == rec.invocation_id
        state = evaluate_containment(
            populated=populated,
            cgroup_gone=rec.cgroup_path is None,
            unit_active=unit_active, unit_same=unit_same)
        return state, populated

    def status(self, record_or_id) -> dict:
        rec = self._record(record_or_id)
        state, populated = self._containment(rec)
        return {"worker_id": rec.worker_id, "unit": rec.unit_name,
                "active": self.systemd.is_active(rec.unit_name),
                "client_alive": rec.proc.poll() is None,
                "cgroup_path": (str(rec.cgroup_path)
                                if rec.cgroup_path else None),
                "cgroup_populated": populated,
                "containment": state.value,
                "digests": dict(rec.digests)}

    def terminate(self, record_or_id, *,
                  deadline_seconds: float = 10.0) -> dict:
        """kernel-owned teardown: stop the unit (SIGTERM -> SIGKILL per
        its KillMode), then PROVE the cgroup subtree emptied via the
        shared evaluator — EMPTY (populated 0) or REMOVED_VERIFIED
        (cgroup gone + unit inactive + same invocation). RUNNING and
        UNKNOWN never verify. /proc scanning is never the claim."""
        rec = self._record(record_or_id)
        deadline = time.monotonic() + float(deadline_seconds)
        stop_rc = getattr(self.systemd.stop(rec.unit_name),
                          "returncode", None)
        while time.monotonic() < deadline:
            if rec.proc.poll() is not None:
                break
            time.sleep(0.05)
        kill_rc = None
        if rec.proc.poll() is None:
            kill_rc = getattr(
                self.systemd.kill(rec.unit_name, "SIGKILL"),
                "returncode", None)
            try:
                rec.proc.wait(timeout=max(1.0, deadline_seconds / 2))
            except Exception:  # noqa: BLE001
                pass
        # verification: the shared evaluator — positive occupancy can
        # NEVER be overridden by an inactive unit or exited client;
        # unknown evidence spins to the deadline and stays unverified.
        state = ContainmentState.UNKNOWN
        populated = None
        while time.monotonic() < deadline:
            state, populated = self._containment(rec)
            if state in (ContainmentState.EMPTY,
                         ContainmentState.REMOVED_VERIFIED):
                break
            time.sleep(0.05)
        verified = state in (ContainmentState.EMPTY,
                             ContainmentState.REMOVED_VERIFIED)
        return {"worker_id": rec.worker_id, "unit": rec.unit_name,
                "verified_empty": bool(verified),
                "containment": state.value,
                "cgroup_populated": populated,
                "stop_rc": stop_rc, "kill_rc": kill_rc,
                "client_exit": rec.proc.poll(),
                "mechanism": "systemd-cgroupv2"}

    def release(self, record_or_id) -> None:
        rec = self._record(record_or_id)
        if rec.stderr_fh is not None:
            try:
                rec.stderr_fh.close()
            except OSError:
                pass
            rec.stderr_fh = None
        if rec.overflow_fd is not None:
            try:
                os.close(rec.overflow_fd)
            except OSError:
                pass
            rec.overflow_fd = None
        self._records.pop(rec.worker_id, None)
        self._workspaces.pop(rec.worker_id, None)
        shutil.rmtree(rec.workspace, ignore_errors=True)

    def _record(self, record_or_id) -> LaunchRecord:
        if isinstance(record_or_id, LaunchRecord):
            return record_or_id
        rec = self._records.get(str(record_or_id))
        if rec is None:
            raise BrokerError(f"unknown worker {record_or_id!r}")
        return rec


__all__ = ["AuthorizedLaunchRecord", "BrokerError", "BrokerRefused",
           "ContainmentState", "LaunchRecord", "LaunchWorkerRequest",
           "MappingRecordResolver", "SystemdClient",
           "WorkerLaunchBroker", "cgroup_populated",
           "evaluate_containment",
           "issue_launch_authorization", "new_worker_id",
           "runtime_closure_paths", "verify_launch_authorization"]
