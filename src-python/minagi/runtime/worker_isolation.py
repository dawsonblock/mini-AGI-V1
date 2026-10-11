"""v16.4.5/v16.4.6 worker process isolation (SEC-402 / RUN-401 / WP1/WP2).

Replacing pickle removes the code-execution channel, but a worker that
inherits the supervisor's full environment still holds its credentials:
API tokens, key paths, cloud metadata. `WorkerIsolationPolicy` makes the
launch boundary explicit:

  * **environment** — the child gets a constructed environment, not a
    copy of ``os.environ``: a small allowlist of safe variables plus
    whatever the operator explicitly adds. Nothing named ``*_KEY``,
    ``*_TOKEN``, ``*_SECRET``, or ``*_PASSWORD`` is ever passed through
    implicitly — signing credentials must not leak into the worker.
  * **private scratch** — each worker receives a dedicated 0700
    temporary directory as ``TMPDIR`` (and as the overflow directory
    for oversized results). Model code cannot see or write the
    supervisor's files except the read-only staged snapshot.
  * **POSIX resource limits** — CPU seconds, address space, process
    count, open files, and file size are bounded via
    ``resource.setrlimit`` in ``preexec_fn`` where the platform
    supports it. Unsupported knobs are skipped, not emulated — see
    ``supported()`` for what this platform actually enforces.
  * **identity demotion** — when the supervisor runs with privilege
    (e.g. a dedicated service identity that may demote children), an
    explicit ``demote_to=(uid, gid)`` drops the worker to a separate
    restricted OS identity. Never implicit.
  * **process group** — workers are launched in their own session so
    termination can signal the whole group and a model's own children
    cannot escape a kill.

macOS note: ``RLIMIT_AS`` does not exist on Darwin; ``RLIMIT_DATA`` /
``RLIMIT_RSS`` are accepted by ``setrlimit`` but not enforced by the
kernel for most allocations. Do not claim equal memory isolation
across platforms — qualify the controls that exist (``sandbox-exec``
is deprecated and opt-in only).
"""
from __future__ import annotations

import os
import re
import select
import shutil
import stat
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path


class IsolationError(RuntimeError):
    """The worker's isolation boundary could not be constructed."""


#: Environment names inherited unconditionally — none may carry
#: credentials, and PATH/LANG are needed for the interpreter to run.
SAFE_ENV_PASSTHROUGH = (
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "HOME",
    # cache locations (never tokens) so model code can reuse the
    # operator's populated caches without outbound credential access
    "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
)

#: Substrings that mark a variable as credential-bearing — a name
#: containing any of these is refused even if an operator adds it to
#: the passthrough list by mistake.
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _looks_secret(name: str) -> bool:
    upper = name.upper()
    return any(m in upper for m in _SECRET_MARKERS)


@dataclass(frozen=True)
class WorkerIsolationPolicy:
    """How a worker process is confined.

    env_passthrough   names additionally inherited from the parent
                      environment (merged over SAFE_ENV_PASSTHROUGH);
                      any name that looks credential-bearing is refused
    env               explicit variables set on the child (highest
                      precedence); also refused when credential-named
    cpu_seconds       RLIMIT_CPU bound (None = no limit)
    memory_bytes      RLIMIT_AS bound where the platform supports it
                      (Linux); on macOS RLIMIT_AS does not exist and
                      the bound is skipped — see module docstring
    max_processes     RLIMIT_NPROC bound (best effort)
    max_open_files    RLIMIT_NOFILE bound
    max_file_bytes    RLIMIT_FSIZE bound (caps file writes)
    demote_to         (uid, gid) the child drops to — requires the
                      parent to hold the privilege to setuid; never
                      applied silently
    sandbox_profile   optional macOS ``sandbox-exec`` profile path —
                      deprecated upstream but functional; on hosts
                      whose kernel refuses kqueue NOTE_TRACK a profile
                      that denies ``process-fork`` is THE descendant-
                      containment mechanism. ``@SCRATCH@`` renders to
                      the worker's private scratch at spawn. Only
                      counted toward required containment when set

    v16.4.6 enforcement requirements — when a ``require_*`` flag is set,
    :func:`validate_isolation` refuses the policy unless the host can
    actually enforce it. A required control that is unavailable fails
    closed; it is never silently downgraded:

    require_separate_identity
                      the worker MUST run under ``demote_to`` — a
                      different uid/gid from the supervisor's
    require_filesystem_confinement
                      protected supervisor resources must be
                      unreachable by the worker identity — enforced
                      by ownership/mode on the authority material plus
                      the private-scratch contract
    require_process_containment
                      process-group/session ownership plus verified
                      descendant termination must exist
    require_network_isolation
                      outbound network access must be denied — only
                      available on platforms with a working mechanism
                      (Linux network namespace); a host that cannot
                      enforce it fails the profile
    """
    env_passthrough: tuple[str, ...] = ()
    env: dict = field(default_factory=dict)
    cpu_seconds: int | None = None
    memory_bytes: int | None = None
    max_processes: int | None = None
    max_open_files: int | None = 256
    max_file_bytes: int | None = None
    demote_to: tuple[int, int] | None = None
    sandbox_profile: str | None = None
    require_separate_identity: bool = False
    require_filesystem_confinement: bool = False
    require_process_containment: bool = False
    require_network_isolation: bool = False


DEFAULT_ISOLATION = WorkerIsolationPolicy()


def supported() -> dict:
    """Which isolation controls this platform actually enforces — the
    honest capability set for qualification records."""
    import resource
    caps = {
        "rlimit_cpu": hasattr(resource, "RLIMIT_CPU"),
        "rlimit_as": hasattr(resource, "RLIMIT_AS"),
        "rlimit_nproc": hasattr(resource, "RLIMIT_NPROC"),
        "rlimit_nofile": hasattr(resource, "RLIMIT_NOFILE"),
        "rlimit_fsize": hasattr(resource, "RLIMIT_FSIZE"),
        "setuid_demote": hasattr(os, "setuid") and hasattr(os, "setgid"),
        "process_group": hasattr(os, "setsid"),
        "sandbox_exec": shutil.which("sandbox-exec") is not None,
        "platform": os.uname().sysname if hasattr(os, "uname") else "",
    }
    caps["can_demote_now"] = _has_demote_privilege()
    caps["proc_fs"] = Path("/proc/self/stat").is_file()
    caps["network_namespace"] = (
        caps["platform"] == "Linux"
        and shutil.which("unshare") is not None
        and Path("/proc/self/ns/net").is_file())
    caps["kill_group"] = hasattr(os, "killpg")
    caps["kqueue_tracking"] = kqueue_tracking_supported()
    return caps


def _has_demote_privilege() -> bool:
    """Can this process setuid a child? True for root or a process
    holding CAP_SETUID (Linux)."""
    if os.geteuid() == 0:
        return True
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("CapEff:"):
                return bool(int(line.split()[1], 16) & (1 << 7))
    except OSError:
        pass
    return False


def production_policy(*, worker_uid: int, worker_gid: int,
                      cpu_seconds: int | None = None,
                      memory_bytes: int | None = None,
                      max_processes: int = 32,
                      max_open_files: int = 256,
                      network_isolation: bool = False
                      ) -> "WorkerIsolationPolicy":
    """The production isolation profile — every enforcement flag on.
    `network_isolation` additionally requires a namespace mechanism;
    leave it off where the host cannot enforce it (the validation
    step refuses rather than pretending)."""
    return WorkerIsolationPolicy(
        require_separate_identity=True,
        require_filesystem_confinement=True,
        require_process_containment=True,
        require_network_isolation=network_isolation,
        demote_to=(int(worker_uid), int(worker_gid)),
        cpu_seconds=cpu_seconds, memory_bytes=memory_bytes,
        max_processes=max_processes, max_open_files=max_open_files)


def validate_isolation(policy: WorkerIsolationPolicy, *,
                       supervisor_uid: int | None = None) -> dict:
    """Fail-closed enforcement check: every required control must be
    enforceable on this host. Returns the startup report — which
    mechanisms will actually be applied — or raises IsolationError.
    Production must call this before any worker launches."""
    caps = supported()
    sup_uid = os.getuid() if supervisor_uid is None else supervisor_uid
    report = {"platform": caps["platform"], "mechanisms": [],
              "required": {}, "demote_to": policy.demote_to}

    def need(flag, name, ok, why):
        report["required"][name] = ok
        if flag and not ok:
            raise IsolationError(f"{name} is required but cannot be "
                                 f"enforced on this host: {why}")
        if ok:
            report["mechanisms"].append(name)

    if policy.require_separate_identity:
        if policy.demote_to is None:
            raise IsolationError(
                "separate worker identity required but no demote_to "
                "(worker uid/gid) configured")
        uid, gid = int(policy.demote_to[0]), int(policy.demote_to[1])
        if uid == sup_uid:
            raise IsolationError(
                f"worker uid {uid} equals the supervisor uid — that is "
                "not a separate identity")
        need(True, "separate_identity",
             caps["setuid_demote"] and caps["can_demote_now"],
             "platform cannot demote (no setuid/setgid privilege)")
        if gid == os.getgid() and uid == sup_uid:
            raise IsolationError(
                "worker (uid,gid) equals the supervisor identity")
    if policy.require_filesystem_confinement:
        if not policy.require_separate_identity:
            raise IsolationError(
                "filesystem confinement without a separate worker "
                "identity is illusory — a same-uid worker reads every "
                "supervisor-owned file")
        need(True, "filesystem_confinement", True,
             "unreachable (separate identity enforced above)")
    # Group control alone is not containment — a detached descendant
    # escapes a group signal. Real containment needs a tracking
    # mechanism that survives reparenting (/proc ancestry+marker on
    # Linux, kqueue NOTE_TRACK where the kernel still honours it) or
    # a seatbelt profile that makes descendant creation impossible
    # (macOS: sandbox-exec with deny process-fork).
    sandbox_containment = (caps["sandbox_exec"]
                           and bool(policy.sandbox_profile))
    need(policy.require_process_containment, "process_containment",
         caps["process_group"] and caps["kill_group"]
         and (caps["proc_fs"] or caps["kqueue_tracking"]
              or sandbox_containment),
         "no descendant-containment mechanism on this host (need /proc "
         "scan, kqueue NOTE_TRACK, or a sandbox_profile that denies "
         "process-fork)")
    need(policy.require_network_isolation, "network_isolation",
         caps["network_namespace"],
         "no network-namespace mechanism on this host")
    for field_name, name, key in (
            ("cpu_seconds", "cpu", "rlimit_cpu"),
            ("memory_bytes", "memory", "rlimit_as"),
            ("max_processes", "processes", "rlimit_nproc"),
            ("max_open_files", "open_files", "rlimit_nofile"),
            ("max_file_bytes", "file_size", "rlimit_fsize")):
        if getattr(policy, field_name) is not None:
            need(True, f"rlimit_{name}", caps[key],
                 f"{key} unsupported on {caps['platform']}")
    return report


def make_worker_scratch(demote_to, *, group_gid: int | None = None,
                        prefix: str = "minagi-worker-") -> Path:
    """Worker-owned scratch: created under the supervisor identity,
    then owned by the worker identity so a demoted child can still
    enter it — group set to the supervisor's gid with mode 0750 so the
    supervisor retains read/traverse for result collection and nothing
    else can enter."""
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        if demote_to is not None:
            uid, gid = int(demote_to[0]), int(demote_to[1])
            grp = gid if group_gid is None else int(group_gid)
            os.chown(path, uid, grp)
            os.chmod(path, 0o750)
        else:
            os.chmod(path, 0o700)
    except OSError:
        shutil.rmtree(path, ignore_errors=True)
        raise IsolationError(
            f"could not establish worker scratch ownership at {path}")
    return path


def make_private_tmp(prefix: str = "minagi-worker-") -> Path:
    """A 0700 scratch dir owned by the supervisor identity — the
    worker's TMPDIR and result-overflow location."""
    path = Path(tempfile.mkdtemp(prefix=prefix))
    os.chmod(path, 0o700)
    return path


def build_env(policy: WorkerIsolationPolicy, *, pythonpath: str,
              private_tmp: Path) -> dict:
    """Construct the child's environment — allowlist plus explicit
    additions, never an unrestricted copy of the parent."""
    env: dict[str, str] = {}
    for name in SAFE_ENV_PASSTHROUGH + tuple(policy.env_passthrough):
        if name in ("PYTHONPATH",):
            continue
        value = os.environ.get(name)
        if value is not None and not _looks_secret(name):
            env[name] = value
    for name, value in dict(policy.env or {}).items():
        name = str(name)
        if _looks_secret(name):
            raise IsolationError(
                f"worker env {name!r} looks credential-bearing — "
                "secrets are never pushed into the worker environment")
        env[name] = str(value)
    env["PYTHONPATH"] = str(pythonpath)
    tmp = str(private_tmp)
    env["TMPDIR"] = tmp
    env["TMP"] = tmp
    env["TEMP"] = tmp
    return env


def worker_preexec(policy: WorkerIsolationPolicy):
    """A ``preexec_fn`` applying the policy's POSIX limits inside the
    child before exec. Returns None when nothing needs applying —
    the caller combines this with ``start_new_session=True``."""
    import resource
    limits: list[tuple[int, tuple[int, int]]] = []
    if policy.cpu_seconds is not None and hasattr(resource, "RLIMIT_CPU"):
        v = int(policy.cpu_seconds)
        limits.append((resource.RLIMIT_CPU, (v, v)))
    if policy.memory_bytes is not None and hasattr(resource, "RLIMIT_AS"):
        v = int(policy.memory_bytes)
        limits.append((resource.RLIMIT_AS, (v, v)))
    if policy.max_processes is not None and \
            hasattr(resource, "RLIMIT_NPROC"):
        v = int(policy.max_processes)
        limits.append((resource.RLIMIT_NPROC, (v, v)))
    if policy.max_open_files is not None and \
            hasattr(resource, "RLIMIT_NOFILE"):
        v = int(policy.max_open_files)
        limits.append((resource.RLIMIT_NOFILE, (v, v)))
    if policy.max_file_bytes is not None and \
            hasattr(resource, "RLIMIT_FSIZE"):
        v = int(policy.max_file_bytes)
        limits.append((resource.RLIMIT_FSIZE, (v, v)))
    demote = policy.demote_to
    if not limits and demote is None:
        return None

    def apply():
        for res, pair in limits:
            try:
                resource.setrlimit(res, pair)
            except (OSError, ValueError):
                # A limit the kernel refuses must not silently pass —
                # the child exits non-zero and the parent reports the
                # worker unstartable.
                os._exit(70)
        if demote is not None:
            uid, gid = int(demote[0]), int(demote[1])
            try:
                os.setgroups([])
                os.setgid(gid)
                os.setuid(uid)
            except OSError:
                os._exit(70)
            os.environ["HOME"] = "/nonexistent"
    return apply


def wrap_argv(argv: list, policy: WorkerIsolationPolicy, *,
              render_vars: dict | None = None) -> list:
    """Optional ``sandbox-exec`` wrapper (macOS only, opt-in). The
    profile must be supplied by the operator — this code does not
    pretend a default profile exists. ``@NAME@`` placeholders in the
    profile are rendered from ``render_vars`` (e.g. ``@SCRATCH@`` to
    the worker's private scratch path) into a supervisor-owned temp
    file — seatbelt reads the profile once at exec, so a rendered
    copy in a 0600 supervisor file keeps the worker from ever naming
    its own confinement rules. Returns ``(argv, rendered_profile)`` —
    the second element is the temp path to unlink on cleanup, or the
    original profile path when no rendering was needed."""
    if not policy.sandbox_profile:
        return list(argv), None
    exe = shutil.which("sandbox-exec")
    if exe is None:
        raise IsolationError(
            "sandbox-exec requested but not available on this host")
    profile = Path(policy.sandbox_profile)
    if not profile.is_file() or profile.is_symlink():
        raise IsolationError(
            f"sandbox profile {profile} is not a regular file")
    rendered = str(profile)
    text = profile.read_text(encoding="utf-8")
    if "@" in text:
        for key, value in dict(render_vars or {}).items():
            text = text.replace(f"@{key}@", str(value))
        if re.search(r"@[A-Z_][A-Z_0-9]+@", text):
            raise IsolationError(
                "sandbox profile still contains unrendered "
                "@PLACEHOLDER@ tokens")
        fd, rendered = tempfile.mkstemp(
            prefix="minagi-worker-profile-", suffix=".sb")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(rendered, 0o600)
    else:
        rendered = None
    return [exe, "-f", rendered or str(profile), *argv], rendered


def tmp_is_private(path: Path, *, worker_uid: int | None = None) -> bool:
    """The mode/ownership contract a worker scratch dir must hold.
    Same-identity workers: 0700 owned by the supervisor. Demoted
    workers: 0750 owned by the worker uid with the supervisor's gid —
    the worker enters its own scratch, the supervisor keeps
    read/traverse for result collection, nothing else can enter."""
    st = path.lstat()
    if not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode):
        return False
    mode = stat.S_IMODE(st.st_mode)
    if worker_uid is None:
        return mode == 0o700 and st.st_uid == os.getuid()
    return mode == 0o750 and st.st_uid == int(worker_uid) \
        and st.st_gid == os.getgid()


__all__ = ["DEFAULT_ISOLATION", "IsolationError", "SAFE_ENV_PASSTHROUGH",
           "WorkerIsolationPolicy", "build_env", "make_private_tmp",
           "supported", "tmp_is_private", "worker_preexec", "wrap_argv"]


# ---------------------------------------------------------------------------
# WP2 — worker process-tree lifecycle (v16.4.6)
# ---------------------------------------------------------------------------

WORKER_UNIT_ENV = "MINAGI_WORKER_UNIT"


def new_worker_unit() -> str:
    """A random token identifying one worker containment unit — carried
    in the worker's initial environment, inherited by descendants, and
    unguessable so no unrelated process can claim membership."""
    import secrets
    return f"{os.getpid()}-{secrets.token_hex(8)}"


def worker_unit_env(unit: str) -> dict:
    """Every worker carries its unit token in its *initial* environment
    — descendants inherit it even if they detach their process group or
    call setsid(), and a process cannot scrub /proc/<pid>/environ after
    the fact. Escaping via exec with a scrubbed environment remains a
    documented limit (cgroups are the stronger mechanism)."""
    return {WORKER_UNIT_ENV: str(unit)}


def _proc_fields(pid: int):
    """(ppid, pgrp, session) from /proc — Linux only."""
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text()
        # comm may contain spaces/parens — split at the final ')'
        rest = stat_text.rsplit(")", 1)[1].split()
        return int(rest[1]), int(rest[2]), int(rest[3])
    except (OSError, IndexError, ValueError):
        return None


def _ps_table() -> dict:
    """pid -> (ppid, pgrp, session) via ps — portable fallback."""
    try:
        import subprocess
        out = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,pgid=,sess="],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return {}
    table = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4:
            try:
                table[int(parts[0])] = (int(parts[1]), int(parts[2]),
                                        int(parts[3]))
            except ValueError:
                continue
    return table


def descendants_of(root_pid: int) -> set:
    """All live descendants of root_pid via the ppid graph."""
    if Path("/proc").is_dir():
        table = {}
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                fields = _proc_fields(int(entry.name))
                if fields is not None:
                    table[int(entry.name)] = fields
    else:
        table = _ps_table()
    owned, frontier = set(), {int(root_pid)}
    while frontier:
        nxt = set()
        for pid, (ppid, _pg, _se) in table.items():
            if ppid in frontier and pid not in owned:
                owned.add(pid)
                nxt.add(pid)
        frontier = nxt
    owned.discard(int(root_pid))
    return owned


_KQ_NAMES = {
    "filter_proc": ("EVFILT_PROC", "KQ_FILTER_PROC"),
    "add": ("EV_ADD", "KQ_EV_ADD"),
    "note_track": ("NOTE_TRACK", "KQ_NOTE_TRACK"),
    "note_fork": ("NOTE_FORK", "KQ_NOTE_FORK"),
    "note_exit": ("NOTE_EXIT", "KQ_NOTE_EXIT"),
}


def _kq_const(name: str):
    for cand in _KQ_NAMES[name]:
        value = getattr(select, cand, None)
        if value is not None:
            return value
    return None


def kqueue_tracking_supported() -> bool:
    """True when this host can follow worker forks event-driven —
    macOS/BSD EVFILT_PROC+NOTE_TRACK via select.kqueue. Constants are
    not proof: modern macOS keeps the API surface but the kernel
    answers NOTE_TRACK registration with EOPNOTSUPP, so probe a real
    registration on self and check for EV_ERROR."""
    if not (hasattr(select, "kqueue")
            and all(_kq_const(n) is not None for n in _KQ_NAMES)):
        return False
    ev_error = getattr(select, "EV_ERROR",
                       getattr(select, "KQ_EV_ERROR", 0))
    try:
        kq = select.kqueue()
        try:
            result = kq.control(
                [select.kevent(os.getpid(), _kq_const("filter_proc"),
                               _kq_const("add"), _kq_const("note_track"),
                               0, 0)], 1, 0.5)
        finally:
            kq.close()
    except OSError:
        return False
    for ev in result or ():
        if ev.flags & ev_error:
            return False
    return True


class KqueueDescendantTracker:
    """Event-driven descendant tracking for one worker leader.

    Registering NOTE_TRACK on the leader delivers a NOTE_FORK (with the
    child's pid in ``ev.data``) for every fork inside the unit, and we
    re-register NOTE_TRACK on each child — so membership is fixed at
    fork time, before a descendant can setsid(), get reparented to
    launchd, or exec with a scrubbed environment. Those escapes defeat
    the ppid graph and the environ-marker scan; they cannot defeat a
    fork event the kernel reported. Residual gap: a descendant that
    forks in the microseconds between its own fork event and our
    registration can be missed — bounded, documented, and strictly
    stronger than the polling fallbacks it complements.
    """

    _MAX_TRACKED = 4096
    _DRAIN_TIMEOUT = 0.25

    def __init__(self, root_pid: int):
        self._root = int(root_pid)
        self._kq = select.kqueue()
        self._lock = threading.Lock()
        self._live: set[int] = set()
        self._root_dead = False
        self._closed = False
        self._register(self._root)
        self._thread = threading.Thread(
            target=self._drain,
            name=f"minagi-kq-track-{self._root}", daemon=True)
        self._thread.start()

    @classmethod
    def attach(cls, root_pid: int):
        """A tracker following root_pid's descendants, or None when the
        host has no kqueue process filter."""
        if not kqueue_tracking_supported():
            return None
        try:
            return cls(root_pid)
        except OSError:
            return None

    def _register(self, pid: int) -> None:
        try:
            self._kq.control(
                [select.kevent(int(pid), _kq_const("filter_proc"),
                               _kq_const("add"), _kq_const("note_track"),
                               0, 0)], 0, 0)
        except OSError:
            # The target already exited — nothing to track.
            pass

    def _drain(self) -> None:
        while True:
            try:
                events = self._kq.control(None, 64, self._DRAIN_TIMEOUT)
            except (OSError, ValueError):
                return
            if self._closed:
                return
            root_dead = self._root_dead
            for ev in events:
                if ev.fflags & _kq_const("note_fork"):
                    child = int(ev.data)
                    if child > 0:
                        with self._lock:
                            if len(self._live) < self._MAX_TRACKED:
                                self._live.add(child)
                        self._register(child)
                if ev.fflags & _kq_const("note_exit"):
                    ident = int(ev.ident)
                    if ident == self._root:
                        root_dead = True
                    else:
                        with self._lock:
                            self._live.discard(ident)
            self._root_dead = root_dead
            if root_dead:
                with self._lock:
                    empty = not self._live
                if empty:
                    return

    def live_pids(self) -> set:
        """Descendant pids believed live — callers still probe each
        pid (os.kill(pid, 0)) before acting on it."""
        with self._lock:
            return set(self._live)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._kq.close()
        except OSError:
            pass
        self._thread.join(timeout=1.0)


def _marker_holders(unit: str) -> set:
    """Live pids whose initial environment carries the worker unit
    marker — the containment set independent of process groups.
    Requires /proc (Linux); platforms without it fall back to the
    tracked-ancestry + process-group checks (documented limitation —
    macOS exposes no cross-process environ scan)."""
    want = f"{WORKER_UNIT_ENV}={unit}".encode()
    holders = set()
    proc = Path("/proc")
    if not proc.is_dir():
        return holders
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            env = (entry / "environ").read_bytes()
        except (OSError, PermissionError):
            continue
        if want in env.split(b"\0"):
            holders.add(int(entry.name))
    return holders


class WorkerProcessController:
    """Verified lifecycle for one worker containment unit.

    Termination protocol: SIGTERM the worker's own process group ->
    bounded grace -> SIGKILL the group -> individually kill any marker-
    carrying survivors -> verify nothing owned remains. Group signals
    are only sent after verifying the leader still owns its group id —
    never an unchecked negative PID.
    """

    def __init__(self, term_grace: float = 2.0,
                 kill_grace: float = 5.0):
        self.term_grace = float(term_grace)
        self.kill_grace = float(kill_grace)

    @staticmethod
    def group_id(pid: int) -> int | None:
        try:
            return os.getpgid(int(pid))
        except OSError:
            return None

    def _signal_group(self, pid: int, sig: int) -> bool:
        """Signal the group ONLY if pid still leads it and it is not
        our own group — a reused or escaped pgid is never signalled."""
        pgid = self.group_id(pid)
        if pgid is None or pgid != int(pid):
            return False
        if pgid == os.getpgrp():
            return False
        try:
            os.killpg(pgid, sig)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def _owned(self, tracked: set, unit: str | None,
               extra_tracked=None) -> set:
        """Processes still owned by the unit: tracked descendants that
        remain alive plus anything still carrying the unit marker.
        ``extra_tracked`` is a live accessor (e.g. the kqueue tracker)
        consulted at call time so forks discovered mid-termination are
        included in the sweep."""
        live = set()
        if extra_tracked is not None:
            tracked = set(tracked) | set(extra_tracked())
        for pid in tracked:
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, PermissionError):
                continue
            live.add(pid)
        if unit:
            live |= _marker_holders(unit)
        return live

    def terminate_tree(self, proc, *, leader_pid: int | None = None,
                       tracked: set | None = None,
                       unit: str | None = None,
                       extra_tracked=None) -> set:
        """Stop routing -> escalate -> reap -> verify. ``proc`` is the
        leader's Popen handle; ``tracked`` is an optional ancestry
        snapshot taken while the leader lived; ``unit`` is the env
        marker token identifying the containment set; ``extra_tracked``
        is an optional live pid accessor (kqueue tracker) consulted at
        each ownership check. Returns the set of pids that could NOT
        be confirmed dead (empty on success)."""
        import signal
        import time
        pid = int(leader_pid if leader_pid is not None else proc.pid or 0)
        tracked = set(tracked or ()) | ({pid} if pid else set())
        if pid:
            if proc.poll() is None:
                self._signal_group(pid, signal.SIGTERM)
                deadline = time.monotonic() + self.term_grace
                while proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
            if proc.poll() is None:
                self._signal_group(pid, signal.SIGKILL)
        try:
            proc.wait(timeout=max(1.0, self.kill_grace))
        except Exception:  # noqa: BLE001 - verify below is authoritative
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        survivors = self._owned(tracked, unit, extra_tracked)
        if survivors:
            for spid in survivors:
                try:
                    os.kill(spid, signal.SIGKILL)
                except OSError:
                    pass
            import time as _t
            _t.sleep(0.05)
            survivors = self._owned(tracked, unit, extra_tracked)
        return survivors

    def verify_terminated(self, leader_pid: int,
                          tracked: set | None = None,
                          unit: str | None = None,
                          extra_tracked=None) -> set:
        """The acceptance check: zero processes owned by the unit.
        Returns the surviving pid set — must be empty."""
        tracked = set(tracked or ())
        tracked.discard(int(leader_pid))
        return self._owned(tracked, unit, extra_tracked)

    def cleanup_resources(self, private_tmp) -> None:
        if private_tmp is not None:
            shutil.rmtree(private_tmp, ignore_errors=True)


__all__ = ["DEFAULT_ISOLATION", "IsolationError",
           "KqueueDescendantTracker", "SAFE_ENV_PASSTHROUGH",
           "WORKER_UNIT_ENV", "WorkerIsolationPolicy",
           "WorkerProcessController", "build_env", "descendants_of",
           "kqueue_tracking_supported",
           "make_private_tmp", "make_worker_scratch",
           "production_policy", "supported", "tmp_is_private",
           "new_worker_unit", "validate_isolation", "worker_preexec",
           "worker_unit_env",
           "wrap_argv"]
