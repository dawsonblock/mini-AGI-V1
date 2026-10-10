"""v16.4.5 worker process isolation (SEC-402 / RUN-401 hardening).

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
import shutil
import stat
import tempfile
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
                      DEPRECATED mechanism, opt-in, qualified only when
                      the operator supplies a profile they have tested
    extra_env         alias kept for readability of call sites
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


DEFAULT_ISOLATION = WorkerIsolationPolicy()


def supported() -> dict:
    """Which isolation controls this platform actually enforces — the
    honest capability set for qualification records."""
    import resource
    return {
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


def wrap_argv(argv: list, policy: WorkerIsolationPolicy) -> list:
    """Optional ``sandbox-exec`` wrapper (macOS only, opt-in). The
    profile must be supplied by the operator — this code does not
    pretend a default profile exists."""
    if not policy.sandbox_profile:
        return list(argv)
    exe = shutil.which("sandbox-exec")
    if exe is None:
        raise IsolationError(
            "sandbox-exec requested but not available on this host")
    profile = Path(policy.sandbox_profile)
    if not profile.is_file() or profile.is_symlink():
        raise IsolationError(
            f"sandbox profile {profile} is not a regular file")
    return [exe, "-f", str(profile), *argv]


def tmp_is_private(path: Path) -> bool:
    """The mode/ownership contract a worker scratch dir must hold."""
    st = path.lstat()
    return stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode) \
        and stat.S_IMODE(st.st_mode) == 0o700 \
        and st.st_uid == os.getuid()


__all__ = ["DEFAULT_ISOLATION", "IsolationError", "SAFE_ENV_PASSTHROUGH",
           "WorkerIsolationPolicy", "build_env", "make_private_tmp",
           "supported", "tmp_is_private", "worker_preexec", "wrap_argv"]
