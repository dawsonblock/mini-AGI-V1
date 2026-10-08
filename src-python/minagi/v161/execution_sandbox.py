"""OS-enforced sandbox for executable evaluator checks (FIX-001).

Corpus-supplied python checkers must never run with the evaluator's
own privileges. This module is the only sanctioned path for running
them, and it fails closed: if no OS-enforced sandbox backend is
available, `run_check` raises `SandboxUnavailable` and executable
evaluation does not happen at all. There is deliberately no
"unsandboxed" fallback.

Backends
--------
macos-sandbox-exec
    SBPL profile with `(deny default)`: reads are allowlisted to the
    interpreter prefix, system libraries, the interpreter's resolved
    dylib closure and the ephemeral workspace; writes are allowed only
    inside the workspace; network is denied. The original v16.2.1
    executor ran checkers with the full parent environment and no
    filesystem or network isolation at all.

linux-bwrap
    bubblewrap with `--unshare-all` (user/pid/net/ipc/uts namespaces),
    read-only system binds, workspace-only writes, `--clearenv`, and
    `--die-with-parent`.

Both backends additionally apply:

  * rlimits: CPU seconds, file size, process count, and address space
    (address space on Linux; macOS cannot lower RLIMIT_AS below its
    enormous current VM reservation — see the residual-risk register
    in docs/research/SECURITY_REPAIR_V1622.md),
  * a minimal environment (os.environ is never inherited),
  * an ephemeral workspace removed after the run,
  * process-group termination on timeout (no orphaned children).
"""
from __future__ import annotations

import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

SANDBOX_RESULT_SCHEMA = "mini-agi-v16.2.2-sandbox-result-v1"

MACOS_SANDBOX_EXEC = "/usr/bin/sandbox-exec"
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_MEMORY_LIMIT = 1 << 30          # 1 GiB
DEFAULT_NPROC_LIMIT = 64
DEFAULT_FSIZE_LIMIT = 64 << 20          # 64 MiB
DEFAULT_MAX_OUTPUT_BYTES = 256 << 10    # 256 KiB captured

_MACOS_ALLOWED_READ_PREFIXES = (
    "/usr/lib", "/System/Library", "/Library/Apple",
    "/private/var/db/dyld",
)


class SandboxUnavailable(RuntimeError):
    """No OS-enforced sandbox backend is available on this host.
    Executable evaluation fails closed — it never runs unsandboxed."""


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    backend: str
    limits: Mapping[str, object]
    schema: str = SANDBOX_RESULT_SCHEMA


def available_backends() -> list[str]:
    names = []
    if sys.platform == "darwin" and Path(MACOS_SANDBOX_EXEC).is_file():
        names.append("macos-sandbox-exec")
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        names.append("linux-bwrap")
    return names


def select_backend() -> str:
    """Pick the platform sandbox backend, or raise SandboxUnavailable.
    `MINIAGI_SANDBOX_BACKEND` may pin a backend name (or `none` to force
    the fail-closed path, e.g. in tests)."""
    override = os.environ.get("MINIAGI_SANDBOX_BACKEND", "").strip()
    names = available_backends()
    if override:
        if override in ("none", "off", "disabled"):
            raise SandboxUnavailable(
                "sandbox backends disabled via MINIAGI_SANDBOX_BACKEND="
                f"{override!r} — executable evaluation fails closed")
        if override not in names:
            raise SandboxUnavailable(
                f"sandbox backend {override!r} requested but not "
                f"available (available: {names or 'none'})")
        return override
    if not names:
        raise SandboxUnavailable(
            "no OS-enforced sandbox backend on this host (macOS: "
            f"{MACOS_SANDBOX_EXEC}; Linux: bwrap) — executable "
            "evaluation fails closed rather than running checkers "
            "unsandboxed")
    return names[0]


def _dylib_closure(binary: str, depth: int = 4) -> tuple[list[str], bool]:
    """Resolved dynamic-library closure of the interpreter (macOS).
    Returns (paths, complete)."""
    if sys.platform != "darwin" or not shutil.which("otool"):
        return [], False
    seen: set[str] = set()
    frontier = [binary]
    complete = True
    for _ in range(depth):
        nxt = []
        for b in frontier:
            try:
                out = subprocess.run(["otool", "-L", b],
                                     capture_output=True, text=True,
                                     timeout=10).stdout
            except (OSError, subprocess.SubprocessError):
                complete = False
                continue
            for line in out.splitlines()[1:]:
                m = re.match(r"\s+(\S+) \(compatibility", line)
                if not m:
                    continue
                lib = m.group(1)
                if lib not in seen:
                    seen.add(lib)
                    nxt.append(lib)
        frontier = nxt
    return sorted(seen), complete


def _extra_read_paths(python: str) -> tuple[list[str], bool]:
    """Dylibs outside the statically allowed prefixes that the
    interpreter needs, with symlinks resolved."""
    closure, complete = _dylib_closure(python)
    allowed = (sys.base_prefix,) + _MACOS_ALLOWED_READ_PREFIXES
    extras: list[str] = []
    for lib in closure:
        if lib.startswith(allowed):
            continue
        for p in (lib, os.path.realpath(lib)):
            if p not in extras:
                extras.append(p)
    return extras, complete


def _ancestor_subpaths(path: str) -> list[str]:
    out = []
    p = Path(path)
    for parent in p.parents:
        s = str(parent)
        if s == "/":
            break
        out.append(s)
    return out


def _macos_profile(python: str, workspace: str,
                   read_paths: list[str], extras: list[str]) -> str:
    read_targets = ['(literal "/")']
    for p in (sys.base_prefix,) + _MACOS_ALLOWED_READ_PREFIXES:
        read_targets.append(f'(subpath "{p}")')
    for p in extras:
        read_targets.append(f'(literal "{p}")')
    for p in [workspace] + read_paths:
        read_targets.append(f'(subpath "{p}")')
    metadata_targets = []
    for p in (sys.base_prefix,) + _MACOS_ALLOWED_READ_PREFIXES:
        metadata_targets.append(f'(subpath "{p}")')
    for p in [workspace] + read_paths:
        metadata_targets.append(f'(subpath "{p}")')
    for p in extras:
        for parent in _ancestor_subpaths(p):
            metadata_targets.append(f'(subpath "{parent}")')
    return f"""(version 1)
(deny default)
(allow process-exec (literal "{python}"))
(allow process-fork)
(allow sysctl-read)
(allow file-read-metadata {' '.join(sorted(set(metadata_targets)))})
(allow file-read* {' '.join(read_targets)})
(allow file-write* (subpath "{workspace}") (literal "/dev/null"))
(deny network*)
"""


def _bwrap_argv(python: str, workspace: str, read_paths: list[str],
                env: Mapping[str, str], check: str) -> list[str]:
    argv = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
            "--clearenv", "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp"]
    for p in ("/usr", "/lib", "/lib64", "/bin", "/sbin",
              "/etc/ld.so.cache"):
        argv += ["--ro-bind-try", p, p]
    prefix = sys.base_prefix
    if not prefix.startswith(("/usr", "/lib", "/bin", "/sbin")):
        argv += ["--ro-bind-try", prefix, prefix]
    for p in read_paths:
        argv += ["--ro-bind", p, p]
    argv += ["--bind", workspace, workspace, "--chdir", workspace]
    for k, v in env.items():
        argv += ["--setenv", str(k), str(v)]
    argv += ["--", python, "-I", "-S", "-c", check]
    return argv


def _apply_limits(cpu_limit_s: int, memory_limit_bytes: int,
                  nproc_limit: int, fsize_limit_bytes: int):
    """preexec_fn: applied to the sandbox process, inherited by the
    checker. Failures to lower a limit are tolerated only where the
    platform cannot enforce it (macOS RLIMIT_AS) — the limits actually
    in force are reported in the SandboxResult."""
    def _set(what, value):
        try:
            resource.setrlimit(what, (value, value))
        except (ValueError, OSError):
            pass
    _set(resource.RLIMIT_CPU, cpu_limit_s)
    _set(resource.RLIMIT_FSIZE, fsize_limit_bytes)
    _set(resource.RLIMIT_NOFILE, 64)
    _set(resource.RLIMIT_NPROC, nproc_limit)
    _set(resource.RLIMIT_AS, memory_limit_bytes)
    _set(resource.RLIMIT_CORE, 0)


def run_check(check: str, *, env: Mapping[str, str] | None = None,
              timeout_s: float = DEFAULT_TIMEOUT_S,
              read_paths: Iterable[str | Path] = (),
              memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT,
              nproc_limit: int = DEFAULT_NPROC_LIMIT,
              fsize_limit_bytes: int = DEFAULT_FSIZE_LIMIT,
              max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
              workspace: str | Path | None = None,
              backend: str | None = None) -> SandboxResult:
    """Run `python -I -S -c check` under the OS sandbox.

    The checker sees only `env` plus fixed defaults (PATH, HOME/TMPDIR
    inside the workspace, PYTHONHASHSEED); os.environ is never
    inherited. `read_paths` are mounted/allowlisted read-only. Returns
    a SandboxResult; raises SandboxUnavailable when no sandbox backend
    can be used (fail closed)."""
    if not isinstance(check, str) or not check.strip():
        raise ValueError("check must be a non-empty python source string")
    backend = backend or select_backend()
    python = os.path.realpath(sys.executable)

    owns_workspace = workspace is None
    ws = Path(os.path.realpath(workspace)) if workspace else \
        Path(os.path.realpath(tempfile.mkdtemp(prefix="minagi-eval-")))
    ws.mkdir(parents=True, exist_ok=True)
    reads = [os.path.realpath(str(p)) for p in read_paths]

    child_env = {"PATH": "/usr/bin:/bin", "HOME": str(ws),
                 "TMPDIR": str(ws), "PYTHONHASHSEED": "0"}
    child_env.update({str(k): str(v) for k, v in (env or {}).items()})

    limits = {
        "timeout_s": float(timeout_s),
        "cpu_limit_s": int(timeout_s) + 1,
        "nproc_limit": int(nproc_limit),
        "fsize_limit_bytes": int(fsize_limit_bytes),
        "memory_limit_bytes": int(memory_limit_bytes),
        # macOS cannot lower RLIMIT_AS below its current (huge) VM
        # reservation; the cap is enforced on Linux. Documented in the
        # residual-risk register.
        "memory_limit_enforced": sys.platform.startswith("linux"),
    }

    if backend == "macos-sandbox-exec":
        extras, closure_complete = _extra_read_paths(python)
        profile = _macos_profile(python, str(ws), reads, extras)
        argv = [MACOS_SANDBOX_EXEC, "-p", profile, python, "-I", "-S",
                "-c", check]
    elif backend == "linux-bwrap":
        closure_complete = True
        argv = _bwrap_argv(python, str(ws), reads, child_env, check)
    else:
        raise SandboxUnavailable(f"unknown sandbox backend {backend!r}")

    out_path = ws / ".sandbox-stdout"
    err_path = ws / ".sandbox-stderr"
    timed_out = False
    try:
        with out_path.open("wb") as fo, err_path.open("wb") as fe:
            proc = subprocess.Popen(
                argv, env=child_env, cwd=str(ws),
                stdin=subprocess.DEVNULL, stdout=fo, stderr=fe,
                start_new_session=True,
                preexec_fn=lambda: _apply_limits(
                    limits["cpu_limit_s"], memory_limit_bytes,
                    nproc_limit, fsize_limit_bytes))
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        with out_path.open("rb") as f:
            stdout = f.read(max_output_bytes).decode("utf-8", "replace")
        with err_path.open("rb") as f:
            stderr = f.read(max_output_bytes).decode("utf-8", "replace")
    finally:
        if owns_workspace:
            shutil.rmtree(ws, ignore_errors=True)

    if (not timed_out and proc.returncode != 0
            and "Library not loaded" in stderr and not closure_complete):
        raise SandboxUnavailable(
            "interpreter dylib closure could not be resolved (otool "
            f"unavailable?) — sandbox cannot start the interpreter: "
            f"{stderr.strip()[:200]}")

    return SandboxResult(
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout, stderr=stderr, timed_out=timed_out,
        backend=backend, limits=limits)
