"""Dedicated worker-launch process (v16.4.7 / isolation enforcement).

v16.4.8: this launcher is the DEVELOPMENT executor's privilege-setup
path only (``worker_executor.DevelopmentWorkerExecutor``). The
production worker boundary is ``nsjail_executor`` — a managed
systemd unit running the pinned NsJail binary — so the hand-rolled
``unshare``/``pivot_root``/cgroup machinery below never executes on
the production path (SEC-702). It remains for trusted local testing
where the platform cannot run NsJail.


The supervisor is multithreaded: doing isolation setup inside
``subprocess.Popen(..., preexec_fn=...)`` is unsafe — a lock held by
another thread at fork() can deadlock the child before it ever execs.
It also meant the trusted supervisor's own fork path performed the
untrusted worker's privilege setup.

This module replaces that: the supervisor spawns *this* launcher with
no ``preexec_fn``. The launcher reads a JSON launch spec over an
inherited fd, applies the isolation contract itself, and then exec's
the real worker entry — so everything the worker becomes was decided
by an independently configured step, not by the supervisor's fork.

Spec (JSON, read from fd named by ``MINAGI_LAUNCH_SPEC_FD``)::

    {
      "exec_argv":   ["/usr/bin/python3", "-m",
                      "minagi.runtime.worker_backend"],
      "uid":         65534,          # drop to (None = keep)
      "gid":         65534,
      "rlimits":     {"cpu": 30, "as": 4_000_000_000, "nproc": 32,
                      "nofile": 256, "fsize": 1_000_000},
      "netns":       true,           # new empty network namespace
      "mountns":     {"ro_binds": ["/usr", "/lib"],
                      "rw_binds": ["/var/lib/minagi/workers/x"]},
      "cgroup":      "/sys/fs/cgroup/minagi/unit-abc",
      "no_new_privs": true,
      "home":        "/nonexistent"
    }

Apply order is deliberate:

1. ``cgroup`` — self-register into the worker's cgroup FIRST so every
   later action (and every descendant) is inside the containment unit.
2. ``netns`` / ``mountns`` — namespace creation needs privilege
   (CAP_SYS_ADMIN); done while the launcher still holds it.
3. ``rlimits`` — bounds inherited across exec.
4. ``no_new_privs`` + demotion — setgroups/setgid/setuid LAST: after
   this the process can never regain privilege.
5. ``exec`` — the worker interpreter replaces the launcher image.

Every failure exits 70 (EX_SOFTWARE) with a one-line stderr reason —
the supervisor reads a dead child + message, never a half-isolated
worker.
"""
from __future__ import annotations

import json
import os

_EXIT_FAIL = 70

#: spec fd env var — a small integer naming an inherited, read-only fd
_SPEC_FD_ENV = "MINAGI_LAUNCH_SPEC_FD"


def _fail(msg: str) -> "None":
    try:
        os.write(2, f"worker-launcher: {msg}\n".encode())
    finally:
        os._exit(_EXIT_FAIL)


def _read_spec() -> dict:
    raw = os.environ.get(_SPEC_FD_ENV)
    if not raw:
        _fail(f"{_SPEC_FD_ENV} not set")
    fd = int(raw)
    try:
        chunks = []
        while True:
            b = os.read(fd, 1 << 16)
            if not b:
                break
            chunks.append(b)
        spec = json.loads(b"".join(chunks))
    except Exception as exc:  # noqa: BLE001
        _fail(f"launch spec unreadable: {exc}")
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    if not isinstance(spec, dict) or not spec.get("exec_argv"):
        _fail("launch spec missing exec_argv")
    return spec


def _apply_cgroup(path) -> None:
    """Write this process's pid into the unit cgroup — membership is
    kernel state, so descendants stay accounted even after detach,
    reparenting, or an env-scrubbed exec."""
    try:
        with open(os.path.join(path, "cgroup.procs"), "w") as fh:
            fh.write(str(os.getpid()))
    except OSError as exc:
        _fail(f"cgroup attach failed for {path}: {exc}")


def _apply_netns() -> None:
    """A fresh, empty network namespace: only a down loopback exists —
    no routes, no sockets reach outside. Requires CAP_SYS_ADMIN while
    still privileged (done pre-demotion deliberately)."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    CLONE_NEWNET = 0x40000000
    if libc.unshare(CLONE_NEWNET) != 0:
        err = ctypes.get_errno()
        _fail(f"unshare(CLONE_NEWNET) failed: errno {err}")


def _apply_mountns(spec: dict) -> None:
    """Confinement via a private mount namespace + pivot_root into a
    minimal root: only approved read-only binds plus worker-writable
    scratch exist — the host filesystem is literally not present."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    CLONE_NEWNS = 0x00020000
    MS_BIND, MS_REC, MS_PRIVATE, MS_RDONLY = 4096, 16384, 1 << 18, 1
    MS_REMOUNT, MNT_DETACH = 32, 2

    def _sys(fn, *args):
        if fn(*args) != 0:
            _fail(f"mountns op failed: errno {ctypes.get_errno()}")

    _sys(libc.unshare, CLONE_NEWNS)
    # no propagation from/to the host
    _sys(libc.mount, None, b"/", None, MS_REC | MS_PRIVATE, None)

    import tempfile
    newroot = tempfile.mkdtemp(prefix="minagi-root-")
    binds = []
    for p in spec.get("ro_binds") or ():
        binds.append((os.path.realpath(p), p, True))
    for p in spec.get("rw_binds") or ():
        binds.append((os.path.realpath(p), p, False))
    for src, target, ro in binds:
        inside = os.path.join(newroot, target.lstrip("/"))
        os.makedirs(inside, exist_ok=True)
        flags = MS_BIND | MS_REC | (MS_RDONLY if ro else 0)
        _sys(libc.mount, src.encode(), inside.encode(), None, flags, None)
        if ro:
            # a bind remount is needed to actually flip it read-only
            _sys(libc.mount, None, inside.encode(), None,
                 flags | MS_REMOUNT, None)
    # minimal device nodes the interpreter needs
    os.makedirs(os.path.join(newroot, "dev"), exist_ok=True)
    for dev in ("null", "zero", "urandom"):
        src = f"/dev/{dev}"
        tgt = os.path.join(newroot, "dev", dev)
        open(tgt, "a").close()
        _sys(libc.mount, src.encode(), tgt.encode(), None, MS_BIND, None)
    old = os.path.join(newroot, ".oldroot")
    os.makedirs(old, exist_ok=True)
    _sys(libc.chdir, newroot.encode())
    _sys(libc.syscall, 155, newroot.encode(), old.encode())  # pivot_root
    _sys(libc.umount2, b"/.oldroot", MNT_DETACH)
    try:
        os.rmdir("/.oldroot")
    except OSError:
        pass


def _apply_rlimits(rl: dict) -> None:
    import resource
    table = {"cpu": "RLIMIT_CPU", "as": "RLIMIT_AS", "nproc":
             "RLIMIT_NPROC", "nofile": "RLIMIT_NOFILE",
             "fsize": "RLIMIT_FSIZE"}
    for key, name in table.items():
        if key in rl and rl[key] is not None and \
                hasattr(resource, name):
            v = int(rl[key])
            try:
                resource.setrlimit(getattr(resource, name), (v, v))
            except (OSError, ValueError) as exc:
                _fail(f"setrlimit {name}={v} failed: {exc}")


def _apply_no_new_privs() -> None:
    if os.uname().sysname != "Linux":
        return  # prctl is Linux-only; seatbelt covers exec control
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    PR_SET_NO_NEW_PRIVS = 38
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        _fail(f"prctl(NO_NEW_PRIVS) failed: errno {ctypes.get_errno()}")


def _apply_demote(uid: int, gid: int) -> None:
    try:
        os.setgroups([])
        os.setgid(int(gid))
        os.setuid(int(uid))
    except OSError as exc:
        _fail(f"demotion to {uid}:{gid} failed: {exc}")


def main() -> "None":
    spec = _read_spec()
    if spec.get("cgroup"):
        _apply_cgroup(str(spec["cgroup"]))
    if spec.get("netns"):
        _apply_netns()
    if spec.get("mountns"):
        _apply_mountns(spec["mountns"])
    if spec.get("rlimits"):
        _apply_rlimits(spec["rlimits"])
    if spec.get("no_new_privs"):
        _apply_no_new_privs()
    if spec.get("uid") is not None:
        _apply_demote(spec["uid"], spec.get("gid", spec["uid"]))
        if spec.get("home"):
            os.environ["HOME"] = str(spec["home"])
    if spec.get("cwd"):
        # pivot_root resets the process cwd to "/" — restore the
        # worker scratch inside the new root (its rw bind keeps the
        # original path). Harmless when no mountns ran.
        try:
            os.chdir(str(spec["cwd"]))
        except OSError as exc:
            _fail(f"cwd {spec['cwd']!r} failed: {exc}")
    os.environ.pop(_SPEC_FD_ENV, None)
    argv = [str(a) for a in spec["exec_argv"]]
    try:
        os.execv(argv[0], argv)
    except OSError as exc:
        _fail(f"exec {argv[0]!r} failed: {exc}")


if __name__ == "__main__":
    main()
