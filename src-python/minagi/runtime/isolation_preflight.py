"""v16.4.8 isolation preflight (WP9) — plan Steps 5 and 8.

``validate_isolation`` reports what the host CAN do; preflight
proves what the boundary DOES. It launches a harmless probe worker
through the SAME executor production uses, and the probe reports —
from inside the boundary — its effective identity, capabilities,
namespace membership, filesystem reach, network reach, and whether
a detached child stays inside the containment unit. The supervisor
then independently verifies the unit's cgroup occupancy.

A host lacking real enforcement is refused here, not falsely
qualified — SEC-701's class of bug cannot pass a measured probe.

Probe channel: the probe writes exactly ONE JSON line to stdout
(before any protocol framing exists). ``run_preflight`` reads the
line, then terminates the unit and checks containment from outside.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _capeff_hex() -> str | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("CapEff:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _ns_ids() -> dict:
    out = {}
    nsdir = Path("/proc/self/ns")
    if not nsdir.is_dir():
        return out
    for entry in ("mnt", "net", "pid", "ipc", "user", "uts", "cgroup"):
        try:
            out[entry] = os.readlink(nsdir / entry)
        except OSError:
            out[entry] = None
    return out


def _probe_read(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            fh.read(16)
        return "read-ok"
    except PermissionError:
        return "denied"
    except OSError:
        return "missing"


def _probe_write(path: str) -> str:
    try:
        probe = Path(path) / f".preflight-{os.getpid()}"
        probe.write_text("x")
        probe.unlink()
        return "write-ok"
    except PermissionError:
        return "denied"
    except OSError:
        return "denied"


def _probe_net(addr: str) -> str:
    import socket
    host, _, port = addr.rpartition(":")
    try:
        s = socket.create_connection((host, int(port)), timeout=3.0)
        s.close()
        return "connect-ok"
    except OSError:
        return "connect-denied"
    except Exception:  # noqa: BLE001
        return "connect-error"


def _probe_detach(marker_path: str) -> str:
    """Fork a child that setsid-detaches, writes its pid to the
    marker, and sleeps briefly — the supervisor checks the unit's
    cgroup from outside to prove the detached child stayed owned."""
    try:
        pid = os.fork()
    except OSError:
        return "fork-denied"
    if pid == 0:
        try:
            os.setsid()
        except OSError:
            pass
        try:
            # pid is NAMESPACE-scoped — the host cannot map it. The
            # kernel-reported cgroup line is the transferable proof:
            # it names the unit the supervisor will verify.
            cg = Path("/proc/self/cgroup").read_text().strip()
            Path(marker_path).write_text(
                f"{os.getpid()}\n{cg}\n")
        except OSError:
            pass
        time.sleep(2.0)
        os._exit(0)
    return "forked"


def _probe_main(argv) -> int:
    """Runs INSIDE the worker. Emits one JSON line of facts."""
    scratch = os.environ.get("MINAGI_PROBE_SCRATCH") or "/work"
    # The launch broker's probe argv is FIXED (module + --probe only)
    # — the caller's probe specs travel through the allowlisted env.
    extra = os.environ.get("MINAGI_PROBE_ARGS")
    if extra and not argv:
        try:
            argv = list(json.loads(extra))
        except ValueError:
            argv = []
    report = {
        "euid": os.geteuid(), "egid": os.getegid(),
        "pid": os.getpid(), "ppid": os.getppid(),
        "capeff": _capeff_hex(),
        "namespaces": _ns_ids(),
        "home": os.environ.get("HOME"),
        "fs": {},
        "scratch_write": _probe_write(scratch),
        "detached_child": None,
        "net": {},
    }
    for spec in argv:
        if spec.startswith("deny:"):
            report["fs"][spec[5:]] = _probe_read(spec[5:])
        elif spec.startswith("read:"):
            report["fs"][spec[5:]] = _probe_read(spec[5:])
        elif spec.startswith("write:"):
            report["fs"][spec[6:]] = _probe_write(spec[6:])
        elif spec.startswith("net:"):
            report["net"][spec[4:]] = _probe_net(spec[4:])
        elif spec.startswith("detach:"):
            report["detached_child"] = _probe_detach(spec[7:])
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    sys.stdout.flush()
    # Hold briefly: the supervisor must find this unit's LIVE cgroup —
    # an instantly-exiting probe races systemd's registration window.
    try:
        hold = float(os.environ.get("MINAGI_PROBE_HOLD") or 2.0)
    except ValueError:
        hold = 2.0
    time.sleep(min(max(hold, 0.0), 10.0))
    return 0


def run_preflight(executor, *, isolation, probe_args=(),
                  worker_id=None, timeout: float = 60.0,
                  build_env_fn=None, expect=None) -> dict:
    """Launch the probe through the production executor and grade
    the boundary it actually ran inside.

    probe_args    ``read:<path>`` / ``write:<dir>`` / ``net:<host:port>``
                  / ``detach:<marker-path>`` specs — the caller supplies
                  the protected paths to prove (synthetic secrets, the
                  authority dir, a host listener, scratch)
    build_env_fn  optional callable(dirs) -> env dict; default builds
                  a minimal env naming the scratch dir
    returns a report: {qualified, checks, probe, containment}
    """
    from .worker_isolation import WorkerIsolationPolicy

    isolation = isolation or WorkerIsolationPolicy()
    probe_args = list(probe_args)
    # The supervisor's own namespace ids are the host's — captured
    # BEFORE the launch so the grader can require the worker to have
    # landed in genuinely different namespaces.
    host_ns = _ns_ids()
    # Expected identity: the executor knows the worker uid/gid it
    # maps into the jail; the supervisor's own euid defines the
    # boundary the worker must be distinct from.
    expect = dict(expect or {})
    broker = getattr(executor, "broker", None)
    if broker is not None:
        expect.setdefault("euid", broker.worker_uid)
        expect.setdefault("egid", broker.worker_gid)
        expect.setdefault("identity_distinct", True)
    expect.setdefault("supervisor_euid", os.geteuid())
    dirs = executor.prepare(worker_id=worker_id or "preflight",
                            isolation=isolation)
    env = (build_env_fn(dirs) if build_env_fn else {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
        "MINAGI_PROBE_SCRATCH": "/work"})
    # The writable location differs per executor: the jail maps the
    # workspace scratch to /work; the development executor's scratch
    # is a private host dir.
    env["MINAGI_PROBE_SCRATCH"] = (
        "/work" if getattr(executor, "name", "") == "nsjail"
        else str(dirs.private_tmp))
    env["MINAGI_PROBE_ARGS"] = json.dumps(list(probe_args))
    argv = [_python(), "-m", "minagi.runtime.isolation_preflight",
            "--probe", *probe_args]
    launched = None
    line = None
    try:
        launched = executor.launch(
            dirs, worker_argv=argv, env=env, isolation=isolation,
            role="probe")
        deadline = time.monotonic() + float(timeout)
        buf = b""
        # select-gated reads: a hung probe cannot outwait the deadline
        # by blocking on an empty pipe.
        while time.monotonic() < deadline:
            import select as _sel
            left = deadline - time.monotonic()
            if launched.rx is None:
                break
            ready, _, _ = _sel.select([launched.rx], [], [],
                                      max(0.0, left))
            if not ready:
                break
            chunk = os.read(launched.rx.fileno(), 65536)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                line = buf.split(b"\n", 1)[0].strip()
                break
        probe = json.loads(line) if line else {"error": "no probe "
                                               "report received"}
    finally:
        containment = (executor.terminate(launched,
                                          deadline_seconds=10.0)
                       if launched is not None else None)
        if launched is not None:
            executor.release(launched)
    checks = _grade(probe if isinstance(probe, dict) else {},
                    expect=expect, host_ns=host_ns,
                    probe_args=probe_args)
    checks["containment_verified"] = bool(
        containment and containment.verified)
    qualified = all(v for k, v in checks.items())
    return {"qualified": qualified, "checks": checks,
            "probe": probe,
            "containment": (containment.mechanism
                            if containment else None)}


# Namespaces a real boundary MUST give the worker — if the host can
# name them, the probe's must differ. Missing probe entries fail.
_REQUIRED_NS = ("mnt", "net", "pid", "user")


def _grade(probe: dict, *, expect: dict, host_ns: dict,
           probe_args) -> dict:
    """The caller's declared expectations meet the worker's measured
    reality. EVERY mandatory check must be present and passing —
    absent evidence (a missing capeff, an unreported namespace, a
    probe spec the worker never answered) is a FAILED check, never an
    implicit pass."""
    checks = {}
    checks["scratch_writable"] = (
        probe.get("scratch_write") == "write-ok")

    # capabilities — a missing report is not "zero caps"
    try:
        checks["no_caps"] = int(probe["capeff"], 16) == 0
    except (KeyError, TypeError, ValueError):
        checks["no_caps"] = False

    # identity — the in-jail ids must be exactly the approved worker
    # ids and distinct from the supervisor's
    if expect.get("euid") is not None:
        checks["worker_euid"] = probe.get("euid") == expect["euid"]
    if expect.get("egid") is not None:
        checks["worker_egid"] = probe.get("egid") == expect["egid"]
    if expect.get("identity_distinct"):
        checks["distinct_worker_identity"] = (
            probe.get("euid") is not None
            and probe.get("euid") != expect.get("supervisor_euid"))

    # namespaces — measured ids must differ from the host's own
    pns = probe.get("namespaces") or {}
    host_ns = host_ns or {}
    for name in _REQUIRED_NS:
        h = host_ns.get(name)
        if h is None:
            continue  # the host cannot name it — nothing to compare
        checks[f"ns:{name}"] = bool(pns.get(name)) and pns.get(name) != h
    # cgroup is absent by design: NsJail delegates a cgroup subtree
    # rather than unsharing a cgroup namespace, so the probe's cgroup
    # ns legitimately equals the host's. Containment proof there is
    # the unit's cgroup membership (containment_verified), not the ns
    # inode.
    for name in ("ipc", "uts"):
        h = host_ns.get(name)
        w = pns.get(name)
        if h is not None and w is not None:
            checks[f"ns:{name}"] = w != h

    # probe specs — each spec carries its OWN acceptance rule:
    #   deny:<path>   the worker must find it unreachable
    #   read:<path>   positive control — must actually be readable
    #   write:<dir>   must be writable
    #   net:<a:p>     the connection must be refused
    #   detach:<p>    the fork must have happened
    fs = probe.get("fs") or {}
    net = probe.get("net") or {}
    for spec in probe_args or ():
        kind, _, target = str(spec).partition(":")
        if kind == "deny":
            checks[f"deny:{target}"] = fs.get(target) in (
                "denied", "missing")
        elif kind == "read":
            checks[f"read:{target}"] = fs.get(target) == "read-ok"
        elif kind == "write":
            checks[f"write:{target}"] = fs.get(target) == "write-ok"
        elif kind == "net":
            checks[f"net:{target}"] = net.get(target) == "connect-denied"
        elif kind == "detach":
            checks["detached_child_forked"] = (
                probe.get("detached_child") == "forked")
    return checks


def _python() -> str:
    return sys.executable


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--probe":
        return _probe_main(argv[1:])
    sys.stderr.write(
        "isolation_preflight: launch probes through an executor; "
        "--probe runs inside the worker\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_preflight"]
