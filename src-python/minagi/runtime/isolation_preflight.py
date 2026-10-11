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
                  build_env_fn=None) -> dict:
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
    checks = _grade(probe if isinstance(probe, dict) else {})
    checks["containment_verified"] = bool(
        containment and containment.verified)
    qualified = all(v for k, v in checks.items())
    return {"qualified": qualified, "checks": checks,
            "probe": probe,
            "containment": (containment.mechanism
                            if containment else None)}


def _grade(probe: dict) -> dict:
    """Turn probe facts into pass/fail checks — the caller's declared
    expectations meet the worker's measured reality here. Raw fs
    outcomes stay in report["probe"]["fs"] for the caller's own
    allowed/denied assertions."""
    checks = {}
    checks["scratch_writable"] = probe.get("scratch_write") == "write-ok"
    net = probe.get("net") or {}
    for addr, outcome in net.items():
        # every probed address must be unreachable under network=deny
        checks[f"net:{addr}"] = outcome == "connect-denied"
    if probe.get("capeff") is not None:
        try:
            checks["no_caps"] = int(probe["capeff"] or "0", 16) == 0
        except ValueError:
            checks["no_caps"] = False
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
