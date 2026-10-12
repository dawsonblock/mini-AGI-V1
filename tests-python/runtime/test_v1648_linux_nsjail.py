"""v16.4.8 — REAL NsJail boundary tests (Linux + systemd + cgroup v2
only). These tests are the G1–G4 qualification evidence: a real
worker inside an OS-enforced isolation boundary, exercised
adversarially. They SKIP honestly anywhere else — a green run on
macOS is not a claim of Linux qualification.

Run on the qualification host:

    PYTHONPATH=src-python:tests-python/helpers \
        python3 -m pytest tests-python/runtime/test_v1648_linux_nsjail.py -v

Environment:
    MINAGI_NSJAIL          nsjail binary path (default
                           /usr/local/bin/nsjail)
    MINAGI_WORKER_UID/GID  separate worker identity (default: euid —
                           user-namespace mapping still confines)
    MINAGI_USER_MANAGER=1  force systemd --user (rootless) launch
"""
import hashlib
import json
import os
import secrets
import shutil
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "tests-python" / "helpers"))

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="NsJail/systemd boundary tests run only on Linux")

from minagi.runtime.isolation_profile import load_profile  # noqa: E402
from minagi.runtime.nsjail_executor import (  # noqa: E402
    NsJailWorkerExecutor)
from minagi.runtime.worker_backend import (  # noqa: E402
    BackendSpec, WorkerBackend)
from minagi.runtime.worker_isolation import (  # noqa: E402
    DEFAULT_ISOLATION)

PROFILE_PATH = (ROOT / "configs" / "isolation"
                / "linux-production-offline-v1.json")

PROBE_SPEC = ("evil_backends", "AccessProbeBackend")
SPAWN_SPEC = ("evil_backends", "SpawnBackend")


def _nsjail():
    path = os.environ.get("MINAGI_NSJAIL") or "/usr/local/bin/nsjail"
    if not Path(path).is_file():
        return None
    return path, "sha256:" + hashlib.sha256(
        Path(path).read_bytes()).hexdigest()




def _traversable(path):
    """The userns-mapped worker uid must traverse every ancestor of
    the bind-mounted scratch — chmod o+x up the chain (traverse-only,
    listing still denied)."""
    p = Path(path).resolve()
    for d in (p, *p.parents):
        try:
            os.chmod(d, os.stat(d).st_mode | 0o001)
        except OSError:
            break
        if d.parent == d or d.name == "tmp":
            break

def _executor(tmp_path, *, denied=()):
    _traversable(tmp_path)
    found = _nsjail()
    if found is None:
        pytest.skip("nsjail binary not found — set MINAGI_NSJAIL")
    if not (shutil.which("systemd-run") and shutil.which("systemctl")):
        pytest.skip("systemd-run/systemctl not available")
    if not Path("/sys/fs/cgroup/cgroup.controllers").is_file():
        pytest.skip("host is not cgroup v2 (unified hierarchy)")
    binary, digest = found
    euid = os.geteuid()
    rootless = euid != 0 or os.environ.get("MINAGI_USER_MANAGER") == "1"
    ex = NsJailWorkerExecutor.from_profile(
        load_profile(PROFILE_PATH),
        worker_state_root=tmp_path / "workers",
        nsjail_path=binary, nsjail_digest=digest,
        worker_uid=int(os.environ.get("MINAGI_WORKER_UID")
                       or (65534 if euid == 0 else euid)),
        worker_gid=int(os.environ.get("MINAGI_WORKER_GID")
                       or os.environ.get("MINAGI_WORKER_UID")
                       or (65534 if euid == 0 else os.getegid())),
        user_manager=rootless,
        python_path_entries=tuple(
            str(p) for p in sys.path if p),
        denied_paths=tuple(str(p) for p in denied))
    report = ex.validate_host()
    if not report.get("ok"):
        pytest.skip(f"host cannot establish the nsjail boundary: "
                    f"{report}")
    return ex


def _snapshot(tmp_path, tag="m"):
    from egai.common.canonical import digest
    from minagi.v161.artifact_closure import close_tree
    from minagi.v161.immutable_snapshot import stage_snapshot
    model = tmp_path / f"model-{tag}"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text('{"model_type": "sim"}')
    adir = tmp_path / f"adapter-{tag}"
    adir.mkdir(exist_ok=True)
    (adir / "adapter_model.safetensors").write_bytes(b"w" + tag.encode())
    digests = {"model": close_tree(model).digest,
               "adapter": close_tree(adir).digest}
    return stage_snapshot(
        tmp_path / "snap" / tag,
        {"model": str(model), "adapter": str(adir)},
        expected_digests=digests,
        manifest_digest=digest({"m": tag}))


def _backend(spec, ex, **kw):
    kw.setdefault("backend_manifest_digest", "sha256:" + "d" * 64)
    return WorkerBackend(
        BackendSpec(module=spec[0], qualname=spec[1], kwargs={}),
        backend_id="hf-peft", executor=ex,
        start_timeout=120.0, probe_timeout=60.0,
        shutdown_grace=10.0, **kw)


# ------------------------------------------------------------- G1

def test_g1_probe_runs_inside_jail(tmp_path):
    """The probe's own syscalls report the jailed identity and a
    fresh pid namespace — G1 evidence."""
    from minagi.runtime.isolation_preflight import run_preflight
    secret = tmp_path / "authority-secret"
    secret.mkdir()
    (secret / "key.pem").write_text("SYNTHETIC")
    ex = _executor(tmp_path, denied=(str(secret),))
    report = run_preflight(
        ex, isolation=DEFAULT_ISOLATION, timeout=120.0,
        probe_args=[f"read:{secret / 'key.pem'}",
                    f"read:{secret}",
                    "net:127.0.0.1:9"])
    probe = report["probe"]
    assert probe["fs"][str(secret / "key.pem")] in (
        "denied", "missing")
    assert probe["fs"][str(secret)] in ("denied", "missing")
    assert probe["net"].get("127.0.0.1:9") == "connect-denied"
    assert probe["namespaces"]["pid"] != os.readlink(
        "/proc/self/ns/pid")
    assert report["checks"]["containment_verified"] is True


def test_g1_authority_dir_never_mounted(tmp_path):
    """G2: a synthetic supervisor secret the worker must never see."""
    from minagi.runtime.isolation_preflight import run_preflight
    auth = tmp_path / "authority"
    auth.mkdir()
    (auth / "promotion.key").write_text("SYNTHETIC-KEY")
    ex = _executor(tmp_path, denied=(str(auth),))
    report = run_preflight(
        ex, isolation=DEFAULT_ISOLATION, timeout=120.0,
        probe_args=[f"read:{auth}", f"read:{auth / 'promotion.key'}",
                    "read:/etc/shadow", "read:/root"])
    fs = report["probe"]["fs"]
    for p in (str(auth), str(auth / "promotion.key")):
        assert fs[p] in ("denied", "missing"), (p, fs[p])
    # even host-world-readable-but-sensitive files are unmounted
    assert fs["/root"] in ("denied", "missing")


def test_g3_network_denied(tmp_path):
    """G3: outbound connections fail — even to a live local
    listener — because the jail owns an empty net namespace."""
    import socket
    import threading
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    threading.Thread(target=lambda: srv.accept(),
                     daemon=True).start()
    try:
        from minagi.runtime.isolation_preflight import run_preflight
        ex = _executor(tmp_path)
        report = run_preflight(
            ex, isolation=DEFAULT_ISOLATION, timeout=120.0,
            probe_args=[f"net:127.0.0.1:{port}", "net:1.1.1.1:443"])
        net = report["probe"]["net"]
        assert net[f"127.0.0.1:{port}"] == "connect-denied"
        assert net["1.1.1.1:443"] == "connect-denied"
    finally:
        srv.close()


# ------------------------------------------------------------- G4

def test_g4_detached_descendant_stays_owned(tmp_path):
    """G4: a child that setsid()-detaches inside the jail remains in
    the unit's cgroup — visible in cgroup.procs from the host —
    and termination empties the unit (cgroup.populated == 0)."""
    ex = _executor(tmp_path)
    dirs = ex.prepare(worker_id="g4", isolation=DEFAULT_ISOLATION)
    env = {"MINAGI_PROBE_ARGS": json.dumps(
        ["detach:/work/marker.pid"])}
    launched = ex.launch(
        dirs,
        worker_argv=[sys.executable, "-m",
                     "minagi.runtime.isolation_preflight", "--probe"],
        env=env, isolation=DEFAULT_ISOLATION, role="probe")
    try:
        # the detached child writes its pid into the scratch mount —
        # visible to the host because the broker owns the workspace
        marker = dirs.private_tmp / "marker.pid"
        deadline = time.monotonic() + 30.0
        cg_line = None
        while time.monotonic() < deadline:
            if marker.exists():
                lines = marker.read_text().splitlines()
                if len(lines) >= 2:
                    cg_line = lines[1]
                    break
            time.sleep(0.1)
        assert cg_line, "detached child never reported its cgroup"
        # the pid is namespace-scoped — unmatchable from the host. The
        # child's OWN /proc/self/cgroup is the kernel's statement of
        # ownership: it must name this unit's cgroup.
        cgroup_path = launched.cgroup_path or ""
        unit_leaf = cgroup_path.rsplit("/", 1)[-1]
        assert unit_leaf and unit_leaf in cg_line, (
            f"detached child's cgroup {cg_line!r} does not name the "
            f"unit cgroup {unit_leaf!r}")
        # and the live process list still carries the whole tree
        if cgroup_path:
            procs = Path(cgroup_path, "cgroup.procs")
            if procs.is_file():
                assert procs.read_text().split(), (
                    f"unit cgroup {cgroup_path} reported empty while "
                    "a detached child was alive")
    finally:
        rep = ex.terminate(launched, deadline_seconds=30.0)
        ex.release(launched)
    assert rep.verified, f"containment not proven empty: {rep.detail}"


def test_g4_spawn_backend_children_all_terminated(tmp_path):
    """A real worker that spawns a setsid escapee AND a SIGTERM-
    ignoring straggler is fully reaped by unit teardown."""
    ex = _executor(tmp_path)
    b = _backend(SPAWN_SPEC, ex)
    h = b.load(_snapshot(tmp_path))
    kids = []
    for kw in ({"setsid": True}, {"ignore_term": True}, {}):
        out = b.infer(h, {"op": "spawn", **kw})
        kids.append(out["pid"])
    assert len(kids) == 3
    record = h.launched.record
    b.terminate(h)
    st = ex.broker.status(record)
    assert st["cgroup_populated"] in (False, None), (
        f"unit still populated after terminate: {st}")
    for pid in kids:
        assert not Path(f"/proc/{pid}").exists() or \
            Path(f"/proc/{pid}/stat").read_text().split(
                ")", 1)[1].split()[2] == "Z"


# ------------------------------------------------- lifecycle

def test_worker_backend_load_infer_unload(tmp_path):
    """Phase-C gate: the real worker entry boots inside the jail,
    exchanges validated frames, and unload proves the unit empty."""
    ex = _executor(tmp_path)
    b = _backend(PROBE_SPEC, ex)
    h = b.load(_snapshot(tmp_path))
    out = b.infer(h, {"paths": ["/etc/hostname"],
                      "write_paths": ["/work/out.txt"]})
    assert out["read"]["/etc/hostname"] in (
        "READABLE", "denied", "missing") or \
        out["read"]["/etc/hostname"].startswith("error:")
    assert out["write"]["/work/out.txt"] == "WRITABLE"
    record = h.launched.record
    b.unload(h)
    st = ex.broker.status(record)
    assert st["cgroup_populated"] in (False, None)


def test_worker_cannot_see_supervisor_env(tmp_path):
    """No supervisor secrets in the jailed environment — the env is
    broker-filtered, never inherited."""
    ex = _executor(tmp_path)
    b = _backend(("evil_backends", "EnvDumpBackend"), ex)
    h = b.load(_snapshot(tmp_path))
    out = b.infer(h, {})
    text = json.dumps(out)
    assert "AWS_SECRET" not in text and "_KEY" not in text
    b.unload(h)
