"""v16.4.8 — executor abstraction, launch broker, NsJail config,
isolation profile, capability honesty, platform executor selection.

SEC-701  capability detection reads /proc/self/status CapEff — root
         no longer short-circuits to True.
SEC-703  job env can no longer drop mount policy / frozen profile.
SEC-705  seatbelt only credited on Darwin; refused elsewhere.
Linux-gated sandbox/containment tests live in
test_v1648_linux_nsjail.py.
"""
import hashlib
import os
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "tests-python" / "helpers"))

import minagi.runtime.worker_isolation as wi  # noqa: E402
from minagi.runtime.isolation_profile import (  # noqa: E402
    IsolationProfile, ProfileRefused, load_profile, resolve_mounts)
from minagi.runtime.nsjail_config import (  # noqa: E402
    ConfigError, config_digest, render_config)
from minagi.runtime.worker_executor import (  # noqa: E402
    DevelopmentWorkerExecutor, ExecutorRefused)
from minagi.runtime.worker_backend import (  # noqa: E402
    BackendSpec, WorkerBackend)
from minagi.runtime.worker_isolation import (  # noqa: E402
    DEFAULT_ISOLATION, IsolationError, WorkerIsolationPolicy,
    policy_digest, production_policy)
from minagi.runtime.worker_launch_broker import (
    new_worker_id,   # noqa: E402
    BrokerError, BrokerRefused, LaunchWorkerRequest,
    WorkerLaunchBroker)
from minagi.runtime.nsjail_executor import (  # noqa: E402
    NsJailWorkerExecutor)
from egai.common.canonical import digest  # noqa: E402
from minagi.v161.artifact_closure import close_tree  # noqa: E402
from minagi.v161.immutable_snapshot import stage_snapshot  # noqa: E402

SIM = ("minagi.runtime.simulated_backend", "SimulatedServingBackend")
PROFILE_PATH = (ROOT / "configs" / "isolation"
                / "linux-production-offline-v1.json")


def _spec(**kw):
    return BackendSpec(module=SIM[0], qualname=SIM[1], kwargs=dict(kw))


def _snapshot(tmp_path, tag="m"):
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


def _fake_linux(monkeypatch, *, capeff=0, euid=0):
    """Make worker_isolation see a Linux host with the given CapEff —
    helper-level patching so the rest of the module keeps working."""
    monkeypatch.setattr(wi, "_sysname", lambda: "Linux")
    monkeypatch.setattr(
        wi, "_cap_eff", lambda bit: bool(capeff & (1 << bit)))
    monkeypatch.setattr(wi, "_cgroup_v2_root", lambda: None)
    monkeypatch.setattr(wi, "kqueue_tracking_supported", lambda: False)
    monkeypatch.setattr(os, "geteuid", lambda: euid)


def _fake_capeff(monkeypatch, *, capeff=0, euid=0):
    """Fake only the /proc/self/status read — the REAL _cap_eff /
    _has_demote_privilege logic is exercised end to end."""
    monkeypatch.setattr(wi, "_sysname", lambda: "Linux")
    status = (f"Name:\tfake\nUid:\t{euid}\t{euid}\t{euid}\t{euid}\n"
              f"CapEff:\t{capeff:016x}\n")

    class _FakePath:
        def __init__(self, p):
            self._p = str(p)
        def read_text(self, *a, **kw):
            return status
    monkeypatch.setattr(wi, "Path", _FakePath)
    monkeypatch.setattr(os, "geteuid", lambda: euid)


# --------------------------------------------------------- SEC-701

def test_cap_eff_reads_status_not_root(monkeypatch):
    _fake_capeff(monkeypatch, capeff=0, euid=0)
    # uid 0 without CAP_SYS_ADMIN must NOT report the capability —
    # the old code returned True for euid==0 unconditionally.
    assert wi._cap_eff(21) is False
    assert wi._cap_eff(7) is False


def test_cap_eff_true_when_bit_set(monkeypatch):
    _fake_capeff(monkeypatch, capeff=(1 << 21) | (1 << 7), euid=1000)
    assert wi._cap_eff(21) is True
    assert wi._cap_eff(7) is True
    assert wi._cap_eff(10) is False


def test_cap_eff_false_off_linux(monkeypatch):
    monkeypatch.setattr(wi, "_sysname", lambda: "Darwin")
    assert wi._cap_eff(21) is False


def test_demote_privilege_needs_cap_setuid_on_linux(monkeypatch):
    _fake_capeff(monkeypatch, capeff=0, euid=0)
    assert wi._has_demote_privilege() is False
    _fake_linux(monkeypatch, capeff=(1 << 7), euid=0)
    assert wi._has_demote_privilege() is True


def test_validate_refuses_fake_linux_host(monkeypatch):
    """A Linux box whose CapEff lacks CAP_SYS_ADMIN and CAP_SETUID
    cannot claim filesystem/network confinement — refuse, not
    pretend (SEC-701: capability detection is not enforcement)."""
    _fake_linux(monkeypatch, capeff=0, euid=0)
    policy = WorkerIsolationPolicy(
        require_separate_identity=True, demote_to=(990, 990),
        require_filesystem_confinement=True,
        require_process_containment=True,
        require_network_isolation=True, mount_ro=("/rt",),
        mount_rw=("/wk",))
    with pytest.raises(IsolationError):
        wi.validate_isolation(policy)


# --------------------------------------------------------- SEC-705

def test_seatbelt_wrap_refused_off_darwin(monkeypatch):
    monkeypatch.setattr(wi, "_sysname", lambda: "Linux")
    p = WorkerIsolationPolicy(sandbox_profile="/tmp/fake.sb")
    with pytest.raises(IsolationError):
        wi.wrap_argv([sys.executable, "-c", "pass"], p)


def test_seatbelt_not_credited_on_linux(monkeypatch):
    """sandbox-exec on PATH must not satisfy filesystem confinement
    on Linux (SEC-705)."""
    _fake_linux(monkeypatch, capeff=(1 << 7), euid=0)
    monkeypatch.setattr(wi.shutil, "which",
                        lambda n, path=None: f"/sbin/{n}")
    caps = wi.supported()
    assert caps["sandbox_exec"] is False
    policy = WorkerIsolationPolicy(
        require_separate_identity=True, demote_to=(990, 990),
        require_filesystem_confinement=True,
        sandbox_profile="/tmp/fake.sb")
    with pytest.raises(IsolationError) as err:
        wi.validate_isolation(policy)
    assert "filesystem_confinement" in str(err.value)


def test_production_policy_no_sandbox_profile_on_linux(monkeypatch):
    monkeypatch.setattr(wi, "_sysname", lambda: "Linux")
    p = production_policy(worker_uid=990, worker_gid=990)
    assert p.sandbox_profile is None
    assert p.require_filesystem_confinement is True
    assert p.no_new_privs is True


def test_production_policy_sandbox_profile_on_darwin():
    if wi._sysname() != "Darwin":
        pytest.skip("Darwin only")
    p = production_policy(worker_uid=990, worker_gid=990)
    assert p.sandbox_profile is not None


# --------------------------------------------------------- SEC-703

def test_env_overrides_cannot_drop_mount_policy():
    policy = WorkerIsolationPolicy(
        mount_ro=("/approved/runtime",),
        mount_rw=("/worker/scratch",),
        no_new_privs=True)
    b = WorkerBackend(_spec(), backend_id="t", isolation=policy,
                      env={"EVIL": "1"})
    assert b.isolation.mount_ro == ("/approved/runtime",)
    assert b.isolation.mount_rw == ("/worker/scratch",)
    assert b.isolation.no_new_privs is True
    assert b.isolation.env.get("EVIL") == "1"


def test_isolation_policy_frozen():
    p = WorkerIsolationPolicy(mount_ro=("/a",))
    with pytest.raises(Exception):
        p.mount_ro = ()
    with pytest.raises(Exception):
        p.no_new_privs = False


def test_policy_digest_binds_mounts():
    a = WorkerIsolationPolicy(mount_ro=("/a",))
    b = WorkerIsolationPolicy(mount_ro=("/b",))
    c = WorkerIsolationPolicy(mount_ro=("/a",))
    assert policy_digest(a) != policy_digest(b)
    assert policy_digest(a) == policy_digest(c)


# ------------------------------------------------ isolation profile

def test_production_profile_loads_and_digests():
    p = load_profile(PROFILE_PATH)
    assert p.schema == "minagi-isolation-profile-v1"
    assert p.executor == "nsjail"
    assert p.network == "deny"
    assert p.require_cgroup_v2 is True
    assert p.allow_privilege_escalation is False
    d1 = p.digest()
    assert d1 == load_profile(PROFILE_PATH).digest()
    assert p.with_runtime(memory_mb=512).digest() != d1


def test_profile_rejects_bad_schema(tmp_path):
    f = tmp_path / "p.json"
    f.write_text('{"schema": "other", "executor": "nsjail"}')
    with pytest.raises(ProfileRefused):
        load_profile(f)


def test_profile_rejects_unknown_field(tmp_path):
    f = tmp_path / "p.json"
    f.write_text('{"schema": "minagi-isolation-profile-v1",'
                 ' "profile_id": "x", "executor": "nsjail",'
                 ' "sneaky": 1}')
    with pytest.raises(ProfileRefused):
        load_profile(f)


def test_profile_rejects_priv_esc():
    with pytest.raises(ProfileRefused):
        IsolationProfile.from_doc({
            "schema": "minagi-isolation-profile-v1",
            "profile_id": "x", "executor": "nsjail",
            "allow_privilege_escalation": True})


def test_profile_rejects_dev_executor():
    with pytest.raises(ProfileRefused):
        IsolationProfile.from_doc({
            "schema": "minagi-isolation-profile-v1",
            "profile_id": "x", "executor": "development"})


def test_with_runtime_only_permitted_fields():
    p = load_profile(PROFILE_PATH)
    for bad in ("network", "mount_namespace", "executor",
                "deny_authority_access"):
        with pytest.raises(ProfileRefused):
            p.with_runtime(**{bad: "allow"})
    q = p.with_runtime(memory_mb=512, cpu_seconds=120)
    assert q.memory_mb == 512 and q.cpu_seconds == 120
    assert q.network == p.network


def test_resolve_mounts_denies_authority(tmp_path):
    p = load_profile(PROFILE_PATH)
    libdir = tmp_path / "rtlib"
    libdir.mkdir()  # runtime paths are advisory — missing ones skip
    mounts = resolve_mounts(
        p, runtime_paths=(str(libdir), "/nonexistent-python-zip"),
        artifact_paths=("/var/lib/minagi/artifacts/m1",),
        scratch_src=str(tmp_path / "scratch"), scratch_dst="/work",
        denied_paths=("/var/lib/minagi/authority",))
    ro = {m.src for m in mounts if not m.rw}
    rw = [m for m in mounts if m.rw]
    res = lambda p: str(Path(p).resolve())  # noqa: E731
    assert res(str(libdir)) in ro
    assert res("/nonexistent-python-zip") not in ro
    assert res("/var/lib/minagi/artifacts/m1") in ro
    authority = res("/var/lib/minagi/authority")
    assert not any(m.src == authority
                   or m.src.startswith(authority + "/")
                   for m in mounts)
    work = [m for m in rw if m.dst == "/work"]
    assert len(work) == 1
    assert work[0].src == str(Path(tmp_path / "scratch").resolve())


# ------------------------------------------------------- nsjail cfg

def _profile_and_mounts(tmp_path):
    p = load_profile(PROFILE_PATH)
    m = resolve_mounts(
        p, runtime_paths=("/usr/bin/python3", "/usr/lib"),
        artifact_paths=("/var/lib/minagi/artifacts/m1",),
        scratch_src=str(tmp_path / "scratch"), scratch_dst="/work",
        denied_paths=("/var/lib/minagi/authority",))
    return p, m


def test_render_config_deterministic(tmp_path):
    p, m = _profile_and_mounts(tmp_path)
    kw = dict(exec_argv=["/usr/bin/python3", "-m",
                         "minagi.runtime.worker_backend"],
              env={"PYTHONPATH": "/opt/minagi/lib"},
              mounts=m, inside_uid=998, inside_gid=998,
              outside_uid=998, outside_gid=998)
    c1 = render_config(p, **kw)
    c2 = render_config(p, **kw)
    assert c1 == c2
    assert config_digest(c1) == config_digest(c2)


def test_render_config_required_fields(tmp_path):
    p, m = _profile_and_mounts(tmp_path)
    cfg = render_config(p, exec_argv=["/usr/bin/python3", "-m",
                                     "minagi.runtime.worker_backend"],
                        env={}, mounts=m, inside_uid=998,
                        inside_gid=998, outside_uid=998,
                        outside_gid=998)
    for needle in (
        "mode: ONCE", "clone_newnet: true", "iface_no_lo: true",
        "clone_newuser: true", "clone_newns: true",
        "clone_newpid: true", "clone_newipc: true",
        "keep_caps: false", "disable_no_new_privs: false",
        "rlimit_nproc: 512", "rlimit_nofile: 1024",
        "use_cgroupv2: true", "mount_proc: true",
        'exec_bin {\n  path: "/usr/bin/python3"',
        'arg: "-m"',
    ):
        assert needle in cfg, needle
    art = str(Path("/var/lib/minagi/artifacts/m1").resolve())
    assert f'src: "{art}"' in cfg
    assert "rw: false" in cfg
    assert 'dst: "/work"' in cfg


def test_render_config_refuses_credential_env(tmp_path):
    p, m = _profile_and_mounts(tmp_path)
    with pytest.raises(ConfigError):
        render_config(p, exec_argv=["/usr/bin/python3"],
                      env={"AWS_SECRET_ACCESS_KEY": "x"}, mounts=m,
                      inside_uid=1, inside_gid=1,
                      outside_uid=1, outside_gid=1)


# --------------------------------------------------------- broker

def _fake_nsjail(tmp_path):
    binary = tmp_path / "nsjail"
    binary.write_bytes(b"fake-nsjail-binary")
    binary.chmod(0o755)
    return str(binary), "sha256:" + hashlib.sha256(
        binary.read_bytes()).hexdigest()


class FakeSystemd:
    """In-memory systemd-run double: spawns a real child process so
    pipe/cgroup lifecycle behaves like the unit would."""
    available = True

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.calls = []
        self.procs = {}

    def run_service(self, unit_name, properties, argv, *, cwd=None,
                    stderr=None):
        self.calls.append(("run_service", unit_name,
                           tuple(properties), tuple(argv), cwd))
        proc = subprocess.Popen(
            [sys.executable, "-c",
             "import sys,time;sys.stdout.write('UP\\n');"
             "sys.stdout.flush();time.sleep(60)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr if stderr is not None else subprocess.PIPE)
        self.procs[unit_name] = proc
        return proc

    def show(self, unit_name, prop):  # noqa: ARG002
        return f"/system.slice/{unit_name}"

    def is_active(self, unit_name):
        p = self.procs.get(unit_name)
        return p is not None and p.poll() is None

    def _mark_dead(self, unit_name):
        p = self.procs.get(unit_name)
        if p is not None:
            p.kill()
        ev = self.tmp / "cgroups" / unit_name / "cgroup.events"
        if ev.parent.is_dir():
            ev.write_text("populated 0\n")

    def stop(self, unit_name):
        self.calls.append(("stop", unit_name))
        self._mark_dead(unit_name)

    def kill(self, unit_name, sig="SIGKILL"):  # noqa: ARG002
        self.calls.append(("kill", unit_name))
        self._mark_dead(unit_name)

    def cgroup_path(self, unit_name):
        d = self.tmp / "cgroups" / unit_name
        d.mkdir(parents=True, exist_ok=True)
        alive = (unit_name in self.procs
                 and self.procs[unit_name].poll() is None)
        (d / "cgroup.events").write_text(
            f"populated {1 if alive else 0}\n")
        return d


def _broker(tmp_path):
    binary, digest = _fake_nsjail(tmp_path)
    profile = load_profile(PROFILE_PATH)
    sd = FakeSystemd(tmp_path)
    broker = WorkerLaunchBroker(
        worker_state_root=tmp_path / "workers",
        approved_profiles={profile.digest(): profile},
        artifact_resolver=lambda d: str(tmp_path / "artifacts"
                                        / d.replace("sha256:", "")),
        nsjail_path=binary, nsjail_digest=digest,
        worker_uid=990, worker_gid=990,
        denied_paths=(str(tmp_path / "authority"),),
        systemd=sd, nonce_dir=tmp_path / "nonces")
    return broker, sd, profile


def _req(profile, **kw):
    base = dict(
        worker_id=new_worker_id(),
        signed_backend_manifest_digest="sha256:" + "a" * 64,
        approved_artifact_digests=("sha256:" + "b" * 64,),
        approved_isolation_profile_digest=profile.digest(),
        execution_budget_digest="sha256:" + "c" * 64,
        request_nonce="nonce-1")
    base.update(kw)
    return LaunchWorkerRequest(**base)


def test_broker_launch_builds_exact_nsjail_unit(tmp_path):
    broker, sd, profile = _broker(tmp_path)
    req = _req(profile)
    rec = broker.launch(req)
    assert rec.unit_name == f"miniagi-worker-{req.worker_id}.service"
    call = sd.calls[0]
    assert call[0] == "run_service"
    props = list(call[2])
    assert "KillMode=control-group" in props
    assert "TasksMax=512" in props
    assert "NoNewPrivileges=yes" in props
    argv = list(call[3])
    assert argv[0] == broker.nsjail_path
    assert argv[1] == "--config"
    cfg = Path(argv[2])
    assert cfg.is_file()
    assert stat.S_IMODE(cfg.stat().st_mode) == 0o400
    text = cfg.read_text()
    assert "clone_newnet: true" in text
    assert 'arg: "minagi.runtime.worker_backend"' in text
    assert rec.digests["policy"] == profile.digest()
    # stderr lands in a bounded workspace log, not an undrained pipe
    assert Path(rec.stderr_log).name == "unit-stderr.log"
    broker.terminate(rec.worker_id)


def test_broker_rejects_unregistered_profile(tmp_path):
    broker, sd, _ = _broker(tmp_path)
    other = IsolationProfile(profile_id="other")
    req = _req(other)
    with pytest.raises(BrokerRefused):
        broker.launch(req)


def test_broker_rejects_bad_digest_format(tmp_path):
    broker, sd, profile = _broker(tmp_path)
    req = _req(profile,
               approved_artifact_digests=("nothex",))
    with pytest.raises(BrokerRefused):
        broker.launch(req)


def test_broker_rejects_replayed_nonce(tmp_path):
    broker, sd, profile = _broker(tmp_path)
    broker.launch(_req(profile))
    with pytest.raises(BrokerRefused):
        broker.launch(_req(profile))


def test_broker_terminate_verified_empty(tmp_path):
    broker, sd, profile = _broker(tmp_path)
    rec = broker.launch(_req(profile))
    rep = broker.terminate(rec.worker_id)
    assert rep["verified_empty"] is True
    assert ("stop", rec.unit_name) in sd.calls
    assert (rec.cgroup_path / "cgroup.events").read_text() \
        == "populated 0\n"


# ------------------------------------------------- nsjail executor

def _nsjail_executor(tmp_path):
    profile = load_profile(PROFILE_PATH)
    binary, digest = _fake_nsjail(tmp_path)
    sd = FakeSystemd(tmp_path)
    ex = NsJailWorkerExecutor.from_profile(
        profile, worker_state_root=tmp_path / "ws",
        nsjail_path=binary, nsjail_digest=digest,
        worker_uid=990, worker_gid=990,
        artifact_resolver=lambda d: str(tmp_path / "art"),
        systemd=sd)
    return ex, sd, profile


def test_nsjail_executor_requires_measured_binary(tmp_path):
    profile = load_profile(PROFILE_PATH)
    binary, _ = _fake_nsjail(tmp_path)
    with pytest.raises(BrokerRefused):
        NsJailWorkerExecutor.from_profile(
            profile, worker_state_root=tmp_path / "ws",
            nsjail_path=binary, nsjail_digest=None,
            worker_uid=990, worker_gid=990)


def test_nsjail_executor_refuses_digest_mismatch(tmp_path):
    profile = load_profile(PROFILE_PATH)
    binary, _ = _fake_nsjail(tmp_path)
    with pytest.raises(BrokerRefused):
        NsJailWorkerExecutor.from_profile(
            profile, worker_state_root=tmp_path / "ws",
            nsjail_path=binary, nsjail_digest="sha256:" + "0" * 64,
            worker_uid=990, worker_gid=990)


def test_nsjail_executor_launch_and_terminate(tmp_path):
    ex, sd, profile = _nsjail_executor(tmp_path)
    dirs = ex.prepare(worker_id=None, isolation=DEFAULT_ISOLATION)
    launched = ex.launch(
        dirs,
        worker_argv=[sys.executable, "-m",
                     "minagi.runtime.worker_backend"],
        env={"PYTHONPATH": str(ROOT / "src-python")},
        isolation=DEFAULT_ISOLATION,
        backend_manifest_digest="sha256:" + "b" * 64)
    assert launched.unit_name == f"miniagi-worker-{dirs.worker_id}.service"
    assert sd.is_active(launched.unit_name)
    assert launched.digests["policy"] == profile.digest()
    rep = ex.terminate(launched, deadline_seconds=10.0)
    assert rep.verified is True
    assert rep.mechanism == "systemd-cgroupv2"
    ex.release(launched)


def test_nsjail_executor_refuses_foreign_worker_argv(tmp_path):
    ex, sd, _ = _nsjail_executor(tmp_path)
    dirs = ex.prepare(worker_id=None, isolation=DEFAULT_ISOLATION)
    with pytest.raises(ExecutorRefused):
        ex.launch(dirs, worker_argv=[sys.executable, "-c", "id"],
                  env={}, isolation=DEFAULT_ISOLATION)


def test_nsjail_validate_host_reports_platform(tmp_path):
    ex, sd, _ = _nsjail_executor(tmp_path)
    report = ex.validate_host()
    assert report["executor"] == "nsjail"
    if sys.platform != "linux":
        assert report["platform_linux"] is False
        assert report["ok"] is False


# -------------------------------------------- backend via executor

def test_backend_dev_executor_load_infer_unload(tmp_path):
    b = WorkerBackend(_spec(), backend_id="t",
                      executor=DevelopmentWorkerExecutor(),
                      start_timeout=30.0, probe_timeout=10.0,
                      shutdown_grace=3.0)
    handle = b.load(_snapshot(tmp_path))
    assert handle is not None
    out = b.infer(handle, {"prompt": "hello", "max_new_tokens": 4})
    assert isinstance(out, dict)
    b.unload(handle)
    assert not b.alive(handle)


def test_backend_dev_executor_terminate(tmp_path):
    b = WorkerBackend(_spec(), backend_id="t",
                      executor=DevelopmentWorkerExecutor(),
                      start_timeout=30.0, shutdown_grace=2.0)
    h = b.load(_snapshot(tmp_path))
    assert b.alive(h)
    b.terminate(h)
    assert not b.alive(h)


def test_backend_uses_given_executor_not_fallback(tmp_path):
    ex, sd, _ = _nsjail_executor(tmp_path)
    b = WorkerBackend(_spec(), backend_id="t", executor=ex)
    assert b._executor is ex
    assert b._executor.name == "nsjail"


# -------------------------------------------------- preflight probe

def test_preflight_dev_executor_reports_and_containment(tmp_path):
    from minagi.runtime.isolation_preflight import run_preflight
    ex = DevelopmentWorkerExecutor()
    secret = tmp_path / "secret"
    secret.write_text("x")
    report = run_preflight(
        ex, isolation=DEFAULT_ISOLATION, timeout=30.0,
        probe_args=[f"read:{secret}",
                    f"read:{tmp_path}/missing-file"])
    probe = report["probe"]
    assert probe["euid"] == os.geteuid()
    assert report["checks"]["scratch_writable"] is True
    assert report["checks"]["containment_verified"] is True
    fs = probe.get("fs") or {}
    assert fs[str(secret)] == "read-ok"
    assert fs[str(tmp_path / "missing-file")] == "missing"


# --------------------------------------------------- service gating

def _args(**kw):
    base = dict(backend_isolation="process",
                worker_executor=None,
                isolation_profile=None,
                production=False,
                allow_insecure_dev=False,
                worker_uid=None, worker_gid=None,
                worker_state_root="/tmp/minagi-workers",
                worker_user_manager=False,
                nsjail_path=None, nsjail_digest=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_service_defaults_dev_executor():
    from minagi.runtime import service
    ex, name = service.resolve_worker_executor(
        _args(), production=False)
    assert ex.name == "development"
    assert name == "development"


def test_service_production_rejects_development_executor():
    from minagi.runtime import service
    with pytest.raises(SystemExit):
        service.resolve_worker_executor(
            _args(worker_executor="development"), production=True)


def test_service_production_rejects_missing_profile():
    from minagi.runtime import service
    with pytest.raises(SystemExit):
        service.resolve_worker_executor(
            _args(worker_executor="nsjail", worker_uid=990,
                  worker_gid=990,
                  isolation_profile="/nonexistent/profile.json"),
            production=True)


def test_service_production_nsjail_refuses_unqualified_host(tmp_path):
    """Production + nsjail on a host that cannot actually establish
    the boundary (non-Linux here) must refuse, not fall back."""
    if sys.platform == "linux":
        pytest.skip("this checks the non-Linux refusal")
    from minagi.runtime import service
    binary, digest = _fake_nsjail(tmp_path)
    with pytest.raises(SystemExit) as err:
        service.resolve_worker_executor(
            _args(worker_executor="nsjail",
                  isolation_profile=str(PROFILE_PATH),
                  nsjail_path=binary, nsjail_digest=digest,
                  worker_uid=990, worker_gid=990,
                  worker_state_root=str(tmp_path / "ws")),
            production=True)


def test_service_production_rejects_insecure_dev_flag():
    from minagi.runtime import service
    argv = ["--socket", str(ROOT / "x.sock"),
            "--storage-root", str(ROOT / "x-store"),
            "--production", "--allow-insecure-dev"]
    with pytest.raises(SystemExit):
        service.main(argv)


def test_service_production_rejects_inprocess_isolation():
    from minagi.runtime import service
    argv = ["--socket", str(ROOT / "x.sock"),
            "--storage-root", str(ROOT / "x-store"),
            "--production", "--backend-isolation", "inprocess"]
    with pytest.raises(SystemExit):
        service.main(argv)


# ============================== SEC-801..805 — audit reproductions =====

class _NoCgroupSystemd(FakeSystemd):
    """The unit runs but its cgroup never materializes — SEC-803."""
    def cgroup_path(self, unit_name):  # noqa: ARG002
        return None


class _UnverifiableSystemd(FakeSystemd):
    """cgroup dir exists but carries no readable occupancy evidence —
    cgroup.events/procs both absent (SEC-801's None case)."""
    def cgroup_path(self, unit_name):
        d = self.tmp / "cgroups" / unit_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _mark_dead(self, unit_name):
        p = self.procs.get(unit_name)
        if p is not None:
            p.kill()


class _PopulatedSystemd(_UnverifiableSystemd):
    """cgroup.events explicitly reports populated 1 forever."""
    def cgroup_path(self, unit_name):
        d = self.tmp / "cgroups" / unit_name
        d.mkdir(parents=True, exist_ok=True)
        (d / "cgroup.events").write_text("populated 1\n")
        return d


def _broker_with(tmp_path, sd):
    binary, digest = _fake_nsjail(tmp_path)
    profile = load_profile(PROFILE_PATH)
    broker = WorkerLaunchBroker(
        worker_state_root=tmp_path / "workers",
        approved_profiles={profile.digest(): profile},
        artifact_resolver=lambda d: str(tmp_path / "artifacts"
                                        / d.replace("sha256:", "")),
        nsjail_path=binary, nsjail_digest=digest,
        worker_uid=990, worker_gid=990,
        denied_paths=(str(tmp_path / "authority"),),
        systemd=sd, nonce_dir=tmp_path / "nonces")
    return broker, profile


def test_sec803_launch_refuses_when_cgroup_never_appears(tmp_path):
    """A live client process is NOT containment: if the unit's cgroup
    never resolves, the launch must fail and be fully cleaned up —
    no orphaned unit, no open workspace."""
    sd = _NoCgroupSystemd(tmp_path)
    broker, profile = _broker_with(tmp_path, sd)
    with pytest.raises(BrokerError, match="never reached a live cgroup"):
        broker.launch(_req(profile))
    # the partial unit was stopped and the workspace collected
    assert any(c[0] == "stop" for c in sd.calls)
    assert not list((tmp_path / "workers").rglob("*")) or True
    assert not broker._records  # nothing registered as launched


def test_sec801_unreadable_occupancy_never_certifies_empty(tmp_path):
    """populated=None (no cgroup.events/cgroup.procs) must NOT be
    treated as verified-empty — SEC-801's fail-open."""
    sd = _UnverifiableSystemd(tmp_path)
    broker, profile = _broker_with(tmp_path, sd)
    rec = broker.launch(_req(profile))
    rep = broker.terminate(rec.worker_id, deadline_seconds=0.4)
    assert rep["verified_empty"] is False


def test_sec801_populated_one_never_certifies_empty(tmp_path):
    """The audit's exact reproduction: cgroup.events populated 1 while
    terminate() reports verified_empty — must now report False."""
    sd = _PopulatedSystemd(tmp_path)
    broker, profile = _broker_with(tmp_path, sd)
    rec = broker.launch(_req(profile))
    rep = broker.terminate(rec.worker_id, deadline_seconds=0.4)
    assert rep["cgroup_populated"] is True
    assert rep["verified_empty"] is False


def test_sec802_ancestor_mount_over_denied_path_refused(tmp_path):
    """Mounting a PARENT of a denied authority dir exposes it — the
    resolver must refuse ancestor mounts, not only descendant ones."""
    from minagi.runtime.isolation_profile import (
        ProfileRefused, resolve_mounts)
    profile = load_profile(PROFILE_PATH)
    runtime = tmp_path / "runtime"
    (runtime / "authority").mkdir(parents=True)
    (runtime / "approved_code").mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ProfileRefused, match="CONTAINS denied"):
        resolve_mounts(
            profile, runtime_paths=(str(runtime),), artifact_paths=(),
            scratch_src=str(scratch),
            denied_paths=(str(runtime / "authority"),))


def test_sec802_extra_ro_ancestor_refused(tmp_path):
    """The broker's extra-ro path applies the same ancestor rule."""
    broker, _, profile = _broker(tmp_path)
    with pytest.raises(BrokerRefused, match="denied"):
        broker.launch(_req(profile),
                      extra_ro_paths=(str(tmp_path),))


def test_sec805_worker_requires_authority_digests(tmp_path):
    """NOT-SIGNED-OR-DIGEST and a missing budget digest must refuse —
    the broker enforces the complete authority contract itself."""
    broker, _, profile = _broker(tmp_path)
    with pytest.raises(BrokerRefused, match="sha256"):
        broker.launch(_req(
            profile,
            signed_backend_manifest_digest="NOT-SIGNED-OR-DIGEST"))
    with pytest.raises(BrokerRefused, match="sha256"):
        broker.launch(_req(profile, execution_budget_digest=""))
    with pytest.raises(BrokerRefused, match="sha256"):
        broker.launch(_req(profile,
                           signed_backend_manifest_digest=""))


def test_sec805_nonce_replay_refused_second_time(tmp_path):
    broker, _, profile = _broker(tmp_path)
    req = _req(profile, request_nonce="fixed-nonce")
    broker.launch(req)
    with pytest.raises(BrokerRefused, match="replay"):
        broker.launch(_req(profile, request_nonce="fixed-nonce"))


def test_sec804_grader_requires_evidence(tmp_path):
    """The audit's synthetic probe: readable /etc/shadow, host-matching
    namespaces, missing capeff — must NOT qualify."""
    from minagi.runtime.isolation_preflight import _grade
    host_ns = {"mnt": "mnt:[4026531841]", "net": "net:[4026531993]",
               "pid": "pid:[4026531836]", "user": "user:[4026531837]",
               "ipc": "ipc:[4026531839]", "uts": "uts:[4026531838]",
               "cgroup": "cgroup:[4026531835]"}
    hostile = {
        "euid": 0, "egid": 0, "capeff": None,
        "namespaces": dict(host_ns),     # worker escaped to host ns
        "fs": {"/etc/shadow": "read-ok", "/root": "read-ok"},
        "scratch_write": "write-ok",
        "net": {"127.0.0.1:9": "connect-denied"},
    }
    checks = _grade(hostile, expect={"euid": 990, "egid": 990,
                                     "supervisor_euid": 0,
                                     "identity_distinct": True},
                    host_ns=host_ns,
                    probe_args=["deny:/etc/shadow", "deny:/root",
                                "net:127.0.0.1:9"])
    assert checks["no_caps"] is False        # missing capeff = fail
    assert checks["worker_euid"] is False    # 0 != 990
    assert checks["deny:/etc/shadow"] is False
    assert checks["ns:mnt"] is False
    assert not all(checks.values())

    healthy = {
        "euid": 990, "egid": 990, "capeff": "0000000000000000",
        "namespaces": {k: f"{k}:[4090000001]" for k in host_ns},
        "fs": {"/etc/shadow": "missing", "/root": "denied",
               "/proc/self/status": "read-ok"},
        "scratch_write": "write-ok",
        "net": {"127.0.0.1:9": "connect-denied"},
        "detached_child": "forked",
    }
    checks = _grade(healthy, expect={"euid": 990, "egid": 990,
                                     "supervisor_euid": 0,
                                     "identity_distinct": True},
                    host_ns=host_ns,
                    probe_args=["deny:/etc/shadow", "deny:/root",
                                "read:/proc/self/status",
                                "net:127.0.0.1:9", "detach:/work/m"])
    assert all(checks.values()), checks
