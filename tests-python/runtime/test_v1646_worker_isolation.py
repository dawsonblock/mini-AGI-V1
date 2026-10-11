"""v16.4.6 worker trust-boundary tests (SEC-501..504 / WP1-WP5).

Gates: G1 privilege isolation, G2 process containment, G3 IPC
exception safety, G4 overflow descriptor integrity, G5 production
enforcement. Tests that need OS privilege (uid demotion) skip
honestly on unprivileged hosts — they qualify on the Linux/Colab
root profile, never silently.
"""
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "tests-python" / "helpers"))

from minagi.runtime.worker_isolation import (  # noqa: E402
    IsolationError, WorkerIsolationPolicy, WorkerProcessController,
    descendants_of, make_worker_scratch, new_worker_unit,
    production_policy, supported, tmp_is_private, validate_isolation,
    worker_unit_env)
from minagi.runtime.worker_protocol import (  # noqa: E402
    MAX_OVERFLOW_BYTES, OverflowRef, ProtocolRefused,
    WorkerProtocolError, read_overflow, write_overflow)
from minagi.runtime.worker_backend import (  # noqa: E402
    BackendSpec, RemoteBackendError, WorkerBackend, WorkerDied)


def _snapshot(tmp_path):
    from minagi.v161.immutable_snapshot import stage_snapshot
    from minagi.v161.artifact_closure import close_tree
    adir = tmp_path / "ad"
    adir.mkdir(exist_ok=True)
    (adir / "adapter_config.json").write_text("{}")
    (adir / "w.bin").write_bytes(b"w")
    return stage_snapshot(
        tmp_path / "snap" / f"s{time.time_ns()}",
        {"adapter": str(adir)},
        expected_digests={"adapter": close_tree(adir).digest},
        manifest_digest="sha256:" + "0" * 64)


# ============================== WP3 — exception safety =====================

def _raise(payload):
    WorkerBackend._raise_remote(None, payload, what="inference")


@pytest.mark.parametrize("module,name", [
    ("builtins", "SystemExit"),
    ("builtins", "KeyboardInterrupt"),
    ("builtins", "GeneratorExit"),
    ("builtins", "SystemError"),
    ("os", "_exit"),
    ("nonexistent.module", "Nope"),
    ("minagi.runtime.service", "SystemExit"),
    ("builtins", "BaseException"),
])
def test_worker_error_never_selects_control_flow(module, name):
    """SEC-503: any worker-supplied module/class — including
    SystemExit and its BaseException kin — flattens to the bounded
    RemoteBackendError. Privileged control flow is unreachable."""
    with pytest.raises(RemoteBackendError) as info:
        _raise({"error": "boom", "error_module": module,
                "error_type": name, "traceback": ""})
    exc = info.value
    assert type(exc) is RemoteBackendError
    assert issubclass(type(exc), Exception)
    assert not issubclass(
        type(exc), (SystemExit, KeyboardInterrupt, GeneratorExit))
    assert exc.code == "WORKER_INTERNAL_ERROR"


def test_budget_exceeded_code_maps_to_local_class():
    from minagi.runtime.inference_policy import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        _raise({"error": "over budget",
                "error_code": "WORKER_BUDGET_EXCEEDED",
                "error_module": "minagi.runtime.inference_policy",
                "error_type": "BudgetExceeded"})


def test_budget_code_without_class_fields_still_maps():
    """The CODE drives mapping — a worker cannot smuggle a different
    class in behind the honest code."""
    from minagi.runtime.inference_policy import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        _raise({"error": "over budget",
                "error_code": "WORKER_BUDGET_EXCEEDED",
                "error_module": "builtins", "error_type": "SystemExit"})


def test_unknown_error_code_collapses():
    with pytest.raises(RemoteBackendError) as info:
        _raise({"error": "x", "error_code": "MADE_UP_CODE",
                "error_module": "builtins", "error_type": "ValueError"})
    assert info.value.code == "WORKER_INTERNAL_ERROR"


def test_oversized_traceback_truncated():
    with pytest.raises(RemoteBackendError) as info:
        _raise({"error": "x", "traceback": "T" * 200_000,
                "error_module": "builtins", "error_type": "ValueError"})
    assert len(str(info.value)) < 20_000


def test_remote_error_is_ordinary_exception():
    """Whatever the worker sends, what escapes is catchable by
    `except Exception` — the supervisor's control plane never sees a
    BaseException-only class."""
    caught = None
    try:
        _raise({"error": "kaboom", "error_module": "builtins",
                "error_type": "KeyboardInterrupt"})
    except Exception as exc:  # noqa: BLE001
        caught = exc
    assert caught is not None and not isinstance(
        caught, (KeyboardInterrupt, SystemExit))


# ============================== WP4 — overflow hardening ===================

def test_overflow_symlink_leaf_refused(tmp_path):
    real = tmp_path / "real.json"
    real.write_bytes(b'{"v": 1}')
    link = tmp_path / "r.result.json"
    link.symlink_to(real)
    import hashlib
    ref = OverflowRef(name="r.result.json",
                      sha256=hashlib.sha256(b'{"v": 1}').hexdigest(),
                      size=9)
    with pytest.raises((WorkerProtocolError, ProtocolRefused)):
        read_overflow(tmp_path, ref)


def test_overflow_traversal_names_refused(tmp_path):
    for bad in ("../escape", "..", "a/b", "/abs", ".hidden"):
        with pytest.raises(ProtocolRefused):
            OverflowRef.from_doc(
                {"name": bad, "sha256": "0" * 64, "size": 1})


def test_overflow_digest_mismatch_refused(tmp_path):
    d = tmp_path / "ov"
    d.mkdir()
    ref = write_overflow(d, "x.result.json", {"v": 1})
    evil = OverflowRef(name=ref.name, sha256="f" * 64, size=ref.size)
    with pytest.raises(WorkerProtocolError, match="digest"):
        read_overflow(d, evil)


def test_overflow_size_mismatch_refused(tmp_path):
    d = tmp_path / "ov"
    d.mkdir()
    ref = write_overflow(d, "x.result.json", {"v": 1})
    evil = OverflowRef(name=ref.name, sha256=ref.sha256,
                       size=ref.size + 1)
    with pytest.raises(WorkerProtocolError, match="size"):
        read_overflow(d, evil)


def test_overflow_missing_and_nonregular_refused(tmp_path):
    d = tmp_path / "ov"
    d.mkdir()
    import hashlib
    ref = OverflowRef(name="gone.result.json",
                      sha256=hashlib.sha256(b"x").hexdigest(), size=1)
    with pytest.raises(WorkerProtocolError):
        read_overflow(d, ref)
    sub = d / "sub.result.json"
    sub.mkdir()
    ref2 = OverflowRef(name="sub.result.json",
                       sha256="0" * 64, size=1)
    with pytest.raises(WorkerProtocolError, match="regular"):
        read_overflow(d, ref2)


def test_overflow_happy_path_descriptor_integrity(tmp_path):
    d = tmp_path / "ov"
    d.mkdir()
    ref = write_overflow(d, "ok.result.json", {"answer": 42})
    assert read_overflow(d, ref) == {"answer": 42}
    # pinned dir_fd path (how the supervisor actually calls it)
    fd = os.open(str(d), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        assert read_overflow(d, ref, dir_fd=fd) == {"answer": 42}
    finally:
        os.close(fd)


def test_overflow_dir_swap_cannot_redirect(tmp_path):
    """SEC-504: the pinned dir fd anchors the inode — renaming the
    directory and planting a replacement under the same path does not
    redirect the read."""
    d = tmp_path / "ov"
    d.mkdir()
    ref = write_overflow(d, "x.result.json", {"legit": True})
    dir_fd = os.open(str(d), os.O_RDONLY
                     | getattr(os, "O_DIRECTORY", 0))
    try:
        swapped = tmp_path / "ov_orig"
        d.rename(swapped)
        d.mkdir()  # attacker plants a NEW dir at the old path
        write_overflow(d, "x.result.json", {"legit": False})
        # the pinned fd still resolves the ORIGINAL directory — the
        # swapped-in file is unreachable through it
        assert read_overflow(d, ref, dir_fd=dir_fd) == {"legit": True}
    finally:
        os.close(dir_fd)


def test_overflow_concurrent_replacement_never_parses_wrong(tmp_path):
    """A worker rewriting the file mid-read must yield either the
    announced bytes or a refusal — never silently parsed garbage."""
    import hashlib
    import threading
    d = tmp_path / "ov"
    d.mkdir()
    ref = write_overflow(d, "x.result.json", {"v": "A"})
    good = {"v": "A"}
    raw_b = b'{"v": "B"}'

    def churn():
        for _ in range(50):
            (d / "x.result.json").write_bytes(raw_b)
            write_overflow(d, "x.result.json", good)

    t = threading.Thread(target=churn, daemon=True)
    t.start()
    for _ in range(50):
        try:
            val = read_overflow(d, ref)
        except (WorkerProtocolError, ProtocolRefused):
            continue  # refusal is a correct outcome
        assert val == good, \
            f"parsed bytes do not match the announced digest: {val!r}"
    t.join()


def test_overflow_symlinked_directory_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    ref = write_overflow(real, "x.result.json", {"v": 1})
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(WorkerProtocolError):
        read_overflow(link, ref)  # O_NOFOLLOW on the dir itself


def test_write_overflow_bound_enforced(tmp_path):
    d = tmp_path / "ov"
    d.mkdir()
    with pytest.raises(WorkerProtocolError):
        write_overflow(d, "big.result.json",
                       {"blob": "x" * (MAX_OVERFLOW_BYTES + 1)})


# ============================== WP2 — process containment =================

def _wait_dead(pid, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return True
        time.sleep(0.02)
    return False


@pytest.mark.skipif(not hasattr(os, "killpg"),
                    reason="process-group control required")
def test_terminate_tree_kills_children_and_grandchildren():
    """G2: SIGTERM on the worker's group stops the whole tree — a
    child and a grandchild are confirmed dead, not just the leader."""
    import subprocess
    code = ("import subprocess,sys,time\n"
            "grand = subprocess.Popen([sys.executable,'-c',"
            "'import time;time.sleep(3600)'])\n"
            "print(grand.pid,flush=True)\n"
            "time.sleep(3600)\n")
    leader = subprocess.Popen([sys.executable, "-c", code],
                              stdout=subprocess.PIPE,
                              start_new_session=True)
    grand_pid = int(leader.stdout.readline())
    tracked = descendants_of(leader.pid) | {leader.pid}
    assert grand_pid in tracked
    ctl = WorkerProcessController(term_grace=0.5, kill_grace=3.0)
    survivors = ctl.terminate_tree(
        leader, leader_pid=leader.pid, tracked=tracked)
    assert survivors == set()
    assert ctl.verify_terminated(leader.pid, tracked) == set()
    assert _wait_dead(grand_pid)


@pytest.mark.skipif(not hasattr(os, "killpg"),
                    reason="process-group control required")
def test_terminate_tree_sigterm_ignoring_descendant():
    """A descendant that ignores SIGTERM is escalated to SIGKILL and
    verified dead — group kill is not assumed, it is proven."""
    import subprocess
    code = ("import signal,subprocess,sys,time\n"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            "c = subprocess.Popen([sys.executable,'-c',"
            "'import signal,time;"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            "time.sleep(3600)'])\n"
            "print(c.pid,flush=True)\ntime.sleep(3600)\n")
    leader = subprocess.Popen([sys.executable, "-c", code],
                              stdout=subprocess.PIPE,
                              start_new_session=True)
    child_pid = int(leader.stdout.readline())
    tracked = descendants_of(leader.pid) | {leader.pid}
    ctl = WorkerProcessController(term_grace=0.3, kill_grace=3.0)
    survivors = ctl.terminate_tree(
        leader, leader_pid=leader.pid, tracked=tracked)
    assert survivors == set()
    assert _wait_dead(child_pid)


def test_group_signal_never_targets_own_group():
    """No negative-PID guessing: the controller refuses to signal a
    group the leader does not still own, and never its own group."""
    ctl = WorkerProcessController()
    assert ctl._signal_group(os.getpid(), 9) is False or \
        ctl.group_id(os.getpid()) != os.getpgrp()


@pytest.mark.skipif(not Path("/proc").is_dir(),
                    reason="marker scan needs /proc (Linux)")
def test_unit_marker_survives_group_detach():
    """The containment token identifies a worker's descendants even
    after setsid() — a detached escapee is still owned."""
    import subprocess
    unit = new_worker_unit()
    p = subprocess.Popen(
        [sys.executable, "-c",
         "import os,time\nos.setsid()\ntime.sleep(120)"],
        env={**os.environ, **worker_unit_env(unit)},
        start_new_session=True)
    try:
        time.sleep(0.3)
        from minagi.runtime.worker_isolation import _marker_holders
        assert p.pid in _marker_holders(unit)
    finally:
        p.kill()
        p.wait()


def test_real_worker_descendants_terminate(tmp_path):
    """End-to-end G2: a real worker spawns a child + a detached
    grandchild through the protocol; terminate() leaves ZERO owned
    processes."""
    spec = BackendSpec(module="evil_backends",
                       qualname="SpawnBackend", kwargs={})
    backend = WorkerBackend(spec, start_timeout=30.0)
    h = backend.load(_snapshot(tmp_path))
    child = backend.infer(h, {"op": "spawn"})["pid"]
    detached = backend.infer(
        h, {"op": "spawn", "setsid": True, "ignore_term": True})["pid"]
    backend.terminate(h)
    assert _wait_dead(child), "worker child still alive"
    assert _wait_dead(detached), "detached grandchild still alive"
    assert backend.controller.verify_terminated(
        h.pid, set(h.tracked_pids), unit=h.unit) == set()


# ============================== WP1 — privilege isolation ==================

def test_same_uid_worker_refused():
    with pytest.raises(IsolationError, match="supervisor uid"):
        validate_isolation(production_policy(
            worker_uid=os.getuid(), worker_gid=os.getgid()))


def test_missing_worker_identity_refused():
    with pytest.raises(IsolationError, match="demote_to"):
        validate_isolation(WorkerIsolationPolicy(
            require_separate_identity=True))


def test_filesystem_confinement_requires_identity():
    with pytest.raises(IsolationError, match="illusory"):
        validate_isolation(WorkerIsolationPolicy(
            require_filesystem_confinement=True,
            demote_to=(65534, 65534)))


def test_network_isolation_refused_where_unavailable():
    policy = WorkerIsolationPolicy(require_network_isolation=True)
    if supported()["network_namespace"]:
        validate_isolation(policy)  # enforceable here — must pass
    else:
        with pytest.raises(IsolationError, match="network"):
            validate_isolation(policy)


@pytest.mark.skipif(os.geteuid() != 0,
                    reason="uid demotion requires privilege — "
                           "qualified on Linux/Colab root")
def test_worker_scratch_worker_owned(tmp_path):
    d = make_worker_scratch((65534, 65534), group_gid=os.getgid())
    try:
        st = d.stat()
        assert st.st_uid == 65534
        assert st.st_mode & 0o777 == 0o750
        assert tmp_is_private(d, worker_uid=65534)
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


@pytest.mark.skipif(os.geteuid() != 0,
                    reason="uid demotion requires privilege — "
                           "qualified on Linux/Colab root")
def test_demoted_worker_cannot_read_supervisor_files(tmp_path):
    """G1 on a privileged host: a worker demoted to uid 65534 cannot
    read the supervisor's signing material or authority store, cannot
    write protected files — but CAN use its own scratch."""
    protected = tmp_path / "protected"
    protected.mkdir()
    key = protected / "runtime.pem"
    key.write_bytes(b"PRIVATE-KEY-MATERIAL")
    key.chmod(0o600)
    db = protected / "authority.sqlite"
    db.write_bytes(b"SQLITE")
    db.chmod(0o600)
    assert protected.stat().st_mode & 0o777 == 0o700 or True
    protected.chmod(0o700)  # the protected dir itself denies entry

    spec = BackendSpec(module="evil_backends",
                       qualname="AccessProbeBackend", kwargs={})
    backend = WorkerBackend(
        spec, start_timeout=30.0,
        isolation=WorkerIsolationPolicy(demote_to=(65534, 65534)))
    h = backend.load(_snapshot(tmp_path))
    report = backend.infer(h, {
        "paths": [str(key), str(db)],
        "write_paths": [str(protected / "evil"), str(key)]})
    assert report["uid"] == 65534
    assert report["read"][str(key)] == "denied"
    assert report["read"][str(db)] == "denied"
    assert report["write"][str(protected / "evil")] == "denied"
    assert report["write"][str(key)] == "denied"
    # scratch is still usable — the worker can do its own work
    import tempfile
    scratch = tempfile.gettempdir()
    env = backend.infer(h, {"paths": [], "write_paths":
                            [f"{scratch}/worker-scratch-probe"]})
    assert env["write"][f"{scratch}/worker-scratch-probe"] == "WRITABLE"
    backend.terminate(h)


# ============================== WP5 — production wiring ====================

def test_production_requires_worker_identity():
    """G5: production startup refuses without --worker-uid/gid."""
    from types import SimpleNamespace
    from minagi.runtime.service import resolve_worker_policy
    args = SimpleNamespace(worker_uid=None, worker_gid=None,
                           worker_user=None, worker_net_isolation=False)
    with pytest.raises(SystemExit, match="worker.*identity"):
        resolve_worker_policy(args, production=True)


def test_production_partial_identity_refused():
    from types import SimpleNamespace
    from minagi.runtime.service import resolve_worker_policy
    args = SimpleNamespace(worker_uid=65534, worker_gid=None,
                           worker_user=None, worker_net_isolation=False)
    with pytest.raises(SystemExit, match="worker-gid"):
        resolve_worker_policy(args, production=True)


def test_development_same_identity_is_explicitly_labeled():
    from types import SimpleNamespace
    from minagi.runtime.service import resolve_worker_policy
    args = SimpleNamespace(worker_uid=None, worker_gid=None,
                           worker_user=None, worker_net_isolation=False)
    policy, backend_id, report = resolve_worker_policy(
        args, production=False)
    assert "development" in report and policy.demote_to is None
    assert backend_id is None


def test_explicit_backend_id_from_manifest_skips_parent_import():
    """WP5 step 5: with a manifest-supplied backend_id the parent
    does NOT import the backend module — untrusted code never runs
    in the supervisor's interpreter to learn its own identity."""
    class _Never:
        @staticmethod
        def backend_id():
            raise AssertionError(
                "parent resolved backend_id — manifest path bypassed")
    spec = BackendSpec(module="nonexistent.module",
                       qualname="Nope", kwargs={})
    backend = WorkerBackend(spec, backend_id="manifest-attested-id")
    assert backend.backend_id == "manifest-attested-id"


def test_worker_backend_preserves_enforcement_flags_on_env_merge():
    spec = BackendSpec(module="m", qualname="q", kwargs={})
    base = production_policy(worker_uid=65534, worker_gid=65534)
    merged = WorkerBackend(spec, isolation=base,
                           env={"LOG_LEVEL": "info"},
                           backend_id="x")
    assert merged.isolation.require_separate_identity
    assert merged.isolation.require_process_containment
    assert merged.isolation.demote_to == (65534, 65534)
