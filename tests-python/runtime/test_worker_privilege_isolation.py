"""v16.4.5 worker privilege-isolation tests (SEC-402 / G3).

A worker must be strictly less privileged than the supervisor: a
constructed environment with no signing credentials, a private 0700
scratch directory, POSIX resource bounds where the platform enforces
them, and optional identity demotion. These tests verify the boundary
is built — and that real spawned workers actually run under it.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "tests-python" / "helpers"))

from minagi.runtime.worker_isolation import (  # noqa: E402
    DEFAULT_ISOLATION, IsolationError, WorkerIsolationPolicy,
    build_env, make_private_tmp, supported, tmp_is_private,
    worker_preexec, wrap_argv)


# ---------- environment boundary --------------------------------------------

def test_env_never_contains_credential_like_names(monkeypatch):
    monkeypatch.setenv("MINAGI_SIGNING_KEY", "supersecret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "supersecret")
    monkeypatch.setenv("HF_TOKEN", "supersecret")
    monkeypatch.setenv("OPENAI_API_KEY", "supersecret")
    policy = WorkerIsolationPolicy(
        env_passthrough=("MINAGI_SIGNING_KEY", "AWS_SECRET_ACCESS_KEY",
                         "HF_TOKEN", "OPENAI_API_KEY"))
    env = build_env(policy, pythonpath="/x",
                    private_tmp=Path("/tmp/w"))
    for name in env:
        assert "KEY" not in name and "TOKEN" not in name \
            and "SECRET" not in name and "PASSWORD" not in name
    assert "supersecret" not in env.values()


def test_env_explicit_secret_refused():
    with pytest.raises(IsolationError, match="credential"):
        build_env(WorkerIsolationPolicy(env={"MY_SECRET": "x"}),
                  pythonpath="/x", private_tmp=Path("/tmp/w"))


def test_env_is_constructed_not_inherited(monkeypatch):
    monkeypatch.setenv("SUPERVISOR_INTERNAL_FLAG", "visible")
    env = build_env(DEFAULT_ISOLATION, pythonpath="/x",
                    private_tmp=Path("/tmp/w"))
    assert "SUPERVISOR_INTERNAL_FLAG" not in env
    assert env["PYTHONPATH"] == "/x"
    assert env["TMPDIR"] == "/tmp/w"


def test_private_tmp_is_0700_and_owned():
    d = make_private_tmp()
    try:
        assert tmp_is_private(d)
        assert oct(d.stat().st_mode & 0o777) == "0o700"
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


# ---------- resource limits --------------------------------------------------

def test_preexec_none_when_no_limits():
    """No limits + no demote → no preexec work. Containment does not
    depend on preexec: the unit token travels in the child's env."""
    assert worker_preexec(
        WorkerIsolationPolicy(max_open_files=None)) is None


def test_worker_unit_token_marks_env():
    """The unit token travels in the worker's initial environment —
    descendants inherit it even after setsid()."""
    from minagi.runtime.worker_isolation import (
        WORKER_UNIT_ENV, new_worker_unit, worker_unit_env)
    unit = new_worker_unit()
    env = worker_unit_env(unit)
    assert env[WORKER_UNIT_ENV] == unit
    unit2 = new_worker_unit()
    assert unit2 != unit  # unguessable per-worker tokens


def test_preexec_reports_platform_support():
    caps = supported()
    assert "rlimit_as" in caps and "platform" in caps
    # the capability record is honest: what exists, not what we wish


@pytest.mark.skipif(not hasattr(os, "fork"),
                    reason="POSIX-only rlimit check")
def test_rlimit_open_files_applies_in_child():
    """The rlimit actually binds: a child under max_open_files=32
    fails to open fd 33+. Verified in a real subprocess."""
    import subprocess
    policy = WorkerIsolationPolicy(max_open_files=32)
    code = ("import os\n"
            "fds=[]\n"
            "try:\n"
            "    for i in range(64): fds.append(os.open('/dev/null',0))\n"
            "    print('NO_LIMIT')\n"
            "except OSError:\n"
            "    print('LIMITED')\n")
    out = subprocess.run(
        [sys.executable, "-c", code],
        preexec_fn=worker_preexec(policy),
        capture_output=True, text=True, timeout=30)
    assert "LIMITED" in out.stdout


def test_sandbox_profile_requires_real_file(tmp_path):
    policy = WorkerIsolationPolicy(
        sandbox_profile=str(tmp_path / "nonexistent.sb"))
    if not supported()["sandbox_exec"]:
        pytest.skip("sandbox-exec absent on this host")
    with pytest.raises(IsolationError):
        wrap_argv(["/bin/true"], policy)


# ---------- real spawned worker posture --------------------------------------

def test_spawned_worker_env_has_no_signing_material(tmp_path,
                                                    monkeypatch):
    """End to end: a real worker's environment carries nothing that
    could sign or reach the authority store — verified by asking the
    worker to dump its own environment through the protocol."""
    monkeypatch.setenv("MINAGI_FAKE_SIGNING_KEY_FOR_TEST", "sekrit")
    from minagi.runtime.worker_backend import BackendSpec, WorkerBackend
    from minagi.v161.immutable_snapshot import stage_snapshot
    from minagi.v161.artifact_closure import close_tree

    spec = BackendSpec(module="evil_backends",
                       qualname="EnvDumpBackend", kwargs={})
    backend = WorkerBackend(spec, start_timeout=30.0)
    adir = tmp_path / "ad"
    adir.mkdir()
    (adir / "adapter_config.json").write_text("{}")
    (adir / "w.bin").write_bytes(b"w")
    snap = stage_snapshot(
        tmp_path / "snap" / "s", {"adapter": str(adir)},
        expected_digests={"adapter": close_tree(adir).digest},
        manifest_digest="sha256:" + "0" * 64)
    h = backend.load(snap)
    env_report = backend.infer(h, {"op": "env"})
    names = set(env_report["env_names"])
    assert "MINAGI_FAKE_SIGNING_KEY_FOR_TEST" not in names
    assert not any("KEY" in n or "TOKEN" in n or "SECRET" in n
                   for n in names)
    # private scratch belongs to the worker, not the supervisor's tmp
    assert env_report["tmpdir"].startswith(
        str(tmp_path) + "/") is False or "minagi-worker-" in \
        env_report["tmpdir"]
    backend.terminate(h)
