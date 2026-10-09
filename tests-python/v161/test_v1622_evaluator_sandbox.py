"""FIX-001 — evaluator isolation.

Adversarial suite for executable evaluator checks. A corpus checker
must not: read evaluator secrets or authority state, write outside its
ephemeral workspace, reach the network, see the evaluator's
environment, or survive its timeout as an orphaned process. When no OS
sandbox backend exists, executable evaluation must fail closed.

The reproducing tests (secret read, env visibility, orphan survival)
pass on the repaired implementation and fail on the v16.2.1 executor,
which ran corpus python with the full parent environment, no
filesystem/network isolation, and no process-tree termination.
"""
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161 import execution_sandbox  # noqa: E402
from minagi.v161.evaluators import (declarative_score, exact_match,  # noqa: E402
                                    executor_score, score_row)
from minagi.v161.execution_sandbox import (SandboxUnavailable,  # noqa: E402
                                           run_check)

BACKENDS = execution_sandbox.available_backends()
requires_sandbox = pytest.mark.skipif(
    not execution_sandbox.sandbox_usable(),
    reason="no usable OS sandbox backend on this host (TEST-001 "
           "classification: intentionally unavailable — backend missing, "
           "or present but unable to start the interpreter)")


def _assert_exit(check: str, *, timeout_s: float = 10.0) -> int:
    return executor_score("x", "", {"type": "python_assert",
                                    "check": check}, timeout_s=timeout_s)


# ---------- declarative evaluator (preferred, code-free) --------------

def test_declarative_ops_allowlisted():
    assert declarative_score("42", "42", {"type": "declarative",
                                          "op": "equals"}) == 1.0
    assert declarative_score("The answer is Paris.", "Paris",
                             {"type": "declarative",
                              "op": "contains"}) == 1.0
    assert declarative_score("42.0000001", "42",
                             {"type": "declarative",
                              "op": "numeric_equals"}) == 1.0
    assert declarative_score('{"a": {"b": 7}}', "7",
                             {"type": "declarative",
                              "op": "json_field_equals",
                              "args": {"path": "a.b"}}) == 1.0
    assert declarative_score("red", "x",
                             {"type": "declarative", "op": "one_of",
                              "args": {"values": ["blue", "red"]}}) == 1.0
    assert declarative_score("benign", "",
                             {"type": "declarative",
                              "op": "contains_none",
                              "args": {"values": ["<unsafe>"]}}) == 1.0
    assert declarative_score("ab12", "x",
                             {"type": "declarative",
                              "op": "regex_fullmatch",
                              "args": {"pattern": r"[a-z]+\d+"}}) == 1.0
    assert declarative_score("7", "x",
                             {"type": "declarative", "op": "all_of",
                              "args": {"specs": [
                                  {"op": "numeric_equals",
                                   "args": {"value": "7"}},
                                  {"op": "not",
                                   "args": {"spec": {"op": "equals",
                                                     "args": {"value": "8"}}}}]}}
                             ) == 1.0


def test_declarative_rejects_unknown_ops():
    with pytest.raises(ValueError, match="unknown declarative op"):
        declarative_score("x", "x", {"type": "declarative", "op": "eval"})
    with pytest.raises(ValueError, match="unknown declarative op"):
        declarative_score("x", "x", {"type": "declarative",
                                     "op": "exec"})
    with pytest.raises(ValueError, match="non-empty list"):
        declarative_score("x", "x", {"type": "declarative",
                                     "op": "one_of",
                                     "args": {"values": []}})


def test_declarative_never_spawns_processes(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("declarative scoring must not spawn processes")
    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(subprocess, "Popen", _boom)
    row = {"id": "d", "expected": "42",
           "verify": {"type": "declarative", "op": "numeric_equals"}}
    assert score_row(row, "42.0", exact_match) == 1.0
    assert score_row(row, "41.0", exact_match) == 0.0


def test_score_row_rejects_unknown_verify_type():
    with pytest.raises(ValueError, match="python_assert"):
        score_row({"id": "x", "expected": "1",
                   "verify": {"type": "shell", "check": "true"}},
                  "1", exact_match)


# ---------- sandboxed execution ---------------------------------------

@requires_sandbox
def test_sandboxed_check_scores_normally():
    assert _assert_exit(
        "import os; assert os.environ['OUTPUT'].strip() == '42'") \
        == 0.0  # prediction "x" -> check fails
    assert executor_score(
        "42", "", {"type": "python_assert",
                   "check": "import os; "
                            "assert os.environ['OUTPUT'].strip() == '42'"}) \
        == 1.0


@requires_sandbox
def test_checker_cannot_read_evaluator_secrets(tmp_path):
    """Reproduces FIX-001: the original executor read this file."""
    secret = tmp_path / "signing-key.pem"
    secret.write_text("PRIVATE-KEY-MATERIAL")
    check = (
        "try:\n"
        f"    data = open({str(secret)!r}).read()\n"
        "except Exception:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    assert _assert_exit(check) == 1.0


@requires_sandbox
def test_checker_cannot_read_authority_state(tmp_path):
    storage = tmp_path / "storage"
    (storage / ".keys").mkdir(parents=True)
    (storage / "trust_root.json").write_text('{"authorities": []}')
    (storage / ".keys" / "promotion.pem").write_text("PRIVATE")
    for target in (storage / "trust_root.json",
                   storage / ".keys" / "promotion.pem"):
        check = (
            "try:\n"
            f"    open({str(target)!r}).read()\n"
            "except Exception:\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit(1)\n"
        )
        assert _assert_exit(check) == 1.0, target


@requires_sandbox
def test_checker_cannot_write_outside_workspace(tmp_path):
    target = tmp_path / "escape.txt"
    check = (
        "try:\n"
        f"    open({str(target)!r}, 'w').write('x')\n"
        "except Exception:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    assert _assert_exit(check) == 1.0
    assert not target.exists()


@requires_sandbox
def test_checker_has_no_network():
    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        check = (
            "import socket\n"
            "try:\n"
            f"    s = socket.create_connection(('127.0.0.1', {port}),"
            " timeout=2)\n"
            "    s.close()\n"
            "except Exception:\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit(1)\n"
        )
        assert _assert_exit(check) == 1.0
    finally:
        listener.close()


@requires_sandbox
def test_checker_cannot_see_evaluator_environment(monkeypatch):
    """Reproduces FIX-001: the original executor passed os.environ."""
    monkeypatch.setenv("MINIAGI_SANDBOX_TEST_SECRET", "leakme")
    check = (
        "import os\n"
        "if 'MINIAGI_SANDBOX_TEST_SECRET' in os.environ:\n"
        "    raise SystemExit(1)\n"
        "raise SystemExit(0)\n"
    )
    assert _assert_exit(check) == 1.0


@requires_sandbox
def test_checker_cannot_read_etc_passwd():
    check = (
        "try:\n"
        "    open('/etc/passwd').read()\n"
        "except Exception:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    assert _assert_exit(check) == 1.0


@requires_sandbox
def test_timeout_leaves_no_orphaned_process(tmp_path):
    """Reproduces FIX-001: subprocess.run(timeout=...) killed only the
    direct child; a grandchild survived and kept running."""
    marker = tmp_path / "orphan.txt"
    child = (f"import time; time.sleep(2); "
             f"open({str(marker)!r}, 'w').write('alive')")
    check = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        "time.sleep(30)\n"
    )
    score = executor_score("x", "", {"type": "python_assert",
                                     "check": check}, timeout_s=1)
    assert score == 0.0
    time.sleep(3.5)  # long enough for a surviving orphan to write
    assert not marker.exists()


@requires_sandbox
def test_process_group_termination_is_explicit(tmp_path):
    marker = tmp_path / "orphan2.txt"
    child = (f"import time; time.sleep(2); "
             f"open({str(marker)!r}, 'w').write('alive')")
    check = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        "time.sleep(30)\n"
    )
    result = run_check(check, timeout_s=1, nproc_limit=16384)
    assert result.timed_out
    time.sleep(3.5)
    assert not marker.exists()


@requires_sandbox
def test_read_paths_are_read_only(tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text('{"answer": 42}')
    check = (
        "import os\n"
        "p = os.environ['REFERENCE']\n"
        "data = open(p).read()\n"
        "assert '\"answer\": 42' in data\n"
        "try:\n"
        "    open(p, 'w').write('tampered')\n"
        "except Exception:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    result = run_check(check, env={"REFERENCE": str(reference)},
                       read_paths=[reference], timeout_s=10)
    assert result.returncode == 0, result.stderr
    assert reference.read_text() == '{"answer": 42}'


@requires_sandbox
def test_result_reports_backend_and_limits():
    result = run_check("print('ok')")
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"
    assert result.backend in BACKENDS
    assert result.limits["memory_limit_enforced"] == \
        sys.platform.startswith("linux")
    assert result.limits["nproc_limit"] == 64


@requires_sandbox
def test_ephemeral_workspace_is_removed(monkeypatch):
    import tempfile as tf
    created = {}
    real = tf.mkdtemp

    def fake(*a, **k):
        path = real(*a, **k)
        created["path"] = path
        return path

    monkeypatch.setattr(tf, "mkdtemp", fake)
    run_check("pass")
    assert created and not Path(created["path"]).exists()


@requires_sandbox
@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="RLIMIT_AS is not enforceable on macOS")
def test_memory_limit_denies_large_allocation():
    result = run_check("x = bytearray(2 << 30)")
    assert result.returncode != 0


# ---------- fail-closed behavior --------------------------------------

def test_fail_closed_when_no_backend(monkeypatch):
    monkeypatch.delenv("MINIAGI_SANDBOX_BACKEND", raising=False)
    monkeypatch.setattr(execution_sandbox, "available_backends", lambda: [])
    with pytest.raises(SandboxUnavailable, match="fail"):
        execution_sandbox.select_backend()
    with pytest.raises(SandboxUnavailable):
        executor_score("x", "", {"type": "python_assert",
                                 "check": "pass"})
    assert execution_sandbox.sandbox_usable() is False


def test_backend_that_cannot_start_is_classified_unavailable(monkeypatch):
    """TEST-001: a backend that exists but cannot start the interpreter
    (e.g. a CI runner whose sandbox profile denies the interpreter's own
    path resolution) is classified as unavailable — fail closed — not as
    a checker failure."""
    monkeypatch.setattr(
        execution_sandbox, "_probe_backend",
        lambda backend, python: (False, "simulated startup failure"))
    assert execution_sandbox.sandbox_usable("macos-sandbox-exec") is False
    with pytest.raises(SandboxUnavailable,
                       match="cannot start the interpreter"):
        run_check("pass", backend="macos-sandbox-exec")
    with pytest.raises(SandboxUnavailable):
        executor_score("x", "", {"type": "python_assert", "check": "pass"})


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="sandbox-exec profile behaviour")
def test_real_probe_detects_a_profile_that_cannot_start_the_interpreter(
        monkeypatch):
    """The CI-macOS failure mode reproduced with the real probe: sandbox-exec
    exists but the profile cannot start the interpreter. The backend must be
    classified unusable, so callers skip instead of failing."""
    monkeypatch.setattr(execution_sandbox, "_PROBE_CACHE", {})
    monkeypatch.setattr(
        execution_sandbox, "_macos_profile",
        lambda python, ws, reads, extras: "(version 1)\n(deny default)\n")
    assert execution_sandbox.sandbox_usable("macos-sandbox-exec") is False
    with pytest.raises(SandboxUnavailable,
                       match="cannot start the interpreter"):
        run_check("pass", backend="macos-sandbox-exec")


def test_backend_override_none_forces_fail_closed(monkeypatch):
    monkeypatch.setenv("MINIAGI_SANDBOX_BACKEND", "none")
    with pytest.raises(SandboxUnavailable):
        execution_sandbox.select_backend()


def test_unknown_backend_override_rejected(monkeypatch):
    monkeypatch.setenv("MINIAGI_SANDBOX_BACKEND", "quantum-sandbox")
    with pytest.raises(SandboxUnavailable, match="not\\s+available|not available"):
        execution_sandbox.select_backend()
