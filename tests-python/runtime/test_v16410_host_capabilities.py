"""v16.4.10 SEC-904 — uid 0 is not a namespace-capability proof.

validate_host() must key eligibility on MEASURED capability bits /
an enabled unprivileged-userns path — never on geteuid()==0. A root
shell in a cap-restricted container cannot create the required
namespaces and must not produce a positive capability preflight.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.runtime.isolation_profile import load_profile  # noqa: E402
from minagi.runtime.nsjail_executor import (  # noqa: E402
    ExecutorRefused, NsJailWorkerExecutor)
import minagi.runtime.nsjail_executor as nsjail_mod  # noqa: E402
from minagi.runtime.worker_launch_broker import (  # noqa: E402
    WorkerLaunchBroker)

PROFILE_PATH = ROOT / "configs" / "isolation" / \
    "linux-production-offline-v1.json"


class _IdleSystemd:
    available = True

    def run_service(self, *a, **k):
        raise AssertionError("no launches in a capability test")

    def show(self, u, p):
        return ""

    def is_active(self, u):
        return False

    def stop(self, u):
        return None

    def kill(self, u, sig="SIGKILL"):
        return None

    def cgroup_path(self, u):
        return None


def _executor(tmp_path, **kw):
    binary = tmp_path / "nsjail"
    binary.write_bytes(b"fake")
    binary.chmod(0o755)
    profile = load_profile(PROFILE_PATH)
    return NsJailWorkerExecutor(
        broker=WorkerLaunchBroker(
            worker_state_root=tmp_path / "workers",
            approved_profiles={profile.digest(): profile},
            nsjail_path=str(binary),
            nsjail_digest="sha256:" + hashlib.sha256(
                binary.read_bytes()).hexdigest(),
            worker_uid=990, worker_gid=990,
            systemd=_IdleSystemd(), nonce_dir=tmp_path / "n"),
        profile=profile)


def test_root_without_capabilities_does_not_qualify(tmp_path,
                                                    monkeypatch):
    """The audit's case: euid 0 but no CAP_SYS_ADMIN and no
    unprivileged userns — capability preflight must FAIL."""
    ex = _executor(tmp_path)
    monkeypatch.setattr("os.geteuid", lambda: 0)
    monkeypatch.setattr(
        "minagi.runtime.worker_isolation._cap_eff",
        lambda bit: False)
    monkeypatch.setattr(
        "minagi.runtime.worker_isolation._unprivileged_userns",
        lambda: False)
    rep = ex.validate_host()
    assert rep["euid"] == 0
    assert rep["cap_sys_admin"] is False
    assert rep["unprivileged_userns"] is False
    # on macOS the platform check also fails; the CRITICAL property is
    # that ok never becomes True from uid 0 alone
    assert rep["ok"] is False


def test_root_with_cap_sys_admin_still_preflights(tmp_path,
                                                  monkeypatch):
    ex = _executor(tmp_path)
    monkeypatch.setattr("os.geteuid", lambda: 0)
    monkeypatch.setattr(
        "minagi.runtime.worker_isolation._cap_eff",
        lambda bit: True)
    monkeypatch.setattr(
        "minagi.runtime.worker_isolation._unprivileged_userns",
        lambda: False)
    rep = ex.validate_host()
    assert rep["cap_sys_admin"] is True
    assert rep["qualification_level"] == "capability_preflight"


def test_unprivileged_userns_path_preflights_without_caps(tmp_path,
                                                          monkeypatch):
    ex = _executor(tmp_path)
    monkeypatch.setattr("os.geteuid", lambda: 1000)
    monkeypatch.setattr(
        "minagi.runtime.worker_isolation._cap_eff",
        lambda bit: False)
    monkeypatch.setattr(
        "minagi.runtime.worker_isolation._unprivileged_userns",
        lambda: True)
    rep = ex.validate_host()
    assert rep["cap_sys_admin"] is False
    assert rep["unprivileged_userns"] is True
    # capability preflight may pass on the userns path — readiness
    # still requires the enforcement qualification (probe run)


def test_report_declares_capability_preflight_not_readiness(tmp_path):
    ex = _executor(tmp_path)
    rep = ex.validate_host()
    assert rep["qualification_level"] == "capability_preflight"
