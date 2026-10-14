"""v16.4.10 SEC-906 — production startup must run the REAL isolation
preflight and persist a host-identity-bound qualification record.
A validate_host() ok is a capability preflight only; readiness
requires the enforcement qualification."""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.runtime import service  # noqa: E402
from minagi.runtime.worker_launch_broker import (  # noqa: E402
    BrokerRefused, WorkerLaunchBroker)


class _FakeBroker:
    user_manager = False
    worker_uid = 65534
    worker_gid = 65534
    _nsjail_digest = "sha256:" + "1" * 64


class _FakeExecutor:
    def __init__(self, qualified=True, probe_ran=None):
        self.broker = _FakeBroker()
        self.profile_digest = "sha256:" + "2" * 64
        self._qualified = qualified
        self.probe_ran = probe_ran if probe_ran is not None else []


def _preflight_result(qualified):
    checks = {
        "worker_euid": True, "worker_egid": True,
        "distinct_worker_identity": True, "no_caps": True,
        "ns:mnt": True, "ns:net": True, "ns:pid": True,
        "ns:user": True, "deny:/etc/shadow": True,
        "read:/proc/self/status": True, "net:127.0.0.1:9": True,
        "scratch_writable": True, "containment_verified": True,
    }
    if not qualified:
        checks["ns:mnt"] = False
    return {"qualified": qualified, "checks": checks,
            "probe": {"namespaces": {"mnt": "mnt:[1]"}}}


def _args(tmp_path):
    return types.SimpleNamespace(
        storage_root=str(tmp_path / "storage"))


def test_production_qualification_runs_real_probe(tmp_path,
                                                  monkeypatch):
    """The gate executes the real preflight — a jailed probe must
    actually launch; ok=True alone never suffices."""
    calls = []

    def fake_preflight(executor, isolation=None, timeout=None,
                       probe_args=()):
        calls.append(probe_args)
        return _preflight_result(True)

    monkeypatch.setattr(
        "minagi.runtime.isolation_preflight.run_preflight",
        fake_preflight)
    ex = _FakeExecutor()
    out = service.qualify_production_host(ex, args=_args(tmp_path))
    assert calls and calls[0], "preflight must run probe args"
    assert out["qualified"] is True
    rec = json.loads(Path(out["record_path"]).read_text())
    assert rec["schema"] == "minagi-host-qualification-v1"
    assert rec["nsjail_binary_digest"] == "sha256:" + "1" * 64
    assert rec["isolation_profile_digest"] == "sha256:" + "2" * 64
    assert rec["worker_uid"] == 65534
    assert rec["qualified"] is True
    assert rec["qualification_record_digest"]


def test_failed_preflight_not_qualified_and_still_recorded(tmp_path,
                                                         monkeypatch):
    monkeypatch.setattr(
        "minagi.runtime.isolation_preflight.run_preflight",
        lambda *a, **k: _preflight_result(False))
    out = service.qualify_production_host(_FakeExecutor(),
                                          args=_args(tmp_path))
    assert out["qualified"] is False
    assert "ns:mnt" in out["failures"]
    rec = json.loads(Path(out["record_path"]).read_text())
    assert rec["qualified"] is False


def test_broker_require_authorization_needs_registry_and_resolver(
        tmp_path):
    binary = tmp_path / "nsjail"
    binary.write_bytes(b"fake")
    binary.chmod(0o755)
    dg = "sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest()
    with pytest.raises(BrokerRefused):
        WorkerLaunchBroker(
            worker_state_root=tmp_path / "w",
            approved_profiles={},
            nsjail_path=str(binary), nsjail_digest=dg,
            systemd=types.SimpleNamespace(available=True),
            require_authorization=True)   # no registry/resolver


def test_require_authorization_defaults_true_when_registry_present(
        tmp_path):
    """A broker holding trusted keys enforces authorization unless an
    operator explicitly disarms it — default follows the registry."""
    from minagi.v161.authority import (AuthorityRegistry,
                                       write_trust_root)
    storage = tmp_path / "storage"
    write_trust_root(storage / ".keys", storage / "trust_root.json")
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    binary = tmp_path / "nsjail"
    binary.write_bytes(b"fake")
    binary.chmod(0o755)
    dg = "sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest()
    from minagi.runtime.worker_launch_broker import (
        MappingRecordResolver)
    b = WorkerLaunchBroker(
        worker_state_root=tmp_path / "w",
        approved_profiles={}, nsjail_path=str(binary),
        nsjail_digest=dg,
        systemd=types.SimpleNamespace(available=True),
        authority_registry=registry,
        record_resolver=MappingRecordResolver({}))
    assert b.require_authorization is True
