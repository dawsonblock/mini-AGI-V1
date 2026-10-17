"""v16.4.10 SEC-903 — worker identifiers are opaque broker-generated
``w-<32 hex>`` values; workspace creation is traversal- and
reuse-proof under the pinned state root.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.runtime.isolation_profile import load_profile  # noqa: E402
from minagi.runtime.nsjail_executor import (  # noqa: E402
    ExecutorRefused, NsJailWorkerExecutor)
from minagi.runtime.worker_isolation import DEFAULT_ISOLATION  # noqa: E402
from minagi.runtime.worker_launch_broker import (  # noqa: E402
    BrokerError, BrokerRefused, LaunchWorkerRequest,
    WorkerLaunchBroker, new_worker_id)

PROFILE_PATH = ROOT / "configs" / "isolation" / \
    "linux-production-offline-v1.json"
D = "sha256:" + "0" * 64


class _IdleSystemd:
    available = True

    def __init__(self, tmp):
        self.tmp = tmp

    def run_service(self, unit_name, properties, argv, *, cwd=None,
                    stderr=None):
        raise AssertionError("launch must refuse before spawning")

    def show(self, unit_name, prop):
        return ""

    def is_active(self, unit_name):
        return False

    def stop(self, unit_name):
        return None

    def kill(self, unit_name, sig="SIGKILL"):
        return None

    def cgroup_path(self, unit_name):
        return None


def _broker(tmp_path):
    binary = tmp_path / "nsjail"
    binary.write_bytes(b"fake")
    binary.chmod(0o755)
    profile = load_profile(PROFILE_PATH)
    b = WorkerLaunchBroker(
        worker_state_root=tmp_path / "workers",
        approved_profiles={profile.digest(): profile},
        artifact_resolver=lambda d: str(tmp_path / "a"),
        nsjail_path=str(binary),
        nsjail_digest="sha256:" + hashlib.sha256(
            binary.read_bytes()).hexdigest(),
        worker_uid=990, worker_gid=990,
        systemd=_IdleSystemd(tmp_path),
        nonce_dir=tmp_path / "nonces")
    return b, profile


def _req(profile, worker_id):
    return LaunchWorkerRequest(
        worker_id=worker_id, signed_backend_manifest_digest=D,
        approved_isolation_profile_digest=profile.digest(),
        execution_budget_digest=D, request_nonce="n1", role="worker")


# ---- identifier rejection table ---------------------------------------

@pytest.mark.parametrize("bad", [
    ".", "..", "../x", "a/b", "a\\b", "w1", "worker-1",
    "..%2f..%2fetc", "w-%41%41", "", " ", "w_" + "a" * 32,
    "w-" + "a" * 31, "w-" + "a" * 33, "w-" + "A" * 32,
    "w-" + "g" * 32, "x" + "a" * 63,
])
def test_unsafe_worker_ids_refused(tmp_path, bad):
    broker, profile = _broker(tmp_path)
    with pytest.raises(BrokerRefused):
        broker.launch(_req(profile, bad))
    with pytest.raises(BrokerRefused):
        broker.create_workspace(bad)


def test_broker_generated_ids_accepted_shape(tmp_path):
    wid = new_worker_id()
    assert wid.startswith("w-") and len(wid) == 34
    int(wid[2:], 16)


def test_caller_supplied_id_refused_by_executor(tmp_path):
    broker, profile = _broker(tmp_path)
    ex = NsJailWorkerExecutor(broker=broker, profile=profile)
    with pytest.raises(ExecutorRefused):
        ex.prepare(worker_id="attacker-named",
                   isolation=DEFAULT_ISOLATION)


def test_workspace_reuse_refused(tmp_path):
    broker, profile = _broker(tmp_path)
    wid = new_worker_id()
    ws, scratch, overflow, fd = broker.create_workspace(wid)
    os.close(fd)
    # idempotent within the owning broker (prepare -> launch flow)...
    again = broker.create_workspace(wid)
    assert again[:3] == (ws, scratch, overflow)
    os.close(again[3])
    # ...but a fresh broker hitting the on-disk dir is refused —
    # O_EXCL means adopt-and-continue is impossible
    broker2, _ = _broker(tmp_path)
    with pytest.raises((BrokerRefused, BrokerError)):
        broker2.create_workspace(wid)


def test_preexisting_workspace_dir_refused(tmp_path):
    """An attacker-planted dir at the workspace location is refused,
    never adopted."""
    broker, profile = _broker(tmp_path)
    root = broker.worker_state_root
    root.mkdir(mode=0o711, parents=True, exist_ok=True)
    wid = new_worker_id()
    (root / wid).mkdir()
    (root / wid / "scratch").mkdir()
    with pytest.raises((BrokerRefused, BrokerError)):
        broker.create_workspace(wid)


def test_workspace_lands_under_pinned_root(tmp_path):
    broker, profile = _broker(tmp_path)
    wid = new_worker_id()
    ws, scratch, overflow, fd = broker.create_workspace(wid)
    try:
        assert Path(os.path.realpath(ws)).parent == \
            Path(os.path.realpath(broker.worker_state_root))
        assert Path(os.path.realpath(scratch)).parent == \
            Path(os.path.realpath(ws))
    finally:
        os.close(fd)


def test_symlinked_worker_dir_boundary_refused(tmp_path):
    """A planted symlink where the workspace would go must not let
    creation escape the root."""
    broker, profile = _broker(tmp_path)
    root = broker.worker_state_root
    root.mkdir(mode=0o711, parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    wid = new_worker_id()
    os.symlink(outside, root / wid)
    with pytest.raises((BrokerRefused, BrokerError)):
        broker.create_workspace(wid)
    # and nothing was created in the attacker-controlled dir
    assert list(outside.iterdir()) == []
