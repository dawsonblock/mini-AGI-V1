"""v16.4.10 SEC-902 — one shared containment state; an occupied cgroup
can NEVER be reported empty because a unit went inactive or the
systemd client exited.

The audit's reproduction: cgroup.events reports ``populated 1``
while ``is_active`` is False and the client proc has exited — the old
``verified_empty`` OR'd those service facts over kernel evidence.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.runtime.isolation_profile import load_profile  # noqa: E402
from minagi.runtime.nsjail_executor import (  # noqa: E402
    NsJailWorkerExecutor)
from minagi.runtime.worker_isolation import DEFAULT_ISOLATION  # noqa: E402
from minagi.runtime.worker_launch_broker import (  # noqa: E402
    BrokerError, LaunchWorkerRequest, WorkerLaunchBroker,
    new_worker_id)

PROFILE_PATH = ROOT / "configs" / "isolation" / \
    "linux-production-offline-v1.json"
D = "sha256:" + "0" * 64


class OccupiedSystemd:
    """Simulated unit whose cgroup stays populated 1 while systemd
    reports the unit inactive and the client proc has exited — the
    service-side story of 'done', the kernel's story of 'occupied'."""
    available = True

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.cg = tmp_path / "cg"
        self.cg.mkdir()
        (self.cg / "cgroup.events").write_text("populated 1\n")

    def run_service(self, unit_name, properties, argv, *, cwd=None,
                    stderr=None):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(60)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr if stderr is not None else subprocess.PIPE)
        self._proc = proc
        return proc

    def show(self, unit_name, prop):
        if prop == "InvocationID":
            return "inv-1"
        return "/system.slice/x.service"

    def is_active(self, unit_name):
        return False     # service says stopped...

    def stop(self, unit_name):
        # the unit stops — the CLIENT proc exits — but the cgroup
        # keeps reporting populated 1 (a detached survivor): service
        # evidence says done, kernel evidence says occupied.
        if getattr(self, "_proc", None) is not None:
            try:
                self._proc.kill()
            except OSError:
                pass
        return None

    def kill(self, unit_name, sig="SIGKILL"):
        return None

    def cgroup_path(self, unit_name):
        return self.cg   # ...but the kernel says occupied


def _broker(tmp_path, sd):
    binary = tmp_path / "nsjail"
    binary.write_bytes(b"fake")
    binary.chmod(0o755)
    profile = load_profile(PROFILE_PATH)
    return WorkerLaunchBroker(
        worker_state_root=tmp_path / "workers",
        approved_profiles={profile.digest(): profile},
        artifact_resolver=lambda d: str(tmp_path / "a"),
        nsjail_path=str(binary),
        nsjail_digest="sha256:" + hashlib.sha256(
            binary.read_bytes()).hexdigest(),
        worker_uid=990, worker_gid=990,
        systemd=sd, nonce_dir=tmp_path / "nonces"), profile


def _req(profile):
    return LaunchWorkerRequest(
        worker_id=new_worker_id(),
        signed_backend_manifest_digest=D,
        approved_isolation_profile_digest=profile.digest(),
        execution_budget_digest=D,
        request_nonce="n-" + new_worker_id(), role="worker")


def test_occupied_unit_never_reports_verified_empty(tmp_path):
    sd = OccupiedSystemd(tmp_path)
    broker, profile = _broker(tmp_path, sd)
    rec = broker.launch(_req(profile))
    # status() must NOT claim empty while the kernel reports occupied
    st = broker.status(rec)
    assert st["cgroup_populated"] is True
    assert st["containment"] == "running"
    rep = broker.terminate(rec, deadline_seconds=1.5)
    assert rep["verified_empty"] is False
    assert rep["containment"] == "running"


def test_executor_containment_status_matches(tmp_path):
    sd = OccupiedSystemd(tmp_path)
    binary = tmp_path / "nsjail"
    binary.write_bytes(b"fake")
    binary.chmod(0o755)
    profile = load_profile(PROFILE_PATH)
    ex = NsJailWorkerExecutor(
        broker=WorkerLaunchBroker(
            worker_state_root=tmp_path / "workers2",
            approved_profiles={profile.digest(): profile},
            artifact_resolver=lambda d: str(tmp_path / "a"),
            nsjail_path=str(binary),
            nsjail_digest="sha256:" + hashlib.sha256(
                binary.read_bytes()).hexdigest(),
            worker_uid=990, worker_gid=990,
            systemd=sd, nonce_dir=tmp_path / "nonces2"),
        profile=profile)
    dirs = ex.prepare(worker_id=None, isolation=DEFAULT_ISOLATION)
    launched = ex.launch(
        dirs,
        worker_argv=[sys.executable, "-m",
                     "minagi.runtime.worker_backend"],
        env={}, isolation=DEFAULT_ISOLATION,
        backend_manifest_digest=D)
    cs = ex.containment_status(launched)
    assert cs["cgroup_populated"] is True
    assert cs["containment"] == "running"
    assert cs["verified_empty"] is False
    rep = ex.terminate(launched, deadline_seconds=1.5)
    assert rep.verified is False


def test_invocation_mismatch_blocks_removed_verified(tmp_path):
    """A reused unit name with a LIVE different invocation must not
    satisfy this worker's teardown — identity binds to the recorded
    InvocationID, not the name."""
    sd = OccupiedSystemd(tmp_path)
    broker, profile = _broker(tmp_path, sd)
    rec = broker.launch(_req(profile))
    assert rec.invocation_id == "inv-1"
    # same unit name, but a NEW invocation is now answering —
    # the cgroup dir is gone yet the unit is active under a
    # different InvocationID
    for f in sd.cg.iterdir():
        f.unlink()
    sd.cg.rmdir()
    sd.cgroup_path = lambda unit_name: None
    sd.is_active = lambda unit_name: True
    sd.show = lambda u, p: "inv-2" if p == "InvocationID" \
        else "/system.slice/x.service"
    rep = broker.terminate(rec, deadline_seconds=1.5)
    assert rep["verified_empty"] is False
    assert rep["containment"] == "unknown"


def test_gone_cgroup_same_invocation_is_removed_verified(tmp_path):
    """The positive case: cgroup dir removed + unit inactive + the
    teardown belongs to THIS invocation -> REMOVED_VERIFIED."""
    sd = OccupiedSystemd(tmp_path)
    broker, profile = _broker(tmp_path, sd)
    rec = broker.launch(_req(profile))
    for f in sd.cg.iterdir():
        f.unlink()
    sd.cg.rmdir()
    sd.cgroup_path = lambda unit_name: None
    rep = broker.terminate(rec, deadline_seconds=1.5)
    assert rep["verified_empty"] is True
    assert rep["containment"] == "removed_verified"
