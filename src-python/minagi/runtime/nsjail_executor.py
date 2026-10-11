"""v16.4.8 NsJail worker executor (WP7/WP8) — plan Step 12.

The production worker boundary: the supervisor's approved worker
specification becomes ONE managed systemd unit whose entire process
tree is kernel-owned (cgroup v2), running the pinned NsJail binary
with a deterministically generated protobuf configuration —
namespaces, read-only mounts, a separate OS identity, a dead
network, and resource bounds the kernel enforces.

The executor never degrades: every launch goes through the
:class:`WorkerLaunchBroker`, which refuses unapproved profiles,
unmeasured binaries, replayed nonces, and any unit that never
reaches a live cgroup. A ``WorkerBackend`` holding this executor
cannot fall back to development semantics.

IPC: ``systemd-run --pipe`` connects the unit's stdin/stdout to the
executor's channel — the SAME framed-JSON v2 protocol as the
development transport. The worker sees only that channel; the
socket never becomes an administrative interface (the closed
protocol vocabulary is enforced in both directions).
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from .isolation_profile import IsolationProfile
from .worker_executor import (ExecutorRefused, LaunchedWorker,
                              TerminationReport, WorkerDirs)
from .worker_isolation import WorkerIsolationPolicy, make_private_tmp
from .worker_launch_broker import (BrokerError, BrokerRefused,
                                   LaunchWorkerRequest,
                                   SystemdClient, WorkerLaunchBroker)


class NsJailWorkerExecutor:
    """Production executor — systemd cgroup v2 + NsJail namespaces.

    broker      a configured WorkerLaunchBroker (approved profile
                registry, pinned nsjail digest, worker identity)
    profile     the IsolationProfile this executor enforces — its
                digest is bound to every launch
    """
    name = "nsjail"
    needs_subreaper = False  # the cgroup is the containment authority

    def __init__(self, *, broker: WorkerLaunchBroker,
                 profile: IsolationProfile,
                 allowed_roles=("worker", "probe")):
        self.broker = broker
        self.profile = profile
        self.profile_digest = profile.digest()
        self.allowed_roles = tuple(allowed_roles)
        self._launched: dict[str, object] = {}

    @classmethod
    def from_profile(cls, profile: IsolationProfile, *,
                     worker_state_root, nsjail_path, nsjail_digest,
                     worker_uid, worker_gid,
                     artifact_resolver=None, user_manager=False,
                     systemd=None, worker_python=None,
                     python_path_entries=(), denied_paths=(),
                     nonce_dir=None) -> "NsJailWorkerExecutor":
        broker = WorkerLaunchBroker(
            worker_state_root=worker_state_root,
            approved_profiles={profile.digest(): profile},
            artifact_resolver=artifact_resolver,
            nsjail_path=nsjail_path, nsjail_digest=nsjail_digest,
            worker_uid=worker_uid, worker_gid=worker_gid,
            worker_python=worker_python,
            python_path_entries=python_path_entries,
            denied_paths=denied_paths, user_manager=user_manager,
            systemd=systemd, nonce_dir=nonce_dir)
        return cls(broker=broker, profile=profile)

    # --- host contract -------------------------------------------------
    def validate_host(self) -> dict:
        """Refuse — loudly — unless every layer of the production
        boundary can actually be established here: Linux, a working
        systemd client, unified cgroup v2, the pinned NsJail binary,
        and a namespace path (CAP_SYS_ADMIN or unprivileged userns)."""
        from .worker_isolation import _cap_eff, _unprivileged_userns
        checks = self.broker.host_report()
        checks["cap_sys_admin"] = _cap_eff(21)
        checks["unprivileged_userns"] = _unprivileged_userns()
        checks["euid"] = os.geteuid()
        checks["executor"] = self.name
        ok = checks["ok"] and (
            self.broker.user_manager
            or checks["cap_sys_admin"]
            or checks["unprivileged_userns"]
            or os.geteuid() == 0)
        checks["ok"] = bool(ok)
        checks["profile_digest"] = self.profile_digest
        return checks

    def require_host(self) -> dict:
        report = self.validate_host()
        if not report.get("ok"):
            raise ExecutorRefused(
                "the NsJail worker boundary cannot be established on "
                f"this host: {report}")
        return report

    # --- lifecycle ------------------------------------------------------
    def prepare(self, *, worker_id: str | None,
                isolation: WorkerIsolationPolicy) -> WorkerDirs:
        """Workspace is broker-owned: ``<root>/<worker_id>/scratch``.
        The private_tmp/overflow contract mirrors the development
        executor so the protocol's overflow channel is unchanged."""
        wid = worker_id or f"w{secrets.token_hex(6)}"
        ws, scratch, overflow, fd = self.broker.create_workspace(wid)
        return WorkerDirs(
            private_tmp=scratch, overflow_dir=overflow,
            overflow_fd=fd, unit_token=f"{os.getpid()}-"
            f"{secrets.token_hex(8)}", worker_id=wid,
            metadata={"workspace": ws})

    def launch(self, dirs: WorkerDirs, *, worker_argv, env: dict,
               isolation: WorkerIsolationPolicy, spec=None,
               policy_digest: str = "",
               artifact_digests: tuple = (),
               backend_manifest_digest: str = "",
               role: str = "worker") -> LaunchedWorker:
        if role not in self.allowed_roles:
            raise ExecutorRefused(f"role {role!r} is not launchable")
        # The jail's exec_bin is FIXED by the broker (pinned
        # interpreter + worker module) — the argv the backend composed
        # only validates that it names the same worker entry.
        tail = [str(a) for a in list(worker_argv)[-2:]]
        if role == "worker" and tail != ["-m",
                                         "minagi.runtime.worker_backend"]:
            raise ExecutorRefused(
                f"nsjail executor refuses argv {list(worker_argv)!r} — "
                "the jail executes only the pinned worker entry")
        nonce = secrets.token_hex(16)
        req = LaunchWorkerRequest(
            worker_id=dirs.worker_id,
            signed_backend_manifest_digest=(
                backend_manifest_digest
                or "sha256:" + "0" * 64),
            approved_artifact_digests=tuple(artifact_digests),
            approved_isolation_profile_digest=self.profile_digest,
            execution_budget_digest="",
            request_nonce=nonce, role=role)
        # the unit token travels in the jail env — defense in depth
        # under the authoritative cgroup boundary
        env = dict(env)
        env["MINAGI_WORKER_UNIT"] = dirs.unit_token
        record = self.broker.launch(
            req, env=env,
            extra_ro_paths=self._extra_ro(isolation),
            unit_token=dirs.unit_token)
        self._launched[dirs.worker_id] = record
        return LaunchedWorker(
            worker_id=dirs.worker_id, proc=record.proc,
            tx=record.proc.stdin, rx=record.proc.stdout,
            dirs=dirs, unit_name=record.unit_name,
            cgroup_path=(str(record.cgroup_path)
                         if record.cgroup_path else None),
            cgroup=None, tracker=None, executor=self.name,
            record=record,
            digests=dict(record.digests))

    def _extra_ro(self, isolation: WorkerIsolationPolicy) -> tuple:
        """The admitted snapshot dirs the WORKER must read — supplied
        through the frozen policy's mount_ro (supervisor-computed),
        still re-screened by the broker against denied paths."""
        return tuple(str(p) for p in (isolation.mount_ro or ()))

    # --- teardown -------------------------------------------------------
    def status(self, launched: LaunchedWorker) -> dict:
        return {"executor": self.name,
                **self.broker.status(launched.record)}

    def terminate(self, launched: LaunchedWorker, *,
                  deadline_seconds: float) -> TerminationReport:
        rep = self.broker.terminate(
            launched.record, deadline_seconds=deadline_seconds)
        survivors = frozenset() if rep["verified_empty"] else \
            frozenset({launched.worker_id})
        return TerminationReport(
            survivors=survivors, verified=bool(rep["verified_empty"]),
            mechanism="systemd-cgroupv2",
            detail=("" if rep["verified_empty"]
                    else f"unit {rep['unit']} still populated "
                    f"(populated={rep['cgroup_populated']}) after "
                    "stop+kill — containment not proven"))

    def containment_status(self, launched: LaunchedWorker) -> dict:
        st = self.broker.status(launched.record)
        return {"mechanism": "systemd-cgroupv2",
                "alive": st["active"] or st["client_alive"],
                "unit": st["unit"],
                "cgroup_path": st["cgroup_path"],
                "cgroup_populated": st["cgroup_populated"],
                "verified_empty": (st["cgroup_populated"] is False)
                or (not st["active"] and not st["client_alive"])}

    def release(self, launched: LaunchedWorker) -> None:
        self._launched.pop(launched.worker_id, None)
        launched.close_streams()
        try:
            self.broker.release(launched.record)
        except (BrokerError, BrokerRefused, OSError):
            pass


__all__ = ["NsJailWorkerExecutor"]
