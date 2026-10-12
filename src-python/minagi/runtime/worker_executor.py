"""v16.4.8 worker-executor contract (WP7) — plan Step 10.

The executor is the ONLY path from an approved worker specification
to a running worker. Two implementations exist:

  * :class:`DevelopmentWorkerExecutor` — the existing local machinery
    (launcher spec, optional cgroup, kqueue tracking, process-group
    teardown). Trusted local testing ONLY; it carries the same
    platform limits v16.4.7 documented.
  * ``NsJailWorkerExecutor`` (``nsjail_executor``) — a systemd-managed
    cgroup running the pinned NsJail binary: namespace, filesystem,
    network, identity and syscall confinement the kernel enforces.
    This is the production boundary.

There is no silent fallback: a backend holding the NsJail executor
cannot degrade to development semantics, and the production service
refuses any configuration that asks for one (Step 15).

Executor lifecycle::

    dirs     = executor.prepare(worker_id=..., isolation=...)
    launched = executor.launch(dirs, worker_argv=..., env=..., ...)
    report   = executor.terminate(launched, deadline_seconds=...)
    status   = executor.containment_status(launched)
    executor.release(launched)

``prepare`` and ``launch`` are split because the supervisor builds
the worker's environment BETWEEN them — the private scratch path and
the unit token must exist before ``build_env`` runs.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .worker_isolation import (
    IsolationError, KqueueDescendantTracker, WorkerCgroup,
    WorkerIsolationPolicy, WorkerProcessController, build_launch_spec,
    descendants_of, make_private_tmp, make_worker_scratch,
    needs_launcher, new_worker_unit, spec_to_fd, supported, wrap_argv)


class ExecutorRefused(PermissionError):
    """The executor cannot provide its declared boundary on this
    host — fail closed, never fall back."""


@dataclass
class WorkerDirs:
    """The worker's private filesystem footprint, created by
    ``prepare`` before the environment is constructed."""
    private_tmp: Path
    overflow_dir: Path
    overflow_fd: int
    unit_token: str
    worker_id: str
    cgroup: object = None          # WorkerCgroup (development path)
    rendered_profile: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TerminationReport:
    """The verified outcome of a unit teardown."""
    survivors: frozenset
    verified: bool
    mechanism: str
    detail: str = ""


class LaunchedWorker:
    """Executor-neutral handle for one running worker containment
    unit. ``tx``/``rx`` are the framed-JSON protocol streams —
    pipes to a direct child (development) or the ``systemd-run
    --pipe`` channel to a managed unit (nsjail); both carry the same
    validated protocol."""
    __slots__ = ("worker_id", "proc", "tx", "rx", "dirs",
                 "unit_name", "cgroup_path", "cgroup", "tracker",
                 "digests", "executor", "record", "tracked_pids")

    def __init__(self, *, worker_id, proc, tx, rx, dirs,
                 unit_name=None, cgroup_path=None, cgroup=None,
                 tracker=None, digests=None, executor="development",
                 record=None):
        self.worker_id = worker_id
        self.proc = proc
        self.tx = tx
        self.rx = rx
        self.dirs = dirs
        self.unit_name = unit_name
        self.cgroup_path = cgroup_path
        self.cgroup = cgroup
        self.tracker = tracker
        self.digests = dict(digests or {})
        self.executor = executor
        self.record = record
        self.tracked_pids = frozenset()

    @property
    def pid(self) -> int:
        """The supervisor-side process identity. For the development
        executor this is the worker pid; for the nsjail executor it is
        the systemd-run client pid — the jail's internal pid is not a
        host-meaningful identity (pid namespace)."""
        return int(self.proc.pid or 0) if self.proc is not None else 0

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def wait(self, timeout: float | None = None):
        try:
            return self.proc.wait(timeout=timeout) \
                if self.proc is not None else 0
        except subprocess.TimeoutExpired:
            return None

    def close_streams(self) -> None:
        for stream in (self.tx, self.rx):
            try:
                if stream is not None:
                    stream.close()
            except Exception:  # noqa: BLE001
                pass


class WorkerExecutor(Protocol):
    """The launch/terminate contract every worker boundary must
    implement — see module docstring for the lifecycle."""
    name: str
    needs_subreaper: bool

    def validate_host(self) -> dict:
        ...

    def prepare(self, *, worker_id: str | None,
                isolation: WorkerIsolationPolicy) -> WorkerDirs:
        ...

    def launch(self, dirs: WorkerDirs, *, worker_argv, env: dict,
               isolation: WorkerIsolationPolicy, spec=None,
               policy_digest: str = "",
               artifact_digests: tuple = (),
               backend_manifest_digest: str = "",
               extra_ro: tuple = (),
               role: str = "worker",
               authorization: dict | None = None) -> LaunchedWorker:
        ...

    def status(self, launched: LaunchedWorker) -> dict:
        ...

    def terminate(self, launched: LaunchedWorker, *,
                  deadline_seconds: float) -> TerminationReport:
        ...

    def containment_status(self, launched: LaunchedWorker) -> dict:
        ...

    def release(self, launched: LaunchedWorker) -> None:
        ...


class DevelopmentWorkerExecutor:
    """The v16.4.6/v16.4.7 local machinery behind the executor
    interface — same launcher spec, cgroup, kqueue tracker and
    process-group teardown as before, now one testable unit.

    TRUSTED TESTING ONLY. Its guarantees are the platform's weakest
    mechanism (env-marker scanning, ancestry graphs); it is refused
    under ``--production`` by the service."""
    name = "development"
    needs_subreaper = True

    def __init__(self, *, executable=None, term_grace: float = 2.0,
                 kill_grace: float = 5.0):
        self.executable = str(executable or sys.executable)
        self.controller = WorkerProcessController(
            term_grace=term_grace, kill_grace=kill_grace)

    # --- host contract -------------------------------------------------
    def validate_host(self) -> dict:
        caps = supported()
        return {"executor": self.name, "ok": True, "platform":
                caps["platform"], "capabilities": caps}

    # --- lifecycle ------------------------------------------------------
    def prepare(self, *, worker_id: str | None,
                isolation: WorkerIsolationPolicy) -> WorkerDirs:
        if isolation.demote_to is not None:
            private_tmp = make_worker_scratch(
                isolation.demote_to, group_gid=os.getgid())
        else:
            private_tmp = make_private_tmp()
        overflow_dir = private_tmp / "overflow"
        try:
            overflow_dir.mkdir(mode=0o750)
            if isolation.demote_to is not None:
                os.chown(overflow_dir, int(isolation.demote_to[0]),
                         os.getgid())
            # SEC-504: pin the directory descriptor NOW, before the
            # worker can influence the path — later result reads open
            # the leaf relative to this inode.
            overflow_fd = os.open(
                str(overflow_dir),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            shutil.rmtree(private_tmp, ignore_errors=True)
            raise IsolationError(
                f"could not establish the worker result directory: "
                f"{exc}") from exc
        token = new_worker_unit()
        return WorkerDirs(
            private_tmp=private_tmp, overflow_dir=overflow_dir,
            overflow_fd=overflow_fd, unit_token=token,
            worker_id=worker_id or token)

    def launch(self, dirs: WorkerDirs, *, worker_argv, env: dict,
               isolation: WorkerIsolationPolicy, spec=None,
               policy_digest: str = "", artifact_digests: tuple = (),
               backend_manifest_digest: str = "",
               extra_ro: tuple = (),
               role: str = "worker",
               authorization: dict | None = None) -> LaunchedWorker:
        # extra_ro is a jail-mount contract — the development
        # executor runs on the host where every path is already
        # reachable; the parameter exists for signature parity.
        argv0, rendered = wrap_argv(list(worker_argv), isolation,
                                    render_vars={
                                        "SCRATCH": str(dirs.private_tmp)})
        dirs.rendered_profile = rendered
        # Kernel-owned containment where available (Linux cgroup v2):
        # created BEFORE spawn so the launcher self-registers and every
        # descendant is born inside the unit.
        cgroup = WorkerCgroup.create(dirs.unit_token)
        dirs.cgroup = cgroup
        spec_fd = None
        release_spec = None
        popen_kwargs = {}
        if needs_launcher(isolation) or cgroup is not None:
            launch_spec = build_launch_spec(
                isolation, exec_argv=argv0, scratch=dirs.private_tmp,
                cgroup_path=(cgroup.path if cgroup else None))
            spec_fd, release_spec = spec_to_fd(launch_spec)
            argv = [self.executable,
                    "-m", "minagi.runtime.worker_launcher"]
            env = dict(env)
            env["MINAGI_LAUNCH_SPEC_FD"] = str(spec_fd)
            popen_kwargs["pass_fds"] = (spec_fd,)
        else:
            argv = argv0
        try:
            proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                env=env, cwd=str(dirs.private_tmp),
                start_new_session=True, **popen_kwargs)
        except (OSError, IsolationError) as exc:
            try:
                os.close(dirs.overflow_fd)
            except OSError:
                pass
            shutil.rmtree(dirs.private_tmp, ignore_errors=True)
            if rendered is not None:
                try:
                    os.unlink(rendered)
                except OSError:
                    pass
            if cgroup is not None:
                cgroup.release()
            raise IsolationError(
                f"worker interpreter could not start: {exc}") from exc
        finally:
            if release_spec is not None:
                try:
                    release_spec()
                except OSError:
                    pass
        tracker = KqueueDescendantTracker.attach(proc.pid)
        return LaunchedWorker(
            worker_id=dirs.worker_id, proc=proc, tx=proc.stdin,
            rx=proc.stdout, dirs=dirs,
            unit_name=None, cgroup_path=(
                str(cgroup.path) if cgroup else None),
            cgroup=cgroup, tracker=tracker, executor=self.name,
            digests={"policy": policy_digest,
                     "artifacts": tuple(artifact_digests),
                     "backend_manifest": backend_manifest_digest})

    # --- teardown --------------------------------------------------------
    def terminate(self, launched: LaunchedWorker, *,
                  deadline_seconds: float) -> TerminationReport:
        proc = launched.proc
        if proc is not None and proc.poll() is None:
            launched.tracked_pids = frozenset(
                descendants_of(proc.pid) | {proc.pid}
                | (launched.tracker.live_pids()
                   if launched.tracker else set()))
        survivors = self.controller.terminate_tree(
            proc, leader_pid=(proc.pid if proc else None),
            tracked=set(launched.tracked_pids),
            unit=launched.dirs.unit_token,
            extra_tracked=(launched.tracker.live_pids
                           if launched.tracker else None),
            cgroup=launched.cgroup)
        return TerminationReport(
            survivors=frozenset(survivors),
            verified=not survivors,
            mechanism=("cgroup" if launched.cgroup is not None
                       else "kqueue+marker+group"),
            detail="" if not survivors else
            f"descendants {sorted(survivors)} could not be confirmed "
            "dead — containment is not proven")

    def status(self, launched: LaunchedWorker) -> dict:
        return {"executor": self.name, "alive": launched.alive(),
                "pid": launched.pid,
                "containment": self.containment_status(launched)}

    def containment_status(self, launched: LaunchedWorker) -> dict:
        survivors = self.controller.verify_terminated(
            launched.pid, tracked=set(launched.tracked_pids),
            unit=launched.dirs.unit_token,
            extra_tracked=(launched.tracker.live_pids
                           if launched.tracker else None),
            cgroup=launched.cgroup)
        return {"mechanism": ("cgroup" if launched.cgroup is not None
                              else "kqueue+marker+group"),
                "alive": launched.alive(),
                "survivors": sorted(survivors),
                "verified_empty": not launched.alive() and not survivors}

    def release(self, launched: LaunchedWorker) -> None:
        launched.close_streams()
        dirs = launched.dirs
        if dirs.overflow_fd is not None:
            try:
                os.close(dirs.overflow_fd)
            except OSError:
                pass
            dirs.overflow_fd = None
        if launched.tracker is not None:
            launched.tracker.close()
            launched.tracker = None
        if launched.cgroup is not None:
            launched.cgroup.release()
            launched.cgroup = None
        if dirs.rendered_profile is not None:
            try:
                os.unlink(dirs.rendered_profile)
            except OSError:
                pass
            dirs.rendered_profile = None
        if dirs.private_tmp is not None:
            shutil.rmtree(dirs.private_tmp, ignore_errors=True)


__all__ = ["DevelopmentWorkerExecutor", "ExecutorRefused",
           "LaunchedWorker", "TerminationReport", "WorkerDirs",
           "WorkerExecutor"]
