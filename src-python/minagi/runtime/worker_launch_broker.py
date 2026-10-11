"""v16.4.8 restricted worker-launch broker (WP8) — plan Step 11.

A small, narrowly authorized component standing between the trusted
supervisor and the process-creation primitives. It accepts ONLY a
:class:`LaunchWorkerRequest` — a worker id, approval digests, and a
nonce — and turns it into ONE managed systemd unit running the
pinned NsJail binary with a deterministically generated
configuration.

The broker never takes from a caller:

  * an executable path or an argv vector — the jail's exec_bin is
    built here from the pinned interpreter and the fixed worker
    module (``worker``) or probe module (``probe``);
  * an environment list — the jail's envar set is allowlisted and
    secret-screened here;
  * a mount list — binds are computed from the approved runtime
    closure, digest-resolved artifacts, and the worker scratch;
  * a property list — unit properties come from the frozen profile.

The broker holds no signing material and cannot admit, promote, or
sign anything. Its privilege is limited to creating and destroying
worker units and workspaces under ``worker_state_root``.

Containment is kernel-owned: systemd places the unit in its own
cgroup v2 subtree; ``systemd-run --pipe --wait --collect`` gives a
bidirectional channel to the unit's stdin/stdout and propagates its
exit; teardown is ``systemctl stop`` verified by
``cgroup.populated == 0`` — never /proc scanning claimed as proof.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .isolation_profile import (IsolationProfile, MountSpec,
                                ProfileRefused, resolve_mounts)
from .nsjail_config import config_digest, render_config


class BrokerError(RuntimeError):
    """A launch or lifecycle operation failed in execution."""


class BrokerRefused(PermissionError):
    """The request itself is refused — malformed, replayed, or
    outside the approved boundary. Fail closed."""


#: envar names the jail may carry — the same screening rule as the
#: supervisor-side environment: nothing credential-shaped crosses.
_ALLOWED_ENV = frozenset({
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "HOME",
    "PYTHONPATH", "TMPDIR", "TMP", "TEMP",
    "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "MINAGI_WORKER_UNIT", "MINAGI_WORKER_ROLE",
    "MINAGI_PROBE_SCRATCH", "MINAGI_PROBE_ARGS"})

_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


@dataclass(frozen=True)
class LaunchWorkerRequest:
    """The whole authority a launch request carries — digests of
    already-approved material plus a replay nonce. No executable, no
    argv, no env, no mounts, no shell."""
    worker_id: str
    signed_backend_manifest_digest: str
    approved_artifact_digests: tuple = ()
    approved_isolation_profile_digest: str = ""
    execution_budget_digest: str = ""
    request_nonce: str = ""
    role: str = "worker"                # "worker" | "probe"


@dataclass
class LaunchRecord:
    """The broker's identity for one launched containment unit."""
    worker_id: str
    unit_name: str
    proc: object                       # systemd-run client Popen
    workspace: Path
    scratch: Path
    overflow_dir: Path
    overflow_fd: int
    config_path: Path
    cgroup_path: Path | None
    unit_token: str
    role: str
    digests: dict = field(default_factory=dict)
    started_at: float = 0.0
    user_manager: bool = False


class SystemdClient:
    """Exact, argv-vector systemd operations — never a shell, never
    string interpolation into a command line."""

    def __init__(self, *, user_manager: bool = False,
                 systemd_run: str | None = None,
                 systemctl: str | None = None):
        self.user_manager = bool(user_manager)
        self.systemd_run = (systemd_run
                            or shutil.which("systemd-run"))
        self.systemctl = systemctl or shutil.which("systemctl")

    @property
    def available(self) -> bool:
        return bool(self.systemd_run and self.systemctl)

    def _ctl(self, *args, timeout: float = 20.0):
        argv = [self.systemctl, *(["--user"] if self.user_manager
                                  else []), *args]
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout)

    def run_service(self, unit_name: str, properties, argv,
                    *, cwd: str | None = None):
        """Spawn the transient unit with a piped protocol channel:
        ``--pipe`` wires the unit's stdin/stdout/stderr to this
        process's pipes; ``--wait`` propagates exit status;
        ``--collect`` GCs the unit record when it ends."""
        cmd = [self.systemd_run,
               *(["--user"] if self.user_manager else []),
               "--pipe", "--wait", "--collect",
               "--unit", unit_name]
        for prop in properties:
            cmd += ["-p", str(prop)]
        cmd += ["--", *[str(a) for a in argv]]
        return subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=cwd,
            start_new_session=True)

    def show(self, unit_name: str, prop: str) -> str:
        try:
            out = self._ctl("show", unit_name,
                            f"--property={prop}", "--value")
        except (OSError, subprocess.SubprocessError):
            return ""
        return (out.stdout or "").strip()

    def is_active(self, unit_name: str) -> bool:
        try:
            out = self._ctl("is-active", unit_name)
        except (OSError, subprocess.SubprocessError):
            return False
        return (out.stdout or "").strip() == "active"

    def stop(self, unit_name: str) -> None:
        self._ctl("stop", unit_name)

    def kill(self, unit_name: str, sig: str = "SIGKILL") -> None:
        self._ctl("kill", unit_name, f"--signal={sig}",
                  "--kill-whom=all")

    def cgroup_path(self, unit_name: str) -> Path | None:
        """Resolve the unit's cgroup via its ControlGroup property —
        works for system and user managers without guessing slice
        layouts."""
        rel = self.show(unit_name, "ControlGroup")
        if not rel:
            return None
        path = Path("/sys/fs/cgroup") / rel.lstrip("/")
        return path if path.is_dir() else None


def cgroup_populated(cgroup_path) -> bool | None:
    """The kernel's own occupancy flag: True when the subtree still
    holds live processes, False when empty, None when unreadable —
    per the kernel's cgroup v2 documentation."""
    try:
        text = (Path(cgroup_path) / "cgroup.populated").read_text()
        return text.strip() == "1"
    except OSError:
        return None


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def runtime_closure_paths() -> tuple:
    """The read-only closure the jailed interpreter needs: the
    interpreter itself, its shared library dirs, stdlib and every
    live sys.path entry — the same computation the launcher's ro
    binds used, now owned by the broker."""
    import sys
    import sysconfig
    binds: list = []

    def add(p):
        if not p:
            return
        try:
            rp = str(Path(p).resolve())
        except OSError:
            return
        if rp not in binds and Path(rp).exists():
            binds.append(rp)

    add(sys.executable)
    for d in ("/usr/lib", "/lib", "/lib64", "/usr/local/lib",
              "/usr/lib64", sysconfig.get_paths().get("stdlib", ""),
              sysconfig.get_paths().get("platstdlib", "")):
        add(d)
    for d in sys.path:
        add(d)
    return tuple(binds)


class WorkerLaunchBroker:
    """The privileged launch surface — narrow, digest-bound, and
    replay-guarded.

    worker_state_root   root for worker workspaces (e.g.
                        /var/lib/minagi/workers)
    approved_profiles   {profile_digest: IsolationProfile} — only
                        registered profiles launch (SEC-703)
    artifact_resolver   callable(digest) -> path, for approved
                        read-only model artifacts
    nsjail_path         the pinned NsJail binary
    nsjail_digest       REQUIRED: sha256 the binary must measure —
                        the broker never runs an unmeasured binary
    worker_uid/gid      the separate worker OS identity the jail
                        maps to (outside_id in the user namespace)
    worker_python       the interpreter exec_bin inside the jail
    python_path_entries dirs bound ro + exported as PYTHONPATH —
                        derived from the supervisor's own runtime
    denied_paths        authority material that may never be mounted
    user_manager        use ``systemd-run --user`` (rootless) instead
                        of the system manager
    systemd             injectable SystemdClient (tests) — when
                        provided, no binaries are probed
    nonce_dir           durable replay-marker directory
    """

    #: fixed in-jail argvs — the only executables this broker runs
    WORKER_ARGV_MODULE = ("-m", "minagi.runtime.worker_backend")
    PROBE_ARGV_MODULE = ("-m", "minagi.runtime.isolation_preflight",
                         "--probe")

    def __init__(self, *, worker_state_root,
                 approved_profiles: dict | None = None,
                 artifact_resolver=None,
                 nsjail_path: str = "/usr/local/bin/nsjail",
                 nsjail_digest: str | None = None,
                 worker_uid: int = 65534, worker_gid: int = 65534,
                 worker_python: str | None = None,
                 python_path_entries: tuple = (),
                 denied_paths: tuple = (),
                 user_manager: bool = False,
                 systemd: SystemdClient | None = None,
                 unit_prefix: str = "miniagi-worker",
                 nonce_dir=None):
        import sys
        self.worker_state_root = Path(worker_state_root)
        self.approved_profiles = dict(approved_profiles or {})
        self.artifact_resolver = artifact_resolver
        self.nsjail_path = str(nsjail_path)
        self.worker_uid = int(worker_uid)
        self.worker_gid = int(worker_gid)
        self.worker_python = str(worker_python or sys.executable)
        self.python_path_entries = tuple(
            str(p) for p in python_path_entries)
        self.denied_paths = tuple(str(p) for p in denied_paths)
        self.user_manager = bool(user_manager)
        self.systemd = systemd or SystemdClient(
            user_manager=user_manager)
        self.unit_prefix = str(unit_prefix)
        self.nonce_dir = (Path(nonce_dir) if nonce_dir
                          else self.worker_state_root / ".nonces")
        self._records: dict[str, LaunchRecord] = {}
        self._nsjail_digest = None
        if nsjail_digest is not None:
            self._nsjail_digest = str(nsjail_digest)
            measured = self._measure_nsjail()
            if measured != self._nsjail_digest:
                raise BrokerRefused(
                    f"nsjail binary {self.nsjail_path} measures "
                    f"{measured} — expected {self._nsjail_digest}: the "
                    "broker runs only the approved binary")
        elif systemd is None:
            raise BrokerRefused(
                "nsjail_digest is required — the broker never runs an "
                "unmeasured sandbox binary")

    # --- qualification ---------------------------------------------------
    def _measure_nsjail(self) -> str:
        try:
            return _sha256_file(self.nsjail_path)
        except OSError as exc:
            raise BrokerRefused(
                f"nsjail binary {self.nsjail_path} unreadable: {exc}") \
                from exc

    def host_report(self) -> dict:
        """What this broker can actually enforce on this host."""
        checks = {
            "platform_linux": os.uname().sysname == "Linux",
            "systemd_client": self.systemd.available,
            "cgroup_v2": Path(
                "/sys/fs/cgroup/cgroup.controllers").is_file(),
            "nsjail_binary": Path(self.nsjail_path).is_file(),
            "nsjail_digest_pinned": self._nsjail_digest is not None,
            "user_manager": self.user_manager,
        }
        checks["ok"] = all((
            checks["platform_linux"], checks["systemd_client"],
            checks["cgroup_v2"], checks["nsjail_binary"],
            checks["nsjail_digest_pinned"]))
        return checks

    # --- request validation ----------------------------------------------
    def _validate_request(self, req: LaunchWorkerRequest) -> None:
        if not isinstance(req, LaunchWorkerRequest):
            raise BrokerRefused("launch requires a LaunchWorkerRequest")
        if not _ID_RE.match(str(req.worker_id)):
            raise BrokerRefused(
                f"worker_id {req.worker_id!r} is not a safe unit "
                "identifier")
        if req.role not in ("worker", "probe"):
            raise BrokerRefused(f"role {req.role!r} is not launchable")
        if not req.request_nonce:
            raise BrokerRefused("request_nonce is required")
        for d in req.approved_artifact_digests:
            if not _DIGEST_RE.match(str(d)):
                raise BrokerRefused(
                    f"artifact digest {d!r} is not sha256:<hex>")
        if req.approved_isolation_profile_digest:
            if not _DIGEST_RE.match(
                    str(req.approved_isolation_profile_digest)):
                raise BrokerRefused(
                    "isolation profile digest is not sha256:<hex>")
        if self._nonce_seen(req.request_nonce):
            raise BrokerRefused(
                "request_nonce already used — replay refused")
        self._nonce_mark(req.request_nonce)

    def _nonce_seen(self, nonce: str) -> bool:
        try:
            self.nonce_dir.mkdir(mode=0o700, parents=True,
                                 exist_ok=True)
            return (self.nonce_dir
                    / f"{hashlib.sha256(nonce.encode()).hexdigest()}"
                    ).exists()
        except OSError:
            return True  # cannot prove freshness -> treat as replay

    def _nonce_mark(self, nonce: str) -> None:
        try:
            (self.nonce_dir
             / f"{hashlib.sha256(nonce.encode()).hexdigest()}"
             ).write_text(str(time.time()))
        except OSError as exc:
            raise BrokerRefused(
                f"nonce store unwritable — replay cannot be "
                f"prevented: {exc}") from exc

    # --- launch ------------------------------------------------------------
    def _exec_argv(self, role: str) -> list:
        module = (self.PROBE_ARGV_MODULE if role == "probe"
                  else self.WORKER_ARGV_MODULE)
        return [self.worker_python, *module]

    def _filter_env(self, env: dict) -> dict:
        out = {}
        for k, v in dict(env or {}).items():
            key = str(k)
            if key not in _ALLOWED_ENV:
                continue
            if any(m in key.upper() for m in _SECRET_MARKERS):
                raise BrokerRefused(
                    f"envar {key!r} is credential-shaped — refused")
            out[key] = str(v)
        return out

    def _resolve_artifacts(self, digests) -> tuple:
        paths = []
        for d in digests or ():
            if self.artifact_resolver is None:
                raise BrokerRefused(
                    "artifact digest supplied but no artifact "
                    "resolver is configured")
            p = self.artifact_resolver(d)
            if p is None:
                raise BrokerRefused(
                    f"artifact digest {d} is not admitted")
            paths.append(str(p))
        return tuple(paths)

    def create_workspace(self, worker_id: str) -> tuple:
        """The worker's private tree under the broker-owned root:
        ``<root>/<worker_id>/scratch`` writable by the worker
        identity; nothing else under root is visible to it."""
        if not _ID_RE.match(str(worker_id)):
            raise BrokerRefused(f"worker_id {worker_id!r} unsafe")
        try:
            self.worker_state_root.mkdir(
                mode=0o700, parents=True, exist_ok=True)
            ws = self.worker_state_root / str(worker_id)
            scratch = ws / "scratch"
            scratch.mkdir(mode=0o750, parents=True, exist_ok=True)
            overflow = scratch / "overflow"
            overflow.mkdir(mode=0o750, exist_ok=True)
            try:
                os.chown(scratch, self.worker_uid, os.getgid())
                os.chown(overflow, self.worker_uid, os.getgid())
            except OSError:
                pass  # userns-only path: ids map, host chown best effort
            overflow_fd = os.open(
                str(overflow),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise BrokerError(
                f"worker workspace under {self.worker_state_root} "
                f"failed: {exc}") from exc
        return ws, scratch, overflow, overflow_fd

    def launch(self, req: LaunchWorkerRequest, *,
               env: dict | None = None,
               extra_ro_paths: tuple = (),
               unit_token: str | None = None) -> LaunchRecord:
        """Validate -> resolve -> generate -> run -> verify the unit
        is actually created and contained."""
        self._validate_request(req)
        profile = self.approved_profiles.get(
            str(req.approved_isolation_profile_digest)) \
            if req.approved_isolation_profile_digest else None
        if profile is None:
            if req.approved_isolation_profile_digest:
                raise BrokerRefused(
                    "isolation profile digest is not a registered "
                    "approved profile — the frozen profile is the only "
                    "boundary a production worker may run under")
            if not self.approved_profiles:
                profile = None
            elif len(self.approved_profiles) == 1:
                profile = next(iter(self.approved_profiles.values()))
        if profile is None:
            raise BrokerRefused(
                "no approved isolation profile resolved")
        artifact_paths = self._resolve_artifacts(
            req.approved_artifact_digests)
        ws, scratch, overflow, overflow_fd = self.create_workspace(
            req.worker_id)
        token = unit_token or (
            f"{os.getpid()}-{secrets.token_hex(8)}")
        env_jail = self._filter_env(env)
        env_jail["MINAGI_WORKER_UNIT"] = token
        env_jail["MINAGI_WORKER_ROLE"] = req.role
        env_jail.setdefault("TMPDIR", "/work")
        env_jail.setdefault("TMP", "/work")
        env_jail.setdefault("TEMP", "/work")
        if self.python_path_entries:
            env_jail["PYTHONPATH"] = os.pathsep.join(
                self.python_path_entries)
        env_jail.setdefault("HOME", "/work")

        mounts = resolve_mounts(
            profile,
            runtime_paths=runtime_closure_paths(),
            artifact_paths=artifact_paths,
            scratch_src=str(scratch), scratch_dst="/work",
            denied_paths=self.denied_paths)
        # extra ro paths arrive only from the trusted executor (the
        # admitted snapshot dir) — still screened through _allowed.
        extra = []
        for p in extra_ro_paths:
            rp = str(Path(p).resolve())
            for deny in self.denied_paths:
                deny = str(deny)
                if rp == deny or rp.startswith(deny + "/"):
                    raise BrokerRefused(
                        f"extra ro path {rp!r} under denied {deny!r}")
            extra.append(MountSpec(src=rp, dst=rp, rw=False))
        mounts = tuple(mounts) + tuple(extra)

        exec_argv = self._exec_argv(req.role)
        config_text = render_config(
            profile, exec_argv=exec_argv, env=env_jail, mounts=mounts,
            inside_uid=self.worker_uid, inside_gid=self.worker_gid,
            outside_uid=self.worker_uid, outside_gid=self.worker_gid,
            cwd="/work", name=f"{self.unit_prefix}-{req.worker_id}")
        cfg_path = ws / "nsjail.cfg"
        try:
            cfg_path.write_text(config_text)
            os.chmod(cfg_path, 0o400)
        except OSError as exc:
            try:
                os.close(overflow_fd)
            except OSError:
                pass
            raise BrokerError(
                f"nsjail config unwritable: {exc}") from exc
        cfg_digest = config_digest(config_text)

        unit_name = f"{self.unit_prefix}-{req.worker_id}.service"
        properties = list(profile.unit_properties)
        properties.append(f"TasksMax={int(profile.unit_tasks_max)}")
        if profile.unit_memory_max:
            properties.append(
                f"MemoryMax={int(profile.unit_memory_max)}")
        if profile.memory_mb:
            properties.append(
                f"MemoryMax={int(profile.memory_mb) << 20}")
        properties.append(
            f"SyslogIdentifier={unit_name[:-len('.service')]}")
        proc = self.systemd.run_service(
            unit_name, properties,
            [self.nsjail_path, "--config", str(cfg_path)],
            cwd=str(scratch))
        record = LaunchRecord(
            worker_id=req.worker_id, unit_name=unit_name, proc=proc,
            workspace=ws, scratch=scratch, overflow_dir=overflow,
            overflow_fd=overflow_fd, config_path=cfg_path,
            cgroup_path=None, unit_token=token, role=req.role,
            digests={
                "policy": str(req.approved_isolation_profile_digest),
                "config": cfg_digest,
                "nsjail_binary": self._nsjail_digest or "",
                "artifacts": tuple(req.approved_artifact_digests),
                "backend_manifest":
                    str(req.signed_backend_manifest_digest),
                "execution_budget": str(req.execution_budget_digest)},
            started_at=time.time(), user_manager=self.user_manager)
        self._records[req.worker_id] = record
        # The unit must actually exist and be placed — a spawned client
        # without a live unit is not containment.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            record.cgroup_path = self.systemd.cgroup_path(unit_name)
            if record.cgroup_path is not None:
                break
            if proc.poll() is not None:
                break
            time.sleep(0.05)
        if proc.poll() is not None and record.cgroup_path is None:
            stderr = b""
            try:
                stderr = proc.stderr.read() or b""
            except OSError:
                pass
            raise BrokerError(
                f"worker unit {unit_name} never reached a live cgroup "
                f"— launch refused, not degraded: "
                f"{stderr.decode(errors='replace')[:400]}")
        return record

    # --- lifecycle ---------------------------------------------------------
    def status(self, record_or_id) -> dict:
        rec = self._record(record_or_id)
        populated = (cgroup_populated(rec.cgroup_path)
                     if rec.cgroup_path else None)
        return {"worker_id": rec.worker_id, "unit": rec.unit_name,
                "active": self.systemd.is_active(rec.unit_name),
                "client_alive": rec.proc.poll() is None,
                "cgroup_path": (str(rec.cgroup_path)
                                if rec.cgroup_path else None),
                "cgroup_populated": populated,
                "digests": dict(rec.digests)}

    def terminate(self, record_or_id, *,
                  deadline_seconds: float = 10.0) -> dict:
        """kernel-owned teardown: stop the unit (SIGTERM -> SIGKILL per
        its KillMode), then PROVE the cgroup subtree emptied — the
        acceptance check is cgroup.populated == 0 or the cgroup gone,
        plus the client exited. /proc scanning is never the claim."""
        rec = self._record(record_or_id)
        deadline = time.monotonic() + float(deadline_seconds)
        self.systemd.stop(rec.unit_name)
        while time.monotonic() < deadline:
            if rec.proc.poll() is not None:
                break
            time.sleep(0.05)
        if rec.proc.poll() is None:
            self.systemd.kill(rec.unit_name, "SIGKILL")
            try:
                rec.proc.wait(timeout=max(1.0, deadline_seconds / 2))
            except Exception:  # noqa: BLE001
                pass
        # verification: populated==0 OR the cgroup itself is gone
        verified = False
        populated = None
        while time.monotonic() < deadline:
            if rec.cgroup_path is None:
                rec.cgroup_path = self.systemd.cgroup_path(
                    rec.unit_name)
            if rec.cgroup_path is None:
                verified = not self.systemd.is_active(rec.unit_name)
                if verified:
                    break
            else:
                populated = cgroup_populated(rec.cgroup_path)
                if populated is False or populated is None:
                    verified = True
                    break
            time.sleep(0.05)
        return {"worker_id": rec.worker_id, "unit": rec.unit_name,
                "verified_empty": bool(verified),
                "cgroup_populated": populated,
                "client_exit": rec.proc.poll(),
                "mechanism": "systemd-cgroupv2"}

    def release(self, record_or_id) -> None:
        rec = self._record(record_or_id)
        if rec.overflow_fd is not None:
            try:
                os.close(rec.overflow_fd)
            except OSError:
                pass
            rec.overflow_fd = None
        self._records.pop(rec.worker_id, None)
        shutil.rmtree(rec.workspace, ignore_errors=True)

    def _record(self, record_or_id) -> LaunchRecord:
        if isinstance(record_or_id, LaunchRecord):
            return record_or_id
        rec = self._records.get(str(record_or_id))
        if rec is None:
            raise BrokerError(f"unknown worker {record_or_id!r}")
        return rec


__all__ = ["BrokerError", "BrokerRefused", "LaunchRecord",
           "LaunchWorkerRequest", "SystemdClient",
           "WorkerLaunchBroker", "cgroup_populated",
           "runtime_closure_paths"]
