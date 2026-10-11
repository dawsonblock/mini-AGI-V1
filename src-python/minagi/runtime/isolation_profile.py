"""v16.4.8 frozen isolation profile — the security contract the
production executor enforces (SEC-703 / WP6).

The JSON document under ``configs/isolation/`` is the ONLY way a
production isolation profile enters the system. It parses into a
frozen dataclass whose canonical digest is bound to every worker
launch — the boundary that runs is provably the boundary that was
approved. Job inputs (model ids, inference settings, environment
overrides) never mutate it: permitted runtime values are applied
with ``dataclasses.replace`` on a copy, and every other field is
unreachable from a request.

This module defines the contract only. ``nsjail_config`` compiles a
resolved profile into NsJail's native protobuf configuration;
``worker_launch_broker`` owns execution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path


PROFILE_SCHEMA = "minagi-isolation-profile-v1"


class ProfileRefused(PermissionError):
    """An isolation profile document is malformed or not approved."""


#: Symbolic resources a profile names; the broker binds concrete,
#: supervisor-resolved paths to each name — a request can never turn
#: a symbolic name into an arbitrary host path.
READ_ONLY_RESOURCES = frozenset(
    {"approved_runtime", "approved_model_artifacts"})
WRITABLE_RESOURCES = frozenset({"worker_scratch"})


@dataclass(frozen=True)
class IsolationProfile:
    """The frozen worker-confinement contract.

    executor                  the only production executor: "nsjail"
    network                   "deny" | "loopback" | "allow" — deny
                              means a fresh empty net namespace with
                              loopback down: no path to any host or
                              remote socket
    separate_os_identity      worker runs under a distinct uid/gid
    mount_namespace           private mount ns with approved binds only
    pid_namespace             worker cannot see host processes
    ipc_namespace             isolated SysV/POSIX IPC
    user_namespace            uid/gid map (also enables rootless use)
    uts_namespace             private hostname
    cgroup_namespace          cgroup view isolation (off: systemd owns
                              the real cgroup; nesting would hide it)
    require_cgroup_v2         refuse hosts without unified cgroup v2
    allow_privilege_escalation  must stay False — disables NNP and
                              no-new-privs protections when True
    mount_proc                fresh /proc inside the pid namespace
    read_only_resources       symbolic ro resources the jail mounts
    writable_resources        symbolic rw resources the jail mounts
    deny_authority_access     supervisor authority paths may never be
                              bound into the jail, even by accident
    max_processes             RLIMIT_NPROC + unit TasksMax bound
    max_open_files            RLIMIT_NOFILE bound
    max_file_mb               RLIMIT_FSIZE bound (None = unbounded)
    memory_mb                 cgroup MemoryMax AND rlimit_as (None =
                              decided by deployment, not the worker)
    cpu_seconds               RLIMIT_CPU bound
    wall_time_seconds         nsjail time_limit bound
    unit_tasks_max            systemd TasksMax on the worker unit
    unit_memory_max           systemd MemoryMax bytes (0 = unset)
    unit_properties           extra systemd properties the OPERATOR
                              pins into the unit (fixed set — callers
                              cannot supply properties)
    seccomp                   "default" (nsjail built-in) or a kafel
                              policy file path approved out-of-band
    nice_level                scheduler niceness inside the jail
    hostname                  uts hostname inside the jail
    """
    profile_id: str
    executor: str = "nsjail"
    network: str = "deny"
    separate_os_identity: bool = True
    mount_namespace: bool = True
    pid_namespace: bool = True
    ipc_namespace: bool = True
    user_namespace: bool = True
    uts_namespace: bool = True
    cgroup_namespace: bool = False
    require_cgroup_v2: bool = True
    allow_privilege_escalation: bool = False
    mount_proc: bool = True
    read_only_resources: tuple = field(
        default=("approved_runtime", "approved_model_artifacts"))
    writable_resources: tuple = field(default=("worker_scratch",))
    deny_authority_access: bool = True
    max_processes: int = 32
    max_open_files: int = 256
    max_file_mb: int | None = None
    memory_mb: int | None = None
    cpu_seconds: int | None = None
    wall_time_seconds: int | None = None
    unit_tasks_max: int = 32
    unit_memory_max: int = 0
    unit_properties: tuple = field(default=(
        "KillMode=control-group",
        "NoNewPrivileges=yes",
        "Delegate=yes"))
    seccomp: str = "default"
    nice_level: int = 19
    hostname: str = "minagi-worker"
    schema: str = PROFILE_SCHEMA

    @classmethod
    def from_doc(cls, doc) -> "IsolationProfile":
        """Strict parse: the schema marker must match, types are
        enforced, and unknown fields are refused — an operator who
        meant to set a control but misspelled its name must not get a
        default instead."""
        if not isinstance(doc, dict):
            raise ProfileRefused("isolation profile must be an object")
        if doc.get("schema") != PROFILE_SCHEMA:
            raise ProfileRefused(
                f"isolation profile schema {doc.get('schema')!r} is "
                f"not {PROFILE_SCHEMA!r}")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(doc) - known
        if unknown:
            raise ProfileRefused(
                f"isolation profile has unknown fields {sorted(unknown)}")
        pid = str(doc.get("profile_id") or "")
        if not pid:
            raise ProfileRefused("isolation profile requires profile_id")
        values = {"profile_id": pid}
        for name in known - {"profile_id", "schema"}:
            if name not in doc:
                continue
            v = doc[name]
            default = getattr(cls, name, None)
            if isinstance(default, bool):
                if not isinstance(v, bool):
                    raise ProfileRefused(
                        f"isolation profile field {name!r} must be bool")
            elif isinstance(default, tuple):
                if not isinstance(v, (list, tuple)):
                    raise ProfileRefused(
                        f"isolation profile field {name!r} must be a list")
                v = tuple(str(x) for x in v)
            elif default is None and name != "schema":
                pass  # optional typed fields validated by use
            elif isinstance(default, str):
                v = str(v)
            elif isinstance(default, int) and not isinstance(v, bool):
                if not isinstance(v, int):
                    raise ProfileRefused(
                        f"isolation profile field {name!r} must be int")
            values[name] = v
        profile = cls(**values)
        if profile.executor != "nsjail":
            raise ProfileRefused(
                f"isolation profile executor {profile.executor!r} is "
                "not an approved executor — production runs 'nsjail'")
        if profile.network not in ("deny", "loopback", "allow"):
            raise ProfileRefused(
                f"isolation profile network {profile.network!r} must be "
                "'deny', 'loopback' or 'allow'")
        if profile.allow_privilege_escalation:
            raise ProfileRefused(
                "allow_privilege_escalation can never be approved in a "
                "production isolation profile")
        unknown_ro = set(profile.read_only_resources) - READ_ONLY_RESOURCES
        unknown_rw = set(profile.writable_resources) - WRITABLE_RESOURCES
        if unknown_ro or unknown_rw:
            raise ProfileRefused(
                "isolation profile names unknown resources "
                f"{sorted(unknown_ro | unknown_rw)} — resources are "
                "symbolic names, not paths")
        return profile

    def to_doc(self) -> dict:
        doc = {"schema": self.schema, "profile_id": self.profile_id}
        for name, f in self.__dataclass_fields__.items():
            if name in ("schema", "profile_id"):
                continue
            v = getattr(self, name)
            doc[name] = list(v) if isinstance(v, tuple) else v
        return doc

    def digest(self) -> str:
        """Canonical digest bound to every launch — SEC-703."""
        from egai.common.canonical import digest
        return digest(self.to_doc())

    def with_runtime(self, **permitted) -> "IsolationProfile":
        """The ONLY mutation path: dataclasses.replace on a copy, and
        only for the whitelisted runtime knobs (budget fields). A
        request can tune its budget; it can never reach a mount, an
        identity, a namespace, or the executor."""
        allowed = {"memory_mb", "cpu_seconds", "wall_time_seconds",
                   "unit_memory_max", "max_file_mb"}
        bad = set(permitted) - allowed
        if bad:
            raise ProfileRefused(
                f"runtime fields {sorted(bad)} are not permitted "
                "adjustments — the security profile is frozen")
        return replace(self, **permitted)


def load_profile(path) -> IsolationProfile:
    """Load + validate a profile JSON document from disk."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProfileRefused(f"isolation profile {p}: {exc}") from exc
    return IsolationProfile.from_doc(doc)


@dataclass(frozen=True)
class MountSpec:
    """One concrete bind the broker computed — never caller-supplied."""
    src: str
    dst: str
    rw: bool = False
    mandatory: bool = True
    nosuid: bool = True
    nodev: bool = True
    noexec: bool = False


def resolve_mounts(profile: IsolationProfile, *, runtime_paths,
                   artifact_paths, scratch_src: str,
                   scratch_dst: str = "/work",
                   denied_paths=()) -> tuple[MountSpec, ...]:
    """Map the profile's symbolic resources to concrete binds.

    runtime_paths    the interpreter/runtime read-only closure the
                     broker computed (never request-supplied)
    artifact_paths   approved, digest-verified model artifact paths —
                     mounted read-only at identical locations
    scratch_src      the worker's private writable directory
    denied_paths     path prefixes a mount may NEVER expose — the
                     supervisor's authority material when
                     ``deny_authority_access`` is set. A resolved bind
                     under a denied prefix refuses the whole launch.
    """
    denied = tuple(str(Path(d).resolve()) for d in denied_paths
                   if d) if profile.deny_authority_access else ()

    def _allowed(src: str) -> str:
        resolved = str(Path(src).resolve())
        for deny in denied:
            if resolved == deny or resolved.startswith(deny + "/"):
                raise ProfileRefused(
                    f"mount source {resolved!r} lies under a denied "
                    f"authority path {deny!r} — the worker can never "
                    "see supervisor authority material")
        return resolved

    mounts: list[MountSpec] = []
    seen = set()

    def add(src, dst, rw, **kw):
        if not str(dst).startswith("/"):
            raise ProfileRefused(
                f"mount destination {dst!r} is not absolute — the "
                "jail filesystem cannot name relative targets")
        src = _allowed(src)
        key = (src, dst)
        if key in seen:
            return
        seen.add(key)
        mounts.append(MountSpec(src=src, dst=dst, rw=rw, **kw))

    if "approved_runtime" in profile.read_only_resources:
        for path in runtime_paths or ():
            try:
                if not Path(str(path)).exists():
                    continue  # advisory closure entry (e.g. zip stubs)
            except OSError:
                continue
            add(str(path), str(path), False)
    if "approved_model_artifacts" in profile.read_only_resources:
        for path in artifact_paths or ():
            add(str(path), str(path), False)
    if "worker_scratch" in profile.writable_resources:
        add(scratch_src, scratch_dst, True,
            nosuid=True, nodev=True, noexec=False)
    # the interpreter needs the device basics even in a minimal root
    for dev in ("null", "zero", "urandom"):
        add(f"/dev/{dev}", f"/dev/{dev}", True, mandatory=False,
            nosuid=True, nodev=False, noexec=False)
    # A mount whose destination sits under an already-emitted
    # same-permission parent is fully covered by that parent's bind —
    # and its later ro-remount would EINVAL on a locked subtree.
    # Parents mount first so remaining children stack correctly.
    covered = {i for i, m in enumerate(mounts)
               if any(o.dst != m.dst
                      and m.dst.startswith(o.dst + "/")
                      and o.rw == m.rw for o in mounts)}
    mounts = sorted((m for i, m in enumerate(mounts)
                     if i not in covered),
                    key=lambda m: (m.dst.count("/"), m.dst))
    return tuple(mounts)


__all__ = ["IsolationProfile", "MountSpec", "PROFILE_SCHEMA",
           "ProfileRefused", "READ_ONLY_RESOURCES", "WRITABLE_RESOURCES",
           "load_profile", "resolve_mounts"]
