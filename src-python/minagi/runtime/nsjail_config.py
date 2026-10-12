"""v16.4.8 deterministic NsJail configuration generator (WP7/WP8).

Compiles a resolved :class:`IsolationProfile` into NsJail's native
protobuf text configuration (``nsjail --config file.cfg``). The
generator is deliberately pure: same resolved inputs -> identical
bytes -> identical digest. Every confinement-relevant field is
emitted EXPLICITLY — undocumented defaults are never relied on for
a security property.

Reference schema: NsJail ``config.proto`` (NsJailConfig, MountPt,
IdMap, Exe, RLimit, Mode, LogLevel).
"""
from __future__ import annotations

import json
from pathlib import Path

from .isolation_profile import IsolationProfile, MountSpec


class ConfigError(RuntimeError):
    """A resolved profile cannot be compiled into a valid config."""


def _q(s: str) -> str:
    """Quote a protobuf text-format string (C-style escapes — the
    same encoding protoc's TextFormat accepts)."""
    return json.dumps(str(s), ensure_ascii=True)


def _bool(v) -> str:
    return "true" if v else "false"


def _mount_block(m: MountSpec) -> str:
    lines = ["mount {"]
    if m.src:
        lines.append(f"  src: {_q(m.src)}")
    lines.append(f"  dst: {_q(m.dst)}")
    lines.append(f"  is_bind: {_bool(True)}")
    lines.append(f"  rw: {_bool(m.rw)}")
    lines.append(f"  mandatory: {_bool(m.mandatory)}")
    lines.append(f"  nosuid: {_bool(m.nosuid)}")
    lines.append(f"  nodev: {_bool(m.nodev)}")
    lines.append(f"  noexec: {_bool(m.noexec)}")
    lines.append("}")
    return lines


def _idmap_block(kind: str, inside, outside, count: int = 1) -> str:
    return [
        f"{kind} {{",
        f"  inside_id: {_q(str(inside))}",
        f"  outside_id: {_q(str(outside))}",
        f"  count: {int(count)}",
        "}",
    ]


_BENIGN_MARKER_NAMES = frozenset({
    # standard performance/verbosity knobs — credential-shaped
    # substring but carry no secret material
    "TOKENIZERS_PARALLELISM", "TORCH_LOGS", "TRANSFORMERS_VERBOSITY",
    "OPENBLAS_NUM_THREADS"})


def render_config(profile: IsolationProfile, *,
                  exec_argv, env: dict | None = None,
                  mounts: tuple[MountSpec, ...] = (),
                  inside_uid: int = 65534, inside_gid: int = 65534,
                  outside_uid: int | None = None,
                  outside_gid: int | None = None,
                  cwd: str = "/work",
                  name: str = "minagi-worker",
                  cgroup_parent: str | None = None,
                  log_level: str = "INFO") -> str:
    """Compile the resolved profile into nsjail textproto.

    exec_argv     the EXACT in-jail argument vector — always the
                  pinned interpreter + module; never shell-derived
    env           allowlisted envar dict (already secret-screened)
    mounts        resolved MountSpec binds
    inside_uid/gid  the identity the jailed process presents in-jail
    outside_uid/gid the host uid/gid the jail maps to — the separate
                  worker identity in production (rootful path); the
                  invoking uid on the rootless userns path
    """
    if not exec_argv or not all(str(a) for a in exec_argv):
        raise ConfigError("exec_argv must be a non-empty argv vector")
    env = dict(env or {})
    for key in env:
        if any(m in key.upper() for m in
               ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")) \
                and key.upper() not in _BENIGN_MARKER_NAMES:
            raise ConfigError(
                f"envar {key!r} looks credential-bearing — secrets are "
                "never configured into the jail")

    out_uid = inside_uid if outside_uid is None else int(outside_uid)
    out_gid = inside_gid if outside_gid is None else int(outside_gid)
    deny_net = profile.network == "deny"

    L: list[str] = []
    A = L.append
    A(f"name: {_q(name)}")
    A('description: "minagi worker confinement — generated '
      'deterministically by minagi.runtime.nsjail_config; do not edit"')
    A("mode: ONCE")
    A(f"hostname: {_q(profile.hostname)}")
    A(f"cwd: {_q(cwd)}")
    # pivot_root is NsJail's tested root-switch path — keep it on;
    # the previous release's hand-rolled pivot_root is what failed.
    A("no_pivotroot: false")
    if profile.wall_time_seconds:
        A(f"time_limit: {int(profile.wall_time_seconds)}")
    A("daemon: false")
    A(f"nice_level: {int(profile.nice_level)}")
    A("keep_env: false")
    for key in sorted(env):
        A(f"envar: {_q(f'{key}={env[key]}')}")
    # Drop the whole capability set: the worker needs none.
    A("keep_caps: false")
    # fds 0/1/2 carry the framed-JSON protocol channel — never silent.
    A("silent: false")
    A("skip_setsid: false")
    A("disable_no_new_privs: "
      + _bool(profile.allow_privilege_escalation))
    # rlimit_as is deliberately not a VALUE bound: virtual-AS accounting
    # counts mmap'd regions (weights, arenas), so it wrongly starves real
    # model runtimes. cgroup MemoryMax is the authoritative memory bound.
    A("rlimit_as_type: INF")
    A("rlimit_core: 0")
    A("rlimit_core_type: VALUE")
    if profile.cpu_seconds is not None:
        A(f"rlimit_cpu: {int(profile.cpu_seconds)}")
        A("rlimit_cpu_type: VALUE")
    if profile.max_file_mb is not None:
        A(f"rlimit_fsize: {int(profile.max_file_mb)}")
        A("rlimit_fsize_type: VALUE")
    A(f"rlimit_nofile: {int(profile.max_open_files)}")
    A("rlimit_nofile_type: VALUE")
    A(f"rlimit_nproc: {int(profile.max_processes)}")
    A("rlimit_nproc_type: VALUE")

    A(f"clone_newnet: {_bool(profile.network != 'allow')}")
    A(f"clone_newuser: {_bool(profile.user_namespace)}")
    A(f"clone_newns: {_bool(profile.mount_namespace)}")
    A(f"clone_newpid: {_bool(profile.pid_namespace)}")
    A(f"clone_newipc: {_bool(profile.ipc_namespace)}")
    A(f"clone_newuts: {_bool(profile.uts_namespace)}")
    # cgroup namespaces are not nested: the systemd unit owns the real
    # cgroup — hiding it would defeat the containment contract.
    A(f"clone_newcgroup: {_bool(profile.cgroup_namespace)}")
    A("clone_newtime: false")

    if profile.user_namespace:
        L += _idmap_block("uidmap", inside_uid, out_uid)
        L += _idmap_block("gidmap", inside_gid, out_gid)

    A(f"mount_proc: {_bool(profile.mount_proc)}")
    for m in mounts:
        L += _mount_block(m)

    A(f"use_cgroupv2: {_bool(profile.require_cgroup_v2)}")
    A("detect_cgroupv2: true")
    A('cgroupv2_mount: "/sys/fs/cgroup"')
    if cgroup_parent:
        A(f"cgroup_mem_parent: {_q(cgroup_parent)}")
        A(f"cgroup_pids_parent: {_q(cgroup_parent)}")
        A(f"cgroup_cpu_parent: {_q(cgroup_parent)}")

    # deny == a fresh net namespace with loopback DOWN: no route to a
    # host listener, a link-local peer, or the internet. "loopback"
    # leaves lo up for an explicitly qualified local channel.
    A(f"iface_no_lo: {_bool(deny_net)}")

    if profile.seccomp and profile.seccomp != "default":
        path = Path(profile.seccomp)
        if not path.is_file():
            raise ConfigError(
                f"seccomp policy {profile.seccomp!r} is not a file")
        A(f"seccomp_policy_file: {_q(str(path))}")
    A("seccomp_log: false")

    exe_path, *exe_args = [str(a) for a in exec_argv]
    A("exec_bin {")
    A(f"  path: {_q(exe_path)}")
    for arg in exe_args:
        A(f"  arg: {_q(arg)}")
    A("}")

    A(f"log_level: {log_level}")
    return "\n".join(L) + "\n"


def config_digest(text: str) -> str:
    """sha256 digest of the rendered configuration — recorded on the
    launch so the executed boundary is provably the generated one."""
    from egai.common.canonical import sha256_bytes
    return sha256_bytes(str(text).encode("utf-8"))


__all__ = ["ConfigError", "config_digest", "render_config"]
