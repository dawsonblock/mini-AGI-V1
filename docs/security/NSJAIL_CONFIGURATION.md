# NsJail Configuration (v16.4.8)

## Source of truth

`configs/isolation/linux-production-offline-v1.json` is the frozen
policy contract (`minagi-isolation-profile-v1`). The broker compiles
it through `minagi.runtime.nsjail_config.render_config` — a
deterministic generator: same inputs, byte-identical textproto, and
`config_digest` is bound into every `LaunchRecord`.

## What the generated config pins (explicitly, never defaults)

| Field | Value | Why |
|-------|-------|-----|
| `mode` | `ONCE` | single exec, no daemon re-entry |
| `clone_newnet` + `iface_no_lo` | `true` | empty net namespace; even loopback is down (G3) |
| `clone_newuser` + `uidmap/gidmap` | `true` / inside↔outside worker identity | jailed uid maps to the separate worker uid |
| `clone_newns/newpid/newipc/newuts` | `true` | mount/pid/IPC/hostname isolation |
| `clone_newcgroup` | `false` | systemd owns the real cgroup; nesting would hide it |
| `keep_caps` | `false` | empty capability set inside |
| `disable_no_new_privs` | `false` | NNP stays ON |
| `mount_proc` | `true` | fresh /proc inside pid ns |
| `rlimit_nproc/nofile` | 32 / 256 | fork/file exhaustion bounds |
| `use_cgroupv2` | `true` | v2-only accounting |
| `mount` entries | runtime ro · artifacts ro (same path) · scratch rw at /work · /dev/{null,zero,urandom} | `denied_paths` can never appear — `resolve_mounts` refuses the launch |
| `exec_bin` | pinned interpreter + `-m minagi.runtime.worker_backend` | fixed argv; never a shell |
| `envar` | allowlisted names only | credential-shaped names refused at render |

## Determinism + binding

- `render_config` output is canonical (sorted mounts/env, fixed
  field order) — diffable and digest-stable.
- `config_digest` is recorded in `LaunchRecord.digests["config"]`
  alongside `policy`, `nsjail_binary`, `artifacts`,
  `backend_manifest`, `execution_budget` — the launch is provably
  the approved one.
- The rendered file is written `0400` inside the broker-owned
  workspace; the worker cannot alter its own rules.

## Pinning the binary

`scripts/qualification/build_nsjail.sh <COMMIT>` builds a reviewed
commit and emits `nsjail-build-record.json` (commit, sha256, dep
versions). The supervisor requires `--nsjail-digest sha256:<hex>`
or `MINAGI_NSJAIL_DIGEST`; the broker measures the binary and
refuses a mismatch — production never runs an unmeasured sandbox.
