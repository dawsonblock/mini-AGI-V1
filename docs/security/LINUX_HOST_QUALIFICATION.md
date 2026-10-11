# Linux Host Qualification (v16.4.8)

## Status: QUALIFIED in a privileged Linux container (nested)

Qualification ran on a real Linux kernel boundary — Debian bookworm
container, **systemd 252 as PID 1, cgroup v2 unified, linuxkit kernel
7.0.14, aarch64** — inside Docker Desktop's Linux VM on the dev host.
All namespace/cgroup/seccomp machinery is the real kernel's; the only
caveat is nesting (the VM is itself containered). A bare-metal or
cloud VM re-run with `MINAGI_WORKER_UID` set to a non-root identity
remains recommended for final sign-off; nothing about the boundary is
macOS-only anymore.

## Recorded evidence (2026-10-11)

- **Host check** — `host_check.sh`: PASS. CapEff `000001ffffffffff`
  (real bits, incl. CAP_SYS_ADMIN); cgroup `cgroup2fs`, root writable;
  `systemd-run`/`systemctl` present; all six namespace nodes exist;
  `max_user_namespaces=64022`.
- **NsJail build** — `build_nsjail.sh` at pinned commit
  `4ff54a6e0d5b65a0e1633d6e2fd1425ba6c882ce` (master, 2026-10-02 —
  the fail-closed explicit-mount-flags fix). Binary
  `sha256:a9408ab22219cb71e9e6d8ff571a594983d098df0acf1e4447494cfa0c1b3af8`
  (aarch64); protoc 3.21.12, gcc 12.2.0, libnl 3.7.0. Build record at
  `/usr/local/bin/nsjail-build-record.json`.
- **Preflight (real boundary)** — `qualified: true`.
  `checks`: `no_caps`, `scratch_writable`, `net:127.0.0.1:9`,
  `containment_verified` all true. Probe facts: `capeff=0`, fresh
  mnt/net/pid/user/uts/ipc inode numbers, `pid=1`/`ppid=0` in its own
  pid namespace, `/etc/shadow` and `/root` `missing` (never mounted),
  `127.0.0.1:9` `connect-denied`, `scratch_write=write-ok`,
  `containment=systemd-cgroupv2`.
- **Adversarial suite** — `test_v1648_linux_nsjail.py`:
  **7 passed, 0 failed.** G1 probe identity/namespaces, authority dir
  never mounted, G3 live-listener + external connect denied, G4
  detached descendant owned by the unit cgroup (verified via the
  child's own `/proc/self/cgroup` — the kernel's statement, not a
  pid guess), G4 setsid+SIGTERM-ignoring children fully reaped,
  worker load→infer→unload through the jail, supervisor-env
  invisibility.
- **Detached-descendant proof** — the jailed child wrote its own
  `/proc/self/cgroup` into the broker-owned scratch; it names
  `system.slice/miniagi-worker-*.service`. After `terminate`,
  `cgroup.events populated=0` / unit gone — `verified=True`.

## Defects qualification caught (all fixed in this cycle)

1. **Symlink-spelling mounts** — the runtime closure mounted only
   *resolved* paths, so `/usr/bin/python3` (symlink to python3.11) and
   merged-usr `/lib` (where the ELF interpreter lives) were absent in
   the jail → `execve` ENOENT. Closure now emits as-spelled absolute
   paths; mount dst keeps the spelled name.
2. **`cgroup.populated` is not a cgroup-v2 file** — occupancy is
   `cgroup.events`' `populated` field (`cgroup.procs` fallback). The
   old read always returned "unreadable" on a real host.
3. **`python_path_entries` never mounted** — exported as the jail's
   `PYTHONPATH` but absent from `runtime_paths`, so `minagi` was
   unimportable in-jail. Now merged into the mount set.
4. **Sub-mounts of an RO-remounted parent EINVAL** — pytest adds the
   repo root to `sys.path`; children under it became locked mounts.
   `resolve_mounts` drops mounts already covered by a same-permission
   parent and orders parents first.
5. **Relative mount destinations** — a relative sys.path entry became
   `dst:"src-python"`. Entries are normalized to absolute; a
   non-absolute dst now raises `ProfileRefused` (fail closed).
6. **Probe/containment race** — the ONCE-mode probe exited in ~50 ms,
   faster than `systemctl show ControlGroup` is reliably queryable.
   The probe now holds briefly (`MINAGI_PROBE_HOLD`, ≤10 s clamp) so
   live-cgroup occupancy is verifiable.
7. **Namespaced pids** — the detached-child check compared a
   jail-namespace pid against host `cgroup.procs` (impossible by
   design). The child now reports its own `/proc/self/cgroup`.

## Still open

- [ ] Bare-metal/cloud-host run with a non-root `MINAGI_WORKER_UID`
- [ ] Real model lifecycle (G10): model/adapter/tokenizer digests,
      load→infer→cancel→rollback→restart under the jail
