# Linux Host Qualification (v16.4.8)

## Status: PENDING — no SSH target supplied yet

The qualification host must be a real Linux machine/VM with native
systemd and cgroup v2. CPU-only for the first pass; GPU after the
fs/net/process boundaries pass.

## Procedure

1. `scripts/qualification/host_check.sh` — kernel, CapEff, cgroup2fs,
   systemd-run/systemctl, namespace nodes, python. Must print
   `HOST CHECK: PASS`.
2. `scripts/qualification/build_nsjail.sh <APPROVED_COMMIT_SHA>` —
   pinned build; records `nsjail-build-record.json` and prints the
   `sha256:` digest to pin via `--nsjail-digest`/`MINAGI_NSJAIL_DIGEST`.
3. `scripts/qualification/run_nsjail_qualification.sh` — preflight
   probe + the Linux test file.

## Evidence to record here

- [ ] host: distro/kernel/arch, systemd version, cgroup mode
- [ ] CapEff at qualification time (prove uid-0 shortcut is gone)
- [ ] nsjail commit, binary sha256, dep versions
- [ ] `run_nsjail_qualification.sh` output — probe report + pytest
- [ ] detached-descendant containment (cgroup.procs before,
      cgroup.populated==0 after)
- [ ] fs: authority dir denied/missing; host home dirs denied
- [ ] net: connect to live local listener + external denied
- [ ] real model lifecycle (G10) — model, adapter, tokenizer digests,
      load/infer/cancel/rollback/restart evidence

_To be filled during qualification. Do not mark items without the
captured output._
