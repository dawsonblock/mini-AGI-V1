# V16.4.8 Repair Report — Linux Isolation and Security Closure

## What changed vs v16.4.7

| Layer | v16.4.7 | v16.4.8 |
|-------|---------|---------|
| Production boundary | custom `unshare`/`pivot_root`/mount launcher | systemd transient unit (cgroup v2) + pinned NsJail |
| Launch authority | backend spawned directly | restricted `WorkerLaunchBroker`: digest-bound request, replay nonces, fixed argv, env allowlist, denied-path mount screening |
| Policy | `WorkerIsolationPolicy` (+ env-merge bug SEC-703) | frozen `IsolationProfile` (`minagi-isolation-profile-v1`), `with_runtime` whitelist, `policy_digest` bound to launch |
| Capability checks | `euid==0` → caps True | `/proc/self/status` CapEff only; preflight probe verifies enforcement |
| Executor selection | one path, seatbelt credited anywhere | `development` vs `nsjail` by platform/profile; production requires nsjail; seatbelt is Darwin-only |
| Termination | process-group + marker + cgroup | cgroup.populated==0 (or cgroup gone) — kernel-owned |
| IPC | framed JSON over stdio | same strict framed JSON over `systemd-run --pipe` |

## New modules

- `runtime/isolation_profile.py` — frozen profile contract + mount resolution
- `runtime/nsjail_config.py` — deterministic textproto generator
- `runtime/worker_executor.py` — executor Protocol, `LaunchedWorker`, dev executor
- `runtime/worker_launch_broker.py` — restricted launch service + `SystemdClient`
- `runtime/nsjail_executor.py` — production executor
- `runtime/isolation_preflight.py` — real-enforcement probe + grader
- `configs/isolation/linux-production-offline-v1.json`
- `tests-python/runtime/test_v1648_executor_isolation.py` (42 tests)
- `tests-python/runtime/test_v1648_linux_nsjail.py` (7, Linux-gated)
- `scripts/qualification/{host_check,build_nsjail,run_nsjail_qualification}.sh`
- `docs/security/*`

## Modified

- `runtime/worker_isolation.py` — SEC-701 (CapEff honesty),
  SEC-705 (seatbelt Darwin-gated), `_unprivileged_userns`,
  `policy_digest`
- `runtime/worker_backend.py` — launch/teardown delegated to the
  executor; `_teardown` keeps synthetic-handle IPC tests working;
  digest binding forwarded at launch
- `runtime/service.py` — `--worker-executor`, `--isolation-profile`,
  `--nsjail-*`, `--worker-state-root`, `--worker-user-manager`;
  production refuses dev executor/in-process/insecure-dev
- `runtime/worker_launcher.py` — documented as dev-path only
- `runtime/isolation_preflight.py` — select-gated reads, env-carried
  probe specs, per-executor scratch

## Verification on this host

- 364/364 runtime tests pass (6 platform skips), incl. 42 new
  v16.4.8 tests and the full IPC-adversarial suite.
- NsJail config renders deterministically; digest stable.
- Broker refuses: unregistered profile, malformed digest, replayed
  nonce, unmeasured binary, digest mismatch, foreign argv.
- Service refuses: `--production` + `development`,
  `--production` + `inprocess`, `--production` + `--allow-insecure-dev`,
  production nsjail on a host that cannot establish the boundary.

## NOT claimed

Linux gates G1–G4 and G10 (real model) are source-complete and
test-authored but unqualified — no SSH/Linux target has been
supplied. This release does not claim them.
