# Mini-AGI v16.4.7 — Runtime Isolation Enforcement

Targeted security release converting v16.4.6's *declared* isolation
controls into *applied* OS enforcement. Driven by an independent
deep-extraction audit of the v16.4.6 package; every finding below was
reproduced in source before repair.

## Audit findings → repairs

| Audit finding | Repair | Proof |
|---|---|---|
| `--production --backend-isolation inprocess` bypassed all worker policy (P0) | Refused at CLI parse time AND at backend construction; no fallback | `test_production_inprocess_refused_at_cli` exercises the real `main()` entry |
| `preexec_fn` inside a multithreaded supervisor (architectural) | `worker_launcher.py` — a dedicated launch process applies the isolation contract then `exec`s the worker; supervisor's `Popen` has no `preexec_fn` | `test_launcher_*` (refusal, spec round-trip, live exec) |
| Network isolation validated but never applied (P0) | Launcher `unshare(CLONE_NEWNET)` before demotion when required; validation probes CAP_SYS_ADMIN, not binary presence | `test_network_isolation_actually_denies_worker` — real worker connect refused (Linux-gated) |
| Detached/env-scrubbed descendants escape discovery (P0) | Linux cgroup v2 `WorkerCgroup` — membership is kernel state, immune to detach/reparent/env-scrub; `cgroup.kill` atomic teardown | `test_cgroup_membership_survives_detach_and_scrub` (Linux-gated) |
| Zombies counted as survivors (P0 — the audit's 2 Linux test failures) | `_pid_state`/`_zombie_pids`: state-Z pids are dead, excluded from `_owned`, cgroup membership, and `verify_terminated` | `test_zombie_never_counts_as_survivor` |
| Normal `unload()` left detached children (P1) | `unload()` snapshots the containment set BEFORE the unload RPC (leader may exit inside it) and sweeps after clean exit | `test_unload_sweeps_detached_descendant` |
| Filesystem confinement = separate uid (P1) | `validate_isolation` splits controls: `separate_identity` vs `filesystem_confinement`; the latter needs a real mount namespace (CAP_SYS_ADMIN + binds) or seatbelt profile — honest refusal otherwise | validate refuses mountns-less confinement; production_policy populates runtime ro-binds + snapshots root |
| `preexec_fn` orphaned descendants reparent to init | `PR_SET_CHILD_SUBREAPER` on the supervisor (Linux) — orphans adopt to the supervisor for teardown+reaping | — |
| Release manifest mismatch (5 files — the lint fixes) | Source frozen, manifest regenerated and re-signed under the pinned key (`0f9d7358`), deterministic rebuild | `verify_release.py` on fresh extraction |

## Mechanism map (what production actually applies, by host)

| Control | Linux production | macOS dev |
|---|---|---|
| Worker identity | uid/gid demotion via launcher | same (root required) |
| Descendant containment | **cgroup v2 kill+verify**; ancestry+marker scan as complement | seatbelt `deny process-fork` (descendants impossible) |
| Network | `CLONE_NEWNET` — no routes, no sockets | seatbelt `deny network*` |
| Filesystem | mount-ns `pivot_root`: ro-binds (interpreter/runtime/snapshots) + rw scratch | seatbelt profile |
| Privilege retention | `NO_NEW_PRIVS` + demotion last | seatbelt + demotion |
| Zombie handling | state-Z excluded everywhere; subreaper reaps orphans | state-Z excluded via `ps` probe |

On any host where a REQUIRED mechanism cannot be applied,
`validate_isolation` raises `IsolationError` and production startup
refuses — capability reporting is never a substitute for enforcement.

## Linux-vs-macOS posture

Linux is the production-security reference: cgroup v2 + netns +
mount-ns give kernel-owned guarantees. macOS qualifies only through
the seatbelt path — `deny process-fork` makes the detached-descendant
class impossible — and sandbox-exec is deprecated upstream, so macOS
production deployments remain independently-qualified-per-host, never
inferred from the reference profile.

## Honest remains-open items

- **Windows**: no restricted-token/Job-Object implementation — POSIX only.
- **Real-model lifecycle under the restricted worker** (G7): the
  synthetic-backend matrix passes; the HF/PEFT GPU matrix under
  demotion+namespaces is the next gate.
- **Learning quality** (G9): unchanged — Campaign 3A stands refused
  (+0.00219 vs +0.0200 required).
- macOS seatbelt profile is a REFERENCE — operators qualify their own.
