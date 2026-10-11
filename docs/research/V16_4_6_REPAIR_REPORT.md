# Mini-AGI v16.4.6 — Worker Isolation and Process Closure — Repair Report

Baseline: v16.4.5 Security and Evidence Closure (pushed, verified on
Colab T4). Target: `mini-AGI-V1-v16.4.6-Worker-Isolation-and-Process-Closure.zip`.

## Defects closed

| ID | Sev | Defect | Resolution |
|---|---|---|---|
| SEC-501 | P0 | Workers shared the supervisor's OS identity → all supervisor-readable files reachable | `WorkerIsolationPolicy` gains `require_separate_identity` / `require_filesystem_confinement` / `require_process_containment` / `require_network_isolation`; `production_policy()` + `validate_isolation()` fail closed; worker scratch re-owned to the demoted uid (0750, supervisor gid) so cross-uid scratch works without world access; `service.py` requires `--worker-uid/--worker-gid` under `--production` and reports enforced mechanisms |
| SEC-502 | P1 | `terminate()`/`_kill_and_reap()` killed only the leader pid | `WorkerProcessController`: ancestry snapshot → verified group SIGTERM → escalate SIGKILL → per-pid + env-marker survivor sweep → `verify_terminated` must be empty; random `MINAGI_WORKER_UNIT` token identifies the containment set through setsid() |
| SEC-503 | P1 | `_remote_exception_class` resolved arbitrary `BaseException` subclasses from worker metadata — worker could select `SystemExit` | Dynamic resolution removed. Worker reports a bounded `error_code`; a static map converts only `WORKER_BUDGET_EXCEEDED → BudgetExceeded` (ordinary `Exception`); everything else → `RemoteBackendError(code)`; traceback bounded at 4 KiB |
| SEC-504 | P1 | `read_overflow` validated path, then read path — TOCTOU | Descriptor-based read: dir fd pinned at spawn (`O_DIRECTORY|O_NOFOLLOW`), leaf `O_NOFOLLOW` relative to it, `fstat`+size+digest+strict-JSON all on the same descriptor; dir-rename/leaf-swap cannot redirect the read |
| VAL-501 | P1 | Production startup used permissive `DEFAULT_ISOLATION` | `resolve_worker_policy()` in `service.py`: production refuses without a distinct worker identity and validates every required control on the host; `backend_id` taken from the verified signed manifest (parent never imports backend code for identity); dev profile explicitly labeled |

## New attack/failure coverage (`test_v1646_worker_isolation.py`, 36 tests)

- Worker error payloads: `SystemExit`, `KeyboardInterrupt`,
  `GeneratorExit`, `SystemError`, `os._exit`, unknown modules,
  smuggled class behind an honest code → all flatten to
  `RemoteBackendError` (never `BaseException` subclasses).
- Overflow: symlink leaf, `..`/nested traversal, digest/size
  mismatch, non-regular files, missing file, symlinked directory,
  directory rename+replacement after fd pin, 50× concurrent
  replacement churn (never parses mismatched bytes), oversize write.
- Process tree: children + grandchildren + `setsid` escapee +
  SIGTERM-ignoring straggler all confirmed dead; group-signal safety
  (never signals supervisor's own group); env-marker survives
  `setsid()` (Linux); real worker end-to-end descendant teardown.
- Isolation: same-uid refusal, missing identity refusal, filesystem-
  confinement-without-identity refusal, network-isolation refusal
  where unavailable, worker-owned scratch mode/ownership (root),
  demoted worker denied supervisor secrets/DB but scratch-writable
  (root).
- Production: identity required, partial identity refused, dev
  labeled, manifest `backend_id` bypasses parent import, enforcement
  flags preserved through env merge.

## Gates

| Gate | Status |
|---|---|
| G1 privilege isolation | Implemented; enforced-by-OS tests run as root (Linux/Colab); on unprivileged hosts the requirement tests pass and the demotion tests skip honestly |
| G2 process containment | PASS — verified empty survivor set incl. detached/ignored descendants |
| G3 IPC exception safety | PASS — no dynamic exception classes remain |
| G4 file safety | PASS — descriptor-pinned reads |
| G5 production enforcement | PASS — startup refuses without a distinct worker identity |
| G6 regression stability | PASS — 946 passed / 4 skipped, zero new failures |
| G8 release provenance | manifest re-issued + signed; deterministic build + fresh-extraction verify |
| G7 real serving | partially (real PEFT e2e still passes; full GPU requalification → v16.4.7) |
| G9 scientific credibility | not claimed — unchanged |

## Skips (honest classification)

4 skipped locally: Linux-only `RLIMIT_AS` test; two root-required
demotion tests; the `/proc` marker-scan test. All runnable on the
Colab Linux root profile.
