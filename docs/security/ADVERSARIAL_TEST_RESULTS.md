# Adversarial Test Results (v16.4.8)

## Executed on this host (macOS, development executor + logic tests)

| Test | Result | Proves |
|------|--------|--------|
| `test_v1648_executor_isolation.py` (42 tests) | PASS | SEC-701/703/705 regressions, profile/schema strictness, deterministic config, broker validation (unregistered profile, bad digest, replay nonce), executor argv pinning, service gating, preflight probe end-to-end on the dev executor |
| `test_worker_ipc_adversarial.py` (9) | PASS | pickle refusal, oversized frame, truncated frame, pending bound, EOF-death, stalled infer |
| `tests-python/runtime` full | 364 PASS / 6 skip | no regression across worker backend, supervisor, router, recovery, v16.4.x suites |

## Authored, gated on the Linux host (Step 16–18 scenarios)

| Scenario | Test | Status |
|----------|------|--------|
| jailed identity + pid ns + fs denies | `test_g1_probe_runs_inside_jail` | authored, unrun |
| authority material unmounted | `test_g1_authority_dir_never_mounted` | authored, unrun |
| network denial incl. live listener | `test_g3_network_denied` | authored, unrun |
| setsid escapee stays in cgroup | `test_g4_detached_descendant_stays_owned` | authored, unrun |
| spawn backend: detached + SIGTERM-ignoring + plain children all reaped | `test_g4_spawn_backend_children_all_terminated` | authored, unrun |
| real worker boot + validated IPC + unload cleanup | `test_worker_backend_load_infer_unload` | authored, unrun |
| env scrubbing inside jail | `test_worker_cannot_see_supervisor_env` | authored, unrun |

## Source-level (not yet exercised under the target kernel)

- `worker_launcher` demoted off the production path — no custom
  `pivot_root`/mount-propagation/process-discovery in
  `nsjail_executor` + `worker_launch_broker`.
- `broker.terminate` accepts only `cgroup.populated==0` or a gone
  cgroup as the emptiness claim.
- Malformed IPC unchanged: strict framed JSON (`minagi-worker-v2`),
  1 MiB bound, depth bound, no pickle — covered by the passing
  adversarial suite.
