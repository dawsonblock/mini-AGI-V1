# RC9.2 KVMem executor scheduler

`KvMemExecutorScheduler` is a host-only affinity scheduler for logical KVMem
sessions. A scheduler slot represents one independently mutable executor
lineage; it is not a CUDA allocation itself.

The scheduler policy is `lru-affinity-fail-fast`:

1. reuse a session already mounted on an idle slot;
2. otherwise take an unused idle slot;
3. otherwise evict the least-recently-used idle slot;
4. if every slot is busy, reject immediately rather than block while owning
   runtime resources.

A lease reports the slot, whether the mount was warm or cold, and any evicted
session. Successful release keeps affinity. Failed release clears affinity and marks the
slot dirty so uncertain executor state cannot be reused as warm state.

The class is tested with two slots, while RC9.2 native serving still certifies one
CUDA slot. The native `QwenExecutor` is now owned by the physical executor pool,
but per-slot CUDA/archive/tier isolation is not yet qualified.
`--kvmem-executor-slots >1` therefore fails closed.

`GET /v1/kvmem/scheduler` returns current slot occupancy plus cumulative
`warm_hits`, `cold_mounts`, `backpressure_rejections`, `faulted_releases`, and
`forced_cold_resets`.

Future executor-pool work should instantiate one isolated executor lineage per
certified slot, then remove the one-slot backend guard without changing the
session or scheduler API.


## RC9.2 fault containment and lease fencing

A failed executor lease now dirties the slot. A dirty slot cannot produce a warm
hit; its next lease is marked `requires_cold_reset` and must reconstruct/reset the
lineage before success clears the dirty bit. This prevents a native exception from
leaving an implicitly trusted warm executor state. Resource admission is separate;
see `kvmem_resource_admission.md`.


## Exact lease fencing

Each busy scheduler slot records `active_lease_id`. `release()` requires the exact
active lease token in addition to slot/session affinity. This prevents an obsolete
lease for the same logical session from releasing a newer owner. The active token
is cleared atomically on successful or faulted release and exposed by the scheduler
status endpoint for diagnosis.
