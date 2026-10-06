# RC9.2 physical executor runtime boundary

RC9.2 retains an explicit ownership object for the physical mutable QW3 executor lineage: `KvMemExecutorSlotRuntime<QwenExecutor>`.

A logical KVMem session is **detachable canonical host state**. A physical executor runtime is **ephemeral compute state** and owns the live `QwenExecutor` object reachable from a scheduler slot.

## Ownership contract

The physical runtime owns one executor object and therefore fences the mutable state reachable from that object, including attention KV, DeltaNet/recurrent state, MTP state, KVMem tier mappings, checkpoints, and other executor-local buffers.

The wrapper tracks:

- physical slot index and stable runtime ID;
- installed/busy/dirty state;
- mounted session affinity;
- runtime generation;
- lease, success, fault, and cold-reset counters;
- an exact active physical lease token and monotonic lease sequence.

A scheduler warm hit is accepted only if the physical runtime agrees that the same session is mounted and the runtime is not dirty. A native execution failure clears physical affinity and marks the runtime dirty. Reuse must pass through cold reconstruction/reset before the runtime can be considered clean again.

## Current certification boundary

RC9.2 still certifies exactly one physical executor runtime in the native backend. The scheduler and runtime ownership ABIs are structured for N slots, but the backend does not enable slot counts greater than one because model/device/tier/archive interactions have not yet been target-GPU qualified for multiple independently instantiated `QwenExecutor` objects.

This is intentional fail-closed behavior. `--kvmem-executor-slots 2` remains rejected.

## Inspection

`GET /v1/kvmem/executors` returns the physical runtime pool. It is separate from `/v1/kvmem/scheduler`: the scheduler describes logical affinity/leases, while the executor-pool endpoint describes the concrete runtime objects that own mutable execution state.

The endpoint reports `scope=persistent-session-runtime`. It describes the affinity-managed executor lineage used for detachable logical sessions; transient request-local continuous-batching executors are a separate execution domain and are not represented as affinity-pool slots.

NEMO should require `runtime_abi=executor-slot-runtime-v2` and verify that installed runtimes are not fewer than certified slots during preflight.


## Runtime-v2 fencing

`executor-slot-runtime-v2` requires an exact physical lease token for both
`mark_cold_reset()` and `release()`. The physical runtime issues its own monotonic lease token and separately records
the scheduler lease ID that caused the bind. This avoids token collisions between
one-shot runtime leases and scheduler leases while retaining cross-layer
traceability. Stale physical releases and stale reset attempts fail closed.
