# RC8 Physical Executor Pool

RC8 introduces an explicit physical executor pool beneath the session scheduler.

## Invariant

Each physical slot owns a distinct mutable executor object. Logical session state remains detachable
host state; CUDA/KV/DeltaNet/MTP state remains slot-local and ephemeral.

## Certified boundary

The pool implementation is N-slot capable and host-tested with two independently owned fake
executors. The QW3 native CUDA backend remains certified for exactly one physical slot. RC8 does not
remove the `--kvmem-executor-slots >1` fail-closed guard because shared device, archive, tier, and
model-service interactions still require target-GPU qualification.

## Next qualification

A CUDA qualification must demonstrate two independently constructed QwenExecutor lineages, separate
mutable KV/recurrent/tier ownership, concurrent leases, fault isolation, deterministic cold recovery,
and resource-envelope enforcement before increasing `certified_slots`.
