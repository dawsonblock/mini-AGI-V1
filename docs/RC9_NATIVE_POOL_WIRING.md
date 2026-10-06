# RC9 Native Physical Pool Wiring

RC9 makes `KvMemPhysicalExecutorPool<QwenExecutor>` the actual owner of the native Qwen executor.

## Ownership

`physical_executor_pool_`
→ `ExecutorSlotRuntime`
→ `QwenExecutor`
→ mutable attention KV / recurrent state / KVMem tier state

The existing generation engine uses a compatibility reference to runtime slot 0. That reference
does not own the executor and cannot outlive the pool.

## Certification boundary

The backend continues to reject `--kvmem-executor-slots >1`. This is intentional. The physical
pool is now wired into the native backend, but archive attachment, tier backing, CUDA device
services, prefix-cache infrastructure, and continuous-batching auxiliaries still contain shared
state that has not been qualified for independently active physical slots.

## RC10 gate

Before increasing `certified_slots`, move per-executor KVMem archive/tier ownership behind a slot
factory and qualify two real QwenExecutor instances concurrently on supported NVIDIA hardware.
