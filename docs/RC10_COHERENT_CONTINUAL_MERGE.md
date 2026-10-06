# RC10 Coherent Hybrid + Continual Memory Merge

RC10 is a merger release built on the qualified RC9.2 native QW3/KVMem
runtime. It does **not** replace that runtime with the earlier Python RC10
prototype. Instead, it adds two deliberately separated layers:

1. a native, host-qualified coherent-hybrid state foundation for segment-level
   affine recurrent summaries and strict cache identity; and
2. a hardened Python continual-learning control plane under `continual/` for
   persistent cognitive memory, candidate adapters, replay, qualification,
   promotion, rollback, and reference reconstruction policy.

## Why the split matters

KVMem execution state, persistent facts/episodes, and learned parameters have
different validity rules. RC10 therefore treats them as three different
memory classes:

- **execution memory**: attention KV, recurrent summaries, seam anchors,
  checkpoints, ReplaySSM-style transient decode inputs;
- **cognitive memory**: episodes, facts, procedures, provenance, temporal
  validity, evidence links;
- **parameter memory**: candidate/promoted adapters and eventually controlled
  base-weight changes.

A semantic fact can remain valid across an adapter promotion. A recurrent
state generally cannot.

## Native coherent-state foundation

`include/qw3/kvmem_coherent_state.hpp` and
`src/kvmem_coherent_state.cpp` implement a reference affine segment summary:

```
S_out = T_C S_in + Z_C
```

and strict ordered composition:

```
A then D:
T_AD = T_D T_A
Z_AD = T_D Z_A + Z_D
```

The implementation is intentionally dense and host-side. It validates the
algebra and fail-closed contracts. Production Qwen Gated DeltaNet integration
must use a structured model-specific transition representation; storing a full
dense transition matrix per block is not claimed as a viable production
format.

Every selected block must contain exactly the same recurrent layer set.
Missing layers are a hard error rather than being silently intersected away.

## Cache identity

Execution caches bind to:

- base-model digest;
- adapter-set digest;
- tokenizer digest;
- layer-layout digest;
- position scheme;
- recurrence implementation.

Compatibility is fail-closed:

- exact identity -> exact reuse;
- adapter-only change -> source replay required;
- structural/model/tokenizer/position/recurrence change -> invalid cache.

There is no heuristic `APPROX_COMPATIBLE` inference. Any future approximate
cross-version reuse must be separately qualified.

## Reconstruction ladder

The target runtime contract remains:

```
compose T/Z
  -> fixed seam repair
  -> adaptive seam repair
  -> suffix/checkpoint replay
  -> exact selected replay
```

RC9.2's existing selected-replay implementation remains the correctness oracle.
The new host layer supplies composition math and repair-policy primitives, but
the model-specific Qwen capture/seam/suffix CUDA integration is still a gated
implementation milestone and is **not** falsely marked complete in this
release.

## Continual-learning plane

The `continual/` package is the repaired version of the RC10 research/control
prototype. It now includes:

- persistent SQLite configuration rather than an in-memory API default;
- temporal `valid_from` + `valid_until` filtering;
- transactional supersession;
- strict recurrent-layer composition;
- fail-closed cache compatibility;
- fixed 8-token FAST seam policy and adaptive BALANCED policy;
- weighted replay without duplicates inside a sample;
- path-safe candidate namespaces;
- candidate-byte rehash at qualification and immediately before promotion;
- digest binding between candidate manifest, adapter and qualification;
- atomic registry metadata writes;
- separate memory API and admin mutation API;
- admin mutation disabled unless `KVCONTINUAL_ADMIN_TOKEN` is configured;
- cache invalidation declaration after adapter promotion/rollback.

## Still not implemented

RC10 intentionally does not claim the following as complete:

- extraction of exact Qwen Gated DeltaNet segment `T_C/Z_C` summaries;
- native seam replay through the full nonlinear hybrid stack;
- Tail-Replay/SuffixReplay kernels;
- ReplaySSM decode representation;
- TreeWY speculative branch integration;
- native QW3 LoRA application to every chosen linear projection;
- target-GPU qualification of coherent reconstruction;
- proof that arbitrary nonlinear deep-layer segment reordering equals replay.

Those items require model weights, CUDA hardware, and empirical comparison
against exact selected replay.

## Intended next integration sequence

1. Capture per-layer segment recurrence summaries in the native Qwen GDN path.
2. Extend tiered KVMem persistence with compact summary payloads and cache
   identity digests.
3. Compare native composition against existing selected replay on real model
   checkpoints.
4. Add fixed seam replay and measure hidden/logit/task divergence.
5. Add adaptive seam and suffix escalation.
6. Add native LoRA inference; treat every adapter promotion as a new execution
   cache epoch.
7. Connect the Python learning worker to native runtime manifests without
   granting it authority to mutate serving state directly.
