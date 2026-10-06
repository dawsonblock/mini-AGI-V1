# KVMem known issues and qualification state

Snapshot: **2026-10-04**  
Release line: **kvmem-qw3-hardening-rc9.2**

This ledger distinguishes a reproducible implementation defect from an unqualified
CUDA path, an explicit product limitation, and an unresolved research problem.
RC9.2 does not mark a CUDA-dependent change as runtime-qualified unless the target
GPU gate has actually run.

## Status summary

| ID | Priority | RC9.2 status | Summary |
| --- | --- | --- | --- |
| KVMI-001 | P1 | PARTIAL / EXPLICIT API BOUNDARY | Rendered message spans, explicit query-message ranges, harness semantic pins, retrieval groups, and mandatory-block budget accounting exist. A fully generic public `context/query/pinned_message_indices` API is still not implemented. |
| KVMI-002 | P1 | IMPLEMENTED / CUDA REQUALIFICATION REQUIRED | Misaligned prefill/resume capture uses exact incremental mean merging. RC9.2 also merges generated suffixes into prompt/response boundary blocks and enables contiguous sub-block decode capture. |
| KVMI-003 | P1 | FIXED | Query replay checkpoint selection is constrained to the block-aligned replay boundary and warm replay is supported. |
| KVMI-004 | P1 | FIXED / FAIL-CLOSED MODE AVAILABLE | Requested/used scorer and failure reason are recorded; strict retrieval can require the requested scorer rather than silently accepting fallback. |
| KVMI-005 | P1 | IMPLEMENTED BOUNDED EXACTNESS / CUDA REQUALIFICATION REQUIRED | ExactMass no longer allocates O(n_blocks) shared memory. Raw fp32 key allocation has an explicit configurable pre-allocation safety cap instead of uncontrolled OOM/fallback. |
| KVMI-006 | P1 | FIXED | Canonical source/build provenance is release-gated rather than relying on an old local binary. |
| KVMI-007 | P2 | RESEARCH | Neighbor and semantic-group expansion mechanisms exist, but optimal retrieval policy remains benchmark-dependent. |
| KVMI-008 | P2 | RESEARCH | Retrieval-query hidden state can still depend on the bounded pressure window. This is a representation-quality problem, not a storage-integrity bug. |
| KVMI-009 | P3 | EXPLICIT LIMITATION | Non-default dtype/tiering/MTP/selector combinations remain intentionally constrained and must fail closed when unsupported. |
| KVMI-010 | P3 | FIXED | Public method names and zero-recent semantics are synchronized with the current CLI in release-critical documentation. Historical benchmark tables retain historical option names as data. |
| KVMI-011 | P0 | IMPLEMENTED / CUDA REQUALIFICATION REQUIRED | Immutable raw-K rebuild bounds repeated re-RoPE drift. |
| KVMI-011A | P0 | IMPLEMENTED / CUDA REQUALIFICATION REQUIRED | MTP logical positions and compact RoPE positions are separated with range checks. |
| KVMI-012 | P0 | RESEARCH / FAIL-CLOSED FOR PERSISTENT SELECTED-REPLAY | Sparse attention-KV reselection cannot in general reconstruct an equivalent DeltaNet recurrent state. One-shot selected-replay rebuilds both representations from selected source tokens; persistent above-budget selected-replay remains rejected rather than claiming false coherence. |

## RC9.2 deterministic repairs

### KVMI-002 — incremental mean-index parity

The incremental builder already had a CUDA merge primitive that combines an
existing prefix mean with a misaligned suffix by exact row count. RC9.2 extends
that invariant through decode:

- a generated response may begin in the middle of a prompt block;
- generated K rows are de-RoPE'd at their actual bake position and staged in a
  position-invariant content frame;
- the staged suffix is merged into the preserved prefix mean using
  `first_block_token_offset`;
- contiguous sub-block mean indexes use the same merge primitive;
- CPU-hosted mean indexes seed the affected boundary block before the merge.

The existing `qw3-kvmem-kmean-merge` CUDA test covers arbitrary split points,
sub-block mode, and replay overwrite parity. The target-GPU gate must run before
this is called CUDA-qualified for RC9.2.

### KVMI-004 — scorer observability and strictness

Every query-conditioned reselect records the requested scorer, scorer actually
used, capture readiness, and a stable failure reason. `--kvmem-strict-retrieval`
/require-scorer behavior allows qualification profiles to reject a degraded
retrieval path rather than silently treating fallback as equivalent.

### KVMI-005 — ExactMass scalability and bounded allocation

The ExactMass kernel now keeps only the query vector in dynamic shared memory.
It performs three bounded-memory passes: global maximum, global denominator, and
normalized per-token mass folded directly into logical block scores. Therefore
the old approximately-12K-block dynamic-shared-memory limit is removed.

Raw per-token keys are still intrinsically expensive because ExactMass needs a
content-frame fp32 key representation. RC9.2 makes that cost explicit with
`--kvmem-exactmass-raw-key-max-mib` (default 8192 MiB, `0` disables the safety
cap). Allocation-size overflow or cap violations fail before allocation with an
explicit diagnostic; they do not silently degrade to a different scorer.

`qw3-kvmem-exactmass-largeblocks` is a CUDA regression that deliberately scores
13,001 one-token blocks, above the historical limit, and verifies normalization.
It is packaged but cannot be counted as passed until run on CUDA hardware.

### Workspace/session integrity repair

RC9.1 correctly made an omitted workspace ID fail for a workspace-bound public
session lookup. That exposed an internal snapshot regression because maintenance
snapshotting also used the public workspace-validated lookup with an empty ID.
RC9.2 splits these contracts:

- `require(...)` is caller-facing and enforces the exact workspace binding;
- `require_internal(...)` is backend-only maintenance lookup and intentionally
  skips caller workspace validation while retaining ID/existence checks.

The host session-manager regression verifies both properties.

### Physical executor ownership scope

`/v1/kvmem/executors` now explicitly reports
`scope="persistent-session-runtime"`. The affinity pool owns the persistent
session runtime lineage. Continuous-batching executors are transient
request-local executors and are intentionally outside that affinity pool. This
removes the previous ambiguity that the pool represented every `QwenExecutor`
instance in the process.

## Remaining research/qualification boundaries

### KVMI-001 — generic message semantics

The runtime has token spans for rendered messages, explicit
`kvmem_query_message_range`, automatic last-user query mapping, harness semantic
pins, semantic retrieval groups, and mandatory-block budget accounting. What it
does not yet expose is one uniform public object that accepts arbitrary context,
query, and pinned message-index sets for every client. This is now an API-product
boundary, not an unbounded silent behavior: explicit query ranges are supported,
and qualification profiles rely on known semantic policies.

A future generic API should map requested messages to exact rendered token spans,
charge every mandatory block against the selected-window budget, and reject a
request whose pinned set alone exceeds the budget.

### KVMI-007 / KVMI-008 — retrieval quality

Neighbor expansion, message/round semantic grouping, source-order refresh, and
multiple query-construction diagnostics exist. They do not prove one retrieval
policy is universally optimal. These entries remain research items and must be
settled by frozen benchmark suites and ablations, not by release metadata.

### KVMI-009 — compatibility/performance limits

Known non-default constraints include dtype/tiering combinations, optional MTP
paths, synchronous portions of NVMe stage-out, host-side selection in some
profiles, and a globally shared standard-attention selected block set. Unsupported
combinations must reject explicitly; they are not treated as correctness-equivalent
fallbacks in a strict qualification profile.

### KVMI-012 — hybrid attention/recurrent coherence

Qwen hybrid models contain normal-attention layers plus DeltaNet recurrent state.
Arbitrarily selecting attention blocks does not provide a mathematically exact
way to delete/reorder the corresponding history already compressed into a
DeltaNet recurrent state. Per-block exact affine transforms would be too large for
the intended memory architecture.

The defensible coherent mode is **selected-replay**: preserve source tokens for
retrieval, select a bounded source-order subset, and fresh-prefill that subset so
both normal-attention KV and recurrent state are rebuilt from the same effective
history. That mode is currently certified only as a one-shot coherence strategy.
Persistent above-budget sessions/transcript/local-cache combinations that cannot
maintain this invariant remain fail-closed.

This is deliberately not marked “fixed.” Claiming exact persistent sparse hybrid
state without a new model/state decomposition would be false.

## RC9.2 qualification requirements

Host qualification covers scheduler/session/resource contracts, release
integrity, CLI fail-closed validation, and model-free tests. CUDA-dependent fixes
require the target GPU gate, including at minimum:

1. `qw3-kvmem-kmean-merge`;
2. `qw3-kvmem-exactmass-largeblocks`;
3. immutable-K and MTP position tests;
4. real-model single-slot session lifecycle;
5. failure -> dirty quarantine -> cold recovery;
6. snapshot/restore and tier spill/reload;
7. selected-replay preflight with strict retrieval.

Native multi-slot CUDA remains disabled until two independently instantiated
physical executors, their archive/tier/device ownership, and fault isolation pass
that target-GPU qualification.
