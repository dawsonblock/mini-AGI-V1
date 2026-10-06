# KVMem runtime profiles and hybrid-state coherence

## Why profiles exist

QW3 exposes a large research surface. Many KVMem, batching, MTP, archive, session,
and scorer options are individually useful but are not uniformly composable.
RC2 introduced named runtime profiles so a deployment can declare a tested
behavioral contract instead of relying on an accidental combination of flags.

Profiles are presets plus validation. The CLI applies preset defaults before
ordinary parsing; a certified profile rejects later overrides that violate its
contract. Use `custom` or `research` when deliberately running an uncertified
combination.

## Profiles

### `agent-safe`

Purpose: correctness-oriented one-shot agent inference.

Contract:

- KVMem enabled.
- retrieval / Mean-K.
- query-conditioned selection.
- strict retrieval (no silent scorer fallback).
- immutable source K.
- message semantic expansion.
- step-boundary reselection.
- query replay enabled.
- serialized generation.
- `selected-replay` hybrid-state coherence.

For an above-budget request, `selected-replay` uses the existing compact-source
`KvAndState` rebuild: after the final KVMem selection, QW3 gathers the selected
historical source tokens in source order, appends the live query suffix, resets
the answer-producing executor state, and densely prefills that compact source.
Both normal-attention KV and DeltaNet recurrent state are therefore rebuilt from
the same selected text for that request.

Current limitation: this mode is supported only for standalone one-shot requests.
RC5 retains detachable logical-session management for query-replay profiles, but
selected-replay still rejects persistent `kvmem_session_id`, transcript replay,
and local-cache requests because its coherent answer cache is not yet detachable.
Below the selection budget, the ordinary dense request is already coherent and
no compact rebuild is needed.

### `agent-fast`

Purpose: lower-latency long-running agent sessions.

Contract is similar to `agent-safe`, but uses `query-replay` instead of selected
source replay. The live query is replayed against the selected attention window,
but arbitrary sparse history is **not** replayed through all recurrent layers.
KVMI-012 therefore still applies. This mode should be treated as an approximation,
not as lossless virtual context.

### `throughput`

Purpose: experimental multi-request throughput.

- KVMem enabled.
- continuous batching enabled.
- `attention-kv` coherence mode.

This profile does not claim exact hybrid-state reconstruction. It exists to make
the trade-off explicit instead of hiding it behind unrelated batching flags.

### `custom`

No preset. The requested combination is validated only for fundamental coherence
constraints.

### `research`

No preset and intended for controlled experiments. It still uses basic parser and
server safety checks, but is not a certified production profile.

## Coherence modes

`attention-kv`
: Selects/remaps standard-attention K/V. DeltaNet state continues to represent
  the original recurrent trajectory. KVMI-012 applies above budget.

`query-replay`
: Replays the live query after semantic selection so the query itself observes
  the selected attention context. This improves current-query construction but
  does not make the complete recurrent history equivalent to the sparse selected
  history.

`selected-replay`
: For an above-budget standalone request, densely replays the selected source
  tokens and live query suffix into a fresh answer-producing cache. This rebuilds
  both standard-attention KV and recurrent state from the same compact source.
  It is the strongest coherence mode currently implemented, at the cost of a
  second prefill over the selected context.

## Readiness and receipts

`GET /readyz` returns process readiness plus the KVMem runtime contract.
`GET /v1/kvmem/status` returns the same KVMem configuration/status object.

Successful Chat Completions responses include a `kvmem` extension with:

- profile;
- state coherence mode;
- logical prompt tokens;
- selection and generation budgets;
- requested retrieval method;
- strict-fallback policy;
- whether the answer-producing request reports coherent hybrid state.

The receipt is telemetry. It is not an authority grant and must not be used by
NEMO/Correct-Once to authorize effects.

## Remaining blockers

RC5 does not close every research issue:

1. KVMI-002 remains open for complete warm-reuse parity, especially sub-block
   and mixed prompt/decode partial-block index paths. Existing incremental
   Mean-K merge/finalization logic mitigates several cases but does not justify
   declaring the issue closed without CUDA/model parity tests across all block
   offsets.
2. RC5 makes canonical host session state detachable and adds an N-slot-capable
   affinity scheduler, but the native backend still certifies one physical
   Qwen executor lineage. Live CUDA KV/DeltaNet/MTP state is not zero-copy
   portable, and persistent selected-replay sessions remain unsupported.
3. Per-request receipts do not yet expose exact selected block counts, tier bytes,
   scorer latency, stage-in latency, or replay latency. Those require a structured
   backend telemetry channel rather than parsing trace logs.
4. CUDA/GPU qualification remains target-hardware dependent.
