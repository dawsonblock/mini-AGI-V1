# RC10.5 — model-specific Gated-DeltaNet `(T,Z)` reference

RC10.5 closes the gap between the earlier generic affine composition prototype and QW3's real recurrent kernel. It adds a host/fp64 correctness implementation derived directly from `src/gated_delta_net.cu`.

## Exact fixed-input recurrence

For one QW3 Gated-DeltaNet value head, the CUDA kernel updates each state column as:

```text
kv[col] = k^T S[:,col]
delta   = beta * (v[col] - g * kv[col])
S'[:,col] = g * S[:,col] + k * delta
```

Therefore the matrix state obeys:

```text
S' = T S + Z
T  = g (I - beta k k^T)
Z  = beta k v^T
```

For a block `C = (1..n)`, RC10.5 composes token summaries to produce:

```text
T_C = T_n ... T_1
Z_C = T_n(...(T_2 Z_1 + Z_2)...) + Z_n
```

and verifies:

```text
S_C = T_C S_0 + Z_C
Z_C = S_{C|0}
```

up to fp64 roundoff.

## What is implemented

- C++ reference: `include/qw3/gdn_reference.hpp`, `src/gdn_reference.cpp`.
- Python independent reference: `continual/src/kvcontinual/recurrent/gdn_reference.py`.
- Native CTest parity coverage: `tests/gdn_reference_test.cpp`.
- Python pytest coverage: `continual/tests/test_gdn_reference.py`.
- User-facing Mac smoke oracle: `./macos/qw3-mac gdn-check`.
- Numerically stable sigmoid/softplus/raw-alpha-to-decay helpers matching CUDA preprocessing semantics.
- Direct token replay oracle and matrix-state error measurement.

## What this proves

It proves the segment-level affine representation is exact **when the block-local `k`, `v`, `g`, and `beta` values are fixed**. This is the algebraic prerequisite for HYPIC-style recurrent-state reuse.

It does **not** prove that a cached block remains exact after arbitrary deletion/reordering in the full deep hybrid network. Moving a block changes the hidden state entering that block, which can change its projected `k/v`, gates, attention outputs, FFN/MoE routing, and therefore its true recurrence operator. RC10.5 deliberately retains seam/suffix/exact-replay escalation as the path for repairing that nonlinear context drift.

## Next native-Mac milestone

The next step is not another abstract recurrence implementation. It is a Metal kernel/parity layer that consumes the same recurrence inputs as QW3, captures per-block structured `(T_C,Z_C)` summaries, and compares them against this CPU oracle and exact selected replay. That work must be qualified on actual Apple Silicon hardware.
