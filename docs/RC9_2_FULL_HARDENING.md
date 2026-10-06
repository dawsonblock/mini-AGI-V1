# RC9.2 full hardening

RC9.2 is a deterministic-correctness and qualification-boundary release built on
RC9.1. It does **not** claim to solve the open hybrid-state research problem or to
have run CUDA qualification in a host-only artifact-generation environment.

Implemented changes:

- exact scheduler and physical-runtime lease fencing retained from RC9.1;
- strict workspace-bound public lookup plus trusted internal snapshot lookup;
- prompt/response boundary mean-index merge during decode, including contiguous
  sub-block mean indexes;
- block-count-scalable ExactMass kernel (no O(n_blocks) dynamic shared memory);
- explicit ExactMass raw-key allocation safety cap and overflow checks;
- >12K-block ExactMass CUDA regression test;
- explicit persistent-session scope on the physical executor pool endpoint/schema;
- RC9.2 source/release/provenance consistency checks;
- current CLI/documentation synchronization and stale-issue-ledger reconciliation.

Still deliberately not enabled/claimed:

- native CUDA executor slots > 1;
- persistent above-budget selected-replay for hybrid DeltaNet models;
- target-GPU qualification where no supported NVIDIA GPU/model is present.

Those boundaries fail closed or are labeled unqualified rather than silently
falling back and being reported as equivalent.
