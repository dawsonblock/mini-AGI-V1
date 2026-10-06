# RC9.1 correctness hardening

RC9.1 is a correctness and release-integrity repair of RC9. It does not enable
multi-slot CUDA serving.

## Fixed

1. **Scheduler stale-lease release** — every slot now records the exact active
   lease ID and release requires an exact token match.
2. **Physical runtime stale-lease release/reset** — runtime ABI v2 adds exact
   lease fencing to `release()` and `mark_cold_reset()`. The physical runtime
   issues its own monotonic token and separately records the triggering scheduler
   lease ID for cross-layer traceability.
3. **Workspace binding omission bypass** — once a session is bound to a
   workspace, an omitted/empty caller workspace no longer bypasses the binding.
4. **Host release gate drift** — the multi-slot fail-closed assertion now matches
   the RC9/RC9.1 runtime diagnostic and the full host gate passes.
5. **Release identity drift** — current release metadata, NEMO ABI checks, GPU
   gate messages, schemas, and qualification records identify RC9.1 consistently.

## Regression coverage

- stale scheduler lease for the same session cannot release a newer lease;
- stale physical runtime token cannot release a newer physical owner;
- bound session access with an empty workspace ID is rejected;
- existing dirty-slot quarantine/cold-reset tests remain passing.

## Explicitly not claimed

- CUDA qualification was not run in the artifact-generation environment;
- more than one concurrently active native CUDA executor remains uncertified and
  rejected;
- persistent selected-replay remains unsupported;
- KVMI-012 hybrid attention/recurrent-state coherence remains an open research
  limitation above the exact supported one-shot path;
- KVMI-002 partial-block retrieval-index parity remains open pending target-GPU
  qualification/fix.
