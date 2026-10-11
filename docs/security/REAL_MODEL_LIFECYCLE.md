# Real Model Lifecycle (v16.4.8) — G10

## Status: PENDING the Linux qualification host

Planned evidence (to be filled with captured output):

1. Pinned HF/PEFT worker env: record torch/transformers/peft/tokenizer
   versions; stage Qwen2.5-0.5B-Instruct + tokenizer + adapter as
   hash-verified read-only artifacts.
2. Base-model smoke: worker under the nsjail executor loads,
   tokenizes, infers within token/time budgets; record outputs,
   memory, latency, worker_id, unit cgroup.
3. Adapter activation: prepare B → private load → health → activate →
   route → infer with B → rollback → B's unit terminated and empty.
4. Recovery + revocation: restart supervisor, verify durable state,
   re-check promotion + revocations; revoke B, prove it is never
   restored.
5. CUDA pass only after CPU passes, with an explicitly qualified
   device/driver/library bind list — never a broad /dev expose.

No part of G10 is claimed until the recorded run exists.
