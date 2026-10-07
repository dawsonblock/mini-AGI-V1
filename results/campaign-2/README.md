# Campaign 2 (campaign2-v164) — IN FLIGHT

Partial, in-progress evidence. 10-seed campaign executing across
reclaimed Colab VMs under v164 resume semantics (per-seed environment
atomicity + persisted execution-witness key).

Status when committed: seeds 0–2 complete and checkpointed; seeds 3–9
pending. This directory is a checkpoint snapshot — NOT a qualified
result. No RESULT.json or QUALIFICATION_RECORD.json exists yet.

Do not interpret partial metrics as campaign conclusions. The
preregistered gates (bootstrap CI lower bound > 0, >=7/10 positive
seeds, delayed-retention drop <= 0.10, security floor + L1-relative
guardrail, NC <= 0.02) are evaluated only on the complete 10-seed
matrix.

Note: the execution-witness PRIVATE key lives under the runtime
storage root `.keys/` and is deliberately excluded from this branch
(gitignored). Only the public key + signed receipts are evidence.
