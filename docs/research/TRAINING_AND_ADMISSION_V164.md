# v16.4.0 — Training Semantics, Runtime Admission, Mechanism Control

> Superseded for the admission/security surface by v16.4.1
> (`docs/research/RUNTIME_SECURITY_CLOSURE_V1641.md`): admission is now
> mandatory via `TrustedRuntimeLauncher`, artifact closure refuses
> symlinks/special/unlisted files, receipts are never unsigned, and
> schemas/protocol bindings are strict. This document remains the record
> of what v16.4.0 shipped and why.

Phases 2, 3, and (early) 5 of the v17 plan. Scientific claims are
unchanged: Campaign 1b remains QUALIFIED, Campaign 2 remains REFUSE,
Campaign 3 remains drafted/unexecuted (its execution is GPU-blocked and
still requires the sealed-campaign procedure).

## Phase 2.1 — explicit training schedule (v16.3.0 scope)

The v16.2.1 structured trainer placed `gradient_accumulation_steps`
examples in each microbatch *and* accumulated over that many
microbatches, silently squaring the effective batch size, and no
explicit microbatch size existed anywhere in the signed surface.

Contract now (bound by `ExperimentProtocolV1`, enforced by
`LoraTrainSpec` and asserted at engine level in tests):

    effective_batch_size = microbatch_size × gradient_accumulation_steps
    optimizer_updates    = steps
    sample_presentations = steps × effective_batch_size

Reproduction on the original code (`steps=3, gacc=4`): 12 forwards of
4 rows = **48** sample presentations — 4× the declared schedule. On the
repaired engine: 12 forwards of 1 row = **12** presentations.

Receipt v2 (`mini-agi-v16.3-training-receipt-v2`) additionally binds:
microbatch size, effective batch size, optimizer updates, sample
presentations, the actual sample-order digest, per-update gradient
norms (+ mean), the malformed-sample policy and rejection count, and
`adapter_artifact_digest` — a digest over the adapter weight+config
bytes only, stable across the later receipt/spec writes (those evidence
files are covered by the directory closure digest that runtime
manifests bind).

Malformed samples: `malformed_policy: reject | fail`. Strict
preregistered campaigns set `fail` — any unapproved rejection raises
and invalidates the run before anything is persisted. Campaign 3A/3B/3C
configs now set `microbatch_size: 1` and `malformed_policy: fail`
explicitly (semantically neutral for the drafted configs, which used
`gradient_accumulation_steps: 1`).

## Phase 3 — runtime admission control (v16.4.0 scope)

New module `minagi.v161.runtime_admission` + `scripts/admit_runtime.py`.
Promotion is not complete until the runtime independently admits the
exact artifact and emits an ActivationReceipt. The controller verifies
the full chain at load time:

    plan (role: plan)
      -> qualification record (role: qualification; decision QUALIFIED)
      -> promotion decision (role: promotion; binds campaign, bundle,
         qualification digest, per-seed adapter digests and runtime
         manifest digests; unexpired; unrevoked)
      -> runtime manifest (binds model / tokenizer / adapter bytes /
         protocol; digest self-consistent)
      -> exact adapter bytes measured on disk

Refusals (all fail closed, each with an adversarial test): unsigned /
forged / expired / revoked decisions; wrong-role signer (research-plane
identity cannot promote); one altered adapter byte; substituted or
stale qualification; different base model or tokenizer; a backend not
covered by the qualification; tampered manifests; incomplete chains;
replayed decisions. Rollback re-admits a previously receipted release
from its signed artifacts (research plane not involved) and records the
lineage; a tampered rollback target is refused.

A sixth authority role, `runtime`, was added to the trust root; the
activation receipt is signed by it when the deployment holds a runtime
key and is recorded in the authority ledger as `activation_receipt`.
`promote.py` decisions now carry `expires_at` (default 30-day window).

## Phase 5 (early) — mechanism controller

`plasticity.py` gains the cost-aware selection layer:
`MechanismEstimate` (ΔQ, cost, confidence, regression risk, latency),
frozen `ObjectiveWeights` (`U = ΔQ − λ_C·C − λ_R·R − λ_L·L`, externally
configured), `rank_candidates` (hard regression-risk cap and confidence
floor applied **before** ranking — safety is not traded for utility),
`select_mechanism` (weight-adaptation estimates require recorded
retrieval + skill attempts; when nothing clears the confidence floor it
selects a diagnostic experiment instead of adapting), and
`AttemptReceipt` chaining

    FailureEvidence → MechanismProposal → AttemptReceipt → EvaluationBundle

by digest. Included ahead of the v16.6.0 milestone: the estimates are
currently external inputs — calibrating them from campaign evidence is
v16.6.0 work.

## Record chain (digest links)

plan → protocol → training spec → training receipt →
`adapter_artifact_digest` → evidence receipt (`adapter_digest`) →
evaluation bundle → qualification record → promotion decision →
runtime manifest → activation receipt. Each link is digest-bound;
directory closure digests cover the evidence files written after the
training receipt.

## Verification

- `test_v163_training_semantics.py` — engine-level schedule counts
  (forward rows, optimizer-step calls), policy behavior, receipt
  bindings, sampling-order digest, gradient statistics, protocol
  binding.
- `test_v164_runtime_admission.py` — the plan's four named attacks
  (altered byte, substituted qualification, revoked replay,
  research-plane identity) plus expiry, forgery, backend/model/
  tokenizer substitution, tampered manifests, incomplete chains,
  rollback, receipt integrity, ledger integration, and the CLI.
- `test_v166_mechanism_controller.py` — utility arithmetic, ranking
  determinism, hard safety cap, confidence floor, ladder enforcement,
  diagnostic fallback, attempt-receipt validation.
- Full suite: 561 passed, 1 skipped (Linux-only `RLIMIT_AS` test on the
  macOS host).

## Open items

| # | Item | Status |
|---|------|--------|
| O1 | Campaign 3A/3B/3C execution | GPU-blocked; requires the sealed-campaign procedure |
| O2 | Estimate calibration for the mechanism controller | v16.6.0 work; needs campaign evidence |
| O3 | Activation receipt signing | **Closed by v16.4.1 (SEC-003)** — receipts are always signed; an unsigned receipt is refused by `write_activation_receipt`/`check_activation_receipt` |
| O4 | Native (qw3) backend admission | Refused by design — the qualification plane has not covered it |
| O5 | Native CTest suite | Not rerun (no C++ changes in this release) |
