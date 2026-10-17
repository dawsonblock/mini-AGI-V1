# Campaign 3B — Repair report

Status: **implementation complete; scientific decision REFUSED**.
This document is the dependency-ordered record of what was repaired at
source, what was tested, and what remains unqualified.

Campaign 3A's evidence and its REFUSED decision are preserved
unmodified. Nothing in 3B retroactively changes a 3A threshold, a 3A
score or the 3A decision.

## 1. Baseline and trust

| item | value |
|---|---|
| runtime base | v16.4.10-containment-and-authority-closure |
| scientific protocol | v166 (`ColabCampaignPlanV166` + `ExperimentProtocolV1`) |
| branch | `research/campaign3b-diagnostics` |
| base commit | `18c1429` (v16.4.10 envelope reissue) |
| model | `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775` |
| positive-control corpus | `configs/positive_control_tasks.jsonl` (78 rows) |
| sealed holdout digest | `sha256:11ea116394fdc1cfffb7d63153dec44c5a8802f344f65a425d7037c9094dcf63` |

The 3A archive embedded the older 16.4.3-hardened-runtime source. The
3B work is built on the **newest** runtime (v16.4.10) and did not
overwrite newer security code with the older implementation. The
runtime security modules were not modified at all (§6).

**Trust limitation, recorded honestly:** the archives were extracted
and inventoried, but no independent cryptographic attestation of the
3A archive's provenance was available to this effort beyond the
manifests it ships with. The release signature in this tree is
verified with the repository's own pinned public key; it is **not** an
independently witnessed attestation. See `REMAINING_DEFECTS.md`.

## 2. Dependency-ordered repair register

| step | repair | defect | status |
|---|---|---|---|
| 3 | shared prompt/tokenization contract | RC-1 (SCI-301) | implemented + tested |
| 4 | answer-only output contract, strict scoring separation | RC-2/RC-8 (SCI-303) | implemented + tested |
| 5 | oracle-verified training replay, L5a preserved | RC-3 (SCI-302) | implemented + tested |
| 6 | corrected experimental arms | — | implemented |
| 7 | positive-control learnability | — | **executed; REFUSED** |
| 8 | family-aware statistics, undefined-metric handling | RC-5/RC-6 (SCI-305) | implemented + tested |
| 9 | governed safety & retention suite | RC-7 | implemented + tested |
| 10 | sealed confirmation executor | RC-4 | implemented + tested (synthetic) |
| 11 | integration with the governed runtime | — | preserved, not weakened |
| 12 | full test sweep | — | see `TEST_RESULTS.md` |

## 3. SCI-301 — shared prompt contract

New module `minagi/v161/prompt_contract.py`. Training and inference now
render through one versioned renderer; the contract binds model and
tokenizer identity, chat-template digest, message construction, the
assistant-generation prefix, response-only targets, EOS/end-of-message
handling, token limits, padding/truncation policy, expected answer
format and a configuration digest.

Parity is *proved*, not assumed: for every valid supervised example the
contract asserts `training_input_ids[:prompt_length] ==
inference_prompt_input_ids` and fails closed on mismatch. Tokenizer
boundary effects are handled explicitly rather than by concatenating
independently tokenized strings. Prompt and padding tokens are masked
with `-100`; intended response tokens are supervised.

`peft_trainer.supervised_tokens` delegates to the contract;
`run_campaign1.generate` renders through `encode_generation`. Full
specification: `PROMPT_CONTRACT_SPEC.md`.

## 4. SCI-303 — answer-only generation and scoring

The frozen output contract (`Return only the transformed result. Do not
provide an explanation.`, 16 tokens, EOS stop) is bound by digest into
signed per-arm metrics and the training receipt. Generation is greedy,
bounded and stops on the end-of-turn token; the finish reason and
max-token truncation are recorded explicitly.

Strict exact match remains the **primary** metric. Diagnostics are
recorded alongside, never instead: correct first line, correct-answer
prefix, output-format validity, generated length and termination
reason. Prefix scoring never substitutes for strict exact match.

Selection was calibrated on development rows only and frozen before
training (`docs/science/SCORING_CALIBRATION.md`). Any scorer or
generation-policy change requires a new protocol version; none of this
amends 3A.

## 5. SCI-302 — an operational non-neural baseline

`OracleVerifiedReplayArm` (L5b) admits TRAIN rows with recorded
provenance — task and family identity, training-split identity, prompt
and answer digests, verifier identity, verification result and
retrieval eligibility. Hidden and sealed-confirmation labels are never
loaded into the store; out-of-split and unverified examples are
rejected.

The original self-verified arm (L5a) is preserved unchanged and
reported as **INACTIVE** when empty rather than presented as an
operational baseline. Measured operation and provenance tests:
`REPLAY_BASELINE_VALIDATION.md`.

## 6. Corrected experimental arms

L1 frozen · L5a self-verified replay · L5b oracle-verified training
replay · L6 fixed-rank LoRA · L7 LoRA + the same verified replay L5b
sees · NC shuffled-label LoRA. All arms share one tokenizer, prompt
contract, scoring rule, model snapshot, task distribution and decoding
policy.

Preregistered primary comparison: **Score(L7) − Score(L5b)**. Secondary:
Score(L6) − Score(L1). Inference-token and compute costs are recorded
per arm (`cost_per_correct`, wall/train seconds, generated tokens,
trainable parameters, arm-state bytes).

## 7. Positive-control learnability (executed)

The diagnostic ran end to end and produced a **REFUSED** outcome with
7 readiness gates assessed, 6 passing. Headline: prompt parity,
supervised targets, numerical stability and adapter reload all pass;
LoRA learns within-family (validation 0.222 → 0.444 against a 0.222
frozen baseline) and the negative control collapses (0.067 train
accuracy vs 0.767 for correct labels), but the 95% fit gate is unmet
(0.767) and cross-family transfer is negative (0.167 vs 0.417 frozen).

Per the plan's decision logic the expensive multi-seed campaign was
**not** launched, and the failure was **not** compensated for by adding
dynamic LoRA, more agents, recursive training or a larger model. Full
evidence: `LEARNABILITY_TEST_REPORT.md`.

## 8. SCI-305 / Step 15 — statistics and scientific validity

`false_activation_rate` now conditions on the baseline-correct
population and returns `NOT_ESTIMABLE` with `rate = null` when that
denominator is empty — never `0.0`. Added: paired baseline comparisons,
per-family performance, seed-level uncertainty, family-aware cluster
bootstrap CIs, leave-one-family-out sensitivity, worst-family
degradation, positive-seed counts, off-target regression rates and
improvement per unit of compute. Repeated seeds over the same tasks are
one cluster, not independent samples. Details:
`SCORING_AND_STATISTICS_REPORT.md`.

## 9. Step 18 — governed safety and retention

New module `minagi/v161/safety_regression.py` governs eight mandatory
categories with objectively verifiable, allowlisted predicates: five
scored from the security partition (instruction hierarchy, protected
information, unauthorized action, prompt injection, off-task drift),
one comparative (previously-correct answers regressing) and two owned
by the retention partition (immediate and delayed retention).

The literal-marker evaluator is retained and labelled a smoke test. A
mandatory category with no scorable evidence is UNDEFINED and blocks
promotion; the qualifier recomputes the rollup from persisted
predictions, so the training agent's own favourable self-assessment is
never safety evidence. Details: `SAFETY_AND_RETENTION_REPORT.md`.

## 10. Step 20-22 — sealed confirmation

`scripts/validation/run_sealed_confirmation.py` freezes an immutable
submission, verifies model/adapter/tokenizer/contract/scorer
identities, operates under evaluator authority, loads sealed tasks,
generates **actual** predictions, scores them under frozen rules,
compares against the preregistered baselines, evaluates family-level
uncertainty plus retention and safety, signs QUALIFIED/REFUSED and
appends to an append-only ledger. A dataset digest alone is never
treated as a completed confirmation.

Research-side access to sealed data is denied by two fail-closed
controls and proven by tests. Details:
`SEALED_CONFIRMATION_PROTOCOL.md`.

## 11. Step 11 — runtime integration

The 3B work adds scientific modules; it does not modify isolation,
admission, signing, revocation or authority. `git diff` over the
runtime security modules is empty; the qualifier gained gate reasons
(17 → 19) and lost none. A scientifically qualified adapter is
*eligible* for deployment, not activated: it must still pass artifact
verification, backend compatibility, signed promotion authorization,
current revocation checks, resource budgets, worker isolation and real
model health checks. Details: `RUNTIME_INTEGRATION_REPORT.md`.

## 12. Evidence and release integrity

Campaign 3A stays immutable. This release separates (1) the corrected
3B source release, (2) diagnostic and training research evidence and
(3) independently controlled sealed-confirmation evidence, each with
its own inventory and SHA-256 verification.

**Signing status.** The envelope was reissued with the repository's
existing release key (`~/.config/miniagi/release-ed25519.pem`, held
outside the tree). The bundled `RELEASE_PUBLIC_KEY.pem` hashes to
`a8cd5cb2…`, which is exactly the attestation's pinned
`public_key_sha256`, and the reissue reported the same fingerprint — so
this is the repository's release authority, not a newly minted key
presented as one. `RELEASE_ATTESTATION.json` was reconciled
(`manifest_sha256` + `reissued_utc` + a 56th `reissue_history` entry),
following the workflow the previous 55 reissues used.

**What the signature does and does not mean.** It establishes that this
envelope was produced by the holder of the release key. It is a
development-machine key, not an independently witnessed third-party
attestation, and it says nothing about whether the scientific claims in
this package are correct. See `REMAINING_DEFECTS.md` and
`RELEASE_CHANGE_MANIFEST.json`.

## 13. Honest summary

**What is proven:** the pipeline is mechanically correct and
instrumented. Prompt parity holds by construction and by test;
supervised masking is correct; generation and scoring are deterministic
and separated; verified replay operates where 3A's buffer was empty;
statistics fail closed on undefined metrics; the safety suite enforces;
the confirmation executor executes and signs.

**What is not proven:** that neural adaptation improves performance on
unseen tasks beyond the non-neural baseline. The one executed
diagnostic says the opposite at its scale. No improvement claim in this
package rests on a training script running or a loss falling — the
executed evidence is reported as it came out.
