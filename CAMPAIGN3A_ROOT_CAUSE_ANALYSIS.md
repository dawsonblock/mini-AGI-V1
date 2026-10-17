# Campaign 3A — Root cause analysis

Status: **complete**. Every defect below was reproduced against the
Campaign 3A source (v16.4.3-hardened-runtime as embedded in the 3A
archive, carried forward into v16.4.10) and repaired at source in
Campaign 3B. Campaign 3A's evidence and its **REFUSED** decision are
preserved unmodified; nothing here amends them.

The 3A result, for the record: neural gain over grounded replay
**+0.00219** against a preregistered **+0.0200** (~11% of threshold),
6/10 positive-transfer seeds → REFUSED.

## Summary

| # | Defect | Class | Evidence it is real | Repair |
|---|---|---|---|---|
| RC-1 | Training and inference rendered different prompt surfaces | P0 — invalidates the measurement | `peft_trainer.supervised_tokens` used `apply_chat_template(..., add_generation_prompt=True)`; `run_campaign1.generate` used `tokenizer(prompt)` on the raw string | shared `prompt_contract` (SCI-301) + token-parity tests |
| RC-2 | The evaluation surface could not produce a scorable answer | P0 — invalidates the measurement | the historical raw surface truncated **100%** of generations and emitted EOS **0%** of the time | frozen answer-only output contract (SCI-303) + calibration |
| RC-3 | The grounded-replay baseline was empty | P0 — the primary comparison had no non-neural comparator | L5 accumulated **zero traces across all ten seeds** | L5b oracle-verified training replay (SCI-302), with L5a preserved as the honest negative |
| RC-4 | "Sealed" confirmation never executed a prediction | P0 — the confirmation was not a confirmation | the qualifier checked a holdout manifest digest only | `run_sealed_confirmation.py` — loads sealed rows, generates, scores, signs |
| RC-5 | `false_activation_rate` used the wrong denominator and reported 0.0 when undefined | P1 — overstates safety | rate divided by all pairs; an empty baseline-correct set yielded `0.0` | baseline-correct denominator; empty → `NOT_ESTIMABLE` / `null` (SCI-305) |
| RC-6 | Repeated seeds on the same tasks were treated as independent samples | P1 — overstates confidence | per-seed bootstrap over rows, no family clustering | family-aware cluster bootstrap, leave-one-family-out, worst-family (Step 15) |
| RC-7 | The only safety evidence was a literal-marker smoke test | P1 — overstates safety | `security_regression` detects `<unsafe>`/`<policy-bypass>`/`<secret-leak>` and nothing else | governed 8-category suite (Step 18) |
| RC-8 | Strict exact match was not separable from diagnostic scoring | P2 — masks the failure mode | one score per row, no format/termination diagnostics | strict EM primary + first-line / prefix / format / EOS / truncation diagnostics (SCI-303) |

## RC-1 — Prompt parity break (the primary defect)

**What the code did.** Training built its input through the chat
template with a generation prompt:

```python
prompt_text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True, tokenize=False)
```

Inference built its input by tokenizing the raw task string:

```python
batch = tokenizer(prompt, return_tensors="pt")
```

**Why it invalidates the measurement.** The adapter was trained on
`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n` and evaluated
on a bare prompt with no role markers and no assistant turn. The model
was scored on a surface it had never been trained on. No amount of
training could show up as improvement on that surface, and any
apparent difference between arms was a difference in how each arm
happened to handle an out-of-distribution input.

**Repair.** `minagi.v161.prompt_contract` is now the single renderer for
both paths (`encode_generation` / `encode_supervised`). Parity is not
asserted in prose: for every valid supervised example the contract
proves `training_input_ids[:prompt_length] == inference_prompt_input_ids`
and fails closed otherwise. See `PROMPT_CONTRACT_SPEC.md`.

## RC-2 — An evaluation surface that could not be scored

`scripts/calibrate_generation.py` measured the historical surface
against the pinned frozen model on development rows (no adapter):

| config | strict EM | valid format | EOS | truncation | mean tokens |
|---|---|---|---|---|---|
| historical-raw-32 | 0.000 | 0.000 | **0.000** | **1.000** | 32.0 |
| answer-only-64 (selected) | 0.167 | **1.000** | **1.000** | **0.000** | 3.2 |

**Why it invalidates the measurement.** Under the 3A surface, *every*
generation ran to the token limit and none terminated on the
end-of-turn token. Strict exact match was 0.000 for the frozen model.
A benchmark on which the baseline scores zero because the harness never
lets it answer cannot detect learning; it can only detect noise.

**Repair.** The answer-only output contract is frozen, bound by digest
into signed metrics and the training receipt, and validated by
`generation_diagnostics` (finish reason, truncation, format validity).
Strict exact match remains the primary metric — prefix/first-line
scores are recorded as diagnostics only.

## RC-3 — The comparator was empty

The 3A L5 arm was a *self-verified* replay buffer: traces were admitted
only if the model's own practice-phase answer was verified. Across all
ten seeds it admitted **zero** traces, so the "grounded replay"
baseline contributed no context and the primary incremental comparison
reduced to adapter-vs-frozen while being *reported* as adapter-vs-replay.

**Repair.** L5b admits every TRAIN row's established label with recorded
provenance (task/family identity, split identity, prompt and answer
digests, verifier identity, verification result, retrieval eligibility)
and never reads hidden or sealed labels. L5a is preserved unchanged and
reported as INACTIVE rather than as an operational baseline. See
`REPLAY_BASELINE_VALIDATION.md`.

## RC-4 — "Sealed confirmation" that never confirmed

The qualifier verified that a final-holdout manifest digest matched the
one bound in the plan. It did not load the holdout, generate a
prediction, or score anything. A digest match is evidence that a file
is the file it claims to be; it is not evidence about a model.

**Repair.** `scripts/validation/run_sealed_confirmation.py` runs the
full confirmation: freeze submission → verify eligibility → verify
candidate artifacts against signed receipts → load sealed rows →
generate real predictions → score under frozen rules → family-level
uncertainty, retention and safety → sign QUALIFIED/REFUSED → append-only
ledger. See `SEALED_CONFIRMATION_PROTOCOL.md`.

## RC-5 — An undefined safety rate reported as zero

`false_activation_rate` divided off-target flips by *all* pairs,
diluting the rate by rows the baseline already failed. Worse, when no
baseline row was correct the denominator was zero and the function
returned `0.0` — reporting "no off-target flips measured" when in fact
no measurement was possible.

**Repair.** The denominator is the baseline-correct population; a zero
denominator returns `rate = null` with `status = NOT_ESTIMABLE`, and the
qualifier refuses rather than accepting an unmeasurable safety claim.
The same rule is applied to the comparative retention category.

## RC-6 — Pseudoreplication

Per-seed bootstrap CIs were computed over rows, and "positive seeds"
counted repeated evaluations of the same task templates as independent
evidence. Ten seeds over the same tasks are not ten independent samples
of task difficulty.

**Repair.** The uncertainty unit is the task **family**
(`cluster_bootstrap_ci`), with leave-one-family-out sensitivity, worst-
family degradation, per-family strict accuracy and an explicit
`families_positive` count. A gain carried by one family is reported as
such and does not support a broad-transfer claim.

## RC-7 — Safety evidence was a smoke test

The only safety evaluator detected three explicit test markers. It is a
useful canary and a useless safety suite: it says nothing about
instruction-hierarchy robustness, protected-information disclosure,
unauthorized action, prompt injection or off-task drift.

**Repair.** The governed suite (Step 18) covers eight mandatory
categories with objectively verifiable, allowlisted predicates and
fail-closed gating. The literal-marker evaluator is retained and
labelled a smoke test. See `SAFETY_AND_RETENTION_REPORT.md`.

## RC-8 — Scoring collapsed into a single number

One score per row meant the failure mode was invisible: a truncated
generation and a confidently wrong answer were indistinguishable.

**Repair.** Strict exact match remains primary; first-line correctness,
answer-prefix correctness, format validity, generated length and finish
reason are recorded separately, so a future negative result can say
*why* it was negative.

## What was deliberately not changed

- **Campaign 3A's evidence and REFUSED decision.** Immutable.
- **The historical surfaces.** `DEFAULT_OUTPUT_CONTRACT` and the raw
  prompt path remain available so 3A-era records replay exactly as they
  were produced.
- **The runtime security surface.** No isolation, admission, signing,
  revocation or authority module was modified (see
  `RUNTIME_INTEGRATION_REPORT.md`).
- **The architecture.** No dynamic LoRA, no extra agents, no recursive
  training, no larger model was added to compensate for the negative
  result.
