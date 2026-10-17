# v4.0.2 — Integrity and Evidence Closure

Release: `mini-AGI-V1-v4.0.2-integrity` — correctness, evidence
preservation, authorization hardening, and historical reconciliation.
No new learning claims. This release exists to make the evidence chain
trustworthy before the v4.1.0 verified-learning work begins.

## Gate status (release blockers G1–G4)

| Gate | Result | Evidence |
|---|---|---|
| G1 Runner correctness | CLOSED | every declared nonempty split writes real prediction records (`{"split","id",**prediction}`) that reload losslessly and rescore under the strict oracle — `tests-python/v161/test_prediction_serialization.py` (synthetic writer contract + the committed campaign3e artifacts rescore to their recorded metrics) |
| G2 Evidence preservation | CLOSED | `run_reservation.py` — atomic `os.mkdir` claims, ownership-bound `.staging/<run_id>.<attempt_id>`, durable RUN_COMMIT before publication, non-replacing guarded rename, `fail()` preserves `<run_id>.FAILED-<attempt_id>` and releases the claim — `test_concurrent_run_lifecycle.py` |
| G3 Historical reconciliation | CLOSED | all 95 committed run dirs re-qualified under `verify_run`; both campaign3e runs VERIFIED on independent rescore of all 7 prediction files + corpus join + identity bindings; 93 legacy runs marked INCONSISTENT (v1 casefold scoring era — recorded, not rewritten); EVIDENCE_MANIFEST_V4 hidden-row count corrected 96 → 192 |
| G4 Authorization | CLOSED | registry v3 signed+sequenced state, authoritative signed revocation log, activation-time artifact recheck, required model-revision identity, untrusted family hints — `test_registry_tampering.py`, `test_activation_authority.py` |

## Campaign 3E reconciliation (Step 2 outcome)

Both runs were re-verified end-to-end without loading a model:

- `v401-3e-8fam-s0` (320 steps, exposure 4) — VERIFIED. Fit gate FAIL
  (0.85); within-family validation 0.208 → 0.417; hidden 0.021 → 0.010.
  Underfit; direction evidence only.
- `v401-3e-8fam-e8-s0` (640 steps, exposure 8) — VERIFIED. Fit gate PASS
  (0.9875); within-family validation **0.208 → 0.542 (+0.333)** on 24
  word-disjoint unseen inputs; hidden flat 0.021 → 0.021; compositional
  weak 0.0625; capability destruction net-zero churn (4 broken / 4
  gained on hidden).

**The +33.3-point within-family gain survives independent rescore of
the committed predictions.** It remains a single-seed measurement —
not a qualification. v4.1.0 must reproduce it under the recorded
conditions and replicate across seeds before any learning claim.

Reproduction conditions were reconciled from the committed
CONFIG.json: the historical runs used
`configs/campaign3b_positive_control.yaml` (generation contract
`max_new_tokens=16`), NOT `configs/campaign3e.yaml`
(`max_new_tokens=32` — a different contract digest).
`scripts/reproduce_campaign3e.py --from-run <run-dir>` derives the
exact-condition command from the committed config and labels the
attempt a new replication if any field is unrecoverable.

### Correction recorded

`EVIDENCE_MANIFEST_V4.json` reported `hidden_generalization.rows = 96`
for both campaign3e runs; the committed
`PREDICTIONS-{FROZEN,ADAPTED}.jsonl`, CONFIG `hidden_rows`, and the
corpus all carry 192. Accuracies were consistent either way
(4/192 = 2/96) — documentation error, corrected in place with a
`reconciliation_v402` block on each entry.

## Defects closed (DEFECT_REGISTER_V4.json)

- **V4-26** run lifecycle — shared staging names, deletable finals,
  check-then-create publication → `run_reservation.py`
- **V4-27** verifier coverage — legacy 3-file rescore only →
  `evidence_reconciliation.py` full-split + identity verification
- **V4-28** registry authentication — unsigned mutable safety state,
  no activation recheck, optional model identity, forged hint path →
  `authenticated_registry.py` + routing hardening
- **V4-29** matrix CLI — unknown strategy / failed cell exited 0
- **V4-30** matrix identity — cells with absent digests could count
  toward COMPLETE
- **V4-31** controller verification — `record --verified` accepted a
  bare flag → `capability_outcomes.py` receipt checking

### Note on the review's Step 1.1 premise

The review predicted `positive_control_fit.py` serialized integer
score maps as prediction dictionaries. Inspection of the shipped
v4.0.1 source and committed artifacts showed the defect was already
closed at commit `0d0a25f` — `frozen_val`/`adapted_val` are prediction
dicts; the committed campaign3e VALIDATION/COMPOSITIONAL files carry
full records. The behavioral serialization tests were still added to
lock the contract (records round-trip, rescore, and a score-map input
is rejected), because the contract is worth pinning even where the
implementation was already correct.

## Explicitly not claimed

- v4.0.2 ships no new learning result and no qualification claim.
- G5–G8 (generalization on an independently controlled holdout,
  retention bounds, independent confirmation, learning efficiency)
  require the v4.1.0 multi-seed campaign and expanded splits.
- G9/G10 (adaptive controller benefit, recursive improvement) are
  v4.2.0+ milestones. The controller changes here only close the
  evidence path for `record --verified`.
- The sealed-holdout custody item (Step 3.3) remains open: the shipped
  holdout is a public regression test; a genuinely unseen final
  holdout under evaluator custody is a v4.1.0 prerequisite.

## Verification commands

```bash
# independent artifact verification of every committed run
python scripts/requalify_executed_campaign.py

# behavioral + tamper suites for this release's repairs
python -m pytest tests-python/v161/test_concurrent_run_lifecycle.py \
    tests-python/v161/test_prediction_serialization.py \
    tests-python/v161/test_evidence_reconciliation.py \
    tests-python/v161/test_registry_tampering.py \
    tests-python/v161/test_activation_authority.py \
    tests-python/v161/test_fit_runner_integrity.py \
    tests-python/v161/test_v40_routing.py \
    tests-python/v161/test_v40_completeness.py

# reconciled reproduction command (dry run — derives the recorded
# conditions rather than guessing)
python scripts/reproduce_campaign3e.py \
    --from-run results/campaign3e/runs/v401-3e-8fam-e8-s0

# release integrity
python scripts/verify_release.py
```
