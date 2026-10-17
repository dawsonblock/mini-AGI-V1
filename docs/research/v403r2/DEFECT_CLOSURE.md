# v4.0.3-r2 — Defect Closure (Diagnostics 9)

Scope: the three Diagnostics 9 blockers that are *engineering
correctness* problems — false scientific qualification, unverified
model execution, unreliable evidence bindings — plus the statistical,
holdout-custody and release-integrity requirements of the r2 release.
The retention-research blocker (v4.1.1) is explicitly NOT claimed here.

Test totals executed: **888 v161 tests passed, 4 skipped**
(+38 new adversarial tests over r1); full repository suite re-run at
commit time.

## Scoreboard

| Gate | Required | Result |
|---|---|---|
| Scientific-gate correctness | PASS | PASS — completeness ≠ qualification |
| Failed-gate adversarial tests | PASS | PASS — 10 Diagnostics-9 scenarios |
| Evaluator-owned inference | PASS on real model | PASS — Qwen2.5-0.5B-Instruct loaded + executed |
| Required evidence identities | PASS | PASS — resolver enforces all 12 |
| Authorization + key isolation | PASS under defined deployment | PASS (mechanism) — custody checks implemented; OS separation remains deployment policy |
| Statistical-policy enforcement | PASS | PASS — versioned policy, CI gates, NOT_EVALUABLE |
| Independent holdout custody | PASS | PASS — sealed evaluator store |
| Full Python regression | all applicable pass | PASS |
| Fresh ZIP source verification | PASS | DEV-VERIFIED — chain executed; signed release ZIP deferred to release cut |

## V4-43 — False scientific qualification (P0, CLOSED)

**Defect.** `matrix_qualified` was `matrix_status == "COMPLETE"`: a
fully receipted run failing its generalization gate still produced a
qualified matrix.

**Repair.** `analyze_v410_matrix.py` now computes two independent
quantities:

```
matrix_complete    = all_required_cells_present
                     and all_cells_verified_at_required_level
                     and protocol_identities_consistent
candidate_qualified = is_candidate and matrix_complete
                     and evidence_level_met and not arm_invalid
                     and not arm_rejected and all_gates_pass
```

Matrix status: `INCOMPLETE` | `INVALID` | `COMPLETE` |
`COMPLETE_WITH_FAILURES` | `QUALIFIED`. Per-arm states include
`NOT_EXECUTED`, `INSUFFICIENT_EVIDENCE`, `PASSES_GATES`,
`COMPLETE_WITH_FAILURES`, `REJECTED`, `QUALIFIED`. Negative controls
(`role: negative_control` in the manifest, or `--control-arms`) may
fail acquisition gates and still contribute to a complete matrix; a
statistically supported gain on a negative control is flagged
`control_integrity: unexpected_gain`. CLI modes: default strict
completeness, `--require-qualified [arm]` for promotion checks,
`--informational` for report-only exits.

**Evidence.** 39 matrix/adversarial tests +
`test_failed_gate_qualification.py` (10 scenarios: failed
generalization/retention/security gates, missing adapter/predictions/
receipt, undefined uncertainty, valid negative control, one qualified
candidate among failures, incomplete multi-seed matrix).

## V4-44 — Unverified model execution (P0, CLOSED)

**Defect.** The evaluator signed a receipt over researcher-submitted
predictions; nothing proved the claimed model+adapter produced them.

**Repair.** `independent_model_executor.py` loads the frozen base
revision (exact `HFLoadSpec` revision) and the adapter bytes in the
evaluator's process, runs `encode_generation`-contract inference on
evaluator-controlled tasks, scores with the declared oracle and issues
`model-execution-receipt-v1` binding: model_id, model_revision,
tokenizer_digest, adapter_digest, dataset_digest,
generation_config_digest, scoring_version,
execution_environment_digest, outputs_digest, scores_digest,
execution_mode. Mock loaders are a test seam — they mark the receipt
`simulated`, which `require_real` (default) refuses as real-model
confirmation. Execution provenance and training provenance are
deliberately separate claims.

**Evidence.** `test_model_execution_provenance.py` — 12 cases:
simulated-never-real, wrong model identity, wrong adapter hash,
invalid adapter format, incompatible adapter base, altered
predictions, stale receipt replay, unauthorized evaluator, load
failure, frozen-baseline null-adapter, and two real-model tests that
load `Qwen/Qwen2.5-0.5B-Instruct@7ae5576` + a committed dev LoRA and
execute real inference (environment-gated).

## V4-45 — Optional evidence identities (P1, CLOSED)

**Defect.** Nothing enforced that every mandatory evidence identity
was bound and non-null.

**Repair.** `trusted_evidence.resolve_identities` checks 12 mandatory
identities — base model, tokenizer, adapter, dataset, scorer,
protocol, generation config, predictions, security, retention,
evaluation receipt, execution receipt (policy-gated). Null permitted
only for declared adapter-free experiment types
(`frozen_baseline`/`foundation_only`). `resolve_trusted_evidence`
rejects missing artifacts, hash mismatches, wrong revisions,
unauthorized receipts, unsupported scoring versions, and
cross-bound receipts — a refused receipt drops the state to INVALID,
not merely below-threshold. `evaluator_custody.py` enforces the
checkable custody half (0600 key, uid ownership, outside research
workspace); `evaluator_policy.yaml` declares the deployment policy.

## V4-46 — Statistical qualification unenforced (P1, CLOSED)

**Defect.** CIs were computed but not consistently used; a point
estimate or degenerate one-cluster interval could look like a pass.

**Repair.** `statistical_policy.yaml` versions the frozen contract —
LCB(Δ) > 0.15 acquisition, UCB(destruction) ≤ 0.10 retention,
min 2 family clusters, min 2 independent seeds, fail-closed missing
data, seeds frozen at preregistration. `--statistical-policy` merges
its `gates:` into the analyzer so the statistical contract lives in
one place; insufficient seeds/clusters/undefined intervals are
`NOT_EVALUABLE`, never passing values.

## V4-47 — Holdout answers not sealed (P1, CLOSED)

**Defect.** The campaign4a final holdout shipped with answers — a
public dev benchmark, not a confirmation surface.

**Repair.** `sealed_evaluator.py`: `SealedHoldoutStore` keeps answers
in a 0700 store with 0600 files; `generate_fresh_tasks` produces
per-eval fresh families; `run_sealed_confirmation` loads
tasks+answers evaluator-side, executes the candidate itself, scores
privately, applies gates, signs the execution receipt and returns
aggregate+receipt only. Attempt logging makes an eval set
single-use — repeated tuning against the final holdout is refused.
`within_holdout` remains the separate shipped surface (evaluator-side
custody + statistical independence, not third-party attestation); a
hardened deployment can place its answers behind this same seal.

## V4-48 — Release manifest drift (P1, OPEN → dev-verified)

**Defect.** The latest ZIP contained source files absent from the
manifest and files modified after inventory — a valid signature over
a stale inventory.

**Repair.** The frozen chain exists and was exercised this release:
`reissue_release.py` regenerates `SOURCE_MANIFEST.json` over
`git ls-files` ∩ disk → sign with the pinned release key →
deterministic zip → fresh-extract `verify_release.py`. Development
verification executed against the committed tree (manifest regenerates
cleanly, signature verifies under the pinned fingerprint). The signed
release ZIP `mini-AGI-V1-v4.0.3-r2-Qualification-Closure.zip` is
deferred to the release cut — packaging an unfrozen tree would repeat
the defect.

## Residual known limitations (honest, not hidden)

- **Custody is deployment policy.** The evaluation/execution signing
  keys live on the same host as the research agent in this workspace.
  Mechanism closed; real confirmation needs the evaluator under a
  separate OS identity (`evaluator_custody` enforces the checkable
  half only).
- **No committed run carries execution provenance.** Every run under
  `results/` tops out at LOCALLY_RECOMPUTED (and only dev-receipted
  cells at INDEPENDENTLY_VERIFIED) — the matrix correctly fails
  closed. A real Campaign 4A confirmation requires evaluator-custody
  runs with exec receipts.
- **Sealed confirmation is simulated in tests.** The sealed-evaluator
  tests use the mock loader; a real-model sealed confirmation against
  the actual candidate artifacts is a confirmation-lane run, not a
  unit test.
- **Release ZIP not yet cut** — the chain is dev-verified; the signed
  artifact requires the frozen release tree.
