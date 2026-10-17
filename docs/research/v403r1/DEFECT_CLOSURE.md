# v4.0.3-r1 — Defect closure report

Release lane: **v4.0.3-r1 Evidence Authority Closure**
Exit criterion under test: *no fabricated experiment can be accepted as
independently verified.*

This report documents what was broken, what now enforces the boundary,
and what was executed to prove it. Nothing below reports a scientific
qualification — this lane is engineering-integrity work, and it
produces zero capability claims.

---

## 1. The defects

### D1 — Local recomputation was labelled independent evaluation (P0)

`evidence_reconciliation.verify_run` can fully rescore a committed run:
every prediction file strict-rescores to the reported metrics. Correct
machinery, wrong label. Downstream, `capability_control.py record
--receipt` treated that local recomputation as *evaluation* — the run
entered verified controller history even though the "evaluator" was the
same process reading files the training pipeline wrote.

A fabricated run — self-authored predictions, self-authored metrics and
a **matching** commit inventory — satisfied every check. The audit's
verdict was correct: the system could generate its own authoritative
evaluation.

**Closed** by splitting verification into four states (V4-38):

| State | Established by | Trusted history? |
|---|---|---|
| `INTEGRITY_VERIFIED` | commit + file-hash inventory consistent | No |
| `LOCALLY_RECOMPUTED` | `verify_run` strict rescore reproduces | No |
| `INDEPENDENTLY_VERIFIED` | signed `independent-evaluation-v1` receipt from an `evaluation`-role authority, binding recomputes exactly | Yes — including negative outcomes |
| `QUALIFIED` | `qualification`-role signed decision over `run_id` + evidence digest, on top of independent verification | Yes — promotion-eligible |

`capability_control.py record` now denies any evidence below
`INDEPENDENTLY_VERIFIED`, rejects caller-supplied numbers beside
`--receipt`, and keeps verified negative outcomes admissible (a failed
rehearsal run is evidence; dropping it would bias the controller).

### D2 — Matrix analyzer trusted asserted aggregates (P0)

`analyze_v410_matrix.py` could mark all four gates passing on a
favorable `METRICS.json` alone — no committed predictions, no adapter,
no closure. Missing retention evidence defaulted to zero regression;
an incomplete matrix (unexecuted arms) qualified because only present
runs were analyzed.

**Closed** (V4-39):

- every cell is evidence-closed through `derive_qualified_outcome`
  before any gate reads it;
- gate inputs are rescore-derived from committed prediction bytes —
  runner-asserted rollups are reported for audit only;
- each metric is three-stated (measured | explicitly optional |
  missing-required) — a gate needing a missing value is
  `NOT_EVALUABLE`, and an undefined confidence interval is not
  statistical evidence;
- `--manifest` declares the expected seeds×arms; absent declared cells
  keep the matrix `INCOMPLETE` and `matrix_qualified: false`;
- `required_evidence_level` (preregistered in `confirmation.yaml` as
  `independently_verified`, overridable for development via
  `--evidence-level`) sets the level every cell must reach;
- exit code is nonzero whenever any cell is missing or below the
  required level.

### D3 — Signed receipts verified without role authorization (P1)

`verify_outcome_receipt` accepted a validly-signed legacy receipt with
only a signature verifier — a key whose evaluation authority could not
be established still counted.

**Closed** (V4-40): an `AuthorityRegistry` is mandatory for signed
receipts; wrong-role, revoked, expired and unregistered signers all
fail with auditable reasons.

### D4 — Qualification could stand on local evidence (P1)

A signed `QUALIFIED` decision elevated a run that was never
independently evaluated.

**Closed** (V4-41): `QUALIFIED` is reachable only from
`INDEPENDENTLY_VERIFIED`; the decision still binds `run_id` and the
exact evidence digest.

### D5 — Security-bound receipts were unverifiable downstream (P1)

Found by dogfooding on real evidence: a receipt issued with the
governed security sidecar bound into `security_metrics_digest` could
never verify through `verify_outcome_receipt` — the recomputation had
no way to receive the same document.

**Closed** (V4-42): `security_doc` now flows through the whole
verification chain; the analyzer presents the sidecar it scored to the
binding check; `capability_control.py` gained `--security-evidence`.

---

## 2. What was executed

All claims below are from actual runs on this checkout.

### 2.1 Regression suite

```
python3 -m pytest tests-python/v161 -q   → 837 passed, 4 skipped
python3 -m pytest tests-python -q        → 1533 passed, 21 skipped
```

(Skips are environment-gated — NsJail/systemd/cgroup/GPU — documented
under RD-8.)

New adversarial coverage, all passing:

- `test_independent_eval_receipt.py` (12) — issue+verify, every binding
  field, forged signature, wrong role, expired/revoked evaluator,
  altered predictions after signing, replayed receipt onto another
  run, incompatible model identity, detached receipt refused for
  trusted import, unverifiable evidence refused at issue time,
  receipt-file lifecycle.
- `test_fabricated_run_rejection.py` (9) — the audit scenario end to
  end: self-authored predictions + metrics + matching commit →
  `LOCALLY_RECOMPUTED` pass / independent verification fail /
  controller import denied; plus receipt signed by a
  research-pipeline-role key denied; forged/altered/replayed/
  model-mismatched receipts denied; a verified **negative** outcome
  admitted; integrity-only fabrication denied.
- `test_evaluator_authority.py` (6) — trust-root loading, role-restricted
  verifier, revocation/expiry falling out of the verifier.
- `test_matrix_false_positive.py` (5) — METRICS-only run `INVALID` +
  all gates `NOT_EVALUABLE` + nonzero exit; scrubbed retention metrics
  → `NOT_EVALUABLE` (never zero-regression); declared-but-unexecuted
  arm → `INCOMPLETE` + `matrix_qualified: false`; full positive
  control qualified; local-only cell below required level.

Updated for the new contract: `test_qualified_outcomes.py`,
`test_activation_authority.py`, `test_v410_security_gate.py`.

### 2.2 Real dev-matrix evidence (`results/v410_dev`)

Every committed dev run verifies at `LOCALLY_RECOMPUTED` — the local
recomputation machinery is sound. What they cannot do is reach
`INDEPENDENTLY_VERIFIED` on their own:

```
analyze_v410_matrix.py --matrix-id v410-dev \
    --gates configs/capability_acquisition/confirmation.yaml \
    (required_evidence_level: independently_verified)
→ INVALID cells ×8, matrix INVALID, exit 1
```

At development level (`--evidence-level local`) the rescore-derived
gates reproduce the recorded dev findings from committed predictions:

| arm | G5 delta | G6 destruction | security regressions | fit |
|---|---|---|---|---|
| fixed | +0.3125 PASS | 0.156 FAIL | 4 FAIL | 0.975 PASS |
| kl | +0.2949 PASS | 0.219 FAIL | 5 FAIL | 0.963 PASS |
| rehearsal | +0.2363 PASS | 0.625 FAIL | 4 FAIL | 0.85 FAIL |
| shuffled | −0.078 FAIL | 0.844 FAIL | 0 PASS | 0.0 FAIL |

The shuffled-label control failing G5 and fit is the negative control
working as designed.

### 2.3 Evaluator pipeline, end to end

`scripts/independent_evaluate.py` was run against
`results/v410_dev/runs/v410-dev-fixed-s0` with a provisioned
`evaluation`-role key, the campaign4a corpus and the governed security
sidecar: receipt issued with all ten binding fields populated;
`verify_outcome_receipt` under the trust root returned
`INDEPENDENTLY_VERIFIED`. The test key and the receipt were removed
afterwards — no permanent authority was minted, and no run in
`results/` carries a receipt.

### 2.4 Acceptance-gate scoreboard

| Gate (from the upgrade plan) | Result |
|---|---|
| Fabricated complete run rejected as independent evidence | PASS — `test_fabricated_run_rejection` |
| Forged evaluation signature rejected | PASS — receipt + outcome tests |
| Missing prediction files → qualification rejected | PASS — closure + `test_missing_prediction_records_are_rejected` |
| Undefined statistical interval cannot pass | PASS — three-state gates; `NOT_EVALUABLE` |
| Missing security evidence → qualification rejected | PASS — fail-closed security block |
| Exposed old holdout not used for fresh final confirmation | PASS — reclassified; see `HOLDOUT_RECLASSIFICATION.md` |
| Independent evaluator: exact evidence and role verification | PASS — receipt binding + role check |
| Signed release: full source inventory PASS | **NOT RUN** — release packaging is Phase 4 and produces the deliverable ZIP; nothing here fakes it |
| Regression suite | PASS — counts above |

### 2.5 What is NOT closed

- **Evaluator custody is development-held.** The key lives on the same
  host as the research agent here — the mechanism is closed, the
  operational separation is not. A real confirmation run must hold the
  `evaluation` key under a different OS identity (RD-1,
  `SEALED_EVALUATOR_CUSTODY.md`).
- **No Campaign 4A qualification evidence exists yet.** Every
  committed run tops out at `LOCALLY_RECOMPUTED`; the matrix cannot
  qualify under `independently_verified` until an evaluator signs.
  That is the intended fail-closed posture, not a bug.
- **Release ZIP not built.** Phase 4 (freeze → test → reports →
  inventory → sign → package → fresh-extract verify) is a packaging
  operation on the frozen tree, outstanding here.
