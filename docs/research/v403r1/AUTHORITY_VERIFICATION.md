# v4.0.3-r1 — Authority & verification model

How a piece of experiment evidence earns trust in this tree, and who is
allowed to grant it.

## The trust boundary

A model-training process must never be able to produce its own
authoritative evaluation. The separation is enforced by **role**, not
by filename, flag or convention:

| Role | May attest |
|---|---|
| `plan` | experiment plans |
| `execution_witness` | that a run executed as declared |
| `evaluation` | **independent recomputation of outcomes** — the only role that can lift a run past `LOCALLY_RECOMPUTED` |
| `qualification` | promotion decisions binding `run_id` + evidence digest |
| `promotion` / `runtime` / `revocation` / `admission` | downstream lifecycle |

Roles live in `AuthorityRegistry` (built from `TRUST_ROOT.json`);
`Ed25519` signatures are checked against the registry's verifier.
Revoked and expired keys are refused at signature-check time.

## The four evidence states

```
INVALID / FAILED_RUN / INCONSISTENT
        │
        ▼ commit + hash inventory internally consistent
INTEGRITY_VERIFIED ───────────── not trusted history
        │ strict rescore of every committed prediction
        │ reproduces the reported metrics
        ▼
LOCALLY_RECOMPUTED ───────────── not trusted history
        │ signed independent-evaluation-v1 receipt:
        │   schema + status fields present
        │   evaluator_authority_id authorized for 'evaluation'
        │   Ed25519 signature verifies
        │   every binding field recomputes EXACTLY against the run
        ▼
INDEPENDENTLY_VERIFIED ───────── trusted history (incl. negatives)
        │ signed qualification decision by a 'qualification' key
        │ binding run_id + this run's evidence digest
        ▼
QUALIFIED ────────────────────── trusted history + promotion-eligible
```

A locally recomputed run's numbers *derive* — the rescored delta,
retention regression and cost are honest measurements — but
`admissible` stays false: local reproduction is not independent
evaluation, however accurate. That is the whole defect: the old code
made "the same process could recompute it" synonymous with "an
evaluator verified it".

## The `independent-evaluation-v1` receipt

`src-python/minagi/v161/independent_eval_receipt.py`. The evaluator
**recomputes the run itself** (`verify_run` must be `VERIFIED` or
issuance is refused) and signs a canonical body carrying ten exact
evidence identities:

```
experiment_id                the committed run this attestation covers
candidate_digest             adapter artifact bytes (the candidate)
base_model_digest            declared frozen model id + revision
dataset_digest               corpus rows the predictions scored against
scorer_digest                scoring-oracle identity
protocol_digest              evaluation protocol the run declared
predictions_digest           every committed prediction file's bytes
evaluation_metrics_digest    METRICS bytes + rescore/reported summary
retention_metrics_digest     recomputed destruction/acquisition accounting
security_metrics_digest      governed security-lane evidence, if bound
evaluator_authority_id       the signing identity
evaluation_status            "INDEPENDENTLY_VERIFIED"
signature                    Ed25519 over the canonical body
```

Verification (`verify_evaluation_receipt`) is four checks, not one:

1. **form** — schema, status and every binding field present;
2. **role** — `evaluator_authority_id` authorized for `evaluation` at
   check time (revoked, expired, wrong-role, unregistered → fail with
   an auditable reason);
3. **signature** — Ed25519 over the canonical body;
4. **binding** — the run reconciles *and* every binding field
   recomputes byte-exactly. Altered predictions, a different model, a
   different corpus, a receipt issued for another run — all surface as
   mismatches. A detached receipt that cannot be bound to evidence is
   refused for trusted import (`allow_detached` exists only for
   tooling introspection and reports `binding_checked: false`).

Issue-side custody: `scripts/independent_evaluate.py` is the
evaluator's tool. It loads a private key, recomputes the run, binds
the evidence (corpus + optional governed security sidecar) and writes
`EVALUATION_RECEIPT.json` — post-commit evidence, deliberately outside
the run's own inventory, exactly like `QUALIFICATION.json`.

## What "independent" can and cannot prove

The cryptography proves *authorization*: a receipt that verifies was
signed by a currently-authorized `evaluation` key, over exactly this
evidence. It cannot prove *custody*: if the private key sits on the
research agent's host, a sufficiently privileged process could sign
with it. That separation is operational — separate OS identity, host
or HSM — and is enforced by deployment, not by this code. On this
development host custody is NOT separated; the receipts here establish
the mechanism, not operational independence.

A model-training process still cannot produce a *valid* authoritative
receipt on a properly deployed system, because the `evaluation` key is
not on its host and the role check rejects every key it does hold.

## Replay, forgery and the negative-outcome rule

- **Replay** is covered by the binding: a receipt names run A's bytes;
  presenting it for run B fails on `experiment_id` and every digest.
- **Forgery** without a registered evaluation key fails at the role
  check or the signature check.
- **Tampering after signing** breaks the binding at verification.
- **Verified negative outcomes are admissible** — a candidate that
  honestly destroyed retention still earns a receipt and enters
  trusted history. Dropping failures would teach the controller only
  from successes; that selection bias is itself a defect.
