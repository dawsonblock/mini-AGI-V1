# Runtime integration report (Step 11)

Status: **preserved — the scientific work did not weaken the governed
runtime.** No isolation, admission, signing, revocation or authority
module was modified.

## What was integrated

| layer | change |
|---|---|
| prompt / tokenization | new `minagi/v161/prompt_contract.py`; `peft_trainer` and `run_campaign1` route through it |
| scoring | `generation_diagnostics`, strict/diagnostic separation, frozen output contract |
| replay | `OracleVerifiedReplayArm` (L5b), `GroundedReplayArm` (L5a) unchanged |
| statistics | `cluster_bootstrap_ci`, `family_delta_analysis`, `false_activation_rate` NOT_ESTIMABLE |
| safety | new `minagi/v161/safety_regression.py`, wired into runner and qualifier |
| confirmation | new `scripts/validation/run_sealed_confirmation.py` |
| runner/qualifier | additional gate reasons; none removed |

## What was not touched

`git diff` between the v16.4.10 release commit and the Campaign 3B head
over the runtime security surface is **empty**:

```
src-python/minagi/v161/authority.py
src-python/minagi/v161/runtime_admission.py
src-python/minagi/v161/trusted_launcher.py
src-python/minagi/v161/execution_sandbox.py
src-python/minagi/v161/runtime_closure3.py
src-python/minagi/v161/artifact_closure.py
src-python/minagi/v161/experiment_budget.py
```

The supervisor, signed admission grants, artifact verification, NsJail
isolation, activation/rollback, revocation handling and the independent
promotion boundary are exactly as the v16.4.10 envelope signed them.

The only security-relevant *new* code is the sealed confirmation
executor, which adds controls (research-custody denial, sealed-set
location rule, artifact verification against signed receipts) and
removes none.

## The runtime version is newer, not older

The Campaign 3A archive embedded the **16.4.3-hardened-runtime** source.
Campaign 3B is built on **v16.4.10-containment-and-authority-closure**.
No newer security code was overwritten with the older implementation;
the 3A source was used only as evidence about how the 3A measurement
was produced.

## Scientific qualification is eligibility, not activation

A scientifically qualified adapter is **eligible for deployment**. It is
not activated. Deployment still requires the full runtime chain:

| required step | mechanism |
|---|---|
| artifact verification | `artifact_closure` — the adapter bytes must match the digest the qualification bound |
| backend compatibility | `backend_manifest` / native adapter tier gate (tier 1 = LM head, tier 2 = attention projections) |
| signed promotion authorization | `scripts/promote.py` — refuses an unsigned record, a record not signed by the registered `qualification` authority, a record whose digest does not match, and any decision other than `QUALIFIED` |
| current revocation checks | `scripts/admit_runtime.py` — versioned revocation list with freshness enforcement (`--max-revocation-age-days`) |
| runtime resource budgets | `experiment_budget` / worker resource limits |
| required worker isolation | `isolation_profile` / `isolation_preflight` / `nsjail_executor` |
| real model health checks | runtime closure tests, including the G10 real-model qualification |
| correct activation and rollback | `activation_state`, `supervisor`, `recovery_manager` |

`promote.py` is explicit that only QUALIFIED evidence may be promoted and
that the record must bind an evaluation bundle; nothing in the 3B work
relaxes any of those checks.

## Authority boundaries, restated

- **Research** may train candidate adapters but cannot independently
  promote them. The runner's decision is a *provisional evidence
  summary*; its own summary text says so.
- **Verification** may evaluate candidates but cannot change historical
  source evidence. The qualifier recomputes metrics from persisted
  predictions and never rewrites them.
- **Promotion** requires independent, authenticated qualification. The
  qualification record is signed by the `qualification` role, which is a
  separate authority from `execution_witness` and from `plan`.
- **Workers** cannot reach protected signing keys or sealed confirmation
  labels. The runner *requires* a provisioned trust root and role keys
  (`authority_bootstrap.py`) rather than self-issuing them, and the
  sealed loader refuses a research-marked process.
- **Evaluation results** are reproducible from immutable predictions and
  exact scorer versions. Predictions are content-addressed
  (`input_sha256`), the scorer identity is digest-bound, and the
  qualifier recomputes rather than trusts.

## If a required condition is unavailable

The runtime returns NOT QUALIFIED rather than proceeding:

- missing trust root or role key → the runner hard-stops
  (`v165 requires a provisioned authority trust root …`);
- missing revocation list when freshness is required → the admission
  tool refuses with a config error;
- a non-`QUALIFIED` qualification record → `promote.py` fails;
- an unmeasurable mandatory safety or retention metric → the qualifier
  refuses.

## Verified on this host

| check | result |
|---|---|
| native CTest (macOS CPU, `QW3_ENABLE_CUDA=OFF`) | **28/28 passed, 0 failed** |
| full Python suite | **1183 passed, 18 skipped, 0 failed** |
| runtime security modules changed | **none** |
| qualifier gate reasons | 17 → 19 (added, none removed) |

## Limitation

The envelope is signed by the repository's release key, and the bundled
`RELEASE_PUBLIC_KEY.pem` hash matches the attestation's pinned
`public_key_sha256` (`a8cd5cb2…`), so the signature is verifiable
against the identity this repository already publishes. It is a
development-machine key held outside the tree, not an independently
witnessed third-party attestation, and it establishes provenance of the
envelope only — not the correctness of the scientific claims. See
`REMAINING_DEFECTS.md`.
