# Mini-AGI — Full Upgrade, Hardening & Self-Improvement Plan

Engineering specification · v16.4.1 baseline → v17.0 target

*Filed as the governing roadmap document for the v16.4.2+ line.
Status of each work package is tracked in
[`REMAINING_DEFECTS_V1641.md`](REMAINING_DEFECTS_V1641.md) and the
CHANGELOG.*

## 1. Executive objective

The next objective is to transform Mini-AGI from a collection of
research, learning, verification, and serving components into one
integrated, independently governed continual-learning system.

The v16.4.1 release has substantial infrastructure. Its main weakness is
no longer a lack of components. It is that authorization, model loading,
artifact integrity, learning decisions, and scientific qualification are
not yet enforced as one complete workflow.

Three particularly important details were confirmed against the source:

- `TrustedRuntimeLauncher` verifies admission and stages model
  artifacts, but backend loading occurs before activation evidence is
  durably recorded.
- `RuntimeAdmissionController` defaults missing qualification backend
  coverage to a supported-backend list.
- `select_mechanism()` validates the *number* of prerequisite attempts
  rather than verifying their actual evidence, and can choose a
  negative-utility intervention.

The engineering priority is to eliminate these weaknesses before
allowing autonomous candidate promotion.

## 2. Target release roadmap

| Priority | Release | Name |
|---|---|---|
| P0 | v16.4.2 | Authority and Runtime Closure |
| P0 | v16.4.3 | Controller Correctness |
| P1 | v16.5.0 | Real-Model Scientific Qualification |
| P1 | v16.6.0 | Governed Continual Learning |
| P2 | v17.0.0 | Experimental Recursive Self-Improvement |

**Non-negotiable architecture rule:**

```
Research authority ≠ Verification authority ≠ Promotion authority
```

The research system must never be able to change its evaluation
requirements, authorize its own promotion, issue production runtime
credentials, or suppress evidence of failed experiments. Each release
has explicit implementation requirements, negative tests, and an
acceptance gate; failure at a gate prevents advancement.

## 3. v16.4.2 — Runtime Authority and Activation Atomicity

*Priority P0 · required before production-serving claims*

### 3.1 Replace the runtime trust model

The v16.4.1 design uses `ApprovedSnapshot` as the object representing
staged artifacts. The problem: an object created by `stage_snapshot()`
can be considered approved without independently proving that a
promotion authority authorized its use, and the loader relies on a
caller-provided object's `path()` method.

Replace it with three distinct concepts:

| Object | Responsibility | Authority |
|---|---|---|
| `MeasuredSnapshot` | Prove artifact content and file structure | No deployment authority |
| `AdmissionGrant` | Authorize specific measured artifacts under a specific policy | Issued by trusted admission service |
| `ActiveRuntime` | Represent an actually admitted and activated model instance | Created by supervised serving control |

`AdmissionGrantV1` binds:

```
schema_version · grant_id · promotion_decision_digest
qualification_digest · runtime_manifest_digest · artifact_root_digest
backend_id · backend_binary_digest · policy_epoch · revocation_epoch
issued_at · expires_at · nonce · audience_runtime_identity
issuer_key_id · signature
```

Signatures use the repository's existing canonicalization and Ed25519
interfaces — no incompatible signing formats.

**Critical design correction:** a signed Python object or an
`isinstance()` check is not an effective security boundary against
malicious code running in the same interpreter. The admission authority
and serving supervisor must operate across an OS-enforced privilege
boundary — a dedicated service over authenticated Unix sockets
(Linux/macOS) or restricted named pipes (Windows). The untrusted
research process must not have access to the runtime's signing
credentials or writable deployment directories.

### 3.2 One protected model-loading path

A supervised runtime service owns every production model load:

```
Research plane ──proposes candidate──▶ Independent admission service
  requests activation                  (promotion · qualification ·
        │                              backend · revocation · policy)
        ▼                                    │
   Protected artifact store                  ▼
   (immutable, measured snapshots)    Serving supervisor
        │                              (verifies grant, prepares
        ▼                               model, controls traffic)
   Authorized inference runtime
   (HF/PEFT or separately qualified native backend)
```

| File | Required change |
|---|---|
| `v161/immutable_snapshot.py` | Produce measured snapshots without implying authorization |
| `v161/runtime_admission.py` | Verify full authorization and issue short-lived admission grants |
| `v161/trusted_launcher.py` | Become a client of the protected serving supervisor |
| `v161/peft_serving.py` | Accept only supervisor-controlled artifact handles |
| `platforms/cuda/hf_runtime.py` | Serving-mode loading behind the supervisor |
| `scripts/trusted_launch.py` | A supervised launch command, not an independent authorization boundary |

Research loaders and serving loaders are separated by process isolation
and permissions, not a `purpose="research"` argument.

### 3.3 Explicit backend qualification

Replace `covered = qual.get("runtime_backends") or
list(QUALIFIED_BACKENDS)` with validation that requires a nonempty,
explicitly signed backend-coverage declaration describing the specific
backend implementation, artifact format, model, tokenizer, quantization
mode, and compatibility evidence. Qualifying `hf-peft` does not
automatically qualify `qwen-native-cuda` or `qwen-native-metal`.

**Acceptance:** a fully signed qualification that omits backend coverage
is refused; a valid HF/PEFT qualification does not authorize native
serving without independently established native compatibility.

### 3.4 Authenticated revocation evidence

The `RevocationList` (digests + generation timestamp) rejects stale
lists but not sufficiently future-dated ones. `RevocationSnapshotV2`:

```
schema · epoch · issued_at · valid_until · revoked_decision_digests
revoked_key_ids · previous_snapshot_digest · issuer_key_id · signature
```

Required checks: dedicated revocation-authority signing role; signature
and full trust chain verify; epoch cannot move backward; not older than
the freshness limit; future timestamps beyond clock-skew allowance
rejected; missing revocation evidence fails closed for production
activation; updates stored atomically and durably. Admission uses the
newest valid authorized snapshot, not an arbitrarily selected older one.

### 3.5 Transactional model activation

Explicit runtime state machine:

```
REQUESTED   candidate activation requested
AUTHORIZED  admission grant verified
STAGED      artifact snapshot verified
PREPARED    model loaded but unavailable to external requests
READY       health and inference probes passed
COMMITTED   durable activation intent recorded
ACTIVE      traffic routed to the new instance
   — any failed transition → ABORTED or QUARANTINED → unload + cleanup
```

The supervisor persists an activation-intent record before changing
traffic routing, atomically switches the active-version pointer, then
persists a signed activation-completion record describing what actually
happened. An intent record is never labelled proof of completed
activation; a crash between routing and receipt persistence is
reconciled by `recover_from_journal()` before accepting traffic.

Required operations: `prepare(candidate)` · `health_check(candidate)` ·
`commit_activation(candidate, expected_previous)` · `abort(candidate)` ·
`rollback(previous_qualified)` · `recover_from_journal()`.

A complete crash-consistent transaction across filesystem, model
processes, and network routing is not automatically possible; the
design uses idempotent transitions and deterministic recovery.

### 3.6 Mandatory rollback

The last independently qualified version remains a protected deployment
target. Rollback covers: failed load or inference probe; receipt/ledger
persistence failure; a revoked active candidate; a broken serving
instance; post-activation regression; interruption during traffic
switching. The supervisor retains the previous healthy model until the
new instance commits; rollback requires valid authorization policy and
produces its own signed event.

### v16.4.2 qualification tests

| Test | Required result |
|---|---|
| Forge a MeasuredSnapshot and request serving | Refused |
| Supply an arbitrary object with `path()` | Refused |
| Omit backend qualification | Refused |
| Qualify HF/PEFT but request native CUDA | Refused |
| Substitute one byte of the adapter | Refused |
| Add a symbolic link to an artifact | Refused |
| Supply unsigned revocations | Refused |
| Supply a revocation list from 2100 | Refused |
| Replay an older revocation epoch | Refused |
| Fail receipt persistence after preparation | No externally active candidate |
| Crash between commit and traffic switch | Recover deterministically |
| Revoke the currently active version | Quarantine or replace per policy |
| Fail the new model health check | Previous healthy version remains active |

**Release gate:** all supported production-serving paths enforce
independent admission, and fault injection demonstrates that
unauthorized or incompletely committed candidates cannot receive
inference traffic.

## 4. v16.4.3 — Learning Controller and Evidence Correctness

*Priority P0 · required before autonomous learning decisions*

Repairs the logic deciding whether Mini-AGI should retrieve information,
create a skill, train an adapter, or make no change.

### 4.1 `NO_CHANGE` as an explicit decision

Extend the action set to `NO_CHANGE · DIAGNOSTIC_EXPERIMENT · RETRIEVAL
· PROCEDURAL_SKILL · WEIGHT_ADAPTATION · HARNESS_REPAIR`. Decision rule:

```
a* = argmax_a U(a)   if max_a U(a) > τ and all gates pass
     NO_CHANGE       otherwise
```

τ is a minimum required utility frozen in the experiment policy.
`DIAGNOSTIC_EXPERIMENT` applies when uncertainty is high enough that
further investigation is valuable and the diagnostic cost is authorized.
Negative expected utility must prevent ordinary automatic promotion.

### 4.2 Verified prerequisite attempts

`VerifiedAttemptRecordV2` extends `AttemptReceipt` with identity and
verification fields (`failure_evidence_digest`, `task_family_digest`,
`baseline_digest`, `execution_receipt_digest`, `qualification_digest`,
`completed_at`, `evaluator_key_id`, `evaluator_signature`). Before
selecting LoRA the controller must verify that prerequisite attempts
exist in the evidence store, match the problem class, actually executed
the claimed mechanism, were independently evaluated, failed to meet the
objective or justify escalation, and are not invalidated or superseded.
An arbitrary digest never satisfies these conditions.

### 4.3 Cost-aware mechanism selection

`U(a) = E[ΔQ_a] − λ_C·C_a − λ_R·R_a − λ_L·L_a` (expected improvement,
cost, regression risk, serving latency). Safety, security, and
authorization limits are hard constraints outside the utility; the
weights are preregistered — the agent may propose new weights but may
not change the active evaluation policy.

### 4.4 Governed experiment budget controller

`ExperimentBudgetV1` with separately enforced limits: GPU/CPU compute,
wall-clock, memory, disk, model/API cost, attempts, evaluation access —
enforced by the execution service, not merely checked by the proposer.
A candidate that exhausts resources becomes `BUDGET_EXHAUSTED` with a
failure receipt and no automatic retry.

### 4.5 Complete evidence chain

`FailureEvidence → MechanismDecision → CandidateManifest →
ExecutionReceipt → EvaluationBundle → QualificationRecord →
PromotionDecision`, every record referencing the exact digests of its
dependencies, resolved to actual artifacts.

**Acceptance:** negative-utility ⇒ `NO_CHANGE`; unsupported prerequisite
digests cannot authorize LoRA escalation; budget exhaustion terminates
execution; a failed candidate cannot construct its own qualification;
changing evidence after evaluation invalidates dependent decisions;
replaying an old evaluation for a new candidate is rejected; decisions
are deterministic under frozen evidence and policy.

## 5. v16.5.0 — Real-Model Learning and Scientific Qualification

*Priority P1 · core scientific validation*

Preserves the Campaign 3A baseline (Qwen2.5-0.5B-Instruct, LoRA
r=16/α=32, lr 5e-5, 100 updates, 10 seeds, L1–L6 + negative control,
+0.02 transfer floor — frozen). Completes the three-stage campaign:

- **3A** — reconstruct/verify the original execution artifacts and
  independently recompute metrics; never relabel a modified run as the
  original.
- **3B** — fresh-family transfer experiment: generate and seal the
  missing `campaign3b_tasks.jsonl` corpus, prohibit tuning against it,
  test the frozen configuration on independent task families.
- **3C** — serving-runtime confirmation: load the exact qualified
  artifact through the protected runtime; record prediction parity,
  artifact identity, load reliability, latency, peak VRAM,
  backend-specific differences. Native serving requires its own
  explicit qualification (`require_native_servable_adapter`).

Colab is an *untrusted GPU execution worker*: frozen configuration +
budgets + model identity stay local; the artifact evidence package
returns to an independent qualification host that recomputes metrics and
applies frozen gates. Promotion keys and hidden confirmation data never
enter the Colab runtime. Hardware/dtype compatibility must be checked
(a T4 does not satisfy a frozen bfloat16 requirement; changing dtype
requires a separately identified experiment).

Evaluation methodology strengthens: more independently constructed task
families (not near-duplicate rows), forward/backward transfer, delayed
retention, false activation, security regression, negative controls,
cost-adjusted utility, seed variation, backend parity, cluster-aware
uncertainty.

A scientifically honest REFUSE is a valid, valuable release outcome.

## 6. v16.6.0 — Integrated Continual-Learning System

*Priority P1 · bounded autonomy*

An `ImprovementOrchestrator` maps observed failures to governed first
interventions (missing information → retrieval; repeated procedural
failure → skill construction; persistent generalizable deficiency →
bounded weight adaptation; broken tool interaction → harness repair;
uncertain cause → diagnostic; insufficient expected improvement → no
change). The orchestrator cannot approve its own proposals.

Three distinct memories — episodic (exact execution history), semantic
(reusable facts with provenance and correction history), procedural
(verified replayable skills) — each write carrying source attribution,
version, and evidence quality; model-generated hypotheses never promote
into trusted labels by repetition alone.

The intervention ladder (diagnose → retrieval/memory → procedural skill
→ justified weight adaptation → independent qualification) permits
justified exceptions. Dynamic-rank LoRA enters only as a *competing
candidate* against fixed-rank and replay baselines under a frozen
objective — it is not assumed better.

**Acceptance:** a full observe → propose → train → evaluate → promote →
activate cycle, plus demonstrated safe abstention, budget exhaustion,
failed-candidate rejection, and rollback.

## 7. v17.0.0 — Bounded Recursive Self-Improvement

*Priority P2 · scientific research milestone*

Tests whether successive generations become demonstrably better at
learning or solving unfamiliar tasks under matched conditions. The
system may propose changes to retrieval ranking, memory consolidation,
skill generation, mechanism selection, training configuration, adapter
rank policy, data-selection policy, and bounded harness components — it
cannot modify the qualification authority, test secrecy, signing keys,
promotion policy, or deployment boundary (those require a separate
operator-controlled release).

Generations G0–G3 run against a **parallel nonrecursive arm with
equivalent compute and tool access**; otherwise extra compute or test
exposure masquerades as recursive gains. Success requires repeated
improvement on fresh tasks, cumulative advantage over matched controls,
preserved prior capability, and independent authorization for every
promoted generation — with failures, plateaus, and regressions reported.
Even success establishes bounded improvement, not general autonomous
intelligence.

## 8. Proposed structure

Extend the existing implementations; no competing second architecture.

```
src-python/minagi/
  v161/            # repaired: runtime_admission, trusted_launcher,
                   # immutable_snapshot, peft_serving, strict_schema
  security/        # NEW: signed_revocations, admission_grants,
                   #      trusted_authority_client, artifact_policy
  runtime/         # NEW: supervisor, activation_state,
                   #      durable_journal, runtime_recovery,
                   #      rollback_controller
  continual/       # NEW (v16.4.3+): attempt_verifier,
                   #      mechanism_policy, budget_enforcer,
                   #      improvement_orchestrator
  experiments/     # NEW (v16.5+): independent_evaluation,
                   #      campaign_evidence, runtime_confirmation
  recursive/       # NEW (v17): generation_controller, matched_controls
tests-python/
  security/  runtime/  learning/  scientific/  adversarial/
```

All new APIs carry explicit ownership, error semantics, serialization
rules, and compatibility tests. Cryptographic verification and promotion
policy live behind one controlled interface — never duplicated.

## 9. Test and validation strategy

| Level | Scope | Expectation |
|---|---|---|
| Unit | hashing, schemas, utility, state transitions | all pass |
| Property-based | allocation invariants, replay rules, serialization | no violations |
| Integration | full signed authority chain, staged activation | all pass |
| Adversarial | forged grants, symlinks, replays, privilege misuse | all refused |
| Fault injection | crashes, write failures, load failures, concurrency | deterministic safe recovery |
| Native CPU | C++ build + registered tests | pass on supported platforms |
| GPU | real load, LoRA updates, inference, parity | pass on qualified hardware |
| Scientific | fresh-family transfer, retention, baselines, controls | preregistered gates |

CI distinguishes passed / skipped / failed / never-executed — missing
platform coverage is not qualification.

## 10. Release engineering and evidence

Release ZIPs include source, tests, configs, docs (ARCHITECTURE,
THREAT_MODEL, SECURITY_REPAIR_REPORT, TEST_RESULTS, REMAINING_DEFECTS,
RELEASE_NOTES), `RELEASE_MANIFEST.json`, `RELEASE_ATTESTATION.json`,
dependency locks, README. Source-integrity verification is distinguished
from cryptographic trust-root provenance — a bundled signing key alone
cannot establish independent release authority. Private signing keys and
secret holdouts are never shipped.

## 11. Engineering work packages

| WP | Scope |
|---|---|
| WP1 — Runtime authority | MeasuredSnapshot/AdmissionGrant/ActiveRuntime contracts; isolated admission service + protected loader; explicit backend qualification |
| WP2 — Runtime lifecycle | prepared/non-serving state; durable activation journal + traffic commit; failed-load cleanup, crash recovery, rollback |
| WP3 — Revocation & protocol | authenticated monotonic revocation snapshots; future-timestamp + stale-epoch rejection; strict signed schema + artifact bindings |
| WP4 — Learning controller | NO_CHANGE + minimum-utility thresholds; verified attempt prerequisites; compute/attempt/evaluation budgets |
| WP5 — Scientific campaigns | reconstruct 3A evidence; seal 3B corpus; run 3C on the real backend; independent GPU + statistical qualification |
| WP6 — Continual learning | failure diagnosis ↔ mechanism selection integration; candidate ↔ independent evaluation; dynamic LoRA vs baselines |
| WP7 — Recursive experiment | governed generation controller; compute-matched nonrecursive control; G0–G3 qualification |
| WP8 — Release verification | adversarial + fault-injection + platform matrices; manifests, provenance, dependency locks; published evidence and residual risks |

## 12. Go/no-go gates

| Release | Must be demonstrated before approval |
|---|---|
| v16.4.2 | Independent admission, explicit backend coverage, signed revocation, safe activation lifecycle |
| v16.4.3 | Evidence-backed learning decisions, safe abstention, verified prerequisites, bounded execution |
| v16.5.0 | Reproducible real-model learning, independently evaluated |
| v16.6.0 | Complete observe–propose–train–evaluate–promote–activate loop |
| v17.0.0 | Repeated cumulative gains vs matched controls across generations |

No release inherits an unqualified capability merely because the code is
present.

## 13. Development priorities

Runtime authority/isolation/activation 40% · controller & evidence
correctness 25% · GPU learning & scientific validation 25% · recursive
improvement 10%. *Relative effort guidance, not a budget.*

## 14. Immediate deliverable

`mini-AGI-V1-v16.4.2-Authority-and-Activation-Closure.zip` with
implementations and regression tests for: independent admission
capabilities (measured files are not deployable by construction);
protected model loading (no unverified paths or caller-created
snapshots); transactional activation and rollback (a failed launch
cannot leave an unauthorized model serving); strict backend and
revocation authorization (missing qualification, invalid revocations,
incompatible backends fail closed). Then v16.4.3 controller correctness,
then real GPU qualification.

v17.0 is a scientific milestone, not a version-number milestone: the
system succeeds when independently measured improvements survive
fresh-task testing, are authorized through an enforceable runtime
boundary, and accumulate across generations without compromising earlier
capabilities. Adding more code will not establish that result;
executing the complete evidence chain will.
