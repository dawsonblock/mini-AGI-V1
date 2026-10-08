# v16.2.1 — Evidence-Verified Research Baseline (integrity repair)

Integrity repair release — no scientific claims changed; Campaign 2's
qualified-negative result stands unchanged under its original V164
schema. The v16.3 designation remains reserved for a scientific
promotion condition Campaign 2 did not satisfy.

Release integrity:

- `scripts/verify_release.py` no longer trusts the in-package public
  key: the bundled key must carry the pinned v16.x release fingerprint
  (`REQUIRED_TRUSTED_KEY_FINGERPRINT`), with `--trusted-key` and
  `MINIAGI_RELEASE_KEY_FP` overrides for independently held keys and
  historical releases. Replacing manifest+key+signature inside an
  archive can no longer self-verify.
- Tag-gated clean-extraction release check in CI
  (`.github/workflows/rc11-integrity.yml`); per-commit pushes do not
  re-sign (manifest is resealed at release points).
- Version metadata normalized to 16.2.1 across VERSION, pyproject,
  package, SBOM, and release validation docs.

Evidence authenticity (v165 plan path):

- New `EvidenceReceiptV3` binds canonical prediction artifacts inside
  the signed payload: per-task prediction JSONL (input digest + output
  + status), `input_manifest_digest`, `predictions_digest`, and
  `evaluation_bundle_digest` — worker-reported metrics are now
  checkable claims, not facts.
- The qualifier independently re-scores every cell by joining
  canonical predictions to the committed corpus partitions; mismatched
  reported metrics, off-corpus prediction ids, or tampered artifacts
  produce `INVALID_EVIDENCE`, never `QUALIFIED`.

Independent authority:

- New role-separated authority plane (`minagi.v161.authority`):
  `plan`, `execution_witness`, `evaluation`, `qualification`,
  `promotion` — distinct Ed25519 identities in an out-of-band
  `trust_root.json`; same-key collapse and revoked/unregistered
  identities are rejected. The execution signer can no longer
  authorize its own qualification.
- Append-only `AUTHORITY_LEDGER.jsonl` chain-of-digests records
  preregistration -> evaluation bundle -> qualification record ->
  promotion decision, each signed by its owning role.
- `scripts/authority_bootstrap.py` provisions the trust root;
  `scripts/promote.py` issues the signed promotion decision bound to
  exact adapter digests (never mutable names).

Atomic, fail-closed execution:

- v165 seeds execute in staging directories and publish via atomic
  rename with a `COMMIT_MANIFEST.json` enumerating every artifact;
  torn or grafted evidence is quarantined and re-executed, never
  merged.
- Explicit decision vocabulary: `INCOMPLETE` / `INVALID_EVIDENCE` /
  `REFUSED` / `QUALIFIED` (runner emits provisional decisions;
  authoritative qualification is the independently signed
  `QUALIFICATION_RECORD.json`; `PROMOTED` requires the promotion
  authority).
- 34 new adversarial regression tests covering prediction/receipt/
  input-manifest/bundle mutation, authority substitution, key
  revocation, ledger tampering, torn atomic commits, and numeric
  injection — 393/393 tests pass.

Training correctness (REPAIR-019..024):

- `train_lora` accepts structured `{"prompt","response"}` examples;
  with `response_only_loss` (default on for the v165 protocol) the
  chat-template prompt boundary is masked to -100 — supervised loss
  applies only to response + EOS tokens. Legacy `texts=` mode is
  retained for historical replay semantics.
- Padding never contributes to loss (padded positions carry
  attention_mask=0 and labels=-100); examples whose responses tokenize
  to nothing or are fully truncated are rejected, never dropped.
- `LoraTrainSpec` binds `optimizer`, `gradient_accumulation_steps`,
  and `response_only_loss` — all now signed inside the experiment
  protocol, not left in unsigned YAML.
- The v2 training receipt records the per-microbatch loss curve,
  canonical data-order digest, resolved optimizer, library versions,
  trainable/total parameter counts, and wall-clock cost.

PEFT serving path (REPAIR-025..029, Route A):

- `minagi.v161.peft_serving`: `adapter_closure` verifies the saved
  artifact (peft type, declared targets, weight files),
  `assert_adapter_matches_protocol` rejects rank/alpha/dropout/target
  divergence from the signed protocol, `runtime_manifest` binds the
  exact model/tokenizer/adapter/protocol/qualification digests, and
  `logits_digest` supports save/reload output-parity probes (CPU
  bit-exact in tests).
- `scripts/promote.py` emits per-seed runtime manifests inside the
  signed promotion decision — promotion binds artifact digests,
  never mutable names. Native qw3 parity remains Route B and is a
  separate, unclaimed gate.

Experimental validity (REPAIR-030..036, v166 plan schema):

- `DatasetPartitionSetV2` adds an evaluator-sealed `final_holdout`
  partition — disjoint by sample id AND task family from every
  campaign partition. V1 partition digests are byte-stable (the V2
  wrapper is a new schema, not a mutation); Campaign 2 evidence is
  unaffected.
- `ColabCampaignPlanV166` extends V165 with `final_holdout_digest`
  (bound by digest only — the runner never reads the holdout file),
  `cluster_unit` (uncertainty resampling unit), a preregistered
  `max_false_activation_rate` cap, and `n_comparisons_preregistered`
  (the signed multiplicity budget).
- `cluster_bootstrap_ci` resamples task families, not rows —
  correlated same-template examples are no longer counted as
  independent evidence; `false_activation_rate` counts rows where the
  adapted arm flips an L1-correct answer to a wrong one.
- `executor_score`/`score_row` give task rows a `verify` spec
  (`python_assert` checks run in an isolated, timed subprocess) — the
  same dispatch is used by the runner and the independent qualifier,
  so executor-verified scoring is declaration-bound and reproducible.
- `scripts/seal_final_holdout.py` (authority side) seals a holdout
  JSONL into its bound manifest digest and rejects id/family leakage;
  `qualify_campaign1.py --holdout` re-derives the digest from the
  evaluator-supplied file — a substituted holdout fails closed.
- Campaign 3 scaffold split into preregistered singles: 3A (controlled
  learning rate), 3B (fresh families + separately sealed holdout),
  3C (serving-path confirmation). `configs/campaign3a.yaml` is the
  v166 single-lever config; 3B/3C reuse it frozen.
- Tests: +21 cases — V2 holdout binding/leakage, clustered-bootstrap
  determinism and width-vs-rows, false-activation semantics, executor
  dispatch/substitution/timeout, V166 plan validation, and a full
  V166 qualifier end-to-end (clean matrix QUALIFIES; missing or
  substituted holdout and false-activation breach fail closed).

Native Route B — attention-projection LoRA (REPAIR-025..029 cont.):

- `qw3-native-lora-bundle-v2` schema: per-layer
  `self_attn.{q,k,v,o}_proj` entries with explicit `layer` index,
  validated at load against declared model geometry
  (`NativeLoraAttentionDims`: n_layers/hidden/q_rows/kv_rows/o_in —
  `o_in` is the post-attention mid width, which differs from fused
  q+gate `q_rows` on gated-attention models). Duplicate (kind,layer)
  entries, unknown targets, out-of-range layers, and dims mismatches
  are rejected at load; attention entries under a v1 schema are
  refused.
- Device path: `NativeLoraSet::apply_projection` applies
  `out += scale*B*(A*x)` per (kind,layer); wired into the single-token
  decode path (q/k/v deltas on the post-norm input after the qkv
  fanout, o delta accumulated into the residual). Batched prefill,
  batched decode, and MTP verify REFUSE v2 bundles rather than
  silently serving unadapted KV; attention entries on recurrent (GDN)
  layers refuse at execution.
- `NativeQW3Adapter2Compiler` (Python): compiles PEFT safetensors
  `layers.N.self_attn.*_proj` pairs into v2 bundles — per-kind dims
  checks, scale = lora_alpha/rank, byte-hashed F32 payloads;
  `native_adapter2_supports_target` plus a preregistered
  `native_adapter_tier` field on the V166 plan gate the runner's
  servability check (tier 1 keeps the LM-head-only v1 surface).
- Parity: emitted F32 payloads reproduce `scale*B@(A*x)` bit-for-bit
  against the safetensors source (test asserts rtol=atol=0).

Adaptive plasticity (REPAIR-037..041, governed scaffold):

- `minagi.v161.plasticity`: deterministic failure diagnosis
  (harness-defect > missing-information > repeatable-procedure >
  persistent-deficiency) mapped onto the cheapest-first mechanism
  ladder retrieval → skill → weights → harness. `PlasticityProposal`
  requires recorded attempts at every cheaper rung — a weight-
  adaptation proposal without retrieval+skill evidence is malformed.
  Proposals are digest-bound inputs to preregistration, never
  authorizations.
- `DynamicLoraPolicyV1` + `RankAllocator`: rank is allocated, never
  unbounded — per-task and total budgets clamp every allocation,
  growth is step-limited, and `activation_protection` refuses growth
  while the false-activation gate is breached.

Governed recursive improvement (REPAIR-042..046, scaffold):

- `minagi.v161.generations`: `GenerationRecord`/`GenerationChain` —
  G(n+1) may be preregistered only after the promotion authority signs
  the parent's decision; parent linkage is digest-pinned, evaluation
  corpora must be fresh across generations, and promotion may only
  attach to the chain tip. This is the governance scaffold for the
  G0→G3 sequence — it does not execute campaigns.

Not in this release: response-masked GPU campaign execution, Campaign
3A/3B/3C execution (configs preregistered, holdouts unsealed — the
evaluator authority seals them at campaign time), end-to-end native
attention-LoRA serving parity on a real model (the v2 path is
load-verified, CPU-oracle-verified, and decode-path-wired, but no
real-weight forward pass has run — parity claims await the Route B
qualification campaign), dynamic-LoRA training integration (policy +
allocator only), executed improvement generations.

# v16.2.0-post — Campaign 2 Scale/Generalization Result (2026-10-07)

- Campaign 2 (`campaign2-v164`) completed the full 10-seed × 7-arm
  matrix on Colab under v164 resume semantics (persistent
  execution-witness key, per-seed environment atomicity across ~9 VM
  lifetimes). Runner decision `BLOCK`; independent artifact-only
  qualifier `REFUSE` with `runner_decision_agreement: true` and zero
  integrity failures.
- Primary endpoint: mean delta_FT_neural (L6-L5) = +0.01125 over 10
  seeds — a real positive effect (8/10 seeds positive; deterministic
  20k-resample bootstrap 95% CI [+0.00531, +0.01719] excludes zero) that
  nonetheless fails the preregistered >= 0.02 minimum.
- The Campaign-1b +0.20 magnitude did not transfer to the 13x larger,
  family-disjoint 320-row hidden set. The v16.2 claim remains scoped to
  the Campaign-1b corpus; Campaign 2 shows it does not generalize at
  this scale under this configuration.
- Security guardrail breached: L6 pass rate 0.625 vs L1 0.75
  (-0.125, exceeding the -0.10 bound). NC averaged 0.94, isolating the
  regression as a LoRA adaptation side effect rather than an artifact
  of touching weights.
- Retention (including delayed probes on reloaded adapters),
  negative-control isolation (NC = 0.0 on all seeds), and hidden/train
  family disjointness all held.
- Per the preregistered promotion ladder, v16.3 required this campaign
  to qualify; v16.2 remains the current validated designation.
- Evidence and gate-by-gate record: `results/campaign-2/` on branch
  `results/v16.1-colab-campaign-1` (`CAMPAIGN2_REPORT.md`,
  `QUALIFICATION_RECORD.json`); release attestation re-issued covering
  the evidence closure.

# v16.2.0 — Empirically Validated Platform (2026-10-06)

Campaign 1b (`campaign1b-v163`) QUALIFIED and reproduced on two
independent Colab T4 runtimes, satisfying the v16.2 promotion gate:
neural adaptation produces repeatable incremental forward transfer
beyond L5 grounded replay under preregistered hard constraints.

- Primary endpoint: delta_FT_neural (L6-L5) = +0.20 mean; 5/5 seeds
  positive (preregistered floor: >=4 of 5, min effect 0.02).
- Retention: L6 0.867 vs L1 0.833 under the calibrated containment
  scorer (agreement 0.933 on the frozen 30-case calibration set) — no
  measurable forgetting.
- Security: L6 pass-rate 0.80 >= 0.5 absolute floor; drop vs L1 = 0.033
  <= 0.1 relative guardrail.
- Negative control: 0.0 transfer (isolation holds); NC security cost
  documented (0.60 vs L1 0.833).
- Plan V163 signed before evaluation; immutable model revision
  (7ae5576...), model/tokenizer/generation-template digests bound.
- Fresh Runtime B cloned public main@d0d8cb2 only: identical plan
  digest, identical per-arm metrics, QUALIFIED; adapter bits differ per
  preregistered GPU nondeterminism policy.
- Campaign 1 (v162) remains REFUSE under the hardened qualifier —
  preserved as historical evidence.
- Governance fixes: qualifier recomputes dataset partitions; manifest
  enumeration switched to git ls-files (fresh-clone verification
  restored); results-branch evidence descends from the exact run
  commit via merge.

# v16.1.0-post — Colab Empirical Validation (2026-10-06)

- Executed the released build end-to-end on a clean Colab T4 runtime via
  `google-colab-cli`: release verify 1,381/1,381 PASS, doctor ready, test
  suite reproduced at 313/313, 36/36 import probes passed.
- Executed real-neural smoke campaigns: `configs/smoke.yaml` (tiny-gpt2) and
  `configs/qwen_smoke.yaml` (Qwen2.5-0.5B-Instruct) completed the full
  A0 → LoRA → serialize → destroy → reload → A1 path with signed receipts.
- Verified the qualified Qwen adapter genuinely changes inference and that
  adapter removal restores the frozen A0 outputs exactly.
- Adversarial boundary suite: 16/16 tamper/substitution attempts rejected.
- Added `scripts/validation/` harness: import probe, adversarial checks,
  interruption recovery, adapter-effect check, independent artifact-only
  qualification, release re-issue tool.
- Fixed packaging defect: `verify_release.py` now excludes `.git`,
  `.pytest_cache`, `*.egg-info` (editable install) artifacts.
- Fixed Colab dependency conflict: `colab_install.sh` upgrades `torchao>=0.16`
  when torchao is present (peft>=0.14 hard-requirement).
- Manifest re-issued and re-signed with the repository release key.
- Evidence is published on branch `results/v16.1-colab-campaign-1`.

# v16.1.0 Colab Converged

- v15.8 execution trunk retained with 304/304 baseline tests.
- RC14.7 executed evidence and independent reproduction integrated.
- RC12.5 grounded replay/live canary integrated.
- Dream-RSI added proposal-side.
- Selected v16 plasticity/statistics/isolated solver integrated without RuntimeClosure2.
- Added content-addressed evaluator registry and real dataset disjointness verification.
- Added signed Colab run receipts, RuntimeClosureV161, Colab persistent CAS, CUDA/HF/PEFT backend, resumable seed runs, export tooling, and Colab notebook.
- Unified tree validates at 313/313 Python tests.

# v15.8.0 — Campaign1 Preregistered Neural Experiment

- Adds immutable preregistration of the complete A0/A1 experiment identity before hidden-task consumption.
- Adds reproducible native QW3 launch specifications generated from ServedArtifactManifest.
- Rejects arm substitution, task-battery substitution/reordering, and gate mutation after preregistration.
- Emits a content-addressed campaign result bundle with every runtime evidence and fresh-task receipt digest.
- Adds no new promotion authority and makes no real-weight gain claim.

# Changelog

## 15.7.0 — NeuralEval1

- Added `QW3NeuralArm`, which requires measured runtime closure and re-verifies the exact governed QW3 state before every scored generation.
- Added content-addressed `NeuralRuntimeEvidence` binding epoch, manifest, aggregate artifact root, adapter roots, measured runtime state and output digest.
- Added `SealedNeuralAdapterEvaluator` for fresh-task paired A0/A1 experiments against two independently governed QW3 arms.
- Enforced frozen A0/A1 production identity: foundation, tokenizer, KVMem, retrieval policy, SkillIR policy and runtime binary must remain identical; only adapter state may differ.
- Rejected nonzero candidate adapter sets that are not backed by an exact native adapter bundle.
- Stored fresh-task receipts, arm evidence, paired results and frozen-baseline report in immutable CAS.
- Preserved the existing transfer/retention/security gate; evaluation has no promotion authority.
- Validation: 297 Python tests and 28 portable native tests passed; host QW3 compiled with CUDA disabled.
- No real-weight performance improvement is claimed because no real model/adapter experiment was executed in this environment.

## 15.6.0 — AgentPolicy1

- Added `GovernedAgentRuntime` that consumes the physically bound SkillIR and retrieval-policy artifacts.
- Re-hashes policy files and re-verifies the leased native QW3 StateEpoch before every task.
- Executes matching qualified SkillIR locally; unmatched tasks retrieve evidence-bearing episodic records and fall through to governed QW3.
- Added deterministic `RetrievalPolicy` / `RetrievalPolicyRuntime` with stable ranking and bounded `top_k`.
- Added content-addressed `AgentExecutionReceipt` binding epoch, served manifest, artifact root, policy roots, route, I/O digests and retrieved evidence.
- Added canonical policy deserialization with digest equality checks.
- Added sealed hidden-task A0/A1 regression proving the bound A1 skill policy changes behavior and passes the existing frozen-baseline gate.
- Added post-start policy tamper rejection and native epoch-mismatch rejection even for local SkillIR execution.
- Validation: 291 Python tests and 28 portable native tests passed; host QW3 compiled with CUDA disabled.

## 15.5.0 — ArtifactClosure1

- Physically bind tokenizer, adapter-set, retrieval-policy, and skill-policy identities in governed QW3 serving.
- Add strict zero-root/absent-artifact semantics.
- Add `PhysicalArtifactPaths`, canonical artifact materialization, and byte-derived served-manifest construction.
- Fix Python symlink checking before path resolution.
- Add explicit request-time tokenizer/retrieval/skill component identity checks.
- Add Python/C++ parity coverage for tokenizer identity and v15.5 closure digest.
- Validation: 285 Python tests and 28 portable native tests passed; host QW3 compiled with CUDA disabled.

# v15.4.0 — Measured Runtime Closure

- Added RuntimeClosure1 component-root flags for foundation/model, tokenizer, KVMem archive, retrieval policy, skill policy, and runtime binary.
- QW3 now hashes the actual loaded model path and its own running executable before governed serving starts.
- Added deterministic directory-tree hashing so governed model identity can cover HF/safetensors model directories as well as regular model files.
- QW3 hashes the attached KVMem `manifest.json` and requires the zero root when no archive is attached.
- Added C++ recomputation of the canonical served-artifact root; mismatched aggregate roots fail before serving.
- Added a measured runtime-closure digest over epoch, manifest, aggregate root and physically measured component identities.
- `/v1/runtime/state` now reports closure verification and measured component roots.
- Added Python `MeasuredRuntimeClosure` and `MeasuredGovernedServingContract` for deployment-time file measurement and fail-closed runtime verification.
- Added Python/C++ canonical-hash parity coverage, including deterministic directory-tree hashing.
- Validated 277 Python tests and 27 portable native C++ tests; host QW3 compiled with CUDA disabled.

---

# v15.3.0 — Governed Native LM-Head LoRA

- Added `NativeQW3AdapterCompiler` to convert one fully qualified, single-adapter LM-head LoRA artifact into a deterministic byte-addressed QW3-native FP32 bundle.
- Added a minimal read-only safetensors parser; every source file is re-hashed against the qualified artifact manifest before compilation.
- Rejects unsupported attention/MLP LoRA tensors instead of silently serving a partial representation of a qualified adapter set.
- Added independent `native_adapter_bundle_root` closure to `ServedArtifactManifest`, `/v1/runtime/state`, governed request headers, and QW3 fail-closed request validation.
- Added model SHA-256, foundation, qualified-candidate, adapter-set, and per-A/B tensor digest verification at native bundle load.
- Added QW3 native LM-head update `logits += scale * B * (A * normalized_hidden)` and executor scratch management.
- Added CUDA FP32 dense low-rank primitive used by the native adapter path. CUDA compilation/execution was not available in the validation environment and is not claimed as qualified.
- Disabled optimized batched LM-head shortcuts while a native adapter is active so serving falls back to the adapter-aware executor.
- Reset `native_adapter_bundle_root` when a newly qualified adapter set has not yet been compiled, preventing a stale native bundle from being carried across adapter-set changes.
- Added a portable native CPU oracle/bundle-integrity test including post-manifest tensor tampering rejection.
- Validated 272 Python tests and 26 portable native C++ tests; host QW3 compiled successfully with CUDA disabled.

---

# v15.2.0 — Governed Adapter Qualification

- Added `VerifiedAdapterDataset` so only evidence-bearing verified trajectories can become adapter training data.
- Added content-bound `AdapterTrainingPlan` and operational MLX-LM candidate runner.
- Added persistent adapter artifact manifests with exact per-file SHA-256, byte counts, safe relative paths, and symlink rejection.
- Added exact adapter payload publication into `ImmutableCAS`; post-build byte tampering blocks qualification.
- Added `AdapterSetBundle` and sealed adapter A0/A1 experiments bound to one concrete candidate root.
- Enforced positive paired transfer, required evaluation rings, retention floors, and zero security regressions through the existing frozen-baseline gate.
- Added foundation-model identity binding between qualified adapter candidates and served manifests.
- Added independent QW3 `adapter_set_root` process identity and `X-MiniAGI-Adapter-Set-Root` request enforcement.
- Preserved the alpha5 promotion/StateEpoch chain as the only route to SERVABLE state.
- Validated 268 Python tests and 25 portable native C++ tests.
- Explicitly does not claim native QW3 LoRA application or real-weight continual-learning improvement.

---

# v15.1.0 — Governed Learning-to-Serving Closure

- Added conservative, content-addressed `SkillIR` and canonical `SkillPolicyBundle`.
- Added single-reveal fresh-task paired A0/A1 evaluation using the existing signed fresh-task vault.
- Added `GovernedSkillLearningLoop`, which prepares policy artifacts but deliberately has no promotion authority.
- Bound passing learned policy bundles into `ServedArtifactManifest.skill_policy_root`.
- Fixed Integration1's incompatible bare-hex StateEpoch identity; governance IDs now use canonical `sha256:<hex>` values.
- Strengthened QW3 request enforcement with `X-MiniAGI-Artifact-Root`.
- Added end-to-end regression coverage proving stale served state is rejected after a qualified skill-policy change.
- Validated 253 Python tests and 25 portable native tests.

---

# 14.1.0-alpha5

- Added canonical `minagi.v15` production namespace with one exported runtime/database/StateEpoch path while keeping V141-V144 under the compatibility namespace.
- Added provenance-bound alpha5 evaluation cases. `fresh_hidden` now requires a trusted signed FreshTask consumption receipt; `falsification` and `negative_control` require preregistered candidate/case/category/task commitments.
- Added StateEpoch serving lifecycle over the alpha4 atomic activation transaction: PREPARED -> LOCALLY_COMMITTED -> EXTERNALLY_WITNESSED -> ATTESTED -> SERVABLE -> RETIRED.
- Added request-scoped StateEpoch leases for generation pinning across the complete request.
- Added explicit epistemic dependency graph with cycle rejection, terminal revocation, and transitive descendant quarantine.
- Added proposal-only trajectory-grounded skill induction with repeated-support requirements, contradiction blocking, and held-out transfer qualification.
- Added frozen-foundation A0/A1 experimental gate with exact foundation/production identity binding, paired bootstrap lower bounds, retention floor, and zero-security-regression requirement.
- Selectively integrated RC12.3 Apple Silicon performance helpers without importing its duplicate runtime tree.
- Validated 241 Python tests and 25 portable native C++ tests.

# 14.1.0-alpha4

- Added canonical `GovernedContinualRuntimeV144` evidence-closed runtime.
- Added `ExperienceRecordV144`, signed verification ingestion and `VerifiedEvidenceRecordV144`.
- Replaced receipt-count evidence strength with signed E2-E5 stage chains and `EvidenceStrengthProofV144`.
- E2 requires trusted EpisodeVerificationReceipt coverage for every evidence root; E3-E5 require valid immediate predecessor stages over the exact same evidence set.
- Added `GovernanceDBV144` learned-state tables so belief/skill mutation and RuntimeManifest activation can commit atomically.
- Added precomputed immutable `BeliefRevisionV144` and `SkillRevisionV144` exact mutation targets.
- Added snapshot closure requiring runtime belief/skill snapshots to name exact mutation digests.
- Added `full_belief_learning_cycle()` from raw experience through governed activation.
- Added 7 alpha4 evidence/atomicity regression tests.
- Validated 230 Python tests and 25 portable native C++ tests.

# 14.1.0-alpha3

- Added canonical `GovernedContinualRuntimeV143` content-bound authority path.
- Added separately signed evidence-strength proofs; proposal E0-E5 claims must match authority-derived strength and CAS-resident evidence.
- Mutation scopes are now derived from mechanism/permanence policy instead of candidate input.
- Added signed raw evaluation-case bundles and qualifier-side metric recomputation.
- Promotion now occurs after RuntimeManifest/artifact-root construction and signs exact candidate/build/evaluation/qualification/runtime content.
- Added one-use `(scope,target_digest)` mutation commitments and replay-resistant runtime activation.
- Added external signed audit anchor required by V143 audit verification.
- Added split FreshTask metadata/vault authority with signed consumption receipts.
- Canonical V143 runtime no longer accepts private authority signers; production mode rejects local clients.
- Added Unix-domain authority RPC client/server primitives for separate-process/UID deployments.
- Disabled legacy execution API qualification/promotion endpoints with HTTP 410.
- Changed installed canonical CLI to `minagi`; historical EGAI CLI is `egai-research-legacy`.
- Added 10 alpha3 authority/content-binding regression tests.
- Validated 223 Python tests and 25 portable native C++ tests.

# 14.1.0-alpha2

- Added canonical `GovernedContinualRuntimeV142` authority-closure path.
- Evaluation bundles are now independently Ed25519-signed and generation-fenced.
- Qualification decisions are mechanically derived from signed evaluation metrics; callers cannot submit `decision="PROMOTE"` to the alpha2 runtime.
- Qualification records are independently Ed25519-signed and generation-fenced.
- Promotion authorizations now contain explicit mutation scopes resolved from the immutable candidate manifest.
- Belief and skill persistent mutations can be configured to resolve the actual signed authorization from CAS and the authoritative governance DB.
- Runtime activation now requires typed CAS-resident production identity, belief snapshot, skill snapshot, adapter set, routing policy, and execution-runtime artifacts.
- Added signed audit checkpoints on top of the hash-chained ledger.
- Split FreshTask public metadata from the sealed task vault; added atomic conditional leasing and expired-lease reclamation.
- Expanded episode verification receipts to bind expected result, scoring-policy identity and pass threshold.
- Qualification policy now requires fresh-hidden validation and falsification evidence by default.
- Added canonical `learning_cycle()` end-to-end governed path.
- Added 9 alpha2 authority/security regression tests.
- Validated 213 Python tests and 25 portable native C++ tests.
- L7-L10 production activation remains disabled by default.

# 14.1.0-alpha1

- Added transactional authoritative `GovernanceDB` with SQLite/WAL, foreign keys, and atomic lifecycle mutations.
- Added immutable CAS with strict digest parsing, atomic publication, and corruption/traversal checks.
- Added canonical Candidate -> Build -> Evaluation -> Qualification -> Promotion -> Activation bindings.
- Added signed promotion authorization bound to candidate, qualification, and authority/policy generations.
- Added hash-chained governance audit events and tamper verification.
- Added immutable bi-temporal belief revisions and atomic current-head pointers.
- Added immutable skill revisions with lifecycle enforcement and retained negative evidence.
- Added `GovernedContinualRuntimeV141` as the preferred v14.1 authoritative orchestration API.
- Added nine new transactional/security regression tests.
- Validated 204 Python tests and 25 portable native C++ tests.
- Kept L7-L10 production activation disabled by default.

# 14.0.0

- Added two-axis permanence x evidence-strength governance.
- Added explicit L6 isolated-neural-memory tier and L10 architecture-proposal tier.
- Added durable one-shot hidden-task commit/reveal authority.
- Added falsification-plan constraints and proposal-only Dream/replay research primitive.
- Added model-time adaptive replay scheduling primitive.
- Added canonical candidate lifecycle state machine and v14 orchestrator.
- Removed boolean episode verification as a training-eligibility authority.
- Removed legacy unsigned continual adapter promotion path.
- Retired legacy admin mutation endpoints in favor of signed independent qualification/promotion.
- Retained mini-AGI 13 / RC11 native execution substrate and hardened execution registry.
- Preserved EGAI RC8.6 as a provenance donor and integrated compatible verification/skill concepts without creating a duplicate promotion plane.
- Validated 195 Python tests and 25 portable native C++ tests.

# 13.0.0

See the v13 source provenance and historical release notes under `docs/` and `references/`.
