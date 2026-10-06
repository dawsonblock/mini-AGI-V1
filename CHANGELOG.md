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
