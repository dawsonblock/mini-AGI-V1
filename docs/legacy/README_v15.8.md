# mini-AGI v15.8.0 — Campaign1

This release makes the v15.7 neural A0/A1 harness operationally reproducible: the experiment plan is frozen in immutable CAS before any hidden task is consumed, and a completed run emits one evidence-closed result bundle. See `docs/NEURAL_CAMPAIGN_V15_8.md`.

**Nonclaim:** no real Qwen + trained adapter performance campaign was executed in this build environment.

# mini-AGI 15.7.0 — Governed Neural A0/A1 Evaluation

v15.7 adds an independently verified A0/A1 QW3 evaluation harness for real adapter experiments. Each arm must be a governed, measured runtime. The evaluator rejects any foundation/tokenizer/KVMem/retrieval/SkillIR/runtime-binary drift and permits only the authorized adapter state to differ. Every scored output is bound to fresh-task provenance plus content-addressed runtime-state evidence before the existing frozen-baseline transfer/retention/security gate can PASS it.

This release does **not** claim a real-weight improvement result. The build environment did not contain a real Qwen weight set plus trained qualified adapter. Its deterministic tests prove the harness and authority invariants, not neural capability. Validation for this exact source tree: **297/297 active Python tests** and **28/28 portable native C++ tests** passed; the host QW3 executable compiled with CUDA disabled. See `docs/NEURAL_EVAL_V15_7.md`.

---

# mini-AGI 15.6.0 — Governed Agent Policy Runtime

v15.6 turns the physically bound retrieval and SkillIR artifacts from v15.5 into executable agent policy. `GovernedAgentRuntime` re-hashes both policy artifacts and re-verifies the leased QW3 state before every task. Matching qualified SkillIR executes locally; unmatched work uses the exact bound retrieval policy to select evidence-bearing episodic records before falling through to the governed QW3 endpoint. Every execution produces a content-addressed receipt bound to the epoch, served manifest, artifact root, policy roots, route, input/output digests and retrieved evidence.

The new sealed regression proves an end-to-end behavior change on hidden tasks: A0 leaves the task unchanged, while the bound A1 SkillIR policy solves unseen reverse-text tasks under the same governed serving contract. This is a deterministic procedural-learning proof, not a claim of general intelligence or real-weight model improvement.

Validation for this exact source tree: **291/291 active Python tests** and **28/28 portable native C++ tests** passed; the host QW3 executable compiled with CUDA disabled. CUDA/GPU qualification, general learned tool use, real-weight unseen-task improvement, all-linear LoRA, long-horizon campaigns and external reproduction remain unclaimed. See `docs/GOVERNED_AGENT_RUNTIME_V15_6.md`.

---

# mini-AGI 15.5.0 — Physical Artifact Closure

v15.5 closes the remaining declared-root gap in governed serving. QW3 now independently measures the physical tokenizer identity plus materialized adapter-set, retrieval-policy, and skill-policy artifacts, combines them with the already measured model/runtime/KVMem identities, and recomputes the exact authorized served-artifact root before entering governed mode. Nonzero governed roots require real bytes; zero roots require the artifact to be absent.

The enforced boundary is:

`authorized StateEpoch -> physical deployment artifacts -> independent native measurements -> canonical aggregate root -> governed QW3`

This release also fixes Python symlink detection before path resolution and adds `build_physically_bound_served_manifest(...)`, which derives the serving manifest from deployed bytes instead of hand-copied hashes. Canonical governance objects can be materialized so their file SHA-256 exactly equals the hex portion of their governance digest.

Validation for this exact source tree: **285/285 active Python tests** and **28/28 portable native C++ tests** passed; the host QW3 executable compiled with CUDA disabled. CUDA/GPU qualification, real-weight unseen-task improvement, all-linear LoRA, native behavioral execution of retrieval/SkillIR policies, 10k–100k continual campaigns, and independent reproduction remain unclaimed. See `docs/PHYSICAL_ARTIFACT_CLOSURE_V15_5.md`.

---

# mini-AGI 15.4.0 — Measured Runtime Closure

v15.4 closes a serving-integrity gap left by v15.3: governed QW3 no longer accepts an aggregate served-artifact identity solely because an operator supplied matching strings. In RuntimeClosure1, QW3 measures the bytes it can independently observe, verifies those measurements against the authorized component roots, recomputes the aggregate served-artifact root with the same canonical hashing rule as Python governance, and refuses to enter governed serving mode when any measured component disagrees.

The enforced boundary is:

`authorized ServedArtifactManifest -> measured model/runtime/KVMem bytes -> native canonical artifact-root recomputation -> closure digest -> QW3 governed serving`

Measured directly by QW3:

- foundation/model bytes (regular GGUF or deterministic model-directory tree digest);
- the currently running QW3 executable bytes;
- the attached sealed KVMem `manifest.json` bytes, or the zero root when no archive is attached;
- the compiled native adapter bundle through the existing v15.3 byte-verified loader.

QW3 also receives the tokenizer, adapter-set, retrieval-policy and skill-policy roots and recomputes the exact aggregate `artifact_root`. The `/v1/runtime/state` endpoint now exposes `runtime_closure_verified`, a closure digest, and the measured model/runtime/KVMem roots. `MeasuredGovernedServingContract` refuses a server that merely echoes the epoch identity but cannot produce those measurements.

Validation for this exact source tree: **277/277 active Python tests** and **27/27 portable native C++ tests** passed; the host QW3 executable compiled with CUDA disabled. The new C++ runtime-closure test verifies canonical root parity against Python. CUDA/GPU execution, all-linear LoRA, real-weight task improvement, 10k–100k continual campaigns, and independent reproduction remain unclaimed. See `docs/RUNTIME_CLOSURE_V15_4.md`.

---

# mini-AGI 15.3.0 — Governed Native LM-Head LoRA

v15.3 closes the next concrete integration gap: a qualified adapter can now be compiled into a byte-verified QW3-native low-rank bundle and applied in the native logits path. The served StateEpoch binds both the qualified `adapter_set_root` and a separate `native_adapter_bundle_root`; QW3 verifies the raw model SHA-256 and every native A/B tensor before load, reports the loaded bundle identity, and rejects governed generation when the request identity does not match the loaded adapter bytes.

The implemented path is:

`VerifiedTrajectory -> qualified adapter artifact -> NativeQW3AdapterCompiler -> exact FP32 A/B bundle -> bound StateEpoch -> QW3 load verification -> logits += scale * B * (A * hidden) -> governed serving`

NativeAdapter1 is deliberately narrow and fail-closed. It supports a fully represented **LM-head/output LoRA only**. If a qualified source adapter contains layer LoRA tensors such as attention/MLP projections, compilation is rejected rather than silently applying a partial adapter while claiming the full adapter-set identity. Optimized batched native paths that bypass the ordinary LM head are disabled while a native LoRA is active so requests fall back to the correctness-preserving executor path.

Validation for this exact source tree: **272/272 active Python tests** and **26/26 portable native C++ tests** passed; the host QW3 executable compiled with CUDA disabled. The new CUDA FP32 low-rank kernel is implemented in source but **CUDA tooling/hardware was unavailable in this validation environment**, so GPU execution, real model weights, numerical GPU parity, end-to-end task improvement, and all-linear LoRA remain unclaimed. See `docs/NATIVE_ADAPTER_V15_3.md`.

---

# mini-AGI 15.2.0 — Governed Adapter Qualification

v15.2 keeps the v15.1 skill-learning integration and closes the next adapter-control gap. Verified trajectories can now form a deterministic training dataset; adapter training is bound to the exact foundation, dataset, trainer identity and hyperparameters; every output byte is hashed and published into immutable CAS; hidden fresh tasks run a paired A0/A1 transfer experiment with retention and security gates; and a passing candidate changes the exact `adapter_set_root` carried into governed serving. QW3 independently requires that adapter-set identity on generation requests.

The tested control path is:

`VerifiedTrajectory -> VerifiedAdapterDataset -> AdapterTrainingPlan -> Adapter bytes -> CAS -> sealed A0/A1 -> PASS -> QualifiedAdapterSetCandidate -> new adapter_set_root -> governed QW3 identity`

This is deliberately **not** described as completed in-weight continual learning. The custom QW3 native engine does not yet apply LoRA matrices to inference. v15.2 proves artifact provenance, byte integrity, qualification, and serving-state binding. Native adapter execution and real-weight improvement remain the next milestone. See `docs/ADAPTER_QUALIFICATION_V15_2.md`.

Validation for this artifact: 268/268 active Python tests and 25/25 portable native C++ tests passed; the host QW3 target compiled with CUDA disabled.

---

# mini-AGI 15.1.0 — Governed Learning-to-Serving Integration

Version 15.1 is the first release in this line that closes a tested procedural-learning loop into the served-state contract. It does **not** claim AGI, autonomous RSI, or in-weight continual learning. It adds one concrete L4 learning path: verified trajectories can become transfer-qualified SkillIR, hidden fresh tasks run a paired frozen A0/A1 experiment, a passing skill-policy bundle changes the exact `skill_policy_root`, and governed QW3 serving rejects requests when the loaded aggregate artifact root does not match the authorized served manifest.

The end-to-end tested path is:

`VerifiedTrajectory -> transfer-qualified candidate -> SkillIR -> sealed fresh-task A0/A1 -> frozen-baseline PASS -> SkillPolicyBundle -> new served artifact root -> governed QW3`

Important corrections in this release:

- StateEpoch and RuntimeManifest governance IDs now use the same canonical `sha256:<64 hex>` representation as the governance database and real `StateEpochLease` objects.
- QW3 now independently checks `X-MiniAGI-Artifact-Root` in addition to epoch and manifest headers.
- `SkillIRCompiler` fails closed on unknown operations and only compiles transfer-qualified trajectory candidates.
- `FreshTaskPairedEvaluator` reveals each hidden task once inside the evaluator and binds both A0/A1 arms to the signed consumption receipt.
- `GovernedSkillLearningLoop` has no promotion authority; it prepares a content-addressed policy candidate only. The existing qualification/promotion/StateEpoch chain remains the sole path to SERVABLE state.

Validation for this artifact: 253/253 active Python tests and 25/25 portable native C++ tests passed. CUDA/Metal real-weight qualification, LoRA parameter learning, long-horizon 10k–100k episode stability, and independent external reproduction remain unclaimed. See `docs/LEARNING_TO_SERVING_V15_1.md`.

---

# mini-AGI 14.1.0-alpha5 — Authority-Converged Evidence-Bound Continual Runtime

Alpha5 uses alpha4 as its transactional trunk and closes the next authority/evidence gaps without adding another mutation path. New production code imports `minagi.v15`, which exposes one canonical runtime and database while the historical V141-V144 APIs remain under `minagi.v14` for compatibility.

The alpha5 additions are mechanically bound fresh-task/falsification evaluation provenance, a witnessed/attested/servable StateEpoch lifecycle with request leases, transitive epistemic invalidation, trajectory-grounded proposal-only skill induction with contradiction blocking, a frozen-foundation A0/A1 transfer gate, and selected Apple Silicon performance helpers. See `ALPHA5_ARCHITECTURE.md` for the exact invariants.

This remains a research runtime. AGI, autonomous RSI, long-horizon continual-learning stability, independent external reproduction, production safety certification, and real-weight Apple/NVIDIA qualification remain unclaimed.

## Alpha5 convergence layer

`GovernedContinualRuntimeV145` wraps the validated alpha4 atomic commit path rather than replacing it. Privileged evaluation labels are no longer evaluator assertions: `fresh_hidden` must resolve to a trusted signed fresh-task consumption receipt, while `falsification` and `negative_control` must resolve to preregistered candidate/case/category/task commitments. After alpha4 activation, a StateEpoch must pass external witness and attestation stages before it becomes servable.

The new canonical import surface is:

```python
from minagi.v15 import GovernedContinualRuntime, GovernanceDB, StateEpochRegistry
```

mini-AGI v14.1-alpha4 builds on alpha3's exact-content authority model and closes the next two integration gaps: evidence-strength receipts are now cryptographically typed/chained instead of existence-counted, and exact learned-state mutations commit atomically with the RuntimeManifest head. It also adds a canonical raw Experience -> signed verification -> evidence-stage -> proposal path and an end-to-end L3 belief-learning cycle.

This remains a research runtime. Real-weight Apple Metal/NVIDIA CUDA qualification, independent third-party reproduction, long-horizon 10k-100k episode evidence, and L7-L10 production activation remain unclaimed.

## Alpha4 evidence closure and atomic learned-state activation

`GovernedContinualRuntimeV144` is the canonical alpha4 control plane. It adds:

- `ExperienceRecordV144` and `VerifiedEvidenceRecordV144` so the canonical learning path begins at raw experience rather than at an already-formed proposal;
- signed `EpisodeVerificationReceipt` ingestion bound to prompt, attempted output, repaired output, expected output, scoring policy and evidence root;
- `EvidenceStageReceiptV144` chains: E2 must be rooted in trusted episode-verification receipts, while E3-E5 must extend the immediately preceding signed evidence stage over the exact same evidence set;
- `EvidenceStrengthProofV144` derived from that signed stage chain, eliminating alpha3's count-by-existence supporting-receipt weakness;
- one authoritative `GovernanceDBV144` for the promotion chain plus learned-state heads;
- precomputed immutable `BeliefRevisionV144` / `SkillRevisionV144` mutation objects whose CAS digest is the exact promotion target;
- runtime snapshot closure: belief/skill snapshots must explicitly include their exact promoted mutation digests;
- atomic learned-state + runtime activation in one SQLite transaction, including one-use mutation-capability consumption;
- `full_belief_learning_cycle()` demonstrating Experience -> Verification -> E2 proof -> Proposal -> Candidate -> Build -> Evaluation -> Qualification -> exact PromotionAuthorization -> belief persistence + runtime activation.

The alpha4 authority chain is:

`Experience -> Signed Verification -> Signed Evidence Stage -> Derived Evidence Strength -> Candidate -> Build -> Signed Evaluation -> Mechanical Qualification -> Exact RuntimeManifest/ArtifactRoot -> One-Use Content Authorization -> Atomic Learned-State + Runtime Activation`

## Alpha3 content-bound authority closure

`GovernedContinualRuntimeV143` is the canonical control-plane API. The alpha3 path adds:

- evidence strength derived by a separately signed `EvidenceStrengthProofV143`; proposals cannot make arbitrary E0-E5 claims or reference nonexistent evidence;
- mutation scopes derived from learning mechanism/permanence policy rather than from candidate payloads;
- signed raw evaluation case results whose summary metrics are recomputed by the qualification authority;
- qualification bound to candidate, build, signed evaluation, raw results and derived metrics;
- a RuntimeManifest and artifact root created before promotion, so promotion signs the exact bytes/state intended for activation;
- `PromotionAuthorizationV143` bound to candidate, build, evaluation, qualification, artifact root and exact RuntimeManifest;
- one-use mutation commitments `(scope,target_digest)` enforced transactionally; a consumed runtime authorization cannot be replayed;
- an external signed audit anchor file required by the V143 audit verifier, so deleting all in-database checkpoints is detectable;
- split FreshTask metadata and secret-vault classes; the learner-facing metadata store has no secret-content handle;
- production-mode enforcement that rejects in-process/local authority clients; authority clients can be hosted behind owner-only Unix-domain sockets under separate processes/UIDs;
- the legacy `kvcontinual.execution.api` qualification/promotion endpoints now return HTTP 410;
- the installed canonical command is `minagi`; the old EGAI CLI is retained only as `egai-research-legacy`.

The exact promotion chain is now:

`EvidenceProof -> Candidate -> Build -> Signed Raw Evaluation -> Mechanically Derived Qualification -> RuntimeManifest/ArtifactRoot -> Exact Content PromotionAuthorization -> One-Use Activation`

## Alpha2 authority closure

`GovernedContinualRuntimeV142` is the preferred control-plane API in this release. It adds:

- independently signed evaluation bundles with evaluator generation fencing;
- mechanically derived qualification decisions (callers no longer supply PROMOTE/REJECT);
- independently signed qualification records with qualifier generation fencing;
- promotion authorizations with explicit mutation scopes;
- belief/skill persistence gates that resolve the actual signed authorization from CAS and the authoritative governance database;
- typed runtime-manifest dependencies that must exist in CAS before activation;
- signed audit checkpoints, making post-checkpoint ledger rewriting detectable without the audit private key;
- a split FreshTask metadata/vault design, atomic lease acquisition, and expired-lease reclamation;
- verification receipts bound to repaired output, expected output, scoring-policy identity, and pass threshold;
- a canonical `learning_cycle()` that executes candidate -> build -> signed evaluation -> mechanical qualification -> scoped promotion -> typed activation.

The alpha1 APIs remain present for compatibility but are not the preferred authority path.

## What changed in v14.1-alpha1

This release hardens the v14 design around one transactional authoritative artifact chain rather than adding broader plasticity. It adds:

- `ImmutableCAS` with strict `sha256:<64 lowercase hex>` parsing, atomic writes, read-time digest verification, and traversal rejection;
- `GovernanceDB`, a SQLite/WAL/FULL-synchronous authority store with foreign-key bindings across Candidate -> Build -> Evaluation -> Qualification -> Promotion -> Activation;
- a hash-chained audit ledger covering privileged lifecycle transitions;
- `PromotionAuthorizationV141`, signed by a dedicated promotion key and bound to candidate, qualification, policy generation, and authority generation;
- `GovernedContinualRuntimeV141`, the canonical v14.1 orchestration surface for the authoritative artifact chain;
- immutable bi-temporal belief revisions with atomic head pointers instead of destructive belief overwrite;
- immutable skill revisions preserving positive and negative evidence with an enforced lifecycle;
- new adversarial tests for skipped-chain promotion, rejected qualifications, signature tampering, CAS corruption/traversal, audit tampering, and failed activation rollback.

The older v14 primitives remain for compatibility and research, but the new v14.1 classes are the preferred control-plane path. L7-L10 activation remains disabled by default.

## What changed from v13

The v14 governance model is two-dimensional:

- **Permanence L0–L10** describes how difficult a learned change is to reverse.
- **Evidence strength E0–E5** describes the quality of support for that change.

Higher permanence requires stronger evidence. These axes are intentionally independent.

The active hierarchy is:

1. L0 working context
2. L1 raw evidence
3. L2 episodic memory
4. L3 semantic belief
5. L4 reusable skill
6. L5 routing/composition policy
7. L6 isolated neural memory
8. L7 shared adapter
9. L8 new module
10. L9 foundation consolidation
11. L10 architecture change

Default policy permits production-directed orchestration only through L6. L7–L10 remain proposal-only until a later release explicitly changes policy.

## Canonical learning path

Persistent changes are expected to follow:

`Experience -> Evidence -> Signed Verification -> Hypothesis/Skill -> LearningProposal -> Candidate -> Falsification -> Fresh Evaluation -> Qualification -> PromotionAuthorization -> Persistence -> RuntimeManifest`

The learning plane can propose and build candidates. It cannot authorize production persistence.

## Authority convergence

v13 retained an older `kvcontinual.continual.AdapterRegistry.promote()` path that could promote a locally qualified adapter without going through the converged promotion authority. v14 removes that path.

`kvcontinual.continual.registry.AdapterRegistry` is now a fail-closed compatibility facade over `kvcontinual.execution.registry.AdapterRegistry` and always requires:

- signed promotion authority;
- independently verified qualification bundles;
- signed production transitions;
- content-bound candidate artifacts.

The legacy admin qualification/promotion/rollback endpoints now return HTTP 410 instead of acting as mutation authorities.

## Signed learning eligibility

Boolean `verified=True` is no longer an authority signal for continual training.

`ExperienceStore` requires an `EpisodeVerificationReceipt` signed by a trusted verifier and bound to:

- episode ID;
- prompt digest;
- attempted output digest;
- repaired output digest;
- evidence root;
- verifier identity;
- score and importance.

The old `/learning/episodes/{id}/verify` endpoint is disabled. The macOS gateway accepts signed receipts only when a trusted verifier public key is explicitly configured.

## New v14 governance primitives

`minagi.v14` adds:

- `GovernancePolicyV14` — permanence x evidence policy;
- `DurableFreshTaskAuthority` — SQLite/WAL one-shot hidden tasks with commit/reveal;
- `FalsificationPlan` — preregistered negative-control, retention and security requirements;
- `DreamPolicyResearcher` — offline proposal-only replay-policy research;
- `ModelTimeClock` and `AdaptiveReplayScheduler` — replay based on parameter displacement and estimated forgetting risk;
- `CandidateStateStore` — enforced REGISTERED -> BUILT -> EVALUATED -> QUALIFIED -> AUTHORIZED -> ACTIVE state machine;
- `GovernedContinualOrchestrator` — one canonical authority-aware lifecycle coordinator.

## EGAI RC8.6 integration

The active tree preserves mini-AGI 13's hardened EGAI scientific kernel and adds compatible RC8.6 mechanisms such as signed repair/verification components and skill synthesis/verification primitives. The complete original RC8.6 donor is retained under `references/egai_rc8_6/` for provenance and comparison, rather than being installed as a competing authority plane.

v14 intentionally does not install RC8.6's standalone promotion/runtime authority over the converged mini-AGI promotion substrate.

## Execution plane

The native execution runtime remains the mini-AGI 13 / RC11-derived substrate:

- Qwen-oriented C++ runtime;
- KVMem/HYPIC execution-memory work;
- persistent CAS and execution data plane;
- exact selected replay baseline;
- CPU/native portable path;
- CUDA implementation retained for supported systems;
- Apple Metal/MLX/macOS integration retained.

Execution memory and semantic continual learning remain separate validity domains.

## Validation performed for this release

In the build environment used to create this archive:

- Python/governance/runtime tests: **204/204 passed**.
- Portable native C++ tests with CUDA and Metal GDN disabled: **25/25 passed**.
- Exact release source membership and hashes are generated into `SOURCE_MANIFEST.json`.
- `RELEASE_ATTESTATION.json` is signed with a newly generated Ed25519 release key; the public key is packaged and the private key is not.

The release signature proves integrity of this generated archive only. It does not establish third-party maintainer identity, independent scientific validation, real-weight Metal/CUDA qualification, or production safety.

## Build and test

Python 3.11+:

```sh
python -m pip install '.[dev]'
python -m pytest -q
python tools/verify_release.py .
```

Portable native validation:

```sh
cmake -S . -B ../mini-agi-v14-native \
  -DQW3_ENABLE_CUDA=OFF \
  -DQW3_ENABLE_METAL_GDN=OFF \
  -DQW3_BUILD_TESTS=ON
cmake --build ../mini-agi-v14-native --parallel 2
ctest --test-dir ../mini-agi-v14-native --output-on-failure
```

On Apple Silicon, use the macOS documentation and enable the supported Metal path in a native macOS build. CUDA tests require an NVIDIA environment and are not claimed by this archive's local validation record.

## Important limits

The release does not claim:

- autonomous safe recursive self-improvement;
- independently reproduced L7+ continual-learning gains;
- production-qualified real-weight HYPIC/Metal/CUDA execution;
- safe base-model consolidation;
- safe architecture self-modification.

Those remain research targets. See `docs/V14_ARCHITECTURE.md`, `FULL_UPGRADED_BUILD_STATUS.md`, and `VALIDATION_SUMMARY.md`.
