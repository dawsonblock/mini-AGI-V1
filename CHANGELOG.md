# v16.4.1 — Runtime Security Closure

Closes the four runtime-security defects from the v16.4.0 audit at the
source, with executable adversarial regression tests for each. No
scientific claim changes (Campaign 1b QUALIFIED, Campaign 2 REFUSE,
Campaign 3A still executing). Details:
`docs/research/RUNTIME_SECURITY_CLOSURE_V1641.md`; open items:
`docs/research/REMAINING_DEFECTS_V1641.md`.

Admission is now mandatory (SEC-001):

- New `TrustedRuntimeLauncher` (`minagi/v161/trusted_launcher.py`) +
  `scripts/trusted_launch.py`: resolves the signed plan/qualification/
  promotion decision and the authorized runtime manifest, verifies every
  signature, role, validity window, expiry, and revocation entry,
  physically measures the model, tokenizer, and adapter, stages an
  immutable snapshot, loads the backend ONLY from that snapshot, and
  emits a signed production activation receipt after the load succeeds.
  Any failure refuses and no receipt exists.
- `PeftServingBackend.load()` accepts only an `ApprovedSnapshot`
  (constructible solely by the staging function); raw paths are
  refused. `minagi/platforms/cuda/hf_runtime.load_causal_lm()` refuses
  `adapter_path=` in serving mode — the research plane must pass
  `purpose="research"` explicitly (the reviewable record that the load
  evaluates a candidate rather than serving it). Call sites updated.

Artifact closure repaired (SEC-002):

- New `artifact_closure.close_tree()`: symlinks and unsupported special
  files anywhere in an authorized tree are REFUSED, not skipped (the
  v16.4.0 walker silently skipped them, so adding a symlink did not
  change the digest); path traversal is rejected; the complete
  (path, size, sha256) listing is returned. `verify_entries()` fails on
  missing, unexpected, resized, or modified files individually.
- Digest compatibility with v16.4.0 semantics is preserved for clean
  trees (existing recorded evidence still verifies); trees that
  previously hashed incompletely are now refused. HF-cache snapshot
  symlinks use the documented `resolve_symlinks=True` policy.
- Runtime manifest v2 (`mini-agi-v16.5-peft-runtime-manifest-v2`) binds
  the explicit authorized `adapter_files` listing; serving requires v2.

Signed activation evidence enforced (SEC-003):

- Receipt schema v2 binds candidate/qualification/promotion/manifest
  digests, the artifact digests actually measured at load, backend id,
  a 128-bit replay nonce, activation time, runtime signer, and
  signature. `write_activation_receipt` refuses unsigned receipts;
  `check_activation_receipt` reports unsigned receipts as problems by
  default and `require_production=True` refuses admission-only
  receipts (signature validity alone is not enough — the signer must
  have measured and loaded the artifacts). `admit_runtime.py` lost its
  unsigned mode and now requires physical `--runtime-model-path` /
  `--runtime-tokenizer-path` measurements.

Protocol made strict (SEC-004):

- New `strict_schema.py`: one versioned validator per authority-bearing
  artifact; unknown versions, absent/malformed digests, unsupported
  backend identifiers, and unsafe listing paths are refused. Admission
  now requires the signed plan's `experiment_protocol_digest` to equal
  the manifest's `protocol_digest` and the qualification's
  `evaluation_bundle_digest` to equal the decision's; `promote.py`
  refuses to sign decisions with unbound evaluation or protocol.
- `RevocationList` carries a generation time; the launcher requires a
  revocation list and refuses a stale one (default 7 days); the
  admission CLI accepts `--max-revocation-age-days`. Production
  receipts carry replay nonces recorded in an append-only journal.

Time-of-check/time-of-use: `immutable_snapshot.stage_snapshot()`
copies-and-verifies in a single pass (the staged bytes are the hashed
bytes), requires the staged digest to equal the authorized digest,
freezes the tree read-only, and re-measures it immediately before load.

Tests: +60 (571 → 631 passing, 1 skipped — Linux-only `RLIMIT_AS` on
the macOS host). New suites: `test_v1641_artifact_closure.py` (21),
`test_v1641_trusted_launcher.py` (20), `test_v1641_strict_schema.py`
(11), `test_v1641_cli.py` (8); `test_v164_runtime_admission.py` updated
to the measured-artifact CLI contract. Release change manifest:
`RELEASE_CHANGE_MANIFEST.json`.

## Review repairs (post-release audit)

A review of this release found four defects, each fixed with a
regression test; no scientific claim changes.

- **Tokenizer identity convention.** The signed plan binds the
  tokenizer as the tokenizer-named files inside the model snapshot
  (`physical_identity_digests`), but the launcher expected a tree
  closure of a separate `--tokenizer-path` — different digests, so no
  plan-derived manifest could ever be admitted (fail-closed, but the
  sanctioned serving path was unusable end-to-end, and the tests
  encoded the launcher's convention only). The convention is now one
  shared implementation (`artifact_closure.tokenizer_artifact_digest`,
  byte-identical to the recorded plans); the launcher re-measures it
  from the staged model, the serving backend loads the tokenizer from
  the verified model artifact, and `--tokenizer-path` /
  `--runtime-tokenizer-path` are optional cross-checks. The runner
  delegates to the shared implementation — plan digests are unchanged,
  so the in-flight Campaign 3A resume semantics are preserved.
- **CUDA loader serving default.** `load_causal_lm()` with the default
  `purpose="serving"` silently loaded a bare base model when no
  snapshot was passed. A serving load without an `ApprovedSnapshot` is
  now refused, and the research-plane base-model call sites pass
  `purpose="research"` explicitly (the flag is now the reviewable
  record for every research-plane load, not only adapter loads).
- **Release verifier symlinks.** `verify_release.py` silently ignored
  symbolic links in the verified tree (the same "skipped, not refused"
  pattern SEC-002 fixed in the artifact closure). Links are now refused
  (`symlinks_in_tree`, exit 2); the builder already excludes them. Tool
  caches are also skipped consistently (`.ruff_cache` joins
  `.pytest_cache` in the verifier, the reseal enumerator, the builder,
  and `.gitignore`), so a local `ruff` run cannot produce a false
  release failure.
- **Snapshot staging cleanup.** `stage_snapshot()` raised a raw
  `KeyError` and left a partially staged, unfrozen tree behind when
  handed an artifact name the manifest did not authorize. Names are now
  validated up front (`SnapshotError`, nothing staged), and any
  staging failure discards the destination.

Docs: `docs/research/RUNTIME_SECURITY_CLOSURE_V1641.md` (tokenizer
convention, verifier policy). Tests: +8 (631 → 639 passing, 1 skipped).

## Lint contract (dedicated hygiene pass)

The repository had no lint configuration, so tools ran with their own
defaults (ruff 0.16's broad rule set, Flake8's 79-column pycodestyle)
and flagged the codebase's deliberate conventions as thousands of
errors. `pyproject.toml` (`[tool.ruff]`) and `.flake8` now declare one
contract for both tools: E4 import hygiene, E7 statement structure, E9
syntax, and pyflakes (F) are enforced. Documented as intentionally not
enforced: the compact one-line statement style (E701/E702/E703), the
post-`sys.path` bootstrap imports (E402 — the package is not installed
on the Colab runners), the ~100-column line style (E501), and the
pycodestyle formatting families (E1/E2/E3/E5/W). `__init__.py`
re-export barrels are exempt from F401; `references/` and
`third_party/` keep their upstream style.

The enforced rules are then satisfied repo-wide: unused imports (F401),
multi-import lines (E401), unused locals (F841), lambda assignments
(E731), ambiguous names (E741), f-strings without placeholders (F541),
a redundant re-import (F811), and an unnecessary `nonlocal` (F824)
were fixed across 180 files. Every fix is behavior-preserving
(side-effecting calls kept; dead assignments removed) — with one
deliberate exception: the F811 "fix" would have broken the v164
qualification path (the first binding serves it when the v165 branch is
skipped), so `ImmutableCAS` is now imported from its defining module
instead. `ruff check .` and `flake8 .` both report zero findings;
tests unchanged at 639 passed (1 skipped).

The pass also surfaced a latent defect in the legacy governed-repair
harness: `egai/bench/sequential_governed.py` constructed
`VerifiedEpisode` with 14 positional arguments against a 12-field
dataclass (every other call site — including its sibling
`egai/bench/sequential.py` — passes 12), so the verified-repair branch
raised `TypeError` before it could run. Fixed to the 12-argument form
and exercised with a probe harness; the branch now executes. The same
harness reached into the ledger's private `_head()`; `EvidenceLedger`
now exposes a documented `head()` accessor and the harness uses it
(the private helper stays for internal use).

Finishing that harness exposed two more missing pieces on
`SandboxSkillMemory`, both called by the run loop and neither existing
anywhere in the codebase: `mark_success` (the success path would raise
`AttributeError` whenever a retrieved skill was credited) and
`health()` (the end-of-run summary). Both are implemented against the
memory's own model: `mark_success` increments `successes`, the exact
mirror of the existing `mark_failure`; `health()` reports
`degraded` = procedures that have failed more often than they have
succeeded (the same condition as `confidence < 0.5`, stated on the
counters so no new threshold is introduced) and `retired` = 0 — the
ephemeral sandbox memory has no retirement mechanism, and the method
docstring says so rather than inventing one. The harness is no longer
uncovered: `tests-python/research/test_sequential_governed.py` drives
the success path, the verified-repair path, the health summary, and
the evidence chain. Tests: 639 → 641 passing (1 skipped).

Pylint — which some editors run alongside ruff/Flake8 — now reads the
same contract from `[tool.pylint]`: the convention and refactor
families and the warning family are disabled (the pyflakes-equivalents
are enforced by ruff's F family, which is clean; Pylint's variants
additionally flag unused loop variables, which the compact style uses
deliberately), the optional extras and platform modules are declared
as ignored modules, and the error category stays enabled. Pylint's
defaults reported 17,896 findings on the release surface; the
configured run reports two — both real, and deliberately not
suppressed: they point at the drifted donor `minagi/rc14/system.py`
(its `GovernedRC14System` passes two keyword arguments the base class
does not accept and reads attributes the base never defines; nothing
imports it). That is recorded as RC14-001 in the remaining-defects
register rather than hidden by the config. Three further Pylint errors
were verified as false positives before being disabled (guarded
subscripts, a variable assigned under the mirrored condition, and
dynamic attributes inferred as `object`).

The contract is now declared and enforced: `flake8` and `pylint` join
`ruff` in the `dev` extras (pinned ranges), and CI runs
`scripts/rc11/lint.sh` — ruff, Flake8 and Pylint over the release
surface. The gate tolerates exactly the two documented RC14-001
findings and fails on anything else (verified: a probe finding in
another file exits 1).

Separately, CI had been red since the Route-B commit (`a2a7295`), which
started importing `torch`/`transformers`/`safetensors` in the test
suites without adding them to the workflow's install line — the
`validate` job could not even collect the tests. The install line now
includes `torch transformers peft safetensors` (the ML dependencies the
suites import; `faiss`/`trl` are imported nowhere and stay out).

# v16.4.0 — Training Semantics, Runtime Admission, Mechanism Control

Phases 2, 3, and (early) 5 of the v17 plan. Scientific claims unchanged
(Campaign 1b QUALIFIED stands, Campaign 2 REFUSE stands, Campaign 3
still drafted/unexecuted). Details and the record chain:
`docs/research/TRAINING_AND_ADMISSION_V164.md`.

Training schedule (Phase 2.1, v16.3.0 scope):

- The structured trainer no longer places `gradient_accumulation_steps`
  examples in each microbatch while also accumulating over that many
  microbatches (which silently squared the effective batch size).
  `LoraTrainSpec` and `ExperimentProtocolV1` gain explicit
  `microbatch_size` / `train_microbatch_size`; effective batch =
  microbatch × accumulation, optimizer updates = steps, presentations =
  steps × effective batch. Reproduced: the original engine ran 48
  presentations for a declared 12 (steps=3, gacc=4); the repaired
  engine runs 12.
- Receipt v2 binds the full schedule, actual sample order, per-update
  gradient statistics, the malformed-sample policy/outcome, and
  `adapter_artifact_digest` (weight+config bytes; the directory closure
  digest in runtime manifests covers the receipt/spec evidence files).
- `malformed_policy: reject | fail` — strict preregistered campaigns
  fail the run on any unapproved sample rejection instead of silently
  changing the training dataset; Campaign 3A/3B/3C configs set
  `microbatch_size: 1` and `malformed_policy: fail` explicitly.

Runtime admission (Phase 3, v16.4.0 scope):

- New `minagi.v161.runtime_admission` + `scripts/admit_runtime.py`:
  `RuntimeAdmissionController` verifies the full chain (plan →
  qualification → promotion decision → runtime manifest → exact
  adapter bytes) before serving and emits an ActivationReceipt;
  rollback re-admits a previously receipted release and records the
  lineage. Refused, with adversarial tests: unsigned/forged/expired/
  revoked decisions, research-plane promotion attempts, one altered
  adapter byte, substituted qualifications, model/tokenizer/backend
  substitution, tampered manifests, incomplete chains, revoked
  replays.
- Sixth authority role `runtime` (trust root + `activation_receipt`
  ledger kind); `promote.py` decisions carry `expires_at` (30-day
  default).

Mechanism controller (Phase 5, ahead of the v16.6.0 milestone):

- `MechanismEstimate` + frozen `ObjectiveWeights`
  (`U = ΔQ − λ_C·C − λ_R·R − λ_L·L`), `rank_candidates` with hard
  regression-risk cap and confidence floor applied before ranking,
  `select_mechanism` (weights require recorded cheaper attempts;
  uncertain causes select a diagnostic experiment), and digest-chained
  `AttemptReceipt`s (FailureEvidence → MechanismProposal →
  AttemptReceipt → EvaluationBundle).

Tests: +40 (engine-level schedule counts, policy behavior, receipt
binding; the plan's four named admission attacks plus expiry/forgery/
substitution/rollback/CLI; controller arithmetic, safety caps, ladder
enforcement, attempt records). Suite: 561 passed, 1 skipped
(Linux-only RLIMIT_AS on macOS). Manifest resealed.

v16.4.0 batch 2 — native suite + Campaign 3A preregistration:

- Native CTest suite rebuilt and rerun for this release: 28/28 PASS
  (macOS CPU/stub build; no C++ sources changed since v16.2.1, so this
  is a toolchain re-verification, not a new native claim).
- Campaign 3A preregistration completed on the CPU side: the sealed
  final holdout (24 rows, 4 fresh families) was generated by the
  evaluation authority (`scripts/generate_holdout.py` — refuses
  in-repo output, asserts a fresh vocabulary, seed + file held outside
  the repository) and bound by manifest digest
  (`sha256:ff9e4340…`) into `configs/campaign3a.yaml`;
  `scripts/seal_final_holdout.py` re-checked id/family disjointness.
  Campaign execution remains GPU-blocked.
- Preregistration regression tests: generator in-repo refusal /
  determinism / vocabulary filtering, sealer round-trip + leakage
  rejection, and campaign-3 binding invariants (the corpus carries no
  holdout rows; 3C must reuse 3B's holdout; the bound digest
  re-derives from the authority-held file when present). Suite:
  571 passed, 1 skipped.

v16.4.0 batch 3 — Campaign 3A launched on Colab:

- The preregistered 3A matrix (10 seeds x 7 arms) is executing on a
  Colab T4 under the signed plan. Verified from the banked plan: the
  campaign binds the sealed holdout digest (sha256:ff9e4340...) and
  the corrected schedule (microbatch 1 / gacc 1 / malformed_policy
  fail; lr 5e-5 as the single lever vs Campaign 2).
- Per-seed evidence banking is armed: completed seeds are tarballed
  (seed evidence + L6/NC adapters + arm states) and pulled locally;
  `scripts/campaign3a_supervisor.py` is the resumable lane supervisor
  (banks completed seeds, relaunches the runner if it dies — COMPLETE
  seeds are skipped by the runner's per-seed atomicity).
- Execution is in progress; independent qualification with the
  authority-held `--holdout` file follows when all seeds are banked.

v16.4.0 batch 4 — VM reclamation recovery + finalize pipeline:

- The first Colab VM was reclaimed mid-seed-1. Recovery validated the
  resume design end to end: the runner was relaunched on a fresh VM
  from the pushed branch with the model revision pinned to the
  originally resolved commit; the recomputed plan digest is
  byte-identical (sha256:e9ecd87e...), so the banked seed-0 evidence
  verifies under it and was counted, and execution resumed at seed 1.
  The authority ledger was reconstructed deterministically (same key +
  same plan doc -> identical signed entry), and restoring it makes any
  plan mismatch fail closed ("refusing to mix campaign identities").
- `scripts/campaign3a_finalize.py`: assemble banked evidence -> run the
  independent qualifier with the authority-held holdout -> write the
  results report -> publish a results branch (evidence force-added per
  repo convention; private keys never committed). Exercised end to end
  on partial evidence: the qualifier failed closed with
  INVALID_EVIDENCE and explicit per-cell reasons, and the results
  branch mechanics were verified on a throwaway branch.

# v16.2.2 — Security and Correctness Repair (Phase 1)

Phase 1 of the v17 plan: the six security/correctness defects from the
audit are repaired, each with a reproducing test that fails on v16.2.1
and passes here. No scientific claims change — Campaign 1b's
qualification and Campaign 2's qualified-negative result stand under
their original schemas; Campaign 3 remains drafted/unexecuted.

Evaluator isolation (FIX-001):

- `executor_score` no longer runs corpus python with the evaluator's
  environment and privileges. Deterministic tasks prefer a declarative
  evaluator (allowlisted ops — equals/contains/numeric/json/regex
  composition — no code execution); `python_assert` checkers run only
  under an OS-enforced sandbox (`minagi.v161.execution_sandbox`):
  macOS `sandbox-exec` SBPL profile or Linux `bwrap --unshare-all`,
  minimal environment (os.environ never inherited), workspace-only
  writes, no network, rlimits, process-group termination on timeout.
- No sandbox backend => `SandboxUnavailable` (fail closed; there is no
  unsandboxed fallback). `MINIAGI_SANDBOX_BACKEND` pins a backend or
  forces the fail-closed path.
- Adversarial suite: secret read, authority-state read, outside write,
  network, env visibility, orphaned-process survival, read-only
  reference inputs, workspace cleanup, fail-closed paths. Residual
  risks (macOS memory-cap best-effort, deprecated sandbox-exec,
  bwrap/userns availability, regex bounds): see
  `docs/research/SECURITY_REPAIR_V1622.md`.

Mathematical correctness:

- FIX-002 — `RankAllocator.grow()` clamps to
  `min(per_task, max_rank, current + headroom)`; growth is strictly
  non-decreasing and blocked growth leaves state unchanged (the
  original could drive a task's rank to zero on budget exhaustion).
  Randomized invariant test added.
- FIX-004 — false-activation rate divides by previously-correct
  baseline cases (matching the preregistered bound); conditional
  regression reported alongside and surfaced in qualifier stats.

Authority and promotion:

- FIX-005 — `not_before`/`not_after` validity windows are enforced at
  an explicit verification time in `is_authorized`/`assert_authorized`/
  `verifier()` and `AuthorityLedger.verify`; expired or not-yet-valid
  keys fail closed; malformed bounds and naive datetimes are rejected.
- FIX-003 — generation promotion is an append-only `PromotionEvent`
  validated against the promotion authority: signed envelope, bound
  generation record/campaign/qualification digests, authorized signer
  valid at verification time, unexpired, unrevoked. A bare
  digest-shaped string no longer unlocks the next generation; the
  chain re-verifies every event on `verify()`.

Release metadata (FIX-006):

- Attestation reconciled with the signed manifest and version
  identities (VERSION, pyproject, package, SBOM, validation docs);
  `scripts/verify_release.py` fails on attestation/manifest/version
  drift (exit 5); reconciliation regression-tested.

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
