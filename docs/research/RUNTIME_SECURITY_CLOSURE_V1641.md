# v16.4.1 — Runtime Security Closure

The v16.4.0 audit identified seven principal problems. Four of them are
runtime-security defects; this release closes those four at the source
and adds executable adversarial regression tests for each. The
remaining three (mechanism-controller evidence, negative-utility
abstention, real-model learning qualification) are correctness and
scientific-qualification work and belong to v16.4.2 and v16.5.0 — they
are carried in `REMAINING_DEFECTS_V1641.md` with severity and the
reproducing tests that currently exist.

Release gate (from the plan): **no authorized model-loading path
bypasses admission; every mutation, substitution, expired authority, or
malformed required record is rejected.**

## SEC-001 — Runtime admission was not mandatory

**Defect.** `RuntimeAdmissionController` existed but nothing required a
model-loading path to call it. `PeftServingBackend` did not exist,
`minagi/platforms/cuda/hf_runtime.load_causal_lm(adapter_path=...)`
opened any adapter directory by path, and no component could prove that
the artifacts it served had been admitted.

**Repair.**

* `minagi/v161/trusted_launcher.py` — `TrustedRuntimeLauncher` is the
  enforcement point. It resolves the signed plan/qualification/decision
  + runtime manifest, verifies every signature, role, validity window,
  expiry and revocation entry, physically measures the model, tokenizer
  and adapter, stages an immutable snapshot, loads the backend from
  that snapshot only, and emits the signed production receipt after the
  load succeeds. Any failure refuses the launch and no receipt exists.
* `minagi/v161/peft_serving.py` — `PeftServingBackend.load()` accepts
  only an `ApprovedSnapshot` (constructible solely by
  `immutable_snapshot.stage_snapshot`); raw paths, strings, and mutable
  directories are refused.
* `minagi/platforms/cuda/hf_runtime.py` — `load_causal_lm()` refuses
  `adapter_path=` in serving mode (the default) and refuses a serving
  load without an `ApprovedSnapshot` (a bare model load is not an
  admitted artifact). Serving passes `snapshot=`; the research plane —
  which evaluates the candidate, the experiment itself, rather than the
  runtime — must pass `purpose="research"` explicitly, which is the
  reviewable record that the load is not a serving activation. Call
  sites updated: `scripts/run_campaign1.py`,
  `scripts/run_colab_campaign.py`,
  `scripts/validation/adapter_effect_check.py`.
* `scripts/trusted_launch.py` — the deployment CLI for the launcher.
* Tokenizer identity: the signed plan binds the tokenizer as the
  tokenizer-named files inside the model snapshot root
  (`physical_identity_digests`); the launcher re-measures that
  convention from the staged model
  (`artifact_closure.tokenizer_artifact_digest`) and the serving
  backend loads the tokenizer from the verified model artifact — there
  is no separately substitutable tokenizer path in the served set.
  `--tokenizer-path` / `--runtime-tokenizer-path` are optional
  cross-check roots.

## SEC-002 — Artifact hashing overlooked symbolic links

**Defect.** `runtime_closure3.sha256_path` walked directories with
`rglob` and *skipped* symlinks and non-regular files. Adding a symlink
to an approved adapter directory did not change its digest, so a
substituted artifact could pass closure checks. The digest also could
not say *which* files it authorized, so an unlisted file was
indistinguishable from an approved one.

**Repair.**

* `minagi/v161/artifact_closure.py` — one canonical closure
  implementation. `close_tree()` refuses symlinks and special files
  (fifo/socket/device) anywhere in the tree, rejects path traversal,
  and returns the ordered (path, size, sha256) entries plus the digest.
  `verify_entries()` compares a measurement against a signed listing:
  missing, unexpected, resized, and modified files each fail
  individually. Digest compatibility is preserved for clean trees, so
  evidence recorded before this release still verifies; trees that
  previously hashed *incompletely* (containing skipped symlinks) are now
  refused — exactly the audited case.
* `runtime_closure3.sha256_path` delegates to that implementation.
* `peft_serving.adapter_closure` refuses symlinked
  `adapter_config.json`/weight files and accepts an `expected_files`
  listing.
* `peft_serving.runtime_manifest` emits schema
  `mini-agi-v16.5-peft-runtime-manifest-v2` binding the complete
  authorized adapter listing (`adapter_files`).
* Hugging Face cache snapshot directories (all symlinks into the blob
  store) are handled by the documented `resolve_symlinks=True` policy,
  which measures the resolved targets and records the policy; the
  strict default refuses them.

**Acceptance test.** `tests-python/v161/test_v1641_artifact_closure.py`
(symlink in tree, symlinked weight file, symlinked directory, top-level
symlink, fifo, unexpected/missing/resized/modified entries, traversal
entries) and the CLI tests in `test_v1641_cli.py`.

## SEC-003 — Unsigned activation receipts passed verification

**Defect.** `check_activation_receipt` only inspected a signature *when
one was present*: a receipt without `signer_key_id` returned "no
problems". `scripts/admit_runtime.py` could write such receipts
(`--no-sign`), and the receipt carried no evidence of what was actually
loaded.

**Repair.**

* `ActivationReceipt` schema v2
  (`mini-agi-v16.4.1-activation-receipt-v2`) binds: promotion decision,
  qualification record, runtime manifest, adapter digest,
  `candidate_digest`, `loaded_artifact_digests` (the model/tokenizer/
  adapter bytes measured at load), `backend_id`, `activation_nonce`,
  `admitted_at`, and the runtime-role signature.
* `write_activation_receipt` refuses to write an unsigned receipt.
* `check_activation_receipt(..., require_signature=True)` (default)
  reports an unsigned receipt as a problem; `require_production=True`
  additionally demands the load-measured digests, the replay nonce, a
  production schema version, and refuses admission-only receipts
  ("signature validity alone is not sufficient unless the signer
  actually measured and loaded the artifacts").
* `scripts/admit_runtime.py` no longer has a `--no-sign` mode; a missing
  runtime key is a configuration error, and its receipts are explicitly
  labelled `production: false`.

## SEC-004 — Schema and protocol requirements were optional

**Defect.** Admission validated only the promotion decision's schema
string. The campaign plan and qualification record were accepted at any
version; `evaluation_bundle_digest` was compared only when present on
*both* sides (omitting one side silently disabled the check); the
protocol link was optional; `runtime_backends` fell back silently; and
revocation evidence had no freshness requirement.

**Repair.**

* `minagi/v161/strict_schema.py` — the single versioned validator for
  every authority-bearing artifact (campaign plan, qualification
  record, promotion decision, runtime manifest, activation receipt).
  Unknown versions, absent or malformed digests, non-integer or
  non-ordered authorization windows, unsupported backend identifiers,
  and unsafe listing paths are refused.
* Admission now requires: a known plan schema that binds an
  `experiment_protocol_digest`; a known qualification schema whose
  `evaluation_bundle_digest` matches the decision; a manifest that binds
  the same protocol digest as the signed plan; a qualification backend
  coverage declaration that is a subset of the supported set (the
  documented default remains `hf-peft`).
* `RevocationList` (`mini-agi-v16.4.1-revocation-list-v1`) carries a
  generation time; the launcher requires a revocation list and refuses
  a stale one (default max age 7 days), and the admission CLI accepts
  `--max-revocation-age-days`.
* Replay protection: each production receipt carries a 128-bit nonce
  recorded in an append-only journal; revoked decisions are refused on
  re-admission.

## Review repairs (post-release audit)

A review of this release found four defects; each is fixed at the
source with a regression test. None changes a security property for
the better or worse in the fail-open direction — all four were
fail-closed or unavailable paths.

* **Tokenizer identity convention (integration blocker).** The plan
  binds the tokenizer with `physical_identity_digests` (a map of
  relative path → content digest over the tokenizer-named files inside
  the model snapshot), while the launcher expected a tree closure of a
  separate `--tokenizer-path`: structurally different digests, so no
  plan-derived manifest could ever be admitted, and the test suite
  encoded only the launcher's convention. The convention now lives in
  one shared implementation (`artifact_closure.tokenizer_artifact_digest`,
  byte-identical to the recorded plans — the runner delegates to it,
  so plan digests are unchanged and Campaign 3A resume stays valid),
  the launcher re-measures it from the staged model, the backend loads
  the tokenizer from the verified model artifact, and the explicit
  tokenizer path is an optional cross-check (the model snapshot and the
  tokenizer path must agree when both are supplied).
* **CUDA loader serving default.** `load_causal_lm(spec)` with the
  default `purpose="serving"` silently loaded a bare base model from
  `spec.model_id` when no snapshot was passed; the guard only covered
  `adapter_path`. A serving load without an `ApprovedSnapshot` is now
  refused, and the research-plane base-model call sites pass
  `purpose="research"` explicitly.
* **Release verifier symlinks.** `verify_release.py` excluded symlinks
  from the measured set — an added link passed verification. Links are
  now refused (`symlinks_in_tree`, exit 2), consistent with the
  artifact-closure policy and with `build_zip.py`, which already
  excludes them. Tool caches are skipped consistently (`.ruff_cache`
  joins `.pytest_cache` in the verifier, the reseal enumerator, the
  builder, and `.gitignore`).
* **Snapshot staging cleanup.** `stage_snapshot()` raised a raw
  `KeyError` and left a partially staged, unfrozen tree behind when
  handed an artifact name the manifest did not authorize. Names are
  validated up front (`SnapshotError`, nothing staged) and any staging
  failure discards the destination.

Regression tests: `test_tokenizer_digest_matches_recorded_plan_convention`
(frozen plan convention), `test_manifest_tokenizer_binding_must_match_model_snapshot`,
`test_tokenizer_cross_check_path_mismatch_refused`,
`test_model_snapshot_without_tokenizer_refused`,
`test_cuda_loader_refuses_serving_a_raw_adapter_path` (extended),
`test_snapshot_refuses_unauthorized_artifact_names`,
`test_symlink_in_tree_fails`, `test_symlink_at_governed_path_fails`,
`test_tool_caches_are_skipped`.

## Time-of-check/time-of-use

`minagi/v161/immutable_snapshot.py` stages every artifact with a
single-pass copy-and-verify (`copy_closure` hashes the bytes it
writes), compares the staged digest against the digest the signed
manifest authorized, freezes the tree read-only, and re-measures it
immediately before load. A source that changes after verification
cannot produce a matching snapshot; a snapshot that changes before load
fails `verify_snapshot` and nothing is served. `ApprovedSnapshot` is
constructible only by `stage_snapshot`.

## Verification

| Suite | Coverage |
|---|---|
| `test_v1641_artifact_closure.py` | symlink/special-file/traversal/unexpected/missing/resized/modified refusals; digest compatibility with v16.4.0 semantics; frozen tokenizer convention; single-pass staging; freeze; forged snapshot; unauthorized artifact names |
| `test_v1641_trusted_launcher.py` | launch happy path + production receipt; backend mismatch/load failure; symlink and byte substitution; tokenizer binding/cross-check refusals; stale/missing revocation list; caller-digest disagreement; source mutation during load; nonce uniqueness; rollback lineage; unsigned/wrong-role/non-production receipts; raw-path backend refusal; CUDA loader gates (adapter path and missing snapshot) |
| `test_v1641_strict_schema.py` | unknown versions, absent digests, unsupported backends, unbound evaluation/protocol, expiry ordering, listing validation, production receipt fields; admission-level refusals |
| `test_v1641_cli.py` | `admit_runtime.py` and `trusted_launch.py` end to end: measured artifacts, mandatory signing, stale revocation, symlink/byte refusals, production receipt verification, backend identity |
| `test_v164_runtime_admission.py` (updated) | the v16.4.0 adversarial set still refuses; the CLI now measures physical paths |
| `test_v1622_verify_release_script.py` | verifier failure paths incl. symlink refusal (extra link, governed path replaced by a link) |

Run: `python -m pytest tests-python -q`.

## Build and verification (deterministic)

Dependencies are declared in `pyproject.toml` (`requires-python >= 3.11`;
runtime pins: numpy>=2.0, cryptography>=43, fastapi/uvicorn/pydantic/httpx,
PyYAML; `[ml]`/`[colab]`/`[mac]` extras for torch/transformers/peft).
The release is verified and archived with three commands:

    python -m pytest tests-python -q          # 641 passed, 1 skipped (macOS)
    python -m ruff check .                    # lint contract (pyproject.toml, .flake8)
    python scripts/verify_release.py          # manifest + signature + metadata
    python scripts/release/build_zip.py --out Runtime-Security-Closure.zip

`verify_release.py` checks every governed file against the signed
`SOURCE_MANIFEST.json`, refuses symbolic links anywhere in the governed
tree, verifies the Ed25519 signature against the
pinned release-key fingerprint, and reconciles the attestation, the
manifest, and `VERSION`. `build_zip.py` produces a byte-reproducible
archive (sorted entries, fixed timestamps/permissions) and prints its
sha256. Module placement note: the release plan proposed new
`minagi/security`, `minagi/runtime`, … trees; this release extends the
existing `minagi/v161` package instead, so there is exactly one
closure, one schema registry, and one admission authority — no parallel
implementations that could disagree.

## Residual risks and open items

* The launcher's immutability is process-level defense in depth
  (read-only modes + staged copies). An adversary with the same OS
  identity can chmod and rewrite a snapshot; `verify_snapshot` before
  load detects that, and the receipt binds what was measured.
* The runtime signing key lives in `<storage>/.keys/runtime.pem` in the
  development scaffold. Operational separation (different OS
  identities/hosts per role) remains a deployment requirement, not a
  code property — see `minagi/v161/authority.py`.
* Research-plane evaluation loads are deliberately outside admission
  (`purpose="research"`); they evaluate the candidate artifact, they do
  not serve it. The flag is greppable and auditable, and it is now the
  only way to load without an `ApprovedSnapshot` — serving loads
  without a snapshot are refused.
* `python -m pytest` was executed on macOS (3.12); Linux CPU, Windows,
  and the GPU/PEFT campaign matrices are unchanged by this release and
  are tracked in the remaining-defects register.
