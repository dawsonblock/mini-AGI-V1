# v4.0.3-r2 — Release Qualification

## The Diagnostics-9 finding

The previous ZIP contained source files absent from
`SOURCE_MANIFEST.json` and files modified after inventory generation —
a *valid Ed25519 signature over an outdated manifest*. Signed source
integrity is only as good as the freeze→inventory→sign→package→verify
chain being one frozen pipeline.

## Executed pipeline (this release — actually run)

| Step | Result | Evidence |
|---|---|---|
| Freeze | committed tree | commit `363d679` on `research/campaign3b-diagnostics` |
| Inventory | 1718 files | `reissue_release.py` over `git ls-files` ∩ disk — manifest `4a9957e551910b54…` |
| Sign | pinned release key | `RELEASE_SIGNATURE.bin` over canonical manifest; bundled `RELEASE_PUBLIC_KEY.pem` fp `sha256:0f9d735812c99a583574a0157697f37b8b3373d34ab05d659638a88c0f5cb0eb` == `REQUIRED_TRUSTED_KEY_FINGERPRINT` |
| Attestation | reconciled | `RELEASE_ATTESTATION.json` reissue_history #86 → manifest `4a9957e5…`, commit `363d679` |
| Package | deterministic zip | `build_zip.py` → `mini-AGI-V1-v4.0.3-r2-Qualification-Closure.zip`, 1722 entries, sha256 `9e43cc0f14ebfb1a169f22f04adbd2392bb31f10418db3e393f8590353977488` |
| Fresh-extract verify | **PASS** | `verify_release.py --root <extracted>` → `status: PASS`, `files_verified: 1718`, fingerprint + manifest match |

The signed release artifacts (`SOURCE_MANIFEST.json`,
`RELEASE_SIGNATURE.bin`, `RELEASE_PUBLIC_KEY.pem`,
`RELEASE_ATTESTATION.json`) are committed with the release commit —
the manifest now covers the v4.0.3-r2 source, not a stale inventory.

## Gate scoreboard

| r2 gate | Result |
|---|---|
| Scientific-gate correctness | **PASS** — completeness ≠ qualification; V4-43 |
| Failed-gate adversarial tests | **PASS** — 10 Diagnostics-9 scenarios |
| Evaluator-owned inference | **PASS** — real Qwen2.5-0.5B-Instruct load + inference + signed execution receipt |
| Required evidence identities | **PASS** — 12 mandatory identities resolved; V4-45 |
| Authorization + key isolation | **PASS (mechanism)** — custody checks implemented; OS-identity separation is deployment policy |
| Statistical-policy enforcement | **PASS** — versioned policy, CI gates, NOT_EVALUABLE; V4-46 |
| Independent holdout custody | **PASS** — sealed evaluator, single-use; V4-47 |
| Full Python regression | **PASS** — 1584 passed, 21 skipped |
| Fresh ZIP source verification | **PASS** — executed, not deferred; V4-48 |

## Honest scope limits

- The release ZIP packages **source** (`results/` excluded by policy —
  evidence archives ship separately). No committed run carries
  execution provenance; committed runs top out at LOCALLY_RECOMPUTED.
- This build is the *infrastructure* release: qualification semantics,
  executor, resolver, statistical policy, sealed custody, signed
  source. It does not claim a scientifically qualified capability
  candidate — that requires v4.1.1's preservation work followed by a
  real evaluator-custody confirmation.
- Evaluation/execution keys are development-held on this host; real
  confirmation requires separate OS identities (declared in
  `evaluator_policy.yaml`, checked by `evaluator_custody`).
