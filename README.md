<div align="center">

# mini-AGI

**A governed continual-learning research platform**

Preregistered, Ed25519-signed experiment plans · evidence-bound qualification ·
a physically closed native inference runtime · adversarially tested admission

[![CI](https://github.com/dawsonblock/mini-AGI-V1/actions/workflows/rc11-integrity.yml/badge.svg)](https://github.com/dawsonblock/mini-AGI-V1/actions/workflows/rc11-integrity.yml)
[![Release](https://img.shields.io/github/v/release/dawsonblock/mini-AGI-V1?include_prereleases&sort=semver&label=release)](https://github.com/dawsonblock/mini-AGI-V1/releases)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.11-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-643%20passed-brightgreen)](#-quality-gates)
[![Signed](https://img.shields.io/badge/release%20integrity-Ed25519%20signed-blueviolet)](scripts/verify_release.py)

</div>

---

> [!IMPORTANT]
> **What mini-AGI is *not*.** This project does **not** claim autonomous
> general intelligence or demonstrated recursive self-improvement. It is
> infrastructure for producing *auditable experimental evidence* about
> those questions — nothing more. A campaign `QUALIFIED` is an
> experimental qualification under a signed plan, never a production
> promotion; promotion requires separate human/operator authority.

## Architecture

```
                          mini-AGI
                            │
   ┌────────────────────────┼─────────────────────────┐
   │                        │                         │
   ▼                        ▼                         ▼
 Native QW3 runtime    Governance plane         Proposal plane
 C++17 · CUDA · Metal  preregistration          (untrusted)
 KVMem · GDN · native  signed receipts          Dream-RSI search
 LoRA · CTest          qualification gates      egai evidence
                       promotion boundary       ingestion
                            │
                            ▼
                     Evidence branches
                results/v16.1-colab-campaign-1
```

Every artifact that crosses a trust boundary is digest-bound and
Ed25519-signed; every evaluator runs inside a fail-closed sandbox;
every release ships a signed manifest verifiable offline.

## Status at a glance

| Item | State |
|---|---|
| Current release | **v16.4.4 — Verified Runtime Closure** (`v16.4.4-runtime-closure`, in flight) · previous: v16.4.3 |
| Release integrity | ✅ **PASS** — `scripts/verify_release.py`: signed manifest, Ed25519 signature, attestation/version reconciliation |
| CI | ✅ **green** — lint (ruff · Flake8 · Pylint error-gate), ubuntu + macOS validation, native CMake/CTest, tag-gated release-integrity |
| Unified Python suite | ✅ **793 passed, 1 skipped** (`tests-python/`) |
| Native CTest | ✅ 28/28 (ubuntu + macOS CPU/stub builds; CUDA paths unverified offline) |
| Runtime admission | ✅ full chain — plan → qualification → promotion → manifest → exact bytes — with signed activation receipts and rollback |
| Evaluator sandbox | ✅ fail-closed — usable / unavailable / misconfigured backends classified explicitly; no unsandboxed fallback |
| Adversarial suites | ✅ pass — altered byte, substituted qualification, revoked replay, research-plane identity, sandbox escape, forged/expired/revoked promotion, rank-budget exhaustion, chain manipulation |
| Campaign 1 — `campaign1-v162` | ❌ **REFUSE** — ΔFT(L6−L5) = +0.20 but retention/security floors tripped on documented calibration artifacts |
| Campaign 1b — `campaign1b-v163` | ✅ **QUALIFIED** — ΔFT = +0.20, 5/5 seeds positive, reproduced bit-for-metric on a second fresh T4 runtime |
| Campaign 2 — `campaign2-v164` | ❌ **REFUSE** — ΔFT = +0.01125 (CI excludes zero, below the ≥0.02 floor); security guardrail breached (−0.125 vs −0.10 bound) |
| Campaign 3A — `campaign3a` | ⏳ **PREREGISTERED + EXECUTING** — halved-optimization-pressure arm, sealed digest-bound holdout, resumable per-seed evidence banking |
| Validated designation | **v16.2** — the v16.3+ promotion gate required Campaign 2 to qualify; it did not |

## What's new in v16.4.4

**Verified runtime closure.** The activation, rollback, routing,
audit, budget and recovery paths are now verified end to end: durable
commit intent precedes any routing publication; deployment generations
CAS-guard every transition and a failed candidate restores the last
committed live predecessor (or durably reports UNAVAILABLE); the router
holds per-activation request leases and unloads only at zero in-flight;
privileged operations land mandatory signed audit records before their
side effect; inference obeys enforceable token/concurrency/deadline
budgets; `--production` refuses to start without signed backend
identity, measured dependency closure, operative policy epoch and fresh
revocation evidence; and `RecoveryManager` restores a cold-started
service through the normal verified path — fresh grant, re-measured
artifacts, isolated load, health probes — rather than trusting a durable
pointer as proof of life.

All seven runtime findings (SEC-301..307) are closed — see
[`RELEASE_CHANGE_MANIFEST.json`](RELEASE_CHANGE_MANIFEST.json).
Controller correctness (SEC-005/006) remains the v16.4.5 gate; real-model
Campaign 3 qualification the v16.5.0 gate.

<details>
<summary><b>v16.4.3 — Hardened Runtime</b></summary>

**Hardened runtime.** Every guarantee the activation architecture

**Hardened runtime.** Every guarantee the activation architecture
claimed is now enforced at the source: durable single-use admission
grants, per-operation role authorization, service-owned storage
paths, a verified hash-chained journal, crash-safe recovery that
never reports a phantom server, backend-identity and policy-epoch
binding, bounded service I/O, and a real serving router — all behind
one transactional authority.

<table>
<tr><td>

**🗄️ One transactional authority**

`authority_store.py` (SQLite, WAL, `BEGIN IMMEDIATE`) owns grant
reservations, the signed hash-chained event log, the active pointer,
and idempotent request outcomes. A consumed grant stays consumed —
across threads, processes, restarts, and crashes.

</td><td>

**🛂 Per-operation authorization**

`PrincipalContext` comes from the authenticated peer uid, never
request JSON. `research.sock` submits proposals; `operator.sock`
runs lifecycle ops; every decision — allowed or refused — is audit
recorded.

</td></tr>
<tr><td>

**📂 Filesystem authority**

Server-generated opaque activation ids; receipts under a
service-owned root; symlinked parents, traversal, and absolute
paths refused at the containment layer.

</td><td>

**🔁 Verified recovery + routing**

Cold start is `UNAVAILABLE` — a durable `ACTIVE` record is evidence
for re-qualification, not a live handle. `ServingRouter` sends
traffic only to the committed+healthy instance; rollback re-checks
the revocation epoch.

</td></tr>
</table>

All ten v16.4.2 findings (SEC-201..207, OPS-001..003) are closed —
evidence in
[`docs/research/RUNTIME_SECURITY_CLOSURE_V1643.md`](docs/research/RUNTIME_SECURITY_CLOSURE_V1643.md)
and the threat model in
[`docs/research/THREAT_MODEL_V1643.md`](docs/research/THREAT_MODEL_V1643.md).
Controller correctness (SEC-005/006) is the v16.4.4 gate; real-model
Campaign 3 qualification is the v16.5.0 gate — see
[`docs/research/UPGRADE_PLAN_V17.md`](docs/research/UPGRADE_PLAN_V17.md).

<details>
<summary><b>v16.4.2 — Authority and Activation Closure</b></summary>

**Authority and activation closure.** Measured files can no longer
become deployable by constructing an object, the production backend
cannot accept unverified paths, a failed launch cannot leave an
unauthorized model serving, and missing backend coverage or invalid
revocations fail closed.

<table>
<tr><td>

**📐 Measurement ≠ authorization**

`MeasuredSnapshot` proves artifact bytes only. The new `admission`
authority issues short-lived, single-use `AdmissionGrantV1`s binding
the promotion decision, measured artifact root, backend, and
revocation epoch — verified independently by the serving supervisor.

</td><td>

**🔁 Transactional activation**

`REQUESTED → AUTHORIZED → STAGED → PREPARED → READY → COMMITTED →
ACTIVE` over a durable journal: intent before routing, atomic pointer
swap, signed completion, guaranteed unload on failure, deterministic
crash recovery, live-handle-only rollback.

</td></tr>
<tr><td>

**🔌 OS-enforced boundary**

`minagi.runtime.service` runs the supervisor + launcher under a
dedicated identity behind an authenticated Unix socket — clients
submit documents and name backends; they never touch signing keys or
loader objects.

</td><td>

**🚫 Strict authorization**

Missing `runtime_backends` coverage is refused outright (hf-peft
never transitively qualifies native backends), and revocation evidence
is now signed, epoch-monotonic, freshness-and-future bounded, and
stored atomically.

</td></tr>
</table>

The full 13-row qualification table from the upgrade spec runs in
`tests-python/v161/test_v1642_activation_closure.py`. Controller
correctness (SEC-005/006) moved to the v16.4.4 gate; real-model
Campaign 3 qualification is the v16.5.0 gate — see
[`docs/research/UPGRADE_PLAN_V17.md`](docs/research/UPGRADE_PLAN_V17.md).

</details>

<table>
<tr><td>

**🔏 Governed training**

Protocol-bound LoRA training semantics — effective batch
(microbatch × accumulation) bound in the *signed* protocol; engine-level
counts assert it; `malformed_policy: fail` invalidates runs on
unapproved rejections.

</td><td>

**🛂 Runtime admission**

`TrustedRuntimeLauncher` makes admission mandatory: physical
measurement, immutable snapshots, snapshot-only loading, signed
production receipts with load-measured digests and replay nonces.
Symlinks, special files, and unlisted files are *refused*, not skipped.

</td></tr>
<tr><td>

**📦 Sandboxed evaluators**

Evaluator execution runs inside a sandbox with explicit backend
classification — usable, unavailable, or misconfigured — and **no path
falls back to running checks unsandboxed**.

</td><td>

**🧭 Mechanism selection**

Controls for adaptive plasticity plus Campaign 3A preregistration with
resumable Colab execution and per-seed evidence banking.

</td></tr>
</table>

**Hardened in this closure:** governed benchmark harness repaired and
test-covered (`VerifiedEpisode` arity, `SandboxSkillMemory.mark_success`
/`health()`, public `EvidenceLedger.head()`); the unwired RC14 donor
class retired (its RC13-era base was never imported); a declared lint
contract (ruff + Flake8 + Pylint error-gate) enforced in CI. Open items:
[`docs/research/REMAINING_DEFECTS_V1641.md`](docs/research/REMAINING_DEFECTS_V1641.md).

## The governed chain

Every campaign runs the same authority-bounded pipeline:

```
 1  PREREGISTER     signed plan binds model revision, digests,
                    dataset partition, evaluator artifacts,
                    seeds, arms, all gates — before hidden eval
        │
 2  EXECUTE         workers run (seed × arm) cells, emit signed
                    receipts; persistent witness key outside VCS;
                    no promotion authority
        │
 3  AGGREGATE       partial matrix ⇒ INCOMPLETE, never PASS
        │
 4  QUALIFY         decision re-derived from artifacts alone;
                    runner and qualifier must agree
        │
 5  PROMOTE         qualified artifacts only, under separate
                    operator authority
```

**Arms:** `L1` frozen · `L2` retrieval · `L3` semantic memory · `L4`
skills · `L5` grounded replay · `L6` neural adapter (LoRA) · `NC`
negative control (label-shuffled LoRA — isolates weight-touching
artifacts from real adaptation).

## Campaigns

| Campaign | Seeds × arms | Decision | Headline result |
|---|---|---|---|
| 1 — `campaign1-v162` | 5 × 6 | **REFUSE** (frozen as evidence) | ΔFT +0.20; retention/security tripped on calibration artifacts |
| 1b — `campaign1b-v163` | 5 × 6 | **QUALIFIED**, reproduced | ΔFT +0.20, 5/5 positive; identical plan digest + metrics on a fresh runtime |
| 2 — `campaign2-v164` | 10 × 7 | **REFUSE** | The +0.20 effect did not transfer to the 13× hidden set; security guardrail breached |
| 3A — `campaign3a` | — | ⏳ executing | Adaptation-footprint experiment, single lever: halved optimization pressure |

<details>
<summary><b>Campaign details</b></summary>

### Campaign 1 — REFUSE (historical evidence)

Six-arm × 5-seed ladder on Qwen2.5-0.5B, 92-task corpus. ΔFT_neural =
+0.20 mean; NC clean. REFUSE stands: the retention scorer could not
grade prose answers and the security floor tripped on baseline marker
echoing — documented calibration artifacts, hardened afterward.

### Campaign 1b — QUALIFIED, reproduced

A *new* preregistered experiment (not a repair of Campaign 1). Plan V163
adds a calibrated containment scorer (frozen 30-case set, agreement
0.933, digest bound into the evaluator artifact), immutable model
revision with model/tokenizer/template digests, and explicit hard
constraints. Result: ΔFT = +0.20, 5/5 seeds positive, all bounds met —
QUALIFIED on Runtime A and independently reproduced on a fresh Runtime B
(adapter bits differ per the preregistered nondeterminism policy).

### Campaign 2 — REFUSE (the scale result)

10 seeds × 7 arms, 464-row corpus, 320 family-disjoint hidden tasks,
20k-resample bootstrap CI, delayed-retention probes on reloaded
adapters, executed across ~9 Colab VM lifetimes via persistent
witness-key resume. ΔFT = +0.01125 — real (CI excludes zero) but below
the preregistered 0.02 floor; L6 breached the security guardrail. v16.2
remains the validated designation.

### Campaign 3 — drafted

Same architecture, corpus difficulty class, and gates as Campaign 2;
single lever — halved optimization pressure (`learning_rate 5e-5`).
Design: [`docs/research/CAMPAIGN3_DESIGN.md`](docs/research/CAMPAIGN3_DESIGN.md).

</details>

## Quick start

### Colab (GPU)

```bash
!unzip -q mini-AGI-v16.1-Colab-Converged-Full-Upgraded.zip -d /content/
%cd /content/mini-AGI-v16.1-Colab-Converged-Full-Upgraded
!bash scripts/colab_install.sh
!python scripts/verify_release.py
!python -m minagi.platforms.colab.doctor --output /content/READINESS_REPORT.json
!python -m pytest tests-python -q
```

Smoke campaign (tiny model, plumbing only — 0.0 thresholds, **not** a
learning claim):

```bash
python3 scripts/run_colab_campaign.py --config configs/smoke.yaml --storage /content/minagi_work
```

Full campaign:

```bash
python3 scripts/run_campaign1.py --config configs/campaign3.yaml --storage /content/campaign3
python3 scripts/validation/qualify_campaign1.py --storage /content/campaign3 --campaign-id campaign3-v165
```

<details>
<summary><b>Lane-parallel execution</b></summary>

Split a campaign across machines by seed; every lane signs receipts
under the shared witness key and emits `INCOMPLETE` alone:

```bash
# each worker runs a strict subset of the preregistered seed list
python3 scripts/run_campaign1.py --config configs/campaign3.yaml \
    --storage /content/campaign3 --execute-seeds 0,1,2

# coordinator: merge all seed dirs, then run once without the flag —
# all seeds verify, complete matrix → real decision
python3 scripts/run_campaign1.py --config configs/campaign3.yaml --storage /content/campaign3
```

Resume rule: completed seed dirs (marker + `SEED_RESULT.json`) are
adopted only within one environment epoch — receipts spanning two
environment digests are refused by the qualifier.

</details>

## Verifying a release

Release archives ship a signed manifest — verifiable offline, no trust
in the transport required:

```bash
unzip Runtime-Security-Closure.zip -d mini-agi-release
cd mini-agi-release
python3 scripts/verify_release.py
# → status: PASS · files_verified: 1455 · Ed25519 signature valid
```

`verify_release.py` checks every governed file against
`SOURCE_MANIFEST.json`, validates `RELEASE_SIGNATURE.bin` against the
bundled `RELEASE_PUBLIC_KEY.pem`, and reconciles attestation, version,
and manifest identities — any drift fails the check.

## Repository layout

| Path | Role |
|---|---|
| `src/`, `include/` | Native QW3/KVMem/GDN inference runtime (C++17, CUDA, Metal) |
| `src-python/minagi/v161/` | Campaign plans, dataset manifests, evaluator registry, signed receipts, bootstrap stats, runtime admission |
| `src-python/minagi/runtime/` | Transactional activation supervisor, durable journal, AF_UNIX supervised launch service |
| `src-python/minagi/security/` | Admission grants, signed revocation snapshots, authority client |
| `src-python/minagi/v14` · `v15` · `v16` | StateEpoch governance, physical runtime closure, donor concepts |
| `src-python/minagi/platforms/` | Colab/CUDA environment probe, CAS storage, HF runtime, PEFT trainer |
| `src-python/minagi/rc14/` | Qualification engine, falsification, authority artifacts (drifted donor class retired in v16.4.1) |
| `src-python/egai/` · `dream_rsi_governed/` · `kvcontinual/` | Proposal-side search, evidence infrastructure, continual-KV research |
| `scripts/` | Campaign runner, validation harness, release tooling |
| `configs/` | Campaign plans + deterministic task corpora (`*_tasks.jsonl`) |
| `tests-python/` | Unified Python suite — 711 tests |
| `docs/` · `docs/research/` | Architecture, protocols, campaign designs, defect register |

## Quality gates

| Gate | What it enforces |
|---|---|
| `pytest tests-python` | 711 tests — campaigns, governance, sandboxing, admission, activation lifecycle, evidence chain |
| `bash scripts/rc11/lint.sh` | ruff + Flake8 at zero findings; Pylint error-category gate at zero findings |
| `cmake --build` + `ctest` | Native runtime build and CTest on ubuntu + macOS |
| `verify_release.py` | Signed-manifest integrity — runs in CI on every `v*` tag |

All four run in [CI](.github/workflows/rc11-integrity.yml) on every
push and pull request; the release-integrity job is tag-gated.

## Known limitations

- **Evidence scope.** The governance substrate is validated for *this*
  experimental workflow, not generally proven. Campaign 1b establishes
  incremental forward transfer on one model/corpus/evaluator
  configuration (~5 extra correct hidden answers/seed on 24 tasks) —
  a mechanism demonstration, not a capability claim.
- **Security metric is a marker-regression smoke check.** A
  preregistered adversarial evaluator is required before security
  numbers carry weight.
- **Adapter ↔ native-serving gap.** Campaigns train LoRA on
  `q/k/v/o_proj`; `NativeAdapter1` deliberately accepts LM-head targets
  only — the qualified PEFT artifact is not yet the artifact the native
  runtime serves.
- **Single substrate.** One model (Qwen2.5-0.5B), one GPU class (T4),
  self-generated corpora — hidden sets are held out, not externally
  administered.
- **Resume is epoch-scoped.** A runtime restart produces a new
  environment digest; cross-epoch resume is refused by design.

## Documentation

| Document | Contents |
|---|---|
| [`docs/research/RUNTIME_SECURITY_CLOSURE_V1643.md`](docs/research/RUNTIME_SECURITY_CLOSURE_V1643.md) | v16.4.3 hardened-runtime design, evidence, and gate status |
| [`docs/research/THREAT_MODEL_V1643.md`](docs/research/THREAT_MODEL_V1643.md) | Trust boundaries, adversaries, and fail-closed invariants |
| [`docs/research/RUNTIME_SECURITY_CLOSURE_V1642.md`](docs/research/RUNTIME_SECURITY_CLOSURE_V1642.md) | v16.4.2 authority + activation-closure design and evidence |
| [`docs/research/UPGRADE_PLAN_V17.md`](docs/research/UPGRADE_PLAN_V17.md) | The v16.4.1 → v17.0 engineering roadmap and release gates |
| [`docs/research/RUNTIME_SECURITY_CLOSURE_V1641.md`](docs/research/RUNTIME_SECURITY_CLOSURE_V1641.md) | v16.4.1 security-closure design and evidence |
| [`docs/research/TRAINING_AND_ADMISSION_V164.md`](docs/research/TRAINING_AND_ADMISSION_V164.md) | v16.4 governed training + admission protocol |
| [`docs/research/REMAINING_DEFECTS_V1641.md`](docs/research/REMAINING_DEFECTS_V1641.md) | Living defect register |
| [`docs/research/SECURITY_REPAIR_V1622.md`](docs/research/SECURITY_REPAIR_V1622.md) | FIX-001..006 adversarial repair record |
| [`docs/research/CAMPAIGN3_DESIGN.md`](docs/research/CAMPAIGN3_DESIGN.md) | Campaign 3 preregistration design |
| [`CHANGELOG.md`](CHANGELOG.md) | Full release history |
| [`ALPHA5_ARCHITECTURE.md`](ALPHA5_ARCHITECTURE.md) · [`docs/V14_ARCHITECTURE.md`](docs/V14_ARCHITECTURE.md) | Architecture overviews |

## License

Apache-2.0 — see [`LICENSE`](LICENSE),
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and
[`LICENSES/`](LICENSES/).

<div align="center">

*mini-AGI produces evidence, not claims. Every number above is backed by
a signed artifact you can re-verify.*

</div>
