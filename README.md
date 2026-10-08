# mini-AGI v16.2

**A governed continual-learning research platform** — preregistered,
Ed25519-signed experiment plans; evidence-bound qualification; and a
physically closed native inference runtime.

mini-AGI does **not** claim autonomous general intelligence or
demonstrated recursive self-improvement. It is infrastructure for
producing auditable experimental evidence about those questions —
nothing more.

```
                mini-AGI
                  │
   ┌──────────────┼──────────────────┐
   │              │                  │
 Native QW3    Governance &      Proposal plane
 runtime       experiment layer  (untrusted)
 C++/CUDA/     plans, receipts,  Dream-RSI search
 Metal, KVMem, qualifiers,      egai evidence
 GDN, native   promotion gates  ingestion
 LoRA
                  │
              Evidence branch
        results/v16.1-colab-campaign-1
```

## Status at a glance

| Item | State |
|---|---|
| Current release | **v16.4.0** — training-schedule repair, runtime admission, mechanism controller (Phases 2/3/5 of the v17 plan); scientific claims unchanged |
| Release integrity (`scripts/verify_release.py`) | **PASS** — 1,439 governed files + Ed25519 signature + attestation/version reconciliation |
| Unified Python suite (`tests-python/`) | **PASS** — 571 passed (1 skipped: Linux-only RLIMIT_AS test on the macOS host) |
| Training schedule (Phase 2.1) | **FIXED + TESTED** — effective batch = microbatch × accumulation bound in the signed protocol; engine-level counts assert it (the original ran 4× the declared sample presentations for accumulation > 1); strict `malformed_policy: fail` invalidates runs on unapproved rejections |
| Runtime admission (Phase 3) | **IMPLEMENTED + TESTED** — full-chain admission (plan → qualification → promotion → manifest → exact bytes) with signed activation receipts and rollback; the plan's four named attacks (altered byte, substituted qualification, revoked replay, research-plane identity) are refused by an adversarial suite |
| Security repair (FIX-001..006) | **IMPLEMENTED + TESTED** — reproducing test per defect, adversarial suites (sandbox escape, forged/expired/revoked promotion, rank-budget exhaustion, chain manipulation); residual risks in `docs/research/SECURITY_REPAIR_V1622.md` |
| Native CTest suite | **PASS** — 28/28 rebuilt and rerun for v16.4.0 (macOS CPU/stub build; CUDA paths unverified offline) |
| Adversarial boundary checks | **PASS** — 16/16 rejections (artifact/evaluator/dataset substitution) |
| Campaign 1 — `campaign1-v162`, 5 seeds | **EXECUTED — REFUSE** — ΔFT(L6−L5) = +0.20 but retention/security floors tripped on documented calibration artifacts |
| Campaign 1b — `campaign1b-v163`, 5 seeds | **EXECUTED — QUALIFIED** — ΔFT = +0.20, 5/5 positive, reproduced bit-for-metric on a second fresh T4 runtime |
| Campaign 2 — `campaign2-v164`, 10 seeds | **EXECUTED — REFUSE** — ΔFT = +0.01125 (95% CI [+0.00531, +0.01719] excludes zero but fails the ≥0.02 floor); security guardrail breached (−0.125 vs bound −0.10) |
| Campaign 3 — `campaign3-v165` | **3A PREREGISTERED + EXECUTING** — adaptation-footprint experiment (single lever: halved optimization pressure, identical gates); sealed holdout digest-bound in `configs/campaign3a.yaml`; running on Colab T4 under the signed plan with per-seed evidence banking; qualification pending |
| Validated designation | **v16.2** — the v16.3 promotion gate required Campaign 2 to qualify; it did not |

A campaign `QUALIFIED` is an *experimental qualification* under a
signed plan — it is never a production promotion. Promotion requires
separate human/operator authority.

## The governed chain

Every campaign runs the same authority-bounded pipeline:

1. **Preregister** — a `ColabCampaignPlanV16x` binds the immutable
   model revision, model/tokenizer/template digests, dataset partition
   digest (family-disjoint hidden split verified), evaluator artifact
   digests, seeds, arms, and all gates — then is Ed25519-signed before
   any hidden evaluation.
2. **Execute** — workers run (seed × arm) cells and emit signed
   receipts carrying adapter digests, arm-state digests, and the
   environment digest. A persistent execution-witness key lives at
   `<storage>/.keys/` — outside the campaign directory, outside VCS.
   Workers have no promotion authority.
3. **Aggregate** — the runner refuses to produce a decision on a
   partial matrix: missing planned cells ⇒ `INCOMPLETE`, never `PASS`.
4. **Qualify** — `scripts/validation/qualify_campaign1.py` re-derives
   the decision **from artifacts alone**: plan signature, every receipt
   signature + digest, environment consistency, complete matrix, all
   gates. Runner and qualifier must agree.
5. **Promote** — only a qualified artifact may cross the promotion
   boundary, and only under separate operator authority.

Arms: `L1` frozen · `L2` retrieval · `L3` semantic memory · `L4`
skills · `L5` grounded replay · `L6` neural adapter (LoRA) · `NC`
negative control (label-shuffled LoRA — isolates weight-touching
artifacts from real adaptation).

## Campaigns

### Campaign 1 — `campaign1-v162` (REFUSE, frozen as historical evidence)

Six-arm × 5-seed ladder on Qwen2.5-0.5B, 92-task corpus. ΔFT_neural =
+0.20 mean; NC clean. REFUSE stands: the retention scorer could not
grade prose answers and the security floor tripped on baseline marker
echoing — documented calibration artifacts, hardened afterward.

### Campaign 1b — `campaign1b-v163` (QUALIFIED, reproduced)

New preregistered experiment (not a repair of Campaign 1). Plan V163
adds a calibrated containment scorer (frozen 30-case set, agreement
0.933, digest bound into the evaluator artifact), immutable model
revision with model/tokenizer/template digests, and explicit hard
constraints (`retention_max_drop`, security dual rule,
`min_seeds_positive_ft`). Result: ΔFT = +0.20, 5/5 seeds positive,
retention/security/NC all in bounds — **QUALIFIED** on Runtime A and
independently reproduced on a fresh Runtime B (identical plan digest
and metrics; adapter bits differ per the preregistered
nondeterminism policy).

### Campaign 2 — `campaign2-v164` (REFUSE — the scale result)

10 seeds × 7 arms, 464-row corpus, 320 family-disjoint hidden tasks,
20k-resample bootstrap CI, delayed-retention probes on reloaded
adapters, executed across ~9 Colab VM lifetimes via persistent
witness-key resume.

The +0.20 Campaign-1b effect **did not transfer** to the 13× larger
hidden set: ΔFT = +0.01125 — real (CI excludes zero) but below the
preregistered 0.02 minimum — and the L6 adapter breached the security
guardrail (−0.125 vs L1). v16.2 remains the validated designation.
Full record: `results/campaign-2/` on the evidence branch.

### Campaign 3 — `campaign3-v165` (drafted)

Adaptation-footprint decision experiment: same architecture, corpus
difficulty class, and gates as Campaign 2; single lever — halved
optimization pressure (`learning_rate 5e-5`). Tests whether the
footprint→(forward transfer, security) tradeoff can be moved without
moving goalposts. Design: `docs/research/CAMPAIGN3_DESIGN.md`.

## Running it

Colab GPU runtime:

```bash
!unzip -q mini-AGI-v16.1-Colab-Converged-Full-Upgraded.zip -d /content/
%cd /content/mini-AGI-v16.1-Colab-Converged-Full-Upgraded
!bash scripts/colab_install.sh
!python scripts/verify_release.py
!python -m minagi.platforms.colab.doctor --output /content/READINESS_REPORT.json
!python -m pytest tests-python -q
```

Smoke campaign (tiny model, plumbing only — 0.0 thresholds, not a
learning claim):

```bash
python3 scripts/run_colab_campaign.py --config configs/smoke.yaml --storage /content/minagi_work
```

Full campaign:

```bash
python3 scripts/run_campaign1.py --config configs/campaign3.yaml --storage /content/campaign3
python3 scripts/validation/qualify_campaign1.py --storage /content/campaign3 --campaign-id campaign3-v165
```

### Lane-parallel execution

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

## Repository layout

| Path | Role |
|---|---|
| `src/` | Native QW3/KVMem/GDN inference runtime (C++17, CUDA, Metal) |
| `src-python/minagi/v161/` | Campaign plans, dataset manifests, evaluator registry, signed receipts, bootstrap stats, runtime closure |
| `src-python/minagi/v14`, `v15`, `v16` | StateEpoch governance, physical runtime closure, donor concepts |
| `src-python/minagi/platforms/` | Colab/CUDA environment probe, CAS storage, HF runtime, PEFT trainer |
| `src-python/minagi/rc14/` | Qualification engine, falsification, authority artifacts (symbolic) |
| `src-python/dream_rsi_governed/`, `egai/` | Proposal-side search and legacy evidence infrastructure |
| `scripts/` | Campaign runner, validation harness, release tooling |
| `configs/` | Campaign plans + deterministic task corpora (`*_tasks.jsonl`) |
| `tests-python/` | Unified Python suite (335 tests) |

## Known limitations

- **Evidence scope.** Governance substrate validated for *this*
  experimental workflow, not generally proven. Campaign 1b establishes
  incremental forward transfer on one model/corpus/evaluator
  configuration (~5 extra correct hidden answers/seed on 24 tasks) —
  a mechanism demonstration, not a capability claim.
- **Security metric is a marker-regression smoke check.** The default
  evaluator detects explicit `<unsafe>`-style markers; a preregistered
  adversarial evaluator is required before security numbers carry
  weight.
- **Adapter ↔ native-serving gap.** Campaigns train LoRA on
  `q/k/v/o_proj`; `NativeAdapter1` deliberately accepts LM-head targets
  only — the qualified PEFT artifact is not yet the artifact the native
  runtime serves. Closing this loop is the primary open work.
- **Single substrate.** One model (Qwen2.5-0.5B), one GPU class (T4),
  self-generated corpora — hidden sets are held out, not externally
  administered.
- **Resume is epoch-scoped.** A runtime restart produces a new
  environment digest; cross-epoch resume is refused by design.
- **Metadata drift (repaired in v16.2.2).** Version identities are
  reconciled across VERSION, pyproject, package, SBOM and validation
  docs; `scripts/verify_release.py` fails on
  attestation/manifest/version drift and the reconciliation is
  regression-tested. The signed manifest remains authoritative.

## License

`LICENSE`, `THIRD_PARTY_NOTICES.md`, `LICENSES/`.
