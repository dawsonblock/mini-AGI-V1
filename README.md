# mini-AGI v16.1 — Converged Governed Continual-Learning Build

mini-AGI v16.1 is an experimental governed continual-learning research
platform. It integrates governed neural adaptation, evidence-backed
qualification, grounded replay, runtime artifact closure, proposal-side
Dream-RSI search, and a Google Colab PyTorch/PEFT execution backend.

**This repository does not claim autonomous general intelligence or
demonstrated recursive self-improvement.** It is infrastructure for producing
auditable A0/A1 evidence about those questions — nothing more.

## Validation status (actual, 2026-10-06)

| Check | Status |
|---|---|
| Release integrity (`scripts/verify_release.py`) | **PASS** — 1,389 files + Ed25519 signature (clean extraction) |
| Unified Python regression suite | **PASS** — 313/313 (macOS arm64 py3.12 **and** Colab py3.13) |
| Clean Colab install | **PASS** — `scripts/colab_install.sh` on fresh T4 runtime |
| CUDA runtime detection | **PASS** — Tesla T4, CUDA 13.0, torch 2.11.0+cu130 |
| Module import probe | **PASS** — 36/36 |
| Tiny-model real LoRA smoke | **PASS (plumbing)** — real gradient updates, signed receipts |
| Qwen2.5-0.5B LoRA smoke | **PASS (plumbing)** — adapter digest `sha256:c25379ed…` |
| Adapter genuinely affects inference | **PASS** — A1≠A0 on 2/2 probes; adapter removal restores A0 exactly |
| Artifact-substitution rejection | **PASS** — corruption of adapter/config/plan/policy rejected |
| Evaluator-substitution rejection | **PASS** — unregistered/forged/tampered evaluators rejected |
| Dataset leakage rejection | **PASS** — sample + task-family leakage rejected |
| A0/A1 artifact reload | **PASS** — physical digest binding verified at reload |
| Independent artifact-only qualification | **PASS** — both smoke campaigns QUALIFIED, runner agrees |
| Campaign 1 — six-arm × 5-seed (v16.2) | **EXECUTED — REFUSE** — ΔFT_neural (L6−L5) = **+0.20** mean (0.125–0.29/seed), negative control clean (0.0), but preregistered retention/security floors not met; see `results/campaign-1/campaign1-v162/` |
| Campaign 1 fresh-runtime reproduction | **PASS** — fresh T4 cloned public `main@ae937a9` only: plan digest, all arm metrics, ΔFT_neural (+0.20), NC (0.0), and REFUSE decision reproduced exactly; adapter bits differ per runtime (policy-compliant) |
| Six-arm ablation (L1–L6 + NC) | **EXECUTED** — see Campaign 1 row |

"PASS (plumbing)" means the execution/evidence path ran end-to-end on real
hardware — it is *not* a learning claim. Smoke thresholds are 0.0 by design.

## Evidence

Experimental evidence is kept off `main` on a dedicated branch:

- **Branch:** [`results/v16.1-colab-campaign-1`](../../tree/results/v16.1-colab-campaign-1)
- **Source commit (Commit A):** `db3f6e5bccc6dd8b76982015b684e54084776542`
- **Results commits:** `364301035acaaac2557b3369cc577f904663c9c9` (Runtime A evidence), `7ebc488189a01727ec67ac7ba50d7d017ef5a3da` (Runtime B reproduction), `0ecee0bdd41be9a8aecef3c425c456df5c57b86d` (Campaign 1 six-arm evidence)
- **Release asset:** [`v16.1.0-colab`](../../releases/tag/v16.1.0-colab) — signed release ZIP + validation pack
- **Bundle:** `results/campaign-1/` — plans, signed run receipts, adapters
  (digest-sealed), qualification records, adversarial results, campaign report

## Architecture sketch

- `src-python/minagi/v161/` — campaign plans, dataset membership manifests,
  evaluator registry, signed executed-run receipts, runtime closure.
- `src-python/minagi/platforms/{colab,cuda}/` — environment probe, Colab
  storage (CAS, campaigns, adapters, evidence), HF runtime, PEFT trainer.
- `src-python/minagi/rc14/` — qualification engine, falsification,
  independent reproduction, authority artifacts (symbolic stack).
- `src-python/egai/`, `src-python/dream_rsi_governed/` — evidence, replay,
  authority, proposal-side search.
- `scripts/validation/` — v16.2 validation harness: import probe,
  adversarial checks, interruption recovery, adapter-effect check,
  artifact-only qualification, release re-issue tool.

Authority boundary: evaluation produces evidence; promotion requires separate
human/operator authority. A campaign `PASS`/`QUALIFIED` is never a production
promotion.

## Colab quick start

```bash
# inside a GPU runtime, with the release ZIP at /content
!unzip -q /content/mini-AGI-v16.1-Colab-Converged-Full-Upgraded.zip -d /content/
%cd /content/mini-AGI-v16.1-Colab-Converged-Full-Upgraded
!bash scripts/colab_install.sh
!python scripts/verify_release.py
!python -m minagi.platforms.colab.doctor --output /content/READINESS_REPORT.json
!python -m pytest tests-python -q
!python scripts/run_colab_campaign.py --config configs/smoke.yaml --storage /content/minagi_work
```

Headless alternative: this validation was driven by `google-colab-cli`
(`colab new --gpu T4`, `colab upload`, `colab exec`); see
`notebooks/Mini_AGI_v16_2_Validation.ipynb` for the gated interactive version.

## Campaign 1 (v16.2)

`scripts/run_campaign1.py` executes the preregistered six-arm ladder —
L1 frozen, L2 retrieval, L3 semantic-memory, L4 skills, L5
grounded-replay, L6 neural-adapter, plus NC negative control
(label-shuffled LoRA) — across 5 seeds. The plan is Ed25519-signed
before any hidden evaluation; per-(seed, arm) receipts carry adapter
and arm-state digests; `scripts/validation/qualify_campaign1.py`
re-derives the decision from artifacts alone.

The preregistered GPU nondeterminism policy defines reproduction as
*statistical equivalence* within `metric_tolerance` (0.05): adapter
bits need not match across runtimes; every adapter carries its own
digest and qualification lineage.

```bash
python3 scripts/run_campaign1.py --config configs/campaign1.yaml --storage /content/campaign1
python3 scripts/validation/qualify_campaign1.py --storage /content/campaign1 --campaign-id campaign1-v162
```

**Observed result (Runtime A, T4):** ΔFT_neural = **+0.20** mean over 5
seeds; NC = 0.0 everywhere; campaign **REFUSE** — the retention scorer
(exact match) cannot grade prose answers from an instruct model, and
the zero-security-regression floor trips on baseline marker echoing.
Both are documented calibration artifacts, not adaptation failures;
see `CAMPAIGN1_REPORT.md` on the results branch.

## Known limitations

- Retention scorer (strict exact-match) cannot measure forgetting for
  prose-answer models; needs a normalized scorer before retention
  claims can be made either way.
- Security floor (zero literal-marker echo) is uncalibrated for
  untuned instruct models; L1 itself flags ~1 lure/seed.
- Six-arm effect size is small in absolute terms (~5 extra correct
  hidden answers/seed on 24 tasks) — a mechanism demonstration, not a
  capability claim.
- Single model (Qwen2.5-0.5B), single GPU class (T4), one corpus.
- Smoke evidence remains single-seed; smoke datasets are toy copy tasks.

## License

See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
