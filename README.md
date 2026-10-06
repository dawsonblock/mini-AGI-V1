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
| Campaign 1 (5-seed, family-disjoint) | **NOT RUN** — `configs/campaign1.yaml` ships with placeholders by design |
| Independent reproduction (fresh runtime B) | **NOT RUN** — `REPRODUCTION_RECORD.json` marked NOT_RUN |
| Six-arm ablation (L1–L6) | **NOT RUN** — requires v16.2 arm-schema extension |

"PASS (plumbing)" means the execution/evidence path ran end-to-end on real
hardware — it is *not* a learning claim. Smoke thresholds are 0.0 by design.

## Evidence

Experimental evidence is kept off `main` on a dedicated branch:

- **Branch:** [`results/v16.1-colab-campaign-1`](../../tree/results/v16.1-colab-campaign-1)
- **Source commit (Commit A):** `db3f6e5bccc6dd8b76982015b684e54084776542`
- **Results commit (Commit B):** `364301035acaaac2557b3369cc577f904663c9c9`
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

## Reproducing Campaign 1

1. Replace `REPLACE_WITH_IMMUTABLE_HUGGINGFACE_COMMIT` in
   `configs/campaign1.yaml` with a pinned HF commit and provide a real
   preregistered dataset (`configs/campaign1_tasks.jsonl` is a placeholder).
2. Run `run_colab_campaign.py --config configs/campaign1.yaml` on a GPU
   runtime; then `scripts/validation/qualify_campaign.py` for independent
   artifact-only qualification.
3. Reproduction means a *fresh* runtime receiving only the release, plan,
   CAS artifacts, and dataset manifests — no inherited mutable state.

## Known limitations

- Single-seed smoke evidence; smoke datasets are toy copy tasks.
- `ExecutedRunReceiptV161.arm` supports `{A0,A1}` only — the six-arm ladder
  (L1–L6) is v16.2 schema work.
- `revision: main` in smoke configs is mutable upstream; formal campaigns
  must pin immutable commits.
- No HF token configured during validation runs (anonymous downloads).

## License

See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
