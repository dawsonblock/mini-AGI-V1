# Campaign Report — v16.1 Colab Validation (2026-10-06)

## Environment

Clean Google Colab GPU runtime provisioned via `google-colab-cli`:

| Property | Value |
|---|---|
| GPU | Tesla T4 |
| VRAM | 15,637,086,208 B (~14.6 GiB) |
| CUDA | 13.0 |
| PyTorch | 2.11.0+cu130 |
| Python | 3.13.15 |
| RAM | 13,605,830,656 B (~12.7 GiB) |
| Free disk | ~193 GiB |
| Platform | Linux-6.6.122+ x86_64 |

## Executed steps and actual results

| Phase | Command / artifact | Result |
|---|---|---|
| Release integrity | `scripts/verify_release.py` on pristine ZIP extraction | **PASS — 1,381/1,381 files, Ed25519 signature valid** |
| Environment gate | `python -m minagi.platforms.colab.doctor` | **PASS — ready_for_gpu_campaign=true** |
| Regression suite | `pytest tests-python -q` | **PASS — 313 passed in 12.26s** (matches reference) |
| Import probe | `scripts/validation/import_probe.py` | **PASS — 36/36 modules** |
| Tiny smoke | `run_colab_campaign.py --config configs/smoke.yaml` | **EXECUTED — PASS (plumbing)** |
| Adversarial checks | `scripts/validation/adversarial_checks.py` | **PASS — 16/16 rejections** |
| Interruption audit | `scripts/validation/interruption_recovery.py` | **PASS** |
| Qwen smoke | `run_colab_campaign.py --config configs/qwen_smoke.yaml` | **EXECUTED — PASS (plumbing)** |
| Adapter effect | `scripts/validation/adapter_effect_check.py` | **PASS — A1≠A0 (2/2), removal restores A0 exactly** |
| Qualification | `scripts/validation/qualify_campaign.py` | **both campaigns QUALIFIED, runner agreement** |

## Real-neural path evidence (per campaign)

Both campaigns executed the full path: frozen HF model → A0 evaluation →
frozen dataset partitions → PEFT LoRA gradient training →
`adapter_model.safetensors` → SHA-256 dir digest → in-memory model destroyed →
fresh base reload → adapter reload → A1 evaluation → signed receipts →
qualification.

| Campaign | Model | LoRA | A0→A1 delta | Adapter digest |
|---|---|---|---|---|
| colab-smoke-v161 | sshleifer/tiny-gpt2 (rev `main`) | r=4 c_attn, 8 steps | 0.0→0.0 | `sha256:38b00bcd…` |
| qwen-smoke-v161 | Qwen/Qwen2.5-0.5B-Instruct (rev `main`) | r=8 q_proj,v_proj, 20 steps | 0.0→0.0 | `sha256:c25379ed…` |

A0/A1 receipts are Ed25519-signed by a per-campaign execution-witness key
(`EXECUTION_PUBLIC_KEY.bin` under `runs/<campaign>/`) and bind campaign plan,
environment, model, tokenizer, adapter, dataset, and evaluator digests.

## Adapter-effect proof (Qwen, seed 0)

- Fresh frozen reload reproduces recorded A0 outputs exactly (2/2).
- Loading the qualified adapter changes generated text on 2/2 probes.
- Adapter reload reproduces recorded A1 outputs exactly (2/2).
- Fresh frozen reload after adapter removal restores A0 exactly (2/2).
- On-disk adapter digest equals the digest bound in the A1 receipt.

## Adversarial results (all rejected)

adapter byte flip, adapter config tamper, missing adapter, model-revision
substitution, tokenizer-identity substitution, retrieval-policy tamper,
evaluator-record tamper, campaign-plan tamper, scorer substitution,
unregistered evaluator, forged registry entry, tampered evaluator source,
hidden-sample leakage, hidden family leakage (when required),
train/validation overlap. One documented acceptance: family overlap is
admitted only when `require_family_disjoint_hidden=false` (smoke posture).

## Defects found during validation (fixed at source commit)

1. `pip install -e` creates `*.egg-info/` inside the tree → release
   verification reported extras. `verify_release.py` now excludes
   `.git`/`.pytest_cache`/`*.egg-info`/`__pycache__`/`*.pyc`.
2. Colab image ships `torchao 0.10.0`; `peft>=0.14` requires `>=0.16` when
   present → `colab_install.sh` now upgrades torchao. Smoke campaigns ran
   after this fix.

## Honest limitations

- Hidden-set acquisition stayed 0.0 → 0.0: smoke proves the pipeline, not
  learning. `PASS`/`QUALIFIED` here means plumbing qualification only.
- Single seed per campaign (smoke scope). Campaign 1 needs N=5 preregistered
  seeds, a real family-disjoint dataset, and a pinned HF commit revision
  (`configs/campaign1.yaml` ships with placeholders by design).
- Fresh-runtime independent reproduction: **REPRODUCED** on a second clean
  Colab T4 session using only the public GitHub release asset — release
  verify 1,389/1,389, tests 313/313, plan digest and metrics identical,
  independent qualifier QUALIFIED. Adapter bits diverged across runtimes
  (GPU nondeterminism — expected; each adapter is digest-bound per run).
  See reproduction/REPRODUCTION_RECORD.json.
- No six-arm ablation (L1–L6) yet; `ExecutedRunReceiptV161.arm` is currently
  restricted to `{A0,A1}` — extending it is a schema change for v16.2.
- The Colab session Google Drive mount required interactive OAuth and was not
  completed during the headless run; evidence was pulled via `colab download`
  and lives on this branch.
