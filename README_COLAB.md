# mini-AGI v16.1 Colab Converged Build

This archive is a single runnable convergence build assembled on the v15.8 execution trunk. It adds RC14.7 executed-evidence components, RC12.5 grounded replay/live-canary semantics, Dream-RSI proposal-side optimization, selected v16 plasticity/convergence components, and a PyTorch/Hugging Face/PEFT Colab backend.

## What this build is

It is a governed continual-learning research platform. It is **not evidence that recursive self-improvement or general continual intelligence has already been achieved**. The included campaign machinery is designed to produce auditable A0/A1 evidence so those claims can be tested rather than assumed.

## Fastest Colab test

Upload the ZIP to `/content`, then run:

```bash
!unzip -q /content/mini-AGI-v16.1-Colab-Converged-Full-Upgraded.zip -d /content/
%cd /content/mini-AGI-v16.1-Colab-Converged-Full-Upgraded
!bash scripts/colab_install.sh
!python scripts/verify_release.py
!python -m minagi.platforms.colab.doctor --output /content/READINESS_REPORT.json
!python scripts/run_colab_campaign.py --config configs/smoke.yaml --storage /content/minagi_work
```

The smoke campaign uses a tiny Hugging Face causal LM and performs a real LoRA update. A scientific PASS is not required for the smoke test; the purpose is to verify that the complete execution/evidence path works.

## Qwen smoke test

After the tiny smoke campaign completes:

```bash
!python scripts/run_colab_campaign.py --config configs/qwen_smoke.yaml --storage /content/minagi_work
```

The included Qwen smoke configuration uses a small Qwen model so it is practical on ordinary Colab GPUs. For a formal campaign, replace `revision: main` with an immutable Hugging Face commit hash before preregistration.

## Persist to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then use:

```bash
!python scripts/run_colab_campaign.py \
  --config configs/qwen_smoke.yaml \
  --storage /content/drive/MyDrive/mini-AGI-v16.1
```

The storage tree contains content-addressed objects, campaign plans, adapter artifacts, execution receipts, evidence, epochs, ledgers, logs, and exports.

## Important authority boundaries

- Dream-RSI and the plasticity controller are proposal-side mechanisms.
- Evaluators are content-addressed artifacts in the v16.1 path; a human-readable evaluator name is not sufficient identity.
- Dataset partition manifests perform actual sample/family overlap checks rather than trusting an opaque disjointness digest.
- A0 and A1 execution receipts bind campaign, environment, model/tokenizer identity, dataset identity, evaluator identity, and adapter identity.
- RuntimeClosureV161 verifies declared roots against physical files/directories.
- A campaign PASS means experimental qualification only. It is not production promotion authority.

## Formal Campaign 1

`configs/campaign1.yaml` is intentionally not immediately executable as a scientific campaign. Replace the model revision placeholder with an immutable upstream commit and replace the placeholder task file with a real preregistered dataset. This prevents an example configuration from being mistaken for a valid experiment.
