# mini-AGI v16.1 — Colab Converged Governed Continual-Learning Build

This repository is the recommended merged build produced from the strongest components of the supplied v15.8, RC14.7, RC12.5, Dream-RSI, v16.0.1, v15.5, RC14.5, and RC14.3 lineages.

The canonical execution trunk remains v15.8. RC14.7 supplies executed-evidence and reproduction machinery; RC12.5 supplies grounded replay/live-canary semantics; Dream-RSI remains proposal-side; and selected v16 modules supply plasticity/statistical/isolated-solver capabilities. The v16.0.1 tree is not used as the execution trunk.

For Google Colab, start with [README_COLAB.md](README_COLAB.md) and `notebooks/Mini_AGI_v16_1_Colab.ipynb`.

## Local validation

```bash
PYTHONPATH=src-python pytest -q tests-python
python scripts/verify_release.py
```

## Colab smoke campaign

```bash
bash scripts/colab_install.sh
python scripts/verify_release.py
python -m minagi.platforms.colab.doctor
python scripts/run_colab_campaign.py --config configs/smoke.yaml --storage /content/minagi_work
```

The smoke campaign performs a real PEFT LoRA update on a tiny Hugging Face causal model when executed in an internet-enabled CUDA Colab runtime.

## Scientific status

This build implements infrastructure for governed continual-learning experiments. It does not itself prove forward transfer, recursive self-improvement, or AGI. Those claims require real campaigns, preregistered evidence, independent reproduction, and bounded forgetting/security results.
