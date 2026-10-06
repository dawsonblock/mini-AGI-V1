# v16.1 Validation Summary

Build: `mini-AGI-v16.1-Colab-Converged-Full-Upgraded`

## Fresh validation performed during assembly

- v15.8 pre-merge baseline: **304/304 Python tests passed**.
- Unified v16.1 tree after convergence: **313/313 Python tests passed**.
- RC14.7 donor: **255/255 Python tests passed**.
- v16.0.1 donor: **293/293 Python tests passed**.
- RC12.5 grounded replay integration test: **7/7 passed**.
- Dream-RSI donor test suite: completed successfully.
- `python -m compileall` over merged Python/source scripts: passed.
- Editable package installation with `--no-build-isolation --no-deps`: passed.
- Release manifest/signature verification: generated and verified after final assembly.

## Post-release empirical validation (2026-10-06, Google Colab T4)

Executed on a clean GPU runtime (Tesla T4, 15.6 GB VRAM, CUDA 13.0,
torch 2.11.0+cu130, Python 3.13.15, 12.7 GB RAM) via `google-colab-cli`:

| Check | Result |
|---|---|
| Release verification (pristine ZIP extraction) | **PASS — 1,381/1,381 files + Ed25519 signature** |
| Environment readiness (`minagi.platforms.colab.doctor`) | **PASS — ready_for_gpu_campaign=true** |
| Unified Python regression suite in Colab | **PASS — 313/313 reproduced** |
| Module import probe (9 groups) | **PASS — 36/36** |
| Tiny real-LoRA smoke campaign (`configs/smoke.yaml`, sshleifer/tiny-gpt2) | **EXECUTED — PASS (plumbing); real gradient updates, adapter digest `sha256:38b00bcd…`, signed A0/A1 receipts** |
| Adversarial boundary checks (artifact corruption, evaluator substitution, dataset leakage) | **PASS — 16/16 rejections fired** |
| Interruption-recovery semantics (COMPLETE/SEED_RESULT rule) | **PASS** |
| Qwen2.5-0.5B LoRA smoke (`configs/qwen_smoke.yaml`) | **EXECUTED — PASS (plumbing); adapter digest `sha256:c25379ed…`** |
| Adapter-effect proof (Phase 9) | **PASS — A1 outputs differ from A0 on 2/2 probes; fresh frozen reload restores A0 exactly; on-disk adapter digest == receipt digest** |
| Independent artifact-only qualification | **Both campaigns QUALIFIED; qualifier agrees with runner** |

### Defect found and repaired during validation

- `pip install -e .[colab]` creates `src-python/*.egg-info/` inside the release
  tree; the unmodified `verify_release.py` then reports EXTRA files and fails.
  `verify_release.py` now excludes `.git`, `.pytest_cache`, `*.egg-info`,
  `__pycache__`, `*.pyc` (build/install artifacts, same class as before).
- Colab images ship `torchao 0.10.0`; `peft>=0.14` hard-requires
  `torchao>=0.16` when torchao is installed. `colab_install.sh` now upgrades
  torchao when present. Smoke campaigns were executed after this fix.

### Honest caveats

- Smoke campaigns validate the *execution/evidence path*, not learning:
  hidden-set scores stayed 0.0→0.0 on a deliberately tiny task; `PASS` means
  plumbing qualification only (smoke thresholds are 0.0 by design).
- Campaign 1 (5 seeds, family-disjoint hidden set, real preregistered dataset)
  and fresh-runtime independent reproduction have **not yet run**.
- This build does not demonstrate RSI or generalized continual learning.

## Trust statement

The release signature is an assembly-integrity signature for this ZIP/source
tree. It is **not** an operator-controlled production promotion key and must
not be treated as scientific or production authority. After post-assembly
source updates the manifest was re-issued and re-signed with the repository
release key (see `RELEASE_ATTESTATION.json` for key fingerprints).
