# Full Upgraded Build Status — v16.1

Status: **assembled and CPU-validated; Colab CUDA smoke pending**.

The merged Python regression suite passes 313/313. Donor scientific/control suites were rerun successfully. Packaging and release integrity are verified by `scripts/verify_release.py`.

The one intentionally open validation item is the real neural Colab smoke campaign because the assembly environment has no CUDA GPU and no outbound internet access. Run `configs/smoke.yaml` in Google Colab to close that item.
