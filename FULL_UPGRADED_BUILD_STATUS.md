# Full Upgraded Build Status — v16.1

Status: **assembled, CPU-validated, and Colab-executed** (2026-10-06).

The merged Python regression suite passes 313/313 locally and reproduces
313/313 on a clean Colab runtime. Release integrity verified at 1,381/1,381
manifest files plus Ed25519 signature on a pristine extraction.

The previously open validation item — the real neural Colab smoke campaign —
is now closed at smoke scope: both `configs/smoke.yaml` (tiny-gpt2) and
`configs/qwen_smoke.yaml` (Qwen2.5-0.5B-Instruct) executed the real
A0 → LoRA → serialize → destroy → reload → A1 path on a Tesla T4 with signed
execution receipts, and the qualified adapter was shown to alter and then
un-alter inference (A0 → A1 → A0′). See `VALIDATION_SUMMARY.md` and the
`results/v16.1-colab-campaign-1` branch for the evidence bundle.

Campaign 1 (multi-seed, family-disjoint hidden evaluation) and fresh-runtime
independent reproduction remain open and are the next gated work items.
