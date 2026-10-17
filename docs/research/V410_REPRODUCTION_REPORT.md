# V4.1.0 — Campaign 3E 640-step reproduction report

Phase 2.3 of the recovery plan: rerun `v401-3e-8fam-e8-s0` under its
recorded contract and independently requalify the fresh evidence.

## Protocol

`reproduce_campaign3e.py` derived the command entirely from the
committed run's `CONFIG.json` (no hand-transcribed hyperparameters).
Every bound field — model id+revision, corpus, seed, rank, learning
rate, exposure, steps, scorer, prompt/output contract digests — was
checked before launch; the script reported `unbound: []` and labelled
the attempt "exact-condition reproduction".

```
python scripts/reproduce_campaign3e.py \
    --from-run results/campaign3e/runs/v401-3e-8fam-e8-s0 \
    --execute --run-id campaign3e-reproduce-s0 --requalify
```

which ran:

```
python scripts/positive_control_fit.py \
    --corpus configs/campaign3e_tasks.jsonl \
    --config configs/campaign3b_positive_control.yaml \
    --families tr3-count-letter,tr3-mirror,tr3-second-char,tr3-strip-first,\
tr3-suffix-ify,tr3-swapcase,tr3-triple,tr3-vowel-echo \
    --seed 0 --rank 16 --learning-rate 0.0002 \
    --run-root results/campaign3e/reproduction \
    --run-id campaign3e-reproduce-s0 --exposure 8
```

## Result — bit-exact reproduction

| quantity | historical | reproduction | match |
|---|---|---|---|
| frozen within-family validation | 0.20833 (5/24) | 0.20833 (5/24) | identical |
| adapted within-family validation | 0.54167 (13/24) | 0.54167 (13/24) | identical |
| frozen hidden | 0.02083 | 0.02083 | identical |
| adapted hidden | 0.02083 | 0.02083 | identical |
| adapted compositional | 0.0625 | 0.0625 | identical |
| fit-gate subset accuracy | 0.9875 | 0.9875 | identical |
| capability churn | 4 broken / 4 gained | 4 broken / 4 gained | identical |
| adapter safetensors sha256 | `da86da82…f73f` | `da86da82…f73f` | **byte-identical** |

`METRICS.json` `frozen`, `adapted`, `fit_gate` and
`capability_destruction` blocks are equal field-for-field, and the
trained adapter hashes to the same digest as the historical artifact —
training is deterministic under this contract on this host (MPS,
torch 2.10.0, transformers 4.57.6, peft 0.20.0, numpy 2.4.6;
train_seconds 330.9).

## Independent requalification

`requalify_executed_campaign.py` over the fresh run:

| run | commit | integrity | rescore vs reported | status |
|---|---|---|---|---|
| campaign3e-reproduce-s0 | COMPLETE | ok | — | VERIFIED |

All committed prediction files (frozen/adapted × validation, hidden,
compositional, retention, security) were independently rescored under
`exact-match-v2` with zero mismatches, and the RUN_COMMIT inventory
matches the bytes on disk.

## Interpretation (unchanged limits)

- The +33.3-pp within-family validation gain (5/24 → 13/24) is
  **reproducible, not yet confirmed**: 24 development-visible rows is
  below the v4.1.0 statistical bar, and the adapted hidden rate stayed
  at 2.08 % — no cross-family transfer.
- The destruction/acquisition split still shows churn, not stability:
  4 of 4 previously-correct hidden answers were lost and 4 different
  ones gained (net-zero). Retention is unproven until the expanded
  retention surface (Phase 5) passes preregistered bounds.
- This reproduction binds only the committed contract. A multi-seed
  replication over the expanded 4A corpus (validation n=24/family,
  within_holdout n=32/family, retention n=108, sealed ho4 holdout) is
  the confirmatory step.
