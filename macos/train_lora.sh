#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/.venv-mac/bin/activate"
: "${QW3_MLX_MODEL:?set QW3_MLX_MODEL to the MLX model/repo used for training}"
DATA="${QW3_TRAIN_DATA:-$ROOT/data/training}"
ADAPTER="${QW3_ADAPTER_OUT:-$ROOT/artifacts/adapters/candidates/mlx-manual}"
ITERS="${QW3_TRAIN_ITERS:-200}"
LAYERS="${QW3_TRAIN_LAYERS:-4}"
mkdir -p "$ADAPTER"
if [[ ! -f "$DATA/train.jsonl" ]]; then
  echo "Missing $DATA/train.jsonl" >&2
  exit 2
fi
exec mlx_lm.lora \
  --model "$QW3_MLX_MODEL" \
  --train \
  --data "$DATA" \
  --adapter-path "$ADAPTER" \
  --iters "$ITERS" \
  --batch-size 1 \
  --num-layers "$LAYERS" \
  --grad-accumulation-steps "${QW3_TRAIN_GRAD_ACCUM:-1}" \
  --learning-rate "${QW3_TRAIN_LR:-2e-5}" \
  --mask-prompt \
  --grad-checkpoint
