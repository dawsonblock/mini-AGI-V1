#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/.venv-mac/bin/activate"

: "${QW3_MLX_MODEL:?set QW3_MLX_MODEL to the MLX base model/repo}"
: "${QW3_BASE_MODEL_DIGEST:?set QW3_BASE_MODEL_DIGEST to the immutable base-model digest}"

DB="${KVCONTINUAL_EXPERIENCE_DB:-$ROOT/data/experience.sqlite3}"
REGISTRY="${KVCONTINUAL_REGISTRY_ROOT:-$ROOT/artifacts/adapters}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DATA_DIR="${QW3_CYCLE_DATA_DIR:-$ROOT/data/training/cycle-$STAMP}"
ADAPTER_DIR="${QW3_CYCLE_ADAPTER_DIR:-$ROOT/artifacts/training/mlx-$STAMP}"
CONFIG_JSON="$DATA_DIR/training_config.json"
mkdir -p "$DATA_DIR" "$ADAPTER_DIR"

EXPORT_JSON="$(python -m kvcontinual.learning_worker export --db "$DB" --out "$DATA_DIR" --limit "${QW3_CYCLE_LIMIT:-500}")"
RUN_ID="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_id"])' "$EXPORT_JSON")"
DATASET_DIGEST="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["dataset_digest"])' "$EXPORT_JSON")"

cat > "$CONFIG_JSON" <<EOF
{
  "backend": "mlx-lm",
  "model": "${QW3_MLX_MODEL}",
  "iters": ${QW3_TRAIN_ITERS:-200},
  "batch_size": ${QW3_TRAIN_BATCH:-1},
  "num_layers": ${QW3_TRAIN_LAYERS:-4},
  "learning_rate": ${QW3_TRAIN_LR:-2e-5},
  "grad_accumulation_steps": ${QW3_TRAIN_GRAD_ACCUM:-1},
  "mask_prompt": true,
  "grad_checkpoint": true
}
EOF

mlx_lm.lora \
  --model "$QW3_MLX_MODEL" \
  --train \
  --data "$DATA_DIR" \
  --adapter-path "$ADAPTER_DIR" \
  --iters "${QW3_TRAIN_ITERS:-200}" \
  --batch-size "${QW3_TRAIN_BATCH:-1}" \
  --num-layers "${QW3_TRAIN_LAYERS:-4}" \
  --grad-accumulation-steps "${QW3_TRAIN_GRAD_ACCUM:-1}" \
  --learning-rate "${QW3_TRAIN_LR:-2e-5}" \
  --mask-prompt \
  --grad-checkpoint

REGISTER_JSON="$(python -m kvcontinual.learning_worker register \
  --registry "$REGISTRY" \
  --adapter "$ADAPTER_DIR" \
  --base-model-digest "$QW3_BASE_MODEL_DIGEST" \
  --dataset-digest "$DATASET_DIGEST" \
  --training-config "$CONFIG_JSON" \
  --db "$DB" \
  --run-id "$RUN_ID")"

echo "$REGISTER_JSON"
echo "Candidate trained and registered. It is NOT promoted. Run evaluation, then learning_worker qualify, then use the protected admin promotion API."
