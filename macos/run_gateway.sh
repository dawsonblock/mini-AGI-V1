#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export KVCONTINUAL_CONFIG_DIR="${KVCONTINUAL_CONFIG_DIR:-$ROOT/runtime/configs-macos}"
export KVCONTINUAL_ADAPTER_ROUTER="${KVCONTINUAL_ADAPTER_ROUTER:-$ROOT/runtime/configs-macos/adapters.json}"
PORT="${PORT:-8000}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
exec "$PYTHON" -m uvicorn kvcontinual.continual.mac_api:app --app-dir "$ROOT/runtime/src" --host 127.0.0.1 --port "$PORT"
