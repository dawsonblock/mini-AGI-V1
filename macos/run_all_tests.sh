#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
(cd "$ROOT/runtime" && PYTHONPATH=src "$PYTHON" -m pytest -q tests)
(cd "$ROOT/authority" && PYTHONPATH=. "$PYTHON" -m pytest -q tests)
(cd "$ROOT/oracle_reference" && PYTHONPATH=. "$PYTHON" -m pytest -q tests)
"$ROOT/scripts/rc11/integration_smoke.py"
