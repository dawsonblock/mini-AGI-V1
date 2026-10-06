#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
(cd "$ROOT" && "$PYTHON" -m pytest -q)
(cd "$ROOT/references/authority" && "$PYTHON" -m pytest -q)
(cd "$ROOT/references/oracle_reference" && "$PYTHON" -m pytest -q)
