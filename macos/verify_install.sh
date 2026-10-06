#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/macos/doctor.sh"
if [[ ! -d "$ROOT/.venv-mac" ]]; then
  echo "missing .venv-mac; run macos/qw3-mac install" >&2
  exit 2
fi
source "$ROOT/.venv-mac/bin/activate"
python - <<'PY'
import kvcontinual
from kvcontinual.experience_store import ExperienceStore
from kvcontinual.backends import BackendCapabilities
print("kvcontinual", kvcontinual.__version__)
print("python import verification: PASS")
PY
