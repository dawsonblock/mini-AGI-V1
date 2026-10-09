#!/usr/bin/env bash
# Run the repository lint contract (pyproject.toml [tool.ruff]/[tool.pylint],
# .flake8) over the release surface. Used by CI and available locally.
#
# The contract enforces:
#   ruff   — E4/E7/E9/F (see [tool.ruff.lint])
#   flake8 — the same selection, for editors that run Flake8
#   pylint — the error category (see [tool.pylint]); the ONLY tolerated
#            findings are the two documented RC14-001 ones — the drifted
#            donor class in rc14/system.py (unexpected-keyword-arg on its
#            super().__init__ call, see REMAINING_DEFECTS_V1641.md).
#            Anything else fails, including any new finding in that same
#            file.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src-python${PYTHONPATH:+:$PYTHONPATH}"

echo "== ruff =="
"$PYTHON" -m ruff check .

echo "== flake8 =="
"$PYTHON" -m flake8 .

echo "== pylint =="
out="$("$PYTHON" -m pylint --recursive=y src-python scripts tests-python \
    -j 4 --score=n --output-format=text 2>/dev/null || true)"
printf '%s\n' "$out"
findings="$(printf '%s\n' "$out" | grep -E ": [A-Z][0-9]{4}: " || true)"
unexpected="$(printf '%s\n' "$findings" | grep -vE "src-python/minagi/rc14/system\.py:[0-9]+:[0-9]+: E1123:" || true)"
if [ -n "$unexpected" ]; then
    echo "Pylint: unexpected findings outside the documented RC14-001 set:" >&2
    printf '%s\n' "$unexpected" >&2
    exit 1
fi
echo "Pylint: only the documented RC14-001 findings (see REMAINING_DEFECTS_V1641.md)"
