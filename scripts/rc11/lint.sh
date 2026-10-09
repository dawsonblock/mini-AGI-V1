#!/usr/bin/env bash
# Run the repository lint contract (pyproject.toml [tool.ruff]/[tool.pylint],
# .flake8) over the release surface. Used by CI and available locally.
#
# The contract enforces:
#   ruff   — E4/E7/E9/F (see [tool.ruff.lint])
#   flake8 — the same selection, for editors that run Flake8
#   pylint — the error category (see [tool.pylint])
# Zero findings are required. (The two RC14-001 unexpected-keyword-arg
# findings that were briefly tolerated here are gone: the drifted donor
# class was retired — see REMAINING_DEFECTS_V1641.md.)
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
if [ -n "$findings" ]; then
    echo "Pylint: findings must be zero:" >&2
    printf '%s\n' "$findings" >&2
    exit 1
fi
echo "Pylint: zero findings"
