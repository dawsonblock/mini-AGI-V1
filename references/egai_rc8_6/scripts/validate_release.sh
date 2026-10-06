#!/usr/bin/env sh
set -eu
ROOT="${1:-.}"
cd "$ROOT"
PYTHONPATH=. python -m compileall -q egai tests scripts
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python scripts/run_rc86_governed_demo.py
