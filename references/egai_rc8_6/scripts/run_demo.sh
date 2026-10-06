#!/usr/bin/env sh
set -eu
python -m egai demo --root "${1:-./egai-run}"
