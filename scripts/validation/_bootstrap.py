"""Shared bootstrap for v16.2 validation scripts.

Resolves the release root so validation drivers work both from inside the
release tree (scripts/validation/) and from a standalone unpack location via
--root /path/to/mini-AGI-v16.1-...
"""
from __future__ import annotations

import sys
from pathlib import Path


def detect_root() -> Path:
    here = Path(__file__).resolve()
    # scripts/validation/_bootstrap.py -> scripts -> release root
    candidate = here.parents[2]
    if (candidate / "SOURCE_MANIFEST.json").is_file():
        return candidate
    return Path.cwd()


def ensure_path(root: Path) -> Path:
    root = Path(root).resolve()
    src = root / "src-python"
    for p in (str(src), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root
