#!/usr/bin/env python3
"""Verify the Mini-AGI signed source manifest.

Delegates to scripts/verify_release.py — the canonical verifier that
checks every governed file's sha256 against SOURCE_MANIFEST.json,
refuses symlinks, verifies the Ed25519 signature against the pinned
release key fingerprint, and reconciles RELEASE_ATTESTATION.json.
Kept as the stable entry name used by host_gate.sh; all arguments are
forwarded (e.g. ``--root <tree>``).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "scripts" / "verify_release.py"


def main() -> int:
    return subprocess.call(
        [sys.executable, str(VERIFIER), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
