#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "continual" / "src"))

import numpy as np
from kvcontinual.recurrent.gdn_reference import GDNToken, compose_tokens, replay, state_error


def main() -> None:
    rng = np.random.default_rng(105)
    dim = 16
    tokens = [
        GDNToken(
            k=rng.normal(0, 0.1, dim),
            v=rng.normal(0, 0.1, dim),
            g=float(rng.uniform(0.85, 0.999)),
            beta=float(rng.uniform(0.05, 0.95)),
        )
        for _ in range(64)
    ]
    s0 = rng.normal(0, 0.05, (dim, dim))
    direct = replay(tokens, s0)
    summary = compose_tokens(tokens)
    composed = summary.apply(s0)
    max_abs, rel_l2 = state_error(composed, direct)
    zero = np.zeros_like(s0)
    z_abs, z_rel = state_error(summary.Z, replay(tokens, zero))
    report = {
        "status": "PASS" if max_abs < 1e-11 and rel_l2 < 1e-11 and z_abs < 1e-11 else "FAIL",
        "dim": dim,
        "tokens": len(tokens),
        "composition_max_abs": max_abs,
        "composition_rel_l2": rel_l2,
        "zero_start_Z_max_abs": z_abs,
        "zero_start_Z_rel_l2": z_rel,
        "meaning": "CPU reference mirrors src/gated_delta_net.cu state-update algebra for fixed token-local k/v/g/beta.",
        "boundary": "This does not prove deep-network arbitrary-segment equivalence; seam/suffix reconstruction remains required when hidden inputs change.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
