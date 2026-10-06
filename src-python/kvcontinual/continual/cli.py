from __future__ import annotations

import argparse
import numpy as np

from kvcontinual.continual.recurrent.affine import AffineSummary, compose_sequence


def demo() -> None:
    rng = np.random.default_rng(7)
    blocks = []
    for _ in range(4):
        T = np.eye(3) * 0.9 + rng.normal(0, 0.01, (3, 3))
        Z = rng.normal(0, 0.05, (3, 3))
        blocks.append(AffineSummary(T, Z))
    state = rng.normal(size=(3, 3))
    replay = state.copy()
    for b in blocks:
        replay = b.apply(replay)
    composed = compose_sequence(blocks).apply(state)
    print("max_abs_error", float(np.max(np.abs(replay - composed))))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["demo"])
    a = p.parse_args()
    if a.command == "demo":
        demo()


if __name__ == "__main__":
    main()
