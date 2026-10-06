from __future__ import annotations

import argparse
import json
import numpy as np

from kvcontinual.execution.recurrent.affine import AffineSummary, compose_sequence


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


def mac_diagnose() -> None:
    from kvcontinual.execution.platforms.macos import detect_macos_capabilities
    print(json.dumps(detect_macos_capabilities().to_dict(), indent=2, sort_keys=True))


def mac_affine_demo() -> None:
    from kvcontinual.execution.recurrent.macos_affine import MacAffineExecutor
    ex = MacAffineExecutor(prefer_mlx=True)
    T = np.array([[0.9, 0.02], [0.01, 0.85]], dtype=np.float32)
    Z = np.array([[0.1, 0.0], [0.0, -0.1]], dtype=np.float32)
    S = np.eye(2, dtype=np.float32)
    out = ex.apply(AffineSummary(T, Z), S)
    print(json.dumps({"backend": ex.info.name, "accelerated": ex.info.accelerated, "result": out.tolist()}, indent=2))



def mac_gdn_demo() -> None:
    from kvcontinual.execution.recurrent.hypic_capture import capture_segment_tail_heads, direct_head_recurrence
    rng = np.random.default_rng(10)
    T, K, V, D, seam = 24, 2, 4, 8, 8
    k = rng.normal(size=(T, K, D)); k /= np.linalg.norm(k, axis=2, keepdims=True)
    v = rng.normal(size=(T, V, D))
    g = np.clip(rng.normal(.94, .015, size=(T, V)), .75, .999)
    b = np.clip(rng.normal(.42, .1, size=(T, V)), .02, .98)
    captured = capture_segment_tail_heads(k, v, g, b, seam_width=seam)
    worst = 0.0
    for vh in range(V):
        state = rng.normal(size=(D, D))
        exact = direct_head_recurrence(state, k[seam:, vh % K], v[seam:, vh], g[seam:, vh], b[seam:, vh])
        got = captured.summary_for_v_head(vh).apply(state)
        err = float(np.linalg.norm(got - exact) / (np.linalg.norm(exact) + 1e-15))
        worst = max(worst, err)
    print(json.dumps({"seam_width": seam, "heads": V, "worst_relative_error": worst}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["demo", "mac-diagnose", "mac-affine-demo", "mac-gdn-demo"])
    a = p.parse_args()
    if a.command == "demo":
        demo()
    elif a.command == "mac-diagnose":
        mac_diagnose()
    elif a.command == "mac-affine-demo":
        mac_affine_demo()
    elif a.command == "mac-gdn-demo":
        mac_gdn_demo()


if __name__ == "__main__":
    main()
