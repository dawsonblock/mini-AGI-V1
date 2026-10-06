from __future__ import annotations

from .models import CalibrationReport


def _ranks(xs: list[float]) -> list[float]:
    # Average ranks for ties.
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Spearman correlation requires equal vectors with >=2 elements")
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx)/len(rx), sum(ry)/len(ry)
    num = sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    dx = sum((a-mx)**2 for a in rx)
    dy = sum((b-my)**2 for b in ry)
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx*dy) ** 0.5


def replay_live_calibration(candidate_ids: list[str], replay_scores: list[float], live_scores: list[float], threshold: float = 0.5) -> CalibrationReport:
    if not (len(candidate_ids) == len(replay_scores) == len(live_scores)):
        raise ValueError("candidate ids and score vectors must have equal length")
    rho = spearman(replay_scores, live_scores)
    return CalibrationReport(tuple(candidate_ids), tuple(replay_scores), tuple(live_scores), rho, rho >= threshold, threshold)
