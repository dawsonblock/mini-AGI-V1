from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable

import numpy as np

from kvcontinual.execution.hardware_qualification import AccelerationQualificationObservation
from kvcontinual.execution.qualification import categorical_kl


@dataclass(frozen=True)
class PairedQualificationCase:
    case_id: str
    payload: Any
    topology: str = "ARBITRARY"
    segment_count: int = 0


def _logits(value: Any) -> np.ndarray:
    if hasattr(value, "last_logits"):
        value = value.last_logits
    return np.asarray(value, dtype=np.float64)


def run_paired_oracle_qualification(
    cases: Iterable[PairedQualificationCase],
    *,
    exact_fn: Callable[[Any], Any],
    accelerated_fn: Callable[[Any], Any],
    max_logit_kl: float = 0.02,
    require_top1: bool = True,
) -> list[AccelerationQualificationObservation]:
    """Run paired accelerated/exact inference and retain quality + TTFT evidence.

    The runner deliberately owns no model implementation. The actual Mac/Qwen
    backend supplies the two callables, so the evidence remains tied to the
    deployed backend rather than to a test double hidden inside this module.
    """
    out: list[AccelerationQualificationObservation] = []
    for case in cases:
        t0 = time.perf_counter()
        exact = exact_fn(case.payload)
        exact_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        accelerated = accelerated_fn(case.payload)
        accelerated_ms = (time.perf_counter() - t0) * 1000.0
        a = _logits(accelerated)
        e = _logits(exact)
        if a.shape != e.shape:
            raise ValueError(f"qualification logit shape mismatch for {case.case_id}")
        kl = categorical_kl(e, a)
        top1 = float(np.argmax(a) == np.argmax(e))
        accepted = kl <= max_logit_kl and (top1 == 1.0 if require_top1 else True)
        out.append(AccelerationQualificationObservation(
            case_id=case.case_id,
            logit_kl=kl,
            top1_agreement=top1,
            accepted=accepted,
            exact_ttft_ms=exact_ms,
            accelerated_ttft_ms=accelerated_ms,
            topology=case.topology,
            segment_count=case.segment_count,
        ))
    return out
