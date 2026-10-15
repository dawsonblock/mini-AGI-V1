"""Capability acquisition controller (V4-13).

The campaign machinery can measure; this module decides WHAT to try next
and records WHY — turning the diagnostics into an evidence-driven
intervention policy.

Pipeline position:

    measurement (qualified evidence in)
        -> diagnose()          : failure evidence -> ranked hypotheses
        -> select_intervention : hypotheses + intervention history
                                 + cost -> a single decision record
        -> record_outcome      : the experiment's verified result is
                                 appended to the intervention history,
                                 closing the loop

Hard rules:

* Diagnoses are hypotheses, not truth — every diagnosis carries the
  evidence features that produced it so a later audit can disagree.
* The controller never qualifies its own choice: decisions carry
  ``candidate_controlled = True`` and feed the *proposal* side. Final
  qualification remains the independent qualifier's job (G7).
* An intervention that previously failed under unchanged conditions
  cannot be reselected without a documented reason (the caller must
  supply ``changed_condition`` describing what is different).
* ``no_intervention`` is always a candidate and wins when nothing else
  beats its (zero-cost) score — the controller can refuse to act.

The module is deliberately dependency-light (no torch, no network): it
consumes metric dicts produced by the runners and produces structured,
digestible decision records.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from egai.common.canonical import digest

SCHEMA = "mini-agi-v4.0-capability-controller-v1"

#: Failure categories the controller is allowed to hypothesize.
HYPOTHESES = (
    "missing_knowledge",
    "incorrect_reasoning",
    "instruction_misunderstanding",
    "representation_limitation",
    "insufficient_generalization",
    "tool_use_failure",
    "memory_retrieval_failure",
    "adapter_interference",
    "catastrophic_forgetting",
    "inference_instability",
)

#: Intervention kinds the policy may select.
INTERVENTIONS = (
    "retrieval_augmentation",
    "verified_skill_memory",
    "in_context_demonstrations",
    "more_training_examples",
    "curriculum_change",
    "supervised_finetuning",
    "fixed_lora",
    "routed_lora",
    "rehearsal",
    "kl_preservation",
    "optimization_change",
    "no_intervention",
)

#: Rough relative cost table (seconds-equivalent). Used to break ties
#: and to record cost-awareness; real cost accounting comes from the
#: executed receipts.
INTERVENTION_COST = {
    "retrieval_augmentation": 1.0,
    "verified_skill_memory": 1.0,
    "in_context_demonstrations": 0.5,
    "more_training_examples": 2.0,
    "curriculum_change": 3.0,
    "supervised_finetuning": 8.0,
    "fixed_lora": 6.0,
    "routed_lora": 8.0,
    "rehearsal": 2.0,
    "kl_preservation": 2.0,
    "optimization_change": 4.0,
    "no_intervention": 0.0,
}


@dataclass(frozen=True)
class FailureEvidence:
    """Quantitative description of how a candidate is failing.

    All rates are fractions in [0,1]; ``None`` means the measurement was
    not taken (distinct from zero).
    """
    train_fit: float | None = None
    within_family: float | None = None
    cross_family: float | None = None
    compositional: float | None = None
    retention: float | None = None
    retention_baseline: float | None = None
    security_regressions: int | None = None
    empty_output_rate: float | None = None
    contract_violation_rate: float | None = None
    frozen_baseline: float | None = None


@dataclass(frozen=True)
class Hypothesis:
    name: str
    confidence: float
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class InterventionRecord:
    """One historical experiment outcome — the controller's memory."""
    signature: str
    intervention: str
    outcome_delta: float
    retention_drop: float
    verified: bool
    cost: float = 0.0
    conditions: str = ""


def diagnose(ev: FailureEvidence) -> tuple[Hypothesis, ...]:
    """Rank failure hypotheses from measured evidence.

    Rules are deliberately simple and each reports its triggers; a
    diagnosis with weak or missing evidence yields a low-confidence
    hypothesis, never a confident guess.
    """
    h: list[Hypothesis] = []

    def add(name, conf, *why):
        if name in HYPOTHESES:
            h.append(Hypothesis(name, round(min(conf, 1.0), 4),
                                tuple(why)))

    tf, wf = ev.train_fit, ev.within_family
    comp = ev.compositional
    fb = ev.frozen_baseline if ev.frozen_baseline is not None else 0.0

    if ev.empty_output_rate and ev.empty_output_rate > 0.5:
        add("inference_instability", min(0.4 + ev.empty_output_rate, 1.0),
            f"empty_output_rate={ev.empty_output_rate:.2f}")
    if ev.contract_violation_rate and ev.contract_violation_rate > 0.3:
        add("instruction_misunderstanding",
            min(0.4 + ev.contract_violation_rate, 1.0),
            f"contract_violation_rate={ev.contract_violation_rate:.2f}")
    if tf is not None and wf is not None:
        if tf < 0.5:
            add("representation_limitation" if tf < 0.2
                else "missing_knowledge",
                0.6 if tf < 0.2 else 0.5,
                f"train_fit={tf:.2f} (cannot even fit the training set)")
        elif wf < tf - 0.25:
            add("insufficient_generalization", 0.7,
                f"train_fit={tf:.2f} vs within_family={wf:.2f}")
    if wf is not None and fb >= wf:
        add("insufficient_generalization", 0.5,
            f"within_family={wf:.2f} not above frozen baseline {fb:.2f}")
    if comp is not None and wf is not None and comp < wf - 0.25:
        add("incorrect_reasoning", 0.5,
            f"compositional={comp:.2f} << within_family={wf:.2f}")
    if (ev.retention is not None and ev.retention_baseline is not None
            and ev.retention_baseline - ev.retention > 0.10):
        add("catastrophic_forgetting",
            min(0.5 + (ev.retention_baseline - ev.retention), 1.0),
            f"retention {ev.retention_baseline:.2f} -> {ev.retention:.2f}")
    if ev.security_regressions:
        add("adapter_interference", 0.5,
            f"security_regressions={ev.security_regressions}")
    if not h:
        add("missing_knowledge", 0.1, "no strong evidence features")
    return tuple(sorted(h, key=lambda x: (-x.confidence, x.name)))


#: Which interventions plausibly address which hypotheses.
_HYPOTHESIS_TO_INTERVENTIONS = {
    "missing_knowledge": ("retrieval_augmentation",
                          "more_training_examples", "supervised_finetuning"),
    "incorrect_reasoning": ("in_context_demonstrations",
                            "curriculum_change", "supervised_finetuning"),
    "instruction_misunderstanding": ("in_context_demonstrations",
                                     "curriculum_change"),
    "representation_limitation": ("fixed_lora", "supervised_finetuning",
                                  "more_training_examples"),
    "insufficient_generalization": ("more_training_examples",
                                    "curriculum_change", "fixed_lora"),
    "tool_use_failure": ("verified_skill_memory", "in_context_demonstrations"),
    "memory_retrieval_failure": ("verified_skill_memory",
                                 "retrieval_augmentation"),
    "adapter_interference": ("routed_lora", "no_intervention"),
    "catastrophic_forgetting": ("rehearsal", "kl_preservation"),
    "inference_instability": ("optimization_change", "no_intervention"),
}


def _signature(intervention: str, target: str, conditions: str) -> str:
    return digest({"intervention": intervention, "target": target,
                   "conditions": conditions})


def intervention_history_load(path: Path) -> list[InterventionRecord]:
    if not Path(path).is_file():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(InterventionRecord(**json.loads(line)))
    return out


def intervention_history_append(path: Path, rec: InterventionRecord) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(asdict(rec), sort_keys=True) + "\n")


@dataclass(frozen=True)
class InterventionDecision:
    """Structured output — the fields the campaign evidence expects."""
    schema: str
    observed_weakness: str
    supporting_evidence: tuple[str, ...]
    candidate_interventions: tuple[str, ...]
    selected_intervention: str
    predicted_improvement: float
    estimated_cost: float
    confidence: float
    rationale: str
    refused: tuple[str, ...]
    candidate_controlled: bool = True  # proposal only — never qualifies
    digest: str = ""

    def sealed(self) -> "InterventionDecision":
        d = dict(asdict(self))
        d.pop("digest", None)
        return InterventionDecision(digest=digest(d), **{k: v for k, v in d.items()
                                    if k != "digest"})


def select_intervention(
        hypotheses: Sequence[Hypothesis],
        history: Iterable[InterventionRecord],
        *,
        target: str = "",
        conditions: str = "",
        changed_condition: str | None = None,
        regression_risk_penalty: float = 0.25) -> InterventionDecision:
    """Choose one intervention from evidence.

    Score = top-hypothesis confidence * expected benefit −
    regression risk penalty * historical retention damage − cost * 0.01.
    ``no_intervention`` scores 0 — anything selected must beat doing
    nothing.
    """
    history = list(history)
    refused: list[str] = []
    candidates: list[str] = []
    for hyp in hypotheses[:2]:
        for iv in _HYPOTHESIS_TO_INTERVENTIONS.get(hyp.name, ()):
            if iv not in candidates:
                candidates.append(iv)
    if "no_intervention" not in candidates:
        candidates.append("no_intervention")

    conf = hypotheses[0].confidence if hypotheses else 0.0
    best, best_score, best_pred = "no_intervention", 0.0, 0.0
    for iv in candidates:
        if iv == "no_intervention":
            continue
        sig = _signature(iv, target, conditions)
        prior = [r for r in history if r.signature == sig]
        failures = [r for r in prior
                    if r.verified and r.outcome_delta <= 0.0]
        if failures and not changed_condition:
            refused.append(
                f"{iv}: {len(failures)} verified non-positive outcome(s) "
                "under unchanged conditions")
            continue
        if failures and changed_condition:
            # Allowed — but only because the caller documented what
            # changed; the record keeps the refusal that would have
            # fired.
            refused.append(
                f"{iv}: prior failures overridden — changed condition: "
                f"{changed_condition}")
        positives = [r.outcome_delta for r in prior
                     if r.verified and r.outcome_delta > 0]
        expected_benefit = (sum(positives) / len(positives)
                            if positives else conf * 0.5)
        risk = sum(max(0.0, r.retention_drop) for r in prior) / \
            max(1, len(prior))
        score = (conf * expected_benefit
                 - regression_risk_penalty * risk
                 - 0.01 * INTERVENTION_COST.get(iv, 5.0))
        if score > best_score:
            best, best_score, best_pred = iv, score, expected_benefit

    dec = InterventionDecision(
        schema=SCHEMA,
        observed_weakness=hypotheses[0].name if hypotheses else "unknown",
        supporting_evidence=hypotheses[0].evidence if hypotheses else (),
        candidate_interventions=tuple(candidates),
        selected_intervention=best,
        predicted_improvement=round(best_pred, 4),
        estimated_cost=INTERVENTION_COST.get(best, 0.0),
        confidence=conf,
        rationale=(
            f"top hypothesis '{hypotheses[0].name}' (conf {conf:.2f}); "
            f"{len(history)} history records; "
            f"{len(refused)} refusal(s)"),
        refused=tuple(refused))
    return dec.sealed()
