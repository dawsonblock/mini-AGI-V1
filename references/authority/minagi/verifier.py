"""Inference-time step verification and bounded search.

A verifier is intentionally separate from the policy model. It can retrieve
external evidence for a partial trace, score the proposed next step, and prune
weak trajectories before errors compound. The package does not pretend a
trained reward model exists; callers provide the scorer/retriever they have
qualified.

v4.1 adds explicit score aggregation and completed-trace handling. Raw reward
sums systematically prefer longer traces when local rewards are positive. The
reasoner can therefore rank by mean, discounted sum, minimum-step reward, or
legacy sum, and it keeps completed traces in a separate pool rather than
forcing them to compete for another generation step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class VerifiedStep:
    text: str
    reward: float
    searched: bool = False
    evidence: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class Trace:
    steps: list[VerifiedStep] = field(default_factory=list)

    @property
    def score(self):
        """Legacy cumulative reward."""
        return sum(float(s.reward) for s in self.steps)

    @property
    def text(self):
        return "\n".join(s.text for s in self.steps)

    def aggregate(self, mode="mean", discount=0.95, length_penalty=0.0):
        if not self.steps:
            return 0.0
        r = [float(s.reward) for s in self.steps]
        if mode == "sum":
            v = sum(r)
        elif mode == "mean":
            v = sum(r) / len(r)
        elif mode == "min":
            v = min(r)
        elif mode == "discounted":
            d = float(discount)
            if not 0 < d <= 1:
                raise ValueError("discount must be in (0,1]")
            # Recent steps receive no privileged scale; discount is from the
            # beginning so pathological tail padding cannot add unlimited gain.
            v = sum((d ** i) * x for i, x in enumerate(r))
        else:
            raise ValueError(f"unknown verifier score mode: {mode}")
        if length_penalty:
            v -= float(length_penalty) * max(0, len(r) - 1)
        return float(v)


class ProcessVerifier:
    """Portable step scorer with optional evidence retrieval.

    scorer(question, partial_trace_text, candidate_step, evidence) -> float
    search_policy(...) -> bool is optional; if absent, retrieval is always used
    when a retriever exists.
    """
    def __init__(self, scorer: Callable, retriever: Callable | None = None,
                 search_policy: Callable | None = None, evidence_k: int = 6,
                 min_reward: float | None = None, max_reward: float | None = None):
        self.scorer = scorer
        self.retriever = retriever
        self.search_policy = search_policy
        self.evidence_k = max(1, int(evidence_k))
        self.min_reward = min_reward
        self.max_reward = max_reward

    def evaluate(self, question: str, trace: Trace, step: str) -> VerifiedStep:
        use_search = bool(self.retriever)
        if self.search_policy is not None:
            use_search = bool(self.search_policy(question, trace.text, step))
        evidence = []
        if use_search and self.retriever is not None:
            evidence = list(self.retriever(question, trace.text, step,
                                           self.evidence_k) or [])[:self.evidence_k]
        reward = float(self.scorer(question, trace.text, step, evidence))
        if reward != reward or reward in (float("inf"), float("-inf")):
            raise ValueError("process verifier returned a non-finite reward")
        if self.min_reward is not None:
            reward = max(float(self.min_reward), reward)
        if self.max_reward is not None:
            reward = min(float(self.max_reward), reward)
        return VerifiedStep(step, reward, searched=use_search, evidence=evidence)


class BeamReasoner:
    """Search over model-proposed reasoning steps using online verification.

    proposer(question, trace_text, branching) -> iterable[str]
    complete(question, trace_text) -> bool
    """
    def __init__(self, proposer: Callable, verifier: ProcessVerifier,
                 beam_width=4, branching=4, max_steps=12,
                 complete: Callable | None = None, score_mode="mean",
                 discount=0.95, length_penalty=0.0, deduplicate=True):
        self.proposer = proposer
        self.verifier = verifier
        self.beam_width = max(1, int(beam_width))
        self.branching = max(1, int(branching))
        self.max_steps = max(1, int(max_steps))
        self.complete = complete or (lambda q, t: False)
        self.score_mode = str(score_mode)
        self.discount = float(discount)
        self.length_penalty = float(length_penalty)
        self.deduplicate = bool(deduplicate)

    def _rank(self, tr: Trace):
        return tr.aggregate(self.score_mode, self.discount, self.length_penalty)

    def run(self, question: str) -> Trace:
        beam = [Trace()]
        completed: list[Trace] = []
        for _ in range(self.max_steps):
            expanded = []
            for tr in beam:
                if self.complete(question, tr.text):
                    completed.append(tr)
                    continue
                proposals = list(self.proposer(question, tr.text,
                                               self.branching) or [])[:self.branching]
                if self.deduplicate:
                    seen = set(); uniq = []
                    for p in proposals:
                        s = str(p)
                        if s not in seen:
                            seen.add(s); uniq.append(s)
                    proposals = uniq
                for p in proposals:
                    vs = self.verifier.evaluate(question, tr, str(p))
                    nt = Trace(tr.steps + [vs])
                    if self.complete(question, nt.text):
                        completed.append(nt)
                    else:
                        expanded.append(nt)
            if not expanded:
                break
            expanded.sort(key=self._rank, reverse=True)
            beam = expanded[:self.beam_width]
        pool = completed or beam
        return max(pool, key=self._rank) if pool else Trace()
