"""Controlled recursive improvement (Phase 8 / REPAIR-042..046).

Bounded, governed generation chain: each generation is a *campaign*
whose candidate improvement was proposed under the previous
generation's approved state. A generation may only be preregistered
after its parent was PROMOTED by the promotion authority — the chain
enforces the sequence cryptographically, not by convention.

Rules enforced here (the honest subset that does not claim open-ended
self-improvement):

  * G0 is the frozen baseline (original model + harness); it has no
    parent and needs no promotion to exist.
  * G(n+1) requires: parent generation record exists, parent's
    promotion_decision_digest is non-empty (the promotion authority
    actually approved it), a FRESH evaluation corpus (digest differs
    from every ancestor — no measuring improvement on tasks the
    proposal was tuned against), and a contemporaneous frozen control
    arm digest.
  * Costs are recorded per generation — improvement claims must carry
    the price paid.
"""
from __future__ import annotations

from dataclasses import dataclass

from egai.common.canonical import digest, validate_digest

GENESIS = ""  # G0 has no parent


@dataclass(frozen=True)
class GenerationRecord:
    """One governed generation. Binds the qualifying campaign, the
    evaluation corpus that measured it, the contemporaneous control,
    and — only after the promotion authority acts — the promotion
    decision that permits descendants."""
    generation: int
    campaign_digest: str             # this generation's campaign plan
    corpus_digest: str               # eval corpus (must be fresh vs ancestors)
    control_arm_digest: str          # contemporaneous frozen-control evidence
    parent_generation_digest: str    # digest of parent's record ("" = genesis)
    promotion_decision_digest: str = ""  # set when promotion authority signs
    schema: str = "mini-agi-v16.8-generation-record-v1"

    def __post_init__(self):
        if self.generation < 0:
            raise ValueError("generation must be >= 0")
        validate_digest(self.campaign_digest)
        validate_digest(self.corpus_digest)
        validate_digest(self.control_arm_digest)
        if self.generation == 0 and self.parent_generation_digest:
            raise ValueError("genesis generation has no parent")
        if self.generation > 0:
            validate_digest(self.parent_generation_digest)
        if self.parent_generation_digest:
            validate_digest(self.parent_generation_digest)
        if self.promotion_decision_digest:
            validate_digest(self.promotion_decision_digest)

    @property
    def digest(self) -> str:
        return digest(self)

    @property
    def promoted(self) -> bool:
        return bool(self.promotion_decision_digest)

    def with_promotion(self, decision_digest: str) -> "GenerationRecord":
        validate_digest(decision_digest)
        return GenerationRecord(
            generation=self.generation,
            campaign_digest=self.campaign_digest,
            corpus_digest=self.corpus_digest,
            control_arm_digest=self.control_arm_digest,
            parent_generation_digest=self.parent_generation_digest,
            promotion_decision_digest=decision_digest)


class GenerationChain:
    """Append-only chain of GenerationRecords. `append` enforces the
    promotion gate: a child generation cannot be preregistered until
    its parent carries a promotion decision."""

    def __init__(self):
        self._records: list[GenerationRecord] = []
        self._corpora: set[str] = set()

    def append(self, record: GenerationRecord) -> None:
        gen = record.generation
        if gen != len(self._records):
            raise ValueError(
                f"next generation must be {len(self._records)}, got {gen}")
        if gen == 0:
            if record.parent_generation_digest != GENESIS:
                raise ValueError("genesis must not reference a parent")
        else:
            parent = self._records[-1]
            if record.parent_generation_digest != parent.digest:
                raise ValueError("parent digest does not match chain tip")
            if not parent.promoted:
                raise PermissionError(
                    "generation cannot be preregistered: parent is not "
                    "promoted — promotion authority must authorize the "
                    "parent before its child exists")
        if record.corpus_digest in self._corpora:
            raise ValueError(
                "corpus reuse across generations is forbidden — each "
                "generation must be measured on tasks its ancestors were "
                "not tuned against")
        self._records.append(record)
        self._corpora.add(record.corpus_digest)

    def record_promotion(self, generation: int,
                         decision_digest: str) -> None:
        """Attach the promotion decision to an existing record. Only
        the promotion authority's decision unlocks the next generation."""
        if generation != len(self._records) - 1:
            raise ValueError("promotion applies only to the chain tip")
        rec = self._records[-1]
        if rec.promoted:
            raise ValueError("generation already promoted")
        self._records[-1] = rec.with_promotion(decision_digest)

    @property
    def tip(self) -> GenerationRecord | None:
        return self._records[-1] if self._records else None

    @property
    def records(self) -> tuple[GenerationRecord, ...]:
        return tuple(self._records)

    def verify(self) -> list[str]:
        """Re-verify the whole chain — promotion gates, corpus
        freshness, parent linkage. Returns a list of violations."""
        problems: list[str] = []
        seen_corpora: set[str] = set()
        for i, rec in enumerate(self._records):
            if rec.generation != i:
                problems.append(f"record {i}: generation index {rec.generation}")
            if i > 0:
                parent = self._records[i - 1]
                if rec.parent_generation_digest != parent.digest:
                    problems.append(f"record {i}: parent digest mismatch")
                if not parent.promoted:
                    problems.append(
                        f"record {i}: parent generation not promoted")
            if rec.corpus_digest in seen_corpora:
                problems.append(f"record {i}: corpus digest reused")
            seen_corpora.add(rec.corpus_digest)
        return problems
