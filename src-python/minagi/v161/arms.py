"""v16.2 six-arm mechanisms for the HF evaluation path.

Each arm is a *prompt-augmentation policy* over a common deterministic
lexical index — honest mechanisms, not trained-learning doubles:

  L1 frozen          : no augmentation
  L2 retrieval       : top-k raw train exemplars for the query
  L3 semantic-memory : consolidated per-family memory entries (one
                       retained exemplar + support count per family)
  L4 skills          : whole-family skill blocks; the nearest family
                       skill is selected and rendered
  L5 grounded-replay : traces verified during a labeled *practice*
                       phase on train rows; hidden labels are never
                       consulted (no leakage)
  L6 neural-adapter  : LoRA trained on train texts (parametric; not
                       implemented here — runner uses peft_trainer)
  NC negative-control: identical LoRA training on label-shuffled
                       train texts; must show ~no forward transfer

Arm states are materialized to disk and digest-bound by
`state_digest` in ExecutedRunReceiptV162 (L1 -> ZERO).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Mapping, Sequence

ARM_IDS = ("L1", "L2", "L3", "L4", "L5", "L6", "NC")
ZERO_DIGEST = "sha256:" + "0" * 64

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _trigrams(text: str) -> set[str]:
    toks = _tokens(text)
    grams: set[str] = set(toks)
    for t in toks:
        for i in range(len(t) - 2):
            grams.add(t[i:i + 3])
    return grams


class LexicalIndex:
    """Deterministic n-gram overlap index over rows.

    Score = weighted Jaccard of token + char-trigram sets.
    Ties break by document position — fully deterministic.
    """

    def __init__(self, docs: Sequence[Mapping]):
        self.docs = list(docs)
        self._grams = [_trigrams(" ".join(str(d.get(f, "")) for f in
                               ("prompt", "exemplar", "prediction", "family")))
                       for d in self.docs]

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        q = _trigrams(query)
        scored = []
        for i, g in enumerate(self._grams):
            inter = len(q & g)
            union = len(q | g) or 1
            scored.append((inter / union, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(i, s) for s, i in scored[:k]]


def _exemplar_block(row: Mapping) -> str:
    return f"{row['prompt']} {row['expected']}"


class FrozenArm:
    arm_id = "L1"
    uses_state = False

    def build_state(self, state_dir: Path, rows: Sequence[Mapping]) -> None:
        return None

    def augment(self, prompt: str, family: str) -> str:
        return prompt


class RetrievalArm:
    """L2: prepend the k lexically-nearest raw train exemplars."""

    arm_id = "L2"
    uses_state = True

    def __init__(self, k: int = 3):
        self.k = int(k)
        self.index: LexicalIndex | None = None

    def build_state(self, state_dir: Path, rows: Sequence[Mapping]) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        self.index = LexicalIndex(rows)
        (state_dir / "INDEX.json").write_text(json.dumps(
            {"schema": "mini-agi-v16.2-retrieval-index-v1", "k": self.k,
             "n_docs": len(rows),
             "doc_ids": [str(d["id"]) for d in rows]},
            indent=2, sort_keys=True))

    def augment(self, prompt: str, family: str) -> str:
        assert self.index is not None
        hits = self.index.rank(prompt, self.k)
        ex = "\n".join(_exemplar_block(self.index.docs[i]) for i, _ in hits)
        return f"Examples:\n{ex}\n\n{prompt}"


class SemanticMemoryArm:
    """L3: consolidated memory — one exemplar retained per train family
    plus its support count. Selection uses the same lexical index over
    the consolidated entries, not raw rows."""

    arm_id = "L3"
    uses_state = True

    def __init__(self, k: int = 2):
        self.k = int(k)
        self.entries: list[dict] = []
        self.index: LexicalIndex | None = None

    def build_state(self, state_dir: Path, rows: Sequence[Mapping]) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        fams: dict[str, list[Mapping]] = {}
        for r in rows:
            fams.setdefault(str(r["family"]), []).append(r)
        self.entries = [
            {"family": fam, "support": len(members),
             "exemplar": _exemplar_block(members[0])}
            for fam, members in sorted(fams.items())]
        self.index = LexicalIndex(self.entries)
        (state_dir / "MEMORY.json").write_text(json.dumps(
            {"schema": "mini-agi-v16.2-semantic-memory-v1",
             "entries": self.entries}, indent=2, sort_keys=True))

    def augment(self, prompt: str, family: str) -> str:
        assert self.index is not None
        hits = self.index.rank(prompt, self.k)
        mem = "\n".join(
            f"Memory[{self.entries[i]['family']} x{self.entries[i]['support']}]: "
            f"{self.entries[i]['exemplar']}" for i, _ in hits)
        return f"{mem}\n\n{prompt}"


class SkillsArm:
    """L4: skill library = one block per train family containing every
    exemplar of that family; the nearest skill is selected."""

    arm_id = "L4"
    uses_state = True

    def __init__(self):
        self.skills: list[dict] = []
        self.index: LexicalIndex | None = None

    def build_state(self, state_dir: Path, rows: Sequence[Mapping]) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        fams: dict[str, list[Mapping]] = {}
        for r in rows:
            fams.setdefault(str(r["family"]), []).append(r)
        self.skills = [
            {"skill_id": f"skill-{fam}", "family": fam,
             "exemplars": [_exemplar_block(r) for r in members]}
            for fam, members in sorted(fams.items())]
        probe_docs = [{"prompt": s["exemplars"][0], "family": s["family"]}
                      for s in self.skills]
        self.index = LexicalIndex(probe_docs)
        (state_dir / "SKILLS.json").write_text(json.dumps(
            {"schema": "mini-agi-v16.2-skill-library-v1",
             "skills": self.skills}, indent=2, sort_keys=True))

    def augment(self, prompt: str, family: str) -> str:
        assert self.index is not None
        i, _ = self.index.rank(prompt, 1)[0]
        skill = self.skills[i]
        block = "\n".join(skill["exemplars"])
        return (f"Skill: {skill['skill_id']}\n{block}\n\n{prompt}")


class GroundedReplayArm:
    """L5: replay buffer of traces verified during a labeled practice
    phase on train rows.

    add_trace() is called by the runner only for *train* rows where the
    model's own prediction scored 1.0 under the preregistered scorer.
    Hidden-set labels never enter the buffer — the arm replays verified
    training-time experience non-parametrically.
    """

    arm_id = "L5"
    uses_state = True

    def __init__(self, k: int = 2):
        self.k = int(k)
        self.traces: list[dict] = []
        self.index: LexicalIndex | None = None

    def build_state(self, state_dir: Path, rows: Sequence[Mapping]) -> None:
        # State is populated via add_trace(); persisted after practice.
        self._state_dir = state_dir
        self.traces = []
        self.index = None

    def add_trace(self, row: Mapping, prediction: str, score: float) -> None:
        if float(score) == 1.0:
            self.traces.append({"id": str(row["id"]),
                                "prompt": str(row["prompt"]),
                                "prediction": prediction})
            self.index = LexicalIndex(self.traces)

    def persist(self) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        (self._state_dir / "REPLAY.json").write_text(json.dumps(
            {"schema": "mini-agi-v16.2-replay-buffer-v1",
             "k": self.k, "traces": self.traces},
            indent=2, sort_keys=True))

    def augment(self, prompt: str, family: str) -> str:
        if self.index is None or not self.traces:
            return prompt
        hits = self.index.rank(prompt, self.k)
        ex = "\n".join(f"{self.traces[i]['prompt']} {self.traces[i]['prediction']}"
                       for i, _ in hits)
        return f"Verified experience:\n{ex}\n\n{prompt}"


ARMS = {"L1": FrozenArm, "L2": RetrievalArm, "L3": SemanticMemoryArm,
        "L4": SkillsArm, "L5": GroundedReplayArm}


def shuffled_label_texts(rows: Sequence[Mapping], seed: int) -> list[str]:
    """NC negative control: same train texts, answers permuted within
    the train set — identical compute, corrupted supervision."""
    answers = [str(r["expected"]) for r in rows]
    n = len(rows)
    rnd = random.Random(seed)
    # strict derangement: no row may keep its own label — otherwise the
    # negative control silently receives partial correct supervision
    for _ in range(64):
        perm = list(range(n))
        rnd.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            break
    else:
        perm = [(i + 1) % n for i in range(n)]
    return [f"{r['prompt']} {answers[perm[i]]}" for i, r in enumerate(rows)]


def shuffled_label_examples(rows: Sequence[Mapping], seed: int) -> list[dict]:
    """Structured NC variant for the corrected trainer: real prompts,
    deranged responses (same permutation semantics as
    shuffled_label_texts)."""
    answers = [str(r["expected"]) for r in rows]
    n = len(rows)
    rnd = random.Random(seed)
    for _ in range(64):
        perm = list(range(n))
        rnd.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            break
    else:
        perm = [(i + 1) % n for i in range(n)]
    return [{"prompt": str(r["prompt"]), "response": answers[perm[i]]}
            for i, r in enumerate(rows)]


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in sorted(Path(path).rglob("*")):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return total
