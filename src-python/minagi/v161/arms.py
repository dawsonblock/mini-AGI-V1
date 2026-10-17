"""v16.2 six-arm mechanisms for the HF evaluation path.

Each arm is a *prompt-augmentation policy* over a common deterministic
lexical index — honest mechanisms, not trained-learning doubles:

  L1 frozen          : no augmentation
  L2 retrieval       : top-k raw train exemplars for the query
  L3 semantic-memory : consolidated per-family memory entries (one
                       retained exemplar + support count per family)
  L4 skills          : whole-family skill blocks; the nearest family
                       skill is selected and rendered
  L5a grounded-replay: traces verified during a labeled *practice*
                       phase on train rows (self-verified); hidden
                       labels are never consulted (no leakage)
  L5b verified-replay: replay of independently verified TRAIN examples
                       with recorded provenance (SCI-302) — the
                       comparator that makes the neural incremental
                       comparison interpretable
  L6 neural-adapter  : LoRA trained on train texts (parametric; not
                       implemented here — runner uses peft_trainer)
  L7 adapter+replay  : the same LoRA, evaluated with the L5b replay
                       context (runner wires it; arms.py provides the
                       augmentation)
  NC negative-control: identical LoRA training on label-shuffled
                       train texts; must show ~no forward transfer

Arm states are materialized to disk and digest-bound by
`state_digest` in ExecutedRunReceiptV162 (L1 -> ZERO).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from egai.common.canonical import sha256_bytes

ARM_IDS = ("L1", "L2", "L3", "L4", "L5", "L6", "NC")
#: v16.6 arm vocabulary: L5 is the historical alias of L5a; L5b and L7
#: are the 3B additions (self-verified vs verified replay; adapter plus
#: verified replay).
SELF_VERIFIED_REPLAY_IDS = ("L5", "L5a")
VERIFIED_REPLAY_ID = "L5b"
ADAPTER_REPLAY_ID = "L7"
PARAMETRIC_ARM_IDS = ("L6", ADAPTER_REPLAY_ID, "NC")
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
        "L4": SkillsArm, "L5": GroundedReplayArm, "L5a": GroundedReplayArm}

REPLAY_RECORD_SCHEMA = "mini-agi-v16.6-verified-replay-record-v1"
#: Digest of the default verification policy: a TRAIN row's established
#: label is the verified answer (no model prediction involved).
LABEL_VERIFIER_DIGEST = sha256_bytes(
    b"mini-agi-v16.6-established-train-labels-v1")


@dataclass(frozen=True)
class VerifiedReplayRecord:
    """Provenance for one admitted replay trace (SCI-302).

    The record binds the task, its prompt/answer content digests, the
    train-split membership digest it was admitted under, and the
    verification policy — the content itself is stored separately for
    retrieval.
    """
    task_id: str
    family_id: str
    prompt_digest: str
    answer_digest: str
    split_digest: str
    verifier_digest: str
    verification_status: str


class OracleVerifiedReplayArm:
    """L5b: replay of independently verified TRAIN examples (SCI-302).

    Unlike L5a (self-verified practice traces, which Campaign 3A left
    empty), this baseline admits every TRAIN row whose membership and
    content verify: the answer is the row's established training label,
    bound by digest — never a model prediction, and never a
    hidden/validation/holdout row. Duplicate or conflicting records are
    rejected with reasons, retrieval is budgeted
    (``max_block_chars``), and every ``augment()`` call is logged (hit
    task ids and scores, injected/trimmed characters, the
    augmented-prompt digest) so the evidence records what the
    comparator actually did.
    """

    arm_id = VERIFIED_REPLAY_ID
    uses_state = True

    def __init__(self, k: int = 2, max_block_chars: int = 2000):
        if int(k) <= 0:
            raise ValueError("k must be >= 1")
        if int(max_block_chars) <= 0:
            raise ValueError("max_block_chars must be >= 1")
        self.k = int(k)
        self.max_block_chars = int(max_block_chars)
        self.records: list[VerifiedReplayRecord] = []
        self.entries: list[dict] = []
        self.rejected: list[dict] = []
        self.retrieval_log: list[dict] = []
        self.index: LexicalIndex | None = None
        self.split_digest = ""
        self.verifier_digest = LABEL_VERIFIER_DIGEST
        self._state_dir: Path | None = None
        #: Optional tokenizer-backed counter (set by the runner) so the
        #: evidence records token cost, not only characters.
        self.token_counter: Callable[[str], int] | None = None

    def build_state(self, state_dir: Path, rows: Sequence[Mapping], *,
                    split_digest: str = "",
                    verifier: Callable[[Mapping], bool] | None = None,
                    verifier_digest: str = "",
                    excluded_ids: Sequence[str] = ()) -> None:
        """Admit verified TRAIN rows; reject everything else with a
        recorded reason."""
        self._state_dir = Path(state_dir)
        self.split_digest = str(split_digest)
        self.verifier_digest = str(verifier_digest) or LABEL_VERIFIER_DIGEST
        excluded = {str(i) for i in excluded_ids}
        self.records, self.entries, self.rejected = [], [], []
        seen_ids: set[str] = set()
        seen_prompts: dict[str, str] = {}
        for row in rows:
            task_id = str(row.get("id", ""))
            if not task_id:
                self.rejected.append({"task_id": "", "reason": "missing id"})
                continue
            if task_id in excluded:
                self.rejected.append({"task_id": task_id,
                                      "reason": "excluded id"})
                continue
            split = str(row.get("split", "train"))
            if split != "train":
                self.rejected.append({"task_id": task_id,
                                      "reason": f"non-train split {split!r}"})
                continue
            prompt = str(row.get("prompt", ""))
            answer = str(row.get("expected", ""))
            if not prompt or not answer:
                self.rejected.append({"task_id": task_id,
                                      "reason": "empty prompt or answer"})
                continue
            p_digest = sha256_bytes(prompt.encode())
            a_digest = sha256_bytes(answer.encode())
            if task_id in seen_ids:
                self.rejected.append({"task_id": task_id,
                                      "reason": "duplicate task id"})
                continue
            prior = seen_prompts.get(p_digest)
            if prior is not None:
                self.rejected.append({
                    "task_id": task_id,
                    "reason": "conflicting answer for prompt"
                    if prior != a_digest else "duplicate prompt"})
                continue
            if verifier is not None and not verifier(row):
                self.rejected.append({"task_id": task_id,
                                      "reason": "verifier rejected"})
                continue
            seen_ids.add(task_id)
            seen_prompts[p_digest] = a_digest
            self.records.append(VerifiedReplayRecord(
                task_id=task_id, family_id=str(row.get("family", "")),
                prompt_digest=p_digest, answer_digest=a_digest,
                split_digest=self.split_digest,
                verifier_digest=self.verifier_digest,
                verification_status="PASS"))
            self.entries.append({"id": task_id, "prompt": prompt,
                                 "answer": answer,
                                 "family": str(row.get("family", ""))})
        self.index = LexicalIndex(self.entries) if self.entries else None

    def persist(self) -> None:
        if self._state_dir is None:
            raise ValueError("build_state() must run before persist()")
        self._state_dir.mkdir(parents=True, exist_ok=True)
        (self._state_dir / "REPLAY.json").write_text(json.dumps(
            {"schema": "mini-agi-v16.6-verified-replay-buffer-v1",
             "k": self.k, "max_block_chars": self.max_block_chars,
             "split_digest": self.split_digest,
             "verifier_digest": self.verifier_digest,
             "records": [asdict(r) for r in self.records],
             "entries": self.entries,
             "rejected": self.rejected},
            indent=2, sort_keys=True))

    def verify_state(self) -> int:
        """Re-derive every persisted digest; raise on tampering.

        Returns the number of verified records.
        """
        if self._state_dir is None:
            raise ValueError("no state directory")
        doc = json.loads((self._state_dir / "REPLAY.json").read_text())
        entries = {str(e["id"]): e for e in doc.get("entries", [])}
        for rec in doc.get("records", []):
            e = entries.get(str(rec["task_id"]))
            if e is None:
                raise ValueError(
                    f"record {rec['task_id']} has no stored content")
            if sha256_bytes(str(e["prompt"]).encode()) \
                    != rec["prompt_digest"]:
                raise ValueError(
                    f"prompt digest mismatch for {rec['task_id']}")
            if sha256_bytes(str(e["answer"]).encode()) \
                    != rec["answer_digest"]:
                raise ValueError(
                    f"answer digest mismatch for {rec['task_id']}")
            if rec["split_digest"] != doc.get("split_digest", ""):
                raise ValueError(
                    f"split digest mismatch for {rec['task_id']}")
        return len(doc.get("records", []))

    def _log(self, prompt: str, *, no_hit: bool, hits: list[dict],
             injected: int, trimmed: int, augmented: str) -> None:
        entry = {
            "query_digest": sha256_bytes(prompt.encode()),
            "no_hit": bool(no_hit), "hits": hits,
            "injected": int(injected), "trimmed": int(trimmed),
            "augmented_digest": sha256_bytes(augmented.encode()),
            "augmented_chars": len(augmented),
            "budget_chars": self.max_block_chars}
        if self.token_counter is not None:
            try:
                entry["prompt_tokens"] = int(self.token_counter(prompt))
                entry["augmented_tokens"] = int(self.token_counter(augmented))
                entry["injected_tokens"] = max(
                    0, entry["augmented_tokens"] - entry["prompt_tokens"])
            except Exception:  # noqa: BLE001 — evidence stays character-only
                pass
        self.retrieval_log.append(entry)

    def augment(self, prompt: str, family: str) -> str:
        if self.index is None or not self.entries:
            self._log(prompt, no_hit=True, hits=[], injected=0, trimmed=0,
                      augmented=prompt)
            return prompt
        hits = self.index.rank(prompt, self.k)
        blocks: list[str] = []
        kept: list[dict] = []
        injected = trimmed = 0
        for i, score in hits:
            entry = self.entries[i]
            block = f"{entry['prompt']} {entry['answer']}"
            room = self.max_block_chars - injected
            if room <= 0:
                trimmed += len(block)
                continue
            if len(block) > room:
                trimmed += len(block) - room
                block = block[:room]
            blocks.append(block)
            injected += len(block)
            kept.append({"task_id": entry["id"],
                         "score": round(float(score), 6)})
        if not blocks:
            self._log(prompt, no_hit=True, hits=[], injected=0, trimmed=trimmed,
                      augmented=prompt)
            return prompt
        augmented = ("Verified experience:\n" + "\n".join(blocks)
                     + f"\n\n{prompt}")
        self._log(prompt, no_hit=False, hits=kept, injected=injected,
                  trimmed=trimmed, augmented=augmented)
        return augmented

    def retrieval_stats(self, since: int = 0) -> dict:
        """Aggregate retrieval evidence for the run's REPLAY_MANIFEST.

        ``since`` selects the window of retrieval-log entries belonging
        to one arm when the instance is shared (L5b and L7).
        """
        window = self.retrieval_log[int(since):]
        hit_rows = [r for r in window if not r["no_hit"]]
        stats = {
            "records": len(self.records),
            "rejected": len(self.rejected),
            "queries": len(window),
            "hit_queries": len(hit_rows),
            "no_hit_queries": len(window) - len(hit_rows),
            "injected_chars": sum(r["injected"] for r in window),
            "trimmed_chars": sum(r["trimmed"] for r in window),
            "retrieved_task_ids": sorted({h["task_id"]
                                          for r in hit_rows
                                          for h in r["hits"]}),
        }
        if self.token_counter is not None:
            stats["prompt_tokens"] = sum(int(r.get("prompt_tokens", 0))
                                         for r in window)
            stats["augmented_tokens"] = sum(int(r.get("augmented_tokens", 0))
                                            for r in window)
            stats["injected_tokens"] = sum(int(r.get("injected_tokens", 0))
                                           for r in window)
        return stats


ARMS[VERIFIED_REPLAY_ID] = OracleVerifiedReplayArm


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
