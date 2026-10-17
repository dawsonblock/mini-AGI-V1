"""Evaluator-owned sealed holdout (v4.0.3-r2 / Phase 5).

The previous Campaign 4A final holdout shipped with expected answers —
it is a public development benchmark, not a confirmation surface. This
module is the replacement custody model:

* the EVALUATOR generates fresh tasks whose answers the training
  agent never sees — answers live in a sealed store (0700 directory,
  0600 files) outside the research workspace;
* the store commits the dataset identity — the confirmation binds the
  commitment, not the plaintext;
* a frozen candidate arrives; the evaluator executes inference itself
  (``independent_model_executor``), scores privately, applies gates
  and signs the outcome;
* the research process receives the aggregate result and the signed
  receipt — never per-row answers or labels;
* an eval set is single-use: repeated adaptive tuning against the
  same final holdout is refused (attempts are appended to a sealed
  log; the second request for the same eval id raises).

``within_holdout`` remains a different surface — its answers ship in
the corpus for evaluator-side rescoring (statistical independence +
evaluator-side custody, not third-party attestation). A hardened
deployment can place those answers behind this same seal with a
second operator.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from pathlib import Path

from egai.common.canonical import digest

SEALED_SCHEMA = "mini-agi-v4.0.3-sealed-evalset-v1"
ANSWERS_FILE = "ANSWERS.json"
PUBLIC_FILE = "TASKS-PUBLIC.json"
ATTEMPTS_FILE = "ATTEMPTS.log"

#: Fresh task families the training agent cannot have memorized —
#: generated per-eval-id, distinct from every shipped corpus family.
#: Each generator is (name, prompt template, answer(w, n)).
FRESH_FAMILY_GENERATORS = (
    ("repeat-word", "Repeat the word {w} twice",
     lambda w, n: f"{w} {w}"),
    ("swap-letters", "Swap the first and last letter of {w}",
     lambda w, n: (w[-1] + w[1:-1] + w[0]) if len(w) > 1 else w),
    ("count-letter", "How many times does 'e' appear in {w}?",
     lambda w, n: str(w.count("e"))),
    ("nth-letter", "What is the {n}th letter of {w}?",
     lambda w, n: w[min(n, len(w)) - 1]),
    ("capitalize-second", "Capitalize every second letter of {w}",
     lambda w, n: "".join(c.upper() if i % 2 else c
                          for i, c in enumerate(w))),
)


def _fresh_vocab(seed: int, n: int) -> list[str]:
    """Deterministic per-eval vocabulary — fresh words that are not
    corpus fixtures."""
    rng = __import__("random").Random(seed)
    letters = "abcdefghijlmnoprstuv"
    return ["".join(rng.choice(letters) for _ in range(6))
            for _ in range(n)]


def generate_fresh_tasks(eval_seed: int, *, n_families: int = 4,
                         items_per_family: int = 3) -> list[dict]:
    """Fresh prompt/expected pairs the training agent has never seen —
    deterministic on the eval seed (committed, not shipped)."""
    rng = __import__("random").Random(eval_seed)
    fams = rng.sample(list(FRESH_FAMILY_GENERATORS),
                      k=min(n_families, len(FRESH_FAMILY_GENERATORS)))
    vocab = _fresh_vocab(eval_seed + 1, n_families * items_per_family)
    tasks: list[dict] = []
    ix = 0
    for fam_name, tmpl, answer in fams:
        for _ in range(items_per_family):
            w = vocab[ix]
            ix += 1
            n = rng.randint(2, min(5, len(w)))
            tasks.append({
                "id": f"{fam_name}-{ix}",
                "family": fam_name,
                "prompt": tmpl.format(w=w, n=n),
                "expected": answer(w, n)})
    return tasks


class SealedHoldoutStore:
    """Evaluator-custody task store. The research agent's access is a
    filesystem boundary — answers never leave this directory."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def _dir(self, eval_id: str) -> Path:
        return self.root / eval_id

    def create_eval_set(self, tasks: list[dict], *,
                        eval_id: str | None = None) -> dict:
        """Seal a fresh task set: prompts go to TASKS-PUBLIC.json (the
        only surface a candidate's I/O sees), answers to ANSWERS.json
        (0600, evaluator-only). Returns the dataset commitment."""
        eval_id = eval_id or f"eval-{uuid.uuid4().hex[:12]}"
        d = self._dir(eval_id)
        d.mkdir(parents=True)
        os.chmod(d, 0o700)
        public = [{"id": t["id"], "family": t["family"],
                   "prompt": t["prompt"]} for t in tasks]
        answers = {t["id"]: t["expected"] for t in tasks}
        (d / PUBLIC_FILE).write_text(
            json.dumps({"schema": SEALED_SCHEMA, "eval_id": eval_id,
                        "tasks": public}, indent=2, sort_keys=True))
        apath = d / ANSWERS_FILE
        apath.write_text(json.dumps(answers, indent=2, sort_keys=True))
        os.chmod(apath, 0o600)
        (d / ATTEMPTS_FILE).touch()
        os.chmod(d / ATTEMPTS_FILE, 0o600)
        return {"eval_id": eval_id,
                "dataset_digest": self.dataset_digest(eval_id),
                "n_tasks": len(tasks),
                "families": sorted({t["family"] for t in tasks})}

    def public_tasks(self, eval_id: str) -> list[dict]:
        doc = json.loads((self._dir(eval_id) / PUBLIC_FILE).read_text())
        return doc["tasks"]

    def load_answers(self, eval_id: str) -> dict:
        """Evaluator-side only — the sealed answers."""
        return json.loads(
            (self._dir(eval_id) / ANSWERS_FILE).read_text())

    def dataset_digest(self, eval_id: str) -> str:
        """Commitment over id+prompt+expected — bound by receipts; the
        plaintext is not."""
        tasks = self.public_tasks(eval_id)
        answers = self.load_answers(eval_id)
        return digest([
            {"id": t["id"],
             "prompt_sha256": hashlib.sha256(
                 t["prompt"].encode()).hexdigest(),
             "expected_sha256": hashlib.sha256(
                 str(answers[t["id"]]).encode()).hexdigest()}
            for t in tasks])

    def record_attempt(self, eval_id: str, *, max_evaluations: int = 1):
        """Append an evaluation attempt; refuse re-use past the cap —
        a final holdout is single-use, not a tuning oracle."""
        d = self._dir(eval_id)
        attempts = (d / ATTEMPTS_FILE).read_text().split()
        if len(attempts) >= max_evaluations:
            raise PermissionError(
                f"eval set {eval_id} already evaluated "
                f"{len(attempts)} time(s) — repeated adaptive tuning "
                "against the final holdout is refused")
        with (d / ATTEMPTS_FILE).open("a") as fh:
            fh.write(f"{uuid.uuid4().hex[:8]}\n")


def run_sealed_confirmation(store: SealedHoldoutStore, eval_id: str,
                            signer, *, model_id: str,
                            model_revision: str, adapter_dir=None,
                            gates: dict | None = None,
                            generation_config: dict | None = None,
                            scoring_version: str = "exact-match-v2",
                            model_loader=None, out_dir=None,
                            max_evaluations: int = 1) -> dict:
    """Evaluator-owned sealed confirmation: load tasks+answers from
    custody, execute the candidate artifact itself, score privately,
    apply the declared gates, sign the execution receipt, and return
    ONLY the aggregate result + receipt — no answers, no labels."""
    from .independent_model_executor import execute_evaluation
    store.record_attempt(eval_id, max_evaluations=max_evaluations)
    public = store.public_tasks(eval_id)
    answers = store.load_answers(eval_id)
    tasks = [{"id": t["id"], "family": t["family"],
              "prompt": t["prompt"],
              "expected": answers[t["id"]]} for t in public]
    res = execute_evaluation(
        signer, model_id=model_id, model_revision=model_revision,
        tasks=tasks, adapter_dir=adapter_dir,
        generation_config=generation_config,
        scoring_version=scoring_version,
        model_loader=model_loader, out_dir=out_dir)
    acc = (sum(res["scores"].values()) / len(res["scores"])) \
        if res["scores"] else 0.0
    gates = dict(gates or {})
    min_acc = float(gates.get("min_accuracy", 0.0))
    return {
        "eval_id": eval_id,
        "dataset_digest": store.dataset_digest(eval_id),
        "n_tasks": res["n_tasks"], "n_correct": res["n_correct"],
        "accuracy": acc,
        "gate": {"min_accuracy": min_acc,
                 "status": "PASS" if acc >= min_acc else "FAIL"},
        "execution_mode": res["execution_mode"],
        "execution_receipt": res["receipt"],
        # deliberate: no prompts, no expected answers, no per-row
        # scores leave the evaluator — only the aggregate + receipt
    }
