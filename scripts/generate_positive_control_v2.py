#!/usr/bin/env python3
"""Campaign 3B.2 — expanded held-out benchmark (v2 corpus).

The v1 held-out block is 12 tasks across 4 families, so one answer moves
accuracy by 8.33 pp and a paired comparison needs a ~50 pp effect to be
detectable at 80% power. That is adequate for the large composition
effect (29 pp) but useless for a rank ladder, where effects are expected
to be a few points: detecting 15 pp needs ~105 tasks.

This generator produces a larger held-out block while keeping the
TRAINING block byte-identical to v1, so a run trained on v2 sees exactly
the same training condition as a run trained on v1 and the two are
comparable. Only the evaluation changes.

  train     unchanged, byte-identical to v1 (6 families x 5 words)
  hidden    8 held-out families x 15 words = 120 tasks
  retention / security  unchanged

Usage:
  python3 scripts/generate_positive_control_v2.py \
      --out configs/positive_control_tasks_v2.jsonl \
      --holdout-out /tmp/pc_v2_holdout.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_positive_control import (WORDS, _rotate, _row,  # noqa: E402
                                       _task, _vowel_upper, build_corpus,
                                       holdout_digest, safety_probes,
                                       smoke_lures)


def _rotate2(w):
    return _rotate(w, 2)


def _last_upper(w):
    return w[:-1] + w[-1].upper() if w else w


def _first3(w):
    return w[:3]


def _no_vowels(w):
    return "".join(c for c in w if c not in "aeiou")


#: The held-out families, v1's four plus four more, all distinct
#: transformations of the input string. Kept disjoint from TRAIN_FAMILIES
#: so the family-disjointness requirement still holds.
HELDOUT_FAMILIES_V2 = {
    "pc-vowels": ("Uppercase only the vowels.", _vowel_upper),
    "pc-dup": ("Repeat the word twice.", lambda w: w + w),
    "pc-len": ("Return the number of characters.", lambda w: str(len(w))),
    "pc-mirror": ("Append the reversed word to the word.",
                  lambda w: w + w[::-1]),
    "pc-rot2": ("Rotate each letter forward by two (y and z wrap).",
                _rotate2),
    "pc-lastupper": ("Uppercase only the last character.", _last_upper),
    "pc-first3": ("Return the first three characters.", _first3),
    "pc-novowel": ("Remove every vowel.", _no_vowels),
}

#: Deterministic word pool. 45 distinct three-letter lowercase words,
#: generated rather than hand-listed so the pool cannot drift by typo.
def _word_pool():
    pool = list(WORDS)
    seen = set(pool)
    for a in "bcdfghjklmnprstvwz":
        for b in "aeiou":
            for c in "bcdfgklmnprst":
                w = a + b + c
                if w not in seen:
                    seen.add(w)
                    pool.append(w)
                if len(pool) >= len(WORDS) + 45:
                    return pool
    return pool


def build_corpus_v2(words_per_family: int = 15) -> list[dict]:
    # Training rows come from v1 unchanged, so the training condition is
    # identical and v1-trained and v2-trained runs are comparable.
    rows = [r for r in build_corpus() if r["split"] == "train"]
    pool = _word_pool()
    # offset each family's slice so families do not all share one word set
    for i, (fam, (line, fn)) in enumerate(sorted(HELDOUT_FAMILIES_V2.items())):
        start = (i * 5) % len(pool)
        picked = [pool[(start + j) % len(pool)]
                  for j in range(words_per_family)]
        for j, word in enumerate(picked):
            rows.append(_row(f"pc2-hd-{fam}-{j}", "hidden", fam,
                             _task(line, word), fn(word),
                             "positive-control-v2"))
    # retention and security blocks carried over unchanged
    v1 = build_corpus()
    rows += [r for r in v1 if r["split"] == "retention"]
    rows += list(safety_probes())
    rows += list(smoke_lures())
    return rows


def build_holdout_v2(words_per_family: int = 4) -> list[dict]:
    """Evaluator-held rows: unseen words under distinct family names."""
    rows, pool = [], _word_pool()
    for i, (fam, (line, fn)) in enumerate(sorted(HELDOUT_FAMILIES_V2.items())):
        start = (i * 5 + 31) % len(pool)
        for j in range(words_per_family):
            word = pool[(start + j) % len(pool)]
            rows.append(_row(f"pc2-ho-{fam}-{j}", "hidden",
                             f"holdout-{fam}", _task(line, word), fn(word),
                             "positive-control-v2"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/positive_control_tasks_v2.jsonl")
    ap.add_argument("--holdout-out", default=None)
    ap.add_argument("--words-per-family", type=int, default=15)
    args = ap.parse_args()

    corpus = build_corpus_v2(args.words_per_family)
    # determinism self-check: every held-out expectation re-derives
    for row in corpus:
        if row["split"] == "hidden":
            fam = row["family"]
            _line, fn = HELDOUT_FAMILIES_V2[fam]
            payload = row["prompt"].split("Input: ", 1)[1] \
                .rsplit("\nAnswer:", 1)[0]
            assert fn(payload) == row["expected"], row["id"]

    v1_train = [r for r in build_corpus() if r["split"] == "train"]
    v2_train = [r for r in corpus if r["split"] == "train"]
    identical = json.dumps(v1_train, sort_keys=True) == \
        json.dumps(v2_train, sort_keys=True)
    if not identical:
        raise SystemExit("training block drifted from v1 — the two corpora "
                         "would not be comparable")
    print("training block byte-identical to v1: True")

    out = ROOT / args.out
    out.write_text("\n".join(json.dumps(r, sort_keys=True)
                             for r in corpus) + "\n")
    hidden = [r for r in corpus if r["split"] == "hidden"]
    fams = sorted({r["family"] for r in hidden})
    print(f"wrote {len(corpus)} rows to {out}")
    print(f"  hidden: {len(hidden)} tasks across {len(fams)} families "
          f"({len(hidden)/len(fams):.0f} words each)")
    print(f"  one answer now moves accuracy by "
          f"{100/len(hidden):.2f} pp (was 8.33)")

    holdout = build_holdout_v2()
    d = holdout_digest(holdout)
    if args.holdout_out:
        hp = Path(args.holdout_out)
        hp.write_text("\n".join(json.dumps(r, sort_keys=True)
                                for r in holdout) + "\n")
        print(f"wrote {len(holdout)} evaluator-held rows to {hp}")
    print(f"final_holdout_digest: {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
