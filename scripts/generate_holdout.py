#!/usr/bin/env python3
"""Authority-side generator for evaluator-sealed final-holdout files.

The final holdout is the confirmation partition the worker never sees:
the signed plan carries only its manifest digest, the FILE stays with
the evaluation authority, and the qualifier re-derives the digest from
the supplied file (``qualify_campaign1.py --holdout``).

This generator produces holdout rows in the campaign corpus shape —
same rule class as the hidden set, NEW task families, and a vocabulary
disjoint from the corpus inputs (asserted against the corpus actually
in use). It REFUSES to write inside the repository: a holdout committed
to the worker tree would no longer be sealed. A ``<out>.meta.json``
sidecar records the generation seed and time for the authority's
records (the seed is deliberately NOT committed to the repository —
the file + sidecar are the reproducible artifacts, the digest is the
public binding).

Usage:
    python scripts/generate_holdout.py \
        --corpus configs/campaign3_tasks.jsonl \
        --out ~/.local/share/minagi/holdouts/campaign3a-holdout.jsonl \
        --families 4 --per-family 6 --seed 20261008
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_campaign2_tasks import HIDDEN_FAMILIES  # noqa: E402

# Fresh vocabulary pool — domains absent from the campaign-2/3 word
# lists (astronomy, music, architecture, mythology, dance). Collisions
# with the corpus in use are filtered and re-asserted at generation.
HOLDOUT_WORDS = tuple(dict.fromkeys(
    ("nebula pulsar quasar zenith nadir equinox solstice eclipse corona "
     "meteor asteroid galaxy parallax aphelion perihelion azimuth "
     "meridian ecliptic albedo crescent gibbous syzygy umbra penumbra "
     "zodiac cepheid heliopause magnetar blazar redshift "
     # music
     "tempo chord octave cadence fugue sonata aria ballad chorus harmony "
     "melody rhythm timbre vibrato staccato legato crescendo overture "
     "prelude minuet rondo etude nocturne serenade cantata motet "
     "madrigal glissando arpeggio counterpoint tessitura "
     # architecture
     "atrium portico rotunda cupola arcade transept nave apse "
     "clerestory buttress cornice frieze pediment pilaster plinth quoins "
     "soffit spandrel tracery vaulting keystone architrave balustrade "
     "colonnade entablature gargoyle mullion newel oculus "
     # mythology
     "chimera griffin phoenix centaur minotaur cyclops titan oracle "
     "sibyl naiad dryad satyr harpy gorgon hydra pegasus amphora "
     "caduceus labyrinth obelisk pantheon "
     # dance
     "waltz tango bolero fandango mazurka polka jig sarabande gavotte "
     "tarantella quadrille courante bourree").split()))

_RULES = {fam: (rule, fn) for fam, rule, fn in HIDDEN_FAMILIES}


def _corpus_index(corpus_rows):
    """(input words, ids, families) actually used by the corpus."""
    words, ids, fams = set(), set(), set()
    for r in corpus_rows:
        ids.add(str(r["id"]))
        fams.add(str(r["family"]))
        m = re.search(r"Input: (\S+)", str(r.get("prompt", "")))
        if m:
            words.add(m.group(1))
    return words, ids, fams


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True,
                    help="holdout path — must be OUTSIDE the repository")
    ap.add_argument("--families", type=int, default=4)
    ap.add_argument("--per-family", type=int, default=6)
    ap.add_argument("--seed", type=int, required=True,
                    help="generation seed (recorded in the sidecar, "
                         "never committed)")
    ap.add_argument("--rules", default="hid-rev-upper,hid-rot1,"
                                       "hid-first-last,hid-sort-chars",
                    help="hidden-class rules to mirror, comma-separated")
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser().resolve()
    if out == ROOT or ROOT in out.parents:
        print("refusing to write the sealed holdout inside the "
              "repository — the worker tree must never contain it",
              file=sys.stderr)
        return 2

    corpus_rows = [json.loads(l) for l in
                   Path(args.corpus).read_text().splitlines() if l.strip()]
    if not corpus_rows:
        print("corpus is empty", file=sys.stderr)
        return 1
    used_words, corpus_ids, corpus_fams = _corpus_index(corpus_rows)

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    unknown = [r for r in rules if r not in _RULES]
    if unknown:
        print(f"unknown rules: {unknown}; available: {sorted(_RULES)}",
              file=sys.stderr)
        return 1

    pool = [w for w in HOLDOUT_WORDS if w not in used_words]
    need = args.families * args.per_family
    if len(pool) < need:
        print(f"word pool too small: {len(pool)} < {need}", file=sys.stderr)
        return 1
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    rows = []
    wi = iter(pool)
    for i in range(args.families):
        rule_name = rules[i % len(rules)]
        rule, fn = _RULES[rule_name]
        fam = f"holdout-{rule_name.removeprefix('hid-')}-{i}"
        for k in range(args.per_family):
            w = next(wi)
            prompt = f"Task: {rule}\nInput: {w}\nAnswer:"
            rows.append({"id": f"{fam}-{k:02d}", "family": fam,
                         "split": "final_holdout", "prompt": prompt,
                         "expected": fn(w)})

    ids = {r["id"] for r in rows}
    fams = {r["family"] for r in rows}
    leak_ids = sorted(ids & corpus_ids)
    leak_fams = sorted(fams & corpus_fams)
    if leak_ids or leak_fams:
        print(json.dumps({"error": "holdout leakage",
                          "overlapping_ids": leak_ids,
                          "overlapping_families": leak_fams}, indent=2),
              file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, sort_keys=True)
                             for r in rows) + "\n")
    meta = {"generator": "scripts/generate_holdout.py",
            "seed": args.seed,
            "utc": datetime.now(timezone.utc).isoformat(),
            "n_rows": len(rows),
            "families": sorted(fams),
            "rules": rules,
            "corpus": str(Path(args.corpus))}
    (out.parent / (out.name + ".meta.json")).write_text(
        json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps({"out": str(out), "n_rows": len(rows),
                      "families": sorted(fams),
                      "next": "seal via scripts/seal_final_holdout.py "
                              "<holdout> <corpus>"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
