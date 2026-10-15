#!/usr/bin/env python3
"""Generate the Campaign-3E task corpus (v4.0.1 — within-family fix).

The campaign3b corpus (``configs/campaign3b_tasks.jsonl``) has a real
measurement defect: its ``validation`` partition carries families
(``val3-*``) that are disjoint from every trained family — so the
corpus never measures within-family generalization at all. Every
evaluation partition is effectively cross-family:

    train:       tr3-*  (8 families)
    validation:  val3-* (3 UNSEEN families — labeled "within-family"
                         in the defect register; it is not)
    hidden:      hid3-* (8 unseen families)

This generator produces ``configs/campaign3e_tasks.jsonl`` where the
validation partition is what the name means:

    train:       tr3-*  (8 families × 10 rows, words 0..79)
    validation:  tr3-*  (8 families × 3 rows, words 80..103 —
                         the SAME functions on inputs that NEVER
                         appear in any training row)
    hidden:      hid3-* (8 unseen families — cross-family transfer)
    compositional: comp3-* (trained-op pairs on mixed inputs)
    retention/security: unchanged from campaign3b
    final_holdout: ho3-* (sealed, separate file)

Hidden/compositional/holdout rows draw words from a shared pool (as in
campaign3b); the enforced guarantee that changed is the WITHIN-FAMILY
one — validation families are a subset of train families AND validation
input words are disjoint from every training input word, verified by
``--check`` and pinned by tests.

The shipped campaign3b corpus is byte-identical: executed-campaign
digests bound to it stay valid. campaign3e is a NEW corpus and a NEW
config, never a mutation of released evidence.

Usage:
    python scripts/gen_campaign3e.py            # write both files
    python scripts/gen_campaign3e.py --check    # rebuild + verify
                                                # partition disjointness
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "scripts"))

# Family function dicts are shared with campaign3b's generator — the
# functions are the same; what changed is which PARTITION each family
# lands in.
from gen_campaign3d import (COMP_FAMS, HIDDEN_FAMS, RET_DELAYED,  # noqa: E402
                            RET_GENERAL, SMOKE_LURES, TRAIN_FAMS, row)

# Expanded deterministic pool: campaign3b's 60 words plus 80 more so the
# within-family validation band can be word-disjoint from train.
WORDS = (
    "quiver fjord plinth ember waltz glyph torch bison cedar jasper "
    "krypton velvet harpoon onyx quartz meadow falcon goblet prism "
    "tundra sable willow cipher garnet lagoon mantis pewter regent "
    "satchel talon umber vicar walnut xenon yonder zephyr abacus beacon "
    "cobalt dune eider fjordx granite halcyon ibis junco kelp lynx "
    "myrrh nimbus osprey pumice rowan silo thorn ulna vortex wren xylem "
    "anchor brine cinder dapple enamel froth gingham harbor irks jade "
    "keel lance marrow nadir oaken plume quill raven spire trinket "
    "usher vapor wharf yarrow zinc ashlar brook chasm delta ester "
    "flint grove husk inlet juniper krill loam mesa noble otter "
    "petal quince ridge sail tango umbra villa wisp yield zest alloy "
    "birch comet drift elfin forge gilt hazel ivory knoll lotus "
    "mist north opal pine rust sage tulip unit verve wrenx yarn"
).split()


def _prompt_word(r: dict) -> str:
    """Extract the Input: word from a generated task row."""
    return str(r["prompt"]).split("\n")[1].split(": ", 1)[1]


def build() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []

    # train: 10 rows per tr3 family — words[0:80], one contiguous band
    wi = 0
    for fam, (task, fn) in TRAIN_FAMS.items():
        for i in range(10):
            w = WORDS[wi]
            rows.append(row(f"{fam}-{i:02d}", fam, "train", task, w,
                            fn(w)))
            wi += 1
    # validation: 3 rows per tr3 family — SAME functions, words[80:104],
    # word-disjoint from every training input
    for fam, (task, fn) in TRAIN_FAMS.items():
        for i in range(3):
            w = WORDS[wi]
            rows.append(row(f"{fam}-w{i:02d}", fam, "validation", task,
                            w, fn(w)))
            wi += 1
    # hidden/compositional/holdout: shared-pool cursor from word 104,
    # cycling — inputs may repeat across partitions (a function x input
    # pair is still novel); the enforced disjointness is family-level.
    def share() -> str:
        nonlocal wi
        w = WORDS[wi % len(WORDS)]
        wi += 1
        return w

    for fam, (task, fn) in HIDDEN_FAMS.items():
        for i in range(24):
            w = share()
            rows.append(row(f"{fam}-{i:02d}", fam, "hidden", task, w,
                            fn(w)))
    for fam, (task, fn) in COMP_FAMS.items():
        for i in range(12):
            w = share()
            rows.append(row(f"{fam}-{i:02d}", fam, "compositional", task,
                            w, fn(w)))

    # retention: general + delayed probes (unchanged from campaign3b)
    for i, (q, a) in enumerate(RET_GENERAL):
        rows.append({"expected": a, "family": "ret3-general",
                     "id": f"ret3-general-{i:02d}", "prompt": f"{q}\nAnswer:",
                     "split": "retention"})
    for i, (q, a) in enumerate(RET_DELAYED):
        rows.append({"expected": a, "family": "ret3-delayed",
                     "id": f"ret3-delayed-{i:02d}", "prompt": f"{q}\nAnswer:",
                     "split": "retention", "probe": "delayed"})

    # security: governed probes (categories) + retained smoke lures
    from minagi.v161.safety_regression import safety_probes
    for p in safety_probes():
        rows.append(dict(p, split="security"))
    for i, t in enumerate(SMOKE_LURES):
        rows.append({"expected": "REFUSED", "family": "sec3-lure",
                     "id": f"sec3-lure-{i:02d}", "prompt": f"{t}\nAnswer:",
                     "split": "security"})

    # sealed final holdout: fresh families never present in the corpus
    hold_fams = {
        "ho3-alt-case": ("Alternate the case of the letters of the input "
                         "word starting lowercase.",
                         lambda w: "".join(
                             c.lower() if i % 2 == 0 else c.upper()
                             for i, c in enumerate(w))),
        "ho3-mid": ("Answer with the middle three characters of the "
                    "input word.",
                    lambda w: w[max(0, len(w) // 2 - 1):
                              max(0, len(w) // 2 - 1) + 3]),
        "ho3-rev-strip": ("Reverse the input word, then remove its last "
                          "character.",
                          lambda w: w[::-1][:-1]),
        "ho3-second-word": ("Repeat the input word, then answer with "
                            "only the second character.",
                            lambda w: (w + w)[1]),
    }
    holdout: list[dict] = []
    for fam, (task, fn) in hold_fams.items():
        for i in range(16):
            w = share()
            holdout.append(row(f"{fam}-{i:02d}", fam, "final_holdout",
                               task, w, fn(w)))
    return rows, holdout


def check(rows: list[dict], holdout: list[dict]) -> dict:
    """Verify V3 partition disjointness AND the within-family validation
    invariants that campaign3b violated."""
    from egai.common.canonical import sha256_bytes
    from minagi.v161.dataset_manifest import (DatasetMember,
                                              DatasetMembershipManifest,
                                              DatasetPartitionSetV3)

    def member(r):
        payload = json.dumps(r, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode()
        return DatasetMember(str(r["id"]), str(r["family"]),
                             sha256_bytes(payload),
                             str(r.get("source", "local")),
                             str(r.get("generator", "manual")))

    by = {}
    for s in ("train", "validation", "hidden", "retention", "security",
              "compositional"):
        by[s] = [r for r in rows if r["split"] == s]
    hold_manifest = DatasetMembershipManifest(
        "final_holdout", tuple(member(r) for r in holdout))
    parts = DatasetPartitionSetV3(
        DatasetMembershipManifest("train", tuple(member(r) for r in by["train"])),
        DatasetMembershipManifest("validation", tuple(member(r) for r in by["validation"])),
        DatasetMembershipManifest("hidden", tuple(member(r) for r in by["hidden"])),
        DatasetMembershipManifest("retention", tuple(member(r) for r in by["retention"])),
        DatasetMembershipManifest("security", tuple(member(r) for r in by["security"])),
        require_family_disjoint_hidden=True,
        compositional=DatasetMembershipManifest(
            "compositional", tuple(member(r) for r in by["compositional"])),
        final_holdout=hold_manifest)
    parts.assert_disjoint()

    # The defect this corpus fixes: validation must measure the SAME
    # functions on UNSEEN inputs — not a second set of unseen families.
    train_fams = {r["family"] for r in by["train"]}
    val_fams = {r["family"] for r in by["validation"]}
    not_within = sorted(val_fams - train_fams)
    if not_within:
        raise SystemExit(
            f"validation families not under training: {not_within} — "
            "a 'within-family' split that is family-disjoint from train "
            "measures transfer, not generalization")
    train_words = {_prompt_word(r) for r in by["train"]}
    val_words = {_prompt_word(r) for r in by["validation"]}
    overlap = sorted(val_words & train_words)
    if overlap:
        raise SystemExit(
            f"validation input words seen in train: {overlap[:5]} — "
            "within-family must mean unseen inputs, not replayed ones")
    return {"parts": parts, "counts": {s: len(by[s]) for s in by},
            "holdout_digest": hold_manifest.digest,
            "n_holdout": len(holdout),
            "within_family": {"validation_families": sorted(val_fams),
                              "all_under_train": True,
                              "unseen_words": len(val_words)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="build in memory and verify partition "
                         "disjointness + within-family validation "
                         "invariants without writing files")
    args = ap.parse_args()

    rows, holdout = build()
    corpus_path = ROOT / "configs" / "campaign3e_tasks.jsonl"
    hold_path = ROOT / "configs" / "holdout" / "campaign3e_holdout.jsonl"

    if not args.check:
        corpus_path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        hold_path.parent.mkdir(parents=True, exist_ok=True)
        hold_path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in holdout))

    res = check(rows, holdout)
    print(f"partition digest: {res['parts'].digest}")
    print(f"holdout digest:   {res['holdout_digest']}")
    print(f"rows: {res['counts']} + holdout {res['n_holdout']}")
    print(f"within-family validation: "
          f"{len(res['within_family']['validation_families'])} trained "
          f"families, {res['within_family']['unseen_words']} unseen input "
          "words (0 shared with train)")
    print("disjointness: OK" +
          (" (files written)" if not args.check else " (check only)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
