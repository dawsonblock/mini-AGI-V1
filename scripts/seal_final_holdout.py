#!/usr/bin/env python3
"""Seal a final-holdout partition for a v16.6 campaign plan.

The evaluator authority — not the executor — controls the final
holdout. This tool runs on the authority side: given a holdout JSONL
and the campaign corpus, it

  1. builds the DatasetMembershipManifest("final_holdout", ...),
  2. verifies the holdout is disjoint (sample ids AND task families)
     from every partition of the campaign corpus, and
  3. prints the manifest digest to bind into the preregistered plan
     (``final_holdout_digest`` in the campaign config).

The holdout FILE stays with the evaluator; the plan carries only its
digest. At qualification time the evaluator supplies the file via
``qualify_campaign1.py --holdout`` and the digest is re-derived.

Usage:
    python scripts/seal_final_holdout.py HOLDOUT.jsonl CORPUS.jsonl
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import sha256_bytes  # noqa: E402
from minagi.v161.dataset_manifest import (DatasetMember,  # noqa: E402
                                          DatasetMembershipManifest)


def _member(row: dict) -> DatasetMember:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    return DatasetMember(str(row["id"]), str(row["family"]),
                         sha256_bytes(payload),
                         str(row.get("source", "local")),
                         str(row.get("generator", "manual")))


def _load(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    holdout_rows, corpus_rows = _load(Path(sys.argv[1])), _load(Path(sys.argv[2]))
    if not holdout_rows:
        print("holdout file is empty", file=sys.stderr)
        return 1

    holdout = DatasetMembershipManifest(
        "final_holdout", tuple(_member(r) for r in holdout_rows))

    # Disjointness: holdout shares no sample id AND no task family with
    # any corpus partition — the confirmation set must be genuinely new.
    corpus_ids = {str(r["id"]) for r in corpus_rows}
    corpus_fams = {str(r["family"]) for r in corpus_rows}
    bad_ids = sorted(corpus_ids & holdout.sample_ids)
    bad_fams = sorted(corpus_fams & holdout.family_ids)
    if bad_ids or bad_fams:
        print(json.dumps({"error": "holdout leakage",
                          "overlapping_ids": bad_ids,
                          "overlapping_families": bad_fams},
                         indent=2), file=sys.stderr)
        return 1

    print(json.dumps({
        "final_holdout_digest": holdout.digest,
        "n_rows": len(holdout_rows),
        "families": sorted(holdout.family_ids),
        "bind_into": "configs/<campaign>.yaml :: final_holdout_digest",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
