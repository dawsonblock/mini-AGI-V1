#!/usr/bin/env python3
"""Finalize Campaign 3A: assemble banked evidence, qualify, publish.

Pipeline (matching the campaign-2 results-branch pattern):

  1. verify every requested seed is banked locally
     (``<ckpt-dir>/ck_seed-*.tar.gz``)
  2. assemble a storage root from the banked tarballs (meta first, then
     each seed: campaign evidence + adapters + arm states + the
     reconstructed authority ledger)
  3. run the INDEPENDENT qualifier with the authority-held holdout file
     (``qualify_campaign1.py --holdout``): the sealed digest is
     re-derived from the file, every metric is recomputed from the
     signed prediction records, and the gates are applied as
     preregistered
  4. write a results report (decision, gate reasons, inventory)
  5. optionally publish a results branch with the evidence + signed
     qualification record + report. Private keys are NEVER committed:
     ``.keys/`` is excluded by construction.

Usage:
  python3 scripts/campaign3a_finalize.py \
      --ckpt-dir /tmp/c3a_ckpt \
      --storage /tmp/c3a_finalize/storage \
      --holdout ~/.local/share/minagi/holdouts/campaign3a-holdout.jsonl \
      --results-branch results/campaign3a-v166 [--push]

Exit codes: 0 qualification completed (any decision), 3 seeds missing,
4 setup/qualifier failure.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[1]
CID_DEFAULT = "campaign3a-v166"


def _run(cmd, cwd=None, timeout=3600):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       timeout=timeout)
    return r.returncode, (r.stdout or ""), (r.stderr or "")


def _decision_of(doc: dict) -> str:
    return doc["value"]["decision"] if "value" in doc else doc["decision"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/tmp/c3a_ckpt")
    ap.add_argument("--storage", default="/tmp/c3a_finalize/storage")
    ap.add_argument("--holdout", required=True,
                    help="authority-held sealed holdout JSONL")
    ap.add_argument("--campaign-id", default=CID_DEFAULT)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--repo", default=str(REPO_DEFAULT))
    ap.add_argument("--results-branch", default=None,
                    help="publish evidence to this branch (created from "
                         "origin/main)")
    ap.add_argument("--push", action="store_true",
                    help="push the results branch to origin")
    ap.add_argument("--force", action="store_true",
                    help="wipe a non-empty --storage first")
    args = ap.parse_args(argv)

    ckpt = Path(args.ckpt_dir).expanduser()
    storage = Path(args.storage).expanduser().resolve()
    holdout = Path(args.holdout).expanduser()
    repo = Path(args.repo).resolve()
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]

    # --- 1. banked-seed completeness ---------------------------------
    missing = [s for s in seeds
               if not (ckpt / f"ck_seed-{s}.tar.gz").is_file()]
    if missing:
        print(f"[finalize] seeds not banked yet: {missing} — "
              "campaign still running?", file=sys.stderr)
        return 3
    if not holdout.is_file():
        print(f"[finalize] holdout file missing: {holdout}", file=sys.stderr)
        return 4
    meta = ckpt / "ck_meta.tar.gz"
    if not meta.is_file():
        print("[finalize] ck_meta.tar.gz missing (trust root/plan/keys)",
              file=sys.stderr)
        return 4

    # --- 2. assemble storage ------------------------------------------
    if storage.exists() and any(storage.iterdir()):
        if not args.force:
            print(f"[finalize] storage {storage} is not empty "
                  "(use --force to wipe)", file=sys.stderr)
            return 4
        shutil.rmtree(storage)
    storage.mkdir(parents=True, exist_ok=True)
    with tarfile.open(meta) as tf:
        tf.extractall(storage)
    for s in seeds:
        with tarfile.open(ckpt / f"ck_seed-{s}.tar.gz") as tf:
            tf.extractall(storage)
    ledger_src = ckpt / "AUTHORITY_LEDGER.jsonl"
    if ledger_src.is_file():
        shutil.copy(ledger_src, storage / "AUTHORITY_LEDGER.jsonl")
    print(f"[finalize] assembled {storage} from {len(seeds)} seeds")

    # --- 3. independent qualification ---------------------------------
    out_path = storage / "QUALIFICATION_RECORD.json"
    cmd = [sys.executable,
           str(repo / "scripts" / "validation" / "qualify_campaign1.py"),
           "--storage", str(storage), "--campaign-id", args.campaign_id,
           "--root", str(repo), "--output", str(out_path),
           "--holdout", str(holdout)]
    rc, out, err = _run(cmd, cwd=repo)
    if not out_path.is_file():
        print(f"[finalize] qualifier produced no record (rc={rc})\n"
              f"{out[-2000:]}\n{err[-2000:]}", file=sys.stderr)
        return 4
    doc = json.loads(out_path.read_text())
    decision = _decision_of(doc)
    print(f"[finalize] qualification decision: {decision} (rc={rc})")

    # --- 4. results report --------------------------------------------
    value = doc.get("value", doc)
    reasons = value.get("reasons") or []
    stats = value.get("stats") or {}
    report = [
        f"# Campaign 3A ({args.campaign_id}) — results",
        "",
        f"- decision: **{decision}** (independent qualification; "
        "experimental qualification only, never production promotion)",
        f"- qualified_at: {datetime.now(timezone.utc).isoformat()}",
        f"- seeds: {seeds}",
        f"- qualification record: `QUALIFICATION_RECORD.json` "
        f"(digest {doc.get('digest', 'n/a')})",
        "",
        "## Gate reasons",
        "",
    ]
    report += [f"- {r}" for r in reasons] or ["- (none reported)"]
    if stats:
        report += ["", "## Reconstructed stats", "",
                   "```json", json.dumps(stats, indent=2, sort_keys=True),
                   "```"]
    report += ["",
               "## Evidence inventory", "",
               f"- campaign evidence + signed receipts: "
               f"`campaigns/{args.campaign_id}/`",
               f"- adapters: `adapters/{args.campaign_id}/`",
               f"- arm states: `arms/{args.campaign_id}/`",
               "- authority ledger: `AUTHORITY_LEDGER.jsonl`",
               "- trust root (public identities): `trust_root.json`",
               "",
               "Private keys are deliberately not published; the sealed "
               "holdout file stays with the evaluation authority.",
               ""]
    (storage / "CAMPAIGN3A_REPORT.md").write_text("\n".join(report))
    print(f"[finalize] wrote {storage / 'CAMPAIGN3A_REPORT.md'}")

    # --- 5. optional results branch -----------------------------------
    if args.results_branch:
        target = repo / "results" / args.campaign_id
        rc, out, err = _run(["git", "rev-parse", "--verify",
                             args.results_branch], cwd=repo)
        if rc != 0:
            rc2, out2, err2 = _run(
                ["git", "checkout", "-b", args.results_branch, "origin/main"],
                cwd=repo)
            if rc2 != 0:
                print(f"[finalize] branch create failed: {err2[-500:]}",
                      file=sys.stderr)
                return 4
        else:
            _run(["git", "checkout", args.results_branch], cwd=repo)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for rel in (f"campaigns/{args.campaign_id}",
                    f"adapters/{args.campaign_id}",
                    f"arms/{args.campaign_id}"):
            src = storage / rel
            if src.is_dir():
                shutil.copytree(src, target / rel)
        for name in ("QUALIFICATION_RECORD.json", "CAMPAIGN3A_REPORT.md",
                     "AUTHORITY_LEDGER.jsonl", "trust_root.json",
                     "DATASET_PROOF.json", "EXPERIMENT_PROTOCOL.json"):
            src = storage / name
            if src.is_file():
                shutil.copy(src, target / name)
        cdir = storage / "campaigns" / args.campaign_id
        for name in ("CAMPAIGN_PLAN.json", "EXECUTION_PUBLIC_KEY.bin",
                     "EXECUTION_KEY_ID.txt"):
            if (cdir / name).is_file():
                shutil.copy(cdir / name, target / name)
        # evidence lives on the results branch, not in source history —
        # .gitignore excludes results/, so force-add (repo convention)
        _run(["git", "add", "-f", str(target.relative_to(repo))], cwd=repo)
        rc, out, err = _run(
            ["git", "commit", "-m",
             f"results({args.campaign_id}): {decision} — banked evidence + "
             "signed qualification record"], cwd=repo)
        print(f"[finalize] committed results branch: rc={rc}")
        if args.push:
            rc, out, err = _run(["git", "push", "-u", "origin",
                                 args.results_branch], cwd=repo)
            print(f"[finalize] push rc={rc}")
            if rc != 0:
                print(err[-800:], file=sys.stderr)
                return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
