#!/usr/bin/env python3
"""Independent evaluation receipt issuer (v4.0.3-r1).

The EVALUATOR side of the trust boundary — this script runs under the
evaluator's custody (a separate user/host/HSM from the research agent;
that separation is operational, not something this script can enforce).

    python3 scripts/independent_evaluate.py \
        --run results/.../runs/<run_id> \
        --evaluator-key ~/.config/miniagi/roles/evaluation.pem \
        [--security-evidence results/.../security/<run_id>.security.json] \
        [--write]

What it does:

1. Independently recomputes the committed run: every prediction file is
   strict-rescored, reported metrics must reproduce, and identity
   bindings (model id+revision, scorer, contract and adapter digests)
   must hold (evidence_reconciliation.verify_run == VERIFIED).
2. Recomputes the evidence binding — experiment id, candidate adapter
   digest, base model digest, dataset digest, scorer digest, protocol
   digest, predictions digest, evaluation/retention/security metrics
   digests.
3. Signs the canonical ``independent-evaluation-v1`` record with the
   evaluator's key and (with --write) stores it as
   EVALUATION_RECEIPT.json inside the run directory — post-commit
   evidence, deliberately outside the run's own inventory.

A run the evaluator's own recomputation cannot verify is REFUSED: no
receipt is produced for inconsistent, incomplete or failed evidence.
A verified run whose measurements are negative still receives a
receipt — a real negative outcome is exactly what the learning
controller must not lose.

Exit codes: 0 receipt issued; 1 recomputation did not verify; 2 usage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Issue an independent-evaluation-v1 receipt for a "
                    "committed run (evaluator-side).")
    ap.add_argument("--run", required=True,
                    help="committed run directory (or its "
                         "RUN_COMMIT.json)")
    ap.add_argument("--evaluator-key", required=True,
                    help="evaluator's Ed25519 private-key file (raw "
                         "bytes, as authority.provision_role writes; "
                         "must be the 'evaluation' role in the "
                         "campaign trust root)")
    ap.add_argument("--repo-root", default=str(ROOT),
                    help="repository root for corpus/protocol "
                         "resolution")
    ap.add_argument("--corpus", default=None,
                    help="corpus JSONL override (default: resolve "
                         "CONFIG.corpus under --repo-root)")
    ap.add_argument("--security-evidence", default=None,
                    help="governed security-lane sidecar JSON to bind "
                         "into the receipt (optional)")
    ap.add_argument("--write", action="store_true",
                    help="write EVALUATION_RECEIPT.json inside the run "
                         "directory (default: print only)")
    ap.add_argument("--output", default=None,
                    help="also write the receipt to this path")
    args = ap.parse_args()

    from egai.common.crypto import Ed25519Signer
    from minagi.v161.evidence_reconciliation import verify_run
    from minagi.v161.independent_eval_receipt import (
        issue_evaluation_receipt, write_receipt_file)

    run_dir = Path(args.run)
    if run_dir.is_file() and run_dir.name == "RUN_COMMIT.json":
        run_dir = run_dir.parent
    if not run_dir.is_dir():
        print(f"error: no such run directory: {run_dir}",
              file=sys.stderr)
        return 2

    key_path = Path(args.evaluator_key)
    if not key_path.is_file():
        print(f"error: evaluator key not found: {key_path}",
              file=sys.stderr)
        return 2
    signer = Ed25519Signer.from_private_bytes(key_path.read_bytes())

    corpus_rows = None
    if args.corpus:
        corpus_rows = [json.loads(ln)
                       for ln in Path(args.corpus).read_text().splitlines()
                       if ln.strip()]

    security_doc = None
    if args.security_evidence:
        security_doc = json.loads(
            Path(args.security_evidence).read_text())

    # The evaluator recomputes FIRST and reports before deciding —
    # a refusal must name the verdict, not just fail.
    verdict = verify_run(run_dir, repo_root=Path(args.repo_root),
                         corpus_rows=corpus_rows)
    if verdict.get("status") != "VERIFIED":
        print(json.dumps({
            "issued": False,
            "run_id": run_dir.name,
            "reconciliation_status": verdict.get("status"),
            "mismatches": (verdict.get("mismatches") or [])[:16],
            "reason": "independent recomputation did not verify — "
                      "no receipt issued for unverifiable evidence"},
            indent=2, sort_keys=True))
        return 1

    try:
        receipt = issue_evaluation_receipt(
            signer, run_dir, repo_root=Path(args.repo_root),
            corpus_rows=corpus_rows, security_doc=security_doc)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.write:
        out = write_receipt_file(run_dir, receipt)
        print(f"wrote {out}", file=sys.stderr)
    if args.output:
        Path(args.output).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"issued": True, "run_id": run_dir.name,
                      "receipt": receipt}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
