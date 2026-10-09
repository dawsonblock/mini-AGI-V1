#!/usr/bin/env python3
"""Migrate a v16.4.2 JSONL activation journal into the v16.4.3
authority store (HARDENING_PLAN WP4 compatibility clause).

    python scripts/migrate_journal_v1_to_v2.py \
        --storage-root STORAGE --journal-dir STORAGE/runtime_journal

Validates every V1 record (record digest recomputation, signature +
runtime role where signed), refuses ambiguous/contradictory histories,
replays them into `state/authority.sqlite` as `migrated` events, and
writes a signed migration checkpoint + anchor. V1 records are never
rewritten to look valid — they are imported as evidence of what the
legacy journal recorded.

Exit codes: 0 migrated (or nothing to migrate), 2 refused.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.runtime.authority_store import AuthorityStore  # noqa: E402
from minagi.runtime.journal_v2 import (JournalRefused,  # noqa: E402
                                       migrate_v1_journal)
from minagi.v161.authority import AuthorityRegistry  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage-root", required=True)
    ap.add_argument("--journal-dir", required=True,
                    help="directory containing activation_journal.jsonl")
    ap.add_argument("--state-dir", default=None,
                    help="authority store dir (default <storage>/state)")
    ap.add_argument("--runtime-key", default=None,
                    help="runtime signing key for the migration "
                         "checkpoint (default <storage>/.keys/runtime.pem)")
    args = ap.parse_args(argv)

    storage = Path(args.storage_root).resolve()
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    key_path = Path(args.runtime_key) if args.runtime_key else \
        storage / ".keys" / "runtime.pem"
    signer = Ed25519Signer.from_private_bytes(key_path.read_bytes())

    store = AuthorityStore(
        (args.state_dir or (storage / "state")) / "authority.sqlite")
    try:
        report = migrate_v1_journal(
            args.journal_dir, store, registry=registry,
            migration_signer=signer)
    except JournalRefused as exc:
        print(f"[MIGRATION REFUSED] {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "MIGRATED", **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
