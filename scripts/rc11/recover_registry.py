#!/usr/bin/env python3
"""Recover production.json from the verified append-only transition ledger."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
from kvcontinual.execution.authority import Ed25519ReceiptVerifier
from kvcontinual.execution.registry import AdapterRegistry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--public-key", required=True)
    ap.add_argument("--key-id", default="rc11-promotion")
    ap.add_argument("--anchor", default=None, help="external anchor exported by export_authority_anchor.py")
    a = ap.parse_args()
    key = Path(a.public_key).read_bytes()
    if len(key) != 32:
        raise SystemExit("public key must contain exactly 32 raw Ed25519 bytes")
    verifier = Ed25519ReceiptVerifier(key, key_id=a.key_id)
    reg = AdapterRegistry(a.registry, require_signed_promotions=True)
    minimum_generation = 0
    expected_tail_digest = None
    if a.anchor:
        anchor = json.loads(Path(a.anchor).read_text())
        if anchor.get("registry_id") != reg.registry_id:
            raise SystemExit("anchor registry_id does not match registry")
        minimum_generation = int(anchor.get("generation", 0))
        expected_tail_digest = anchor.get("transition_record_digest")
    try:
        state = reg.recover_current(
            verifier=verifier,
            minimum_generation=minimum_generation,
            expected_tail_digest=expected_tail_digest,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2)); return 1
    print(json.dumps({
        "ok": True,
        "registry_id": reg.registry_id,
        "current": state["current"],
        "previous": state.get("previous"),
        "generation": state["generation"],
        "transition_record_digest": state["transition_record_digest"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
