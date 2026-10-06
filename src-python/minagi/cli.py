from __future__ import annotations
import argparse
import json
from pathlib import Path


def _release_version() -> str:
    root = Path(__file__).resolve().parents[2]
    version = root / "VERSION"
    return version.read_text().strip() if version.is_file() else "15.2.0"


def main():
    p = argparse.ArgumentParser(
        prog="minagi",
        description="mini-AGI governed continual runtime and serving utility",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status"); s.add_argument("--root", default=".")
    v = sub.add_parser("verify-layout"); v.add_argument("--root", default=".")
    a = p.parse_args(); root = Path(a.root)
    if a.cmd == "status":
        print(json.dumps({
            "canonical_api": "minagi.v15",
            "canonical_runtime": "GovernedContinualRuntime",
            "implementation": "GovernedContinualRuntimeV145",
            "release": _release_version(),
            "root": str(root.resolve()),
            "historical_authority_api": "legacy-only under minagi.v14",
            "private_authority_keys_in_runtime": "forbidden",
            "evaluation_provenance": "fresh/falsification mechanically bound",
            "learned_state_activation": "atomic-with-runtime-head",
            "serving_state": "witnessed+attested StateEpoch with request leases",
            "epistemic_invalidation": "transitive dependency quarantine/revocation",
            "procedural_learning": "sealed SkillIR A0/A1 qualification",
            "adapter_learning": "verified dataset + immutable payload + sealed transfer/retention/security qualification",
            "native_adapter_application": "not implemented; adapter-set identity is process-bound only",
        }, indent=2)); return 0
    required = [root / "governance.sqlite3", root / "cas"]
    print(json.dumps({"ok": all(x.exists() for x in required), "required": [str(x) for x in required]}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
