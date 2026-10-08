#!/usr/bin/env python3
"""Promotion authority — the last link in the artifact chain.

    Preregistered Experiment (plan, signed by `plan`)
        -> evaluation receipts + predictions (signed by `execution_witness`)
        -> EVALUATION_BUNDLE.json (signed by `evaluation`)
        -> QUALIFICATION_RECORD.json (signed by `qualification`)
        -> PromotionDecision + RuntimeManifest (signed by `promotion`)

`promotion` lives outside the research/verification planes in
principle; in this development scaffold it is still a key file — but a
*distinct* one, checked against the trust root, recorded in the
authority ledger, and bound to exact artifact digests (never names
like "latest"). Deployment MUST read the decision, not the presence
of files.

Usage:
  python scripts/promote.py --storage-root ROOT --campaign-id ID \
      --adapter L6 --out RuntimeManifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer, SignedEnvelope  # noqa: E402
from minagi.v161.authority import (AuthorityLedger,  # noqa: E402
                                   AuthorityRegistry)


def _fail(msg):
    print(f"[PROMOTE REFUSED] {msg}", file=sys.stderr)
    return 2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage-root", required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--adapter", default="L6",
                    help="adapter arm to promote (subject to gates)")
    ap.add_argument("--out", required=True, help="RuntimeManifest path")
    args = ap.parse_args(argv)

    storage = Path(args.storage_root).resolve()
    cdir = storage / "evidence" / args.campaign_id
    trust_path = storage / "trust_root.json"
    if not trust_path.is_file():
        return _fail("trust_root.json missing")
    try:
        registry = AuthorityRegistry.load(trust_path)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"trust root load: {exc}")
    verifier = registry.verifier()

    # --- verify qualification chain ---
    qdoc_path = cdir / "QUALIFICATION_RECORD.json"
    if not qdoc_path.is_file():
        return _fail("QUALIFICATION_RECORD.json missing")
    qdoc = json.loads(qdoc_path.read_text())
    if "value" in qdoc:  # signed envelope form
        if not registry.is_authorized("qualification",
                                      qdoc.get("signer_key_id", "")):
            return _fail("qualification record not signed by "
                         "registered qualification authority")
        env = SignedEnvelope(qdoc["signer_key_id"],
                             qdoc["signature_b64"])
        if not verifier.verify(qdoc["value"], env):
            return _fail("qualification record signature invalid")
        if qdoc.get("digest") != digest(qdoc["value"]):
            return _fail("qualification record digest mismatch")
        record = qdoc["value"]
    else:  # legacy unsigned form — refuse on principle for promotion
        return _fail("unsigned qualification record cannot ground "
                     "a promotion decision")
    if record.get("decision") != "QUALIFIED":
        return _fail(f"campaign decision {record.get('decision')!r} "
                     "— only QUALIFIED evidence may be promoted")

    # adapter must be among the receipts actually qualified — build a
    # per-seed runtime manifest binding model/tokenizer/adapter/protocol
    plan_value = {}
    plan_doc_path = cdir / "CAMPAIGN_PLAN.json"
    if plan_doc_path.is_file():
        plan_value = json.loads(plan_doc_path.read_text()).get(
            "value", {})
    protocol_doc = cdir / "EXPERIMENT_PROTOCOL.json"
    protocol = None
    if protocol_doc.is_file():
        try:
            from minagi.v161.experiment_protocol import \
                ExperimentProtocolV1
            protocol = ExperimentProtocolV1(
                **json.loads(protocol_doc.read_text())["value"])
        except Exception:
            protocol = None
    from minagi.v161.peft_serving import (AdapterClosureError,
                                          runtime_manifest)
    promoted = {}
    adir = storage / "adapters" / args.campaign_id / args.adapter
    if not adir.is_dir():
        return _fail(f"adapter dir {adir} missing")
    seeds = {}
    manifests = {}
    for sd in sorted(adir.iterdir()):
        if not (sd.is_dir() and sd.name.startswith("seed-")):
            continue
        try:
            manifests[sd.name] = runtime_manifest(
                model_id=plan_value.get("model_id", ""),
                model_revision=plan_value.get("model_revision", ""),
                model_digest=plan_value.get("model_digest", ""),
                tokenizer_digest=plan_value.get("tokenizer_digest", ""),
                adapter_dir=sd, protocol=protocol,
                campaign_digest=record.get("campaign_plan_digest"),
                qualification_record_digest=qdoc["digest"])
        except AdapterClosureError as exc:
            return _fail(f"{sd.name}: adapter closure failed: {exc}")
        seeds[sd.name] = manifests[sd.name]["adapter_digest"]
    if not seeds:
        return _fail("no adapter seed artifacts found")
    promoted[args.adapter] = seeds

    # --- promotion authority signs ---
    priv = storage / ".keys" / "promotion.pem"
    if not priv.is_file():
        return _fail("promotion key missing")
    signer = Ed25519Signer.from_private_bytes(priv.read_bytes())
    if not registry.is_authorized("promotion", signer.key_id):
        return _fail("promotion key not registered in trust root")

    decision = {
        "schema": "mini-agi-v16.5-promotion-decision-v1",
        "campaign_id": args.campaign_id,
        "campaign_plan_digest": record["campaign_plan_digest"],
        "qualification_record_digest": qdoc["digest"],
        "evaluation_bundle_digest": record.get("evaluation_bundle_digest"),
        "adapter": args.adapter,
        "adapter_artifact_digests": seeds,
        "runtime_manifest_digests": {k: v["digest"]
                                    for k, v in manifests.items()},
        "authorized_at": int(time.time()),
        "note": "Binds the EXACT qualified adapter digests. Promotion "
                "authority only; runtime admission control must still "
                "enforce it.",
    }
    env = signer.sign(decision)
    doc = {"value": decision, "digest": digest(decision),
           "signer_key_id": env.key_id,
           "signature_b64": env.signature_b64,
           "runtime_manifests": manifests}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True))
    AuthorityLedger(storage / "AUTHORITY_LEDGER.jsonl") \
        .append(signer, "promotion_decision", decision)
    print(json.dumps({"schema": decision["schema"],
                      "campaign_id": decision["campaign_id"],
                      "adapter": args.adapter,
                      "digest": doc["digest"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
