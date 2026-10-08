"""v16.5 evidence receipt v3 — the predictions are inside the signature.

ExecutedRunReceiptV162 signed model/dataset/evaluator/adapter/state
identifiers plus self-reported metrics. Predictions lived beside the
receipt, outside its authenticated payload — a worker could alter what
it claims the model produced without breaking the signature.

EvidenceReceiptV3 binds the evidence chain end to end:

    corpus rows
        -> canonical prediction records (input digest + output + status)
        -> predictions_digest   (sha256 of the canonical JSONL bytes)
        -> input_manifest_digest (digest of [{id, input_sha256}])
        -> evaluation_bundle_digest (digest of bundle body incl. the
           WORKER-REPORTED metrics — kept as claims, never trusted)
        -> this signed receipt

The qualifier independently recomputes scores from the canonical
prediction records joined with corpus rows, so reported metrics become
checkable claims rather than facts.

Canonical prediction record (one JSONL line per evaluated task):

    {"id":             task id
     "block":          "hidden" | "retention" | "security" |
                       "retention_delayed"
     "input":          exact model input text (post-augmentation)
     "input_sha256":   sha256 of the UTF-8 input bytes
     "output":         decoded output text
     "generated_tokens": int
     "decoding":       {"mode": "greedy", "max_new_tokens": int}
     "status":         "ok" | "error:<code>"}

Hidden reference answers never enter the prediction artifact — scoring
joins records against corpus rows by id at qualification time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Any

from egai.common.canonical import (digest, sha256_bytes, validate_digest,
                                   canonical_bytes)
from egai.common.crypto import Ed25519Signer, SignedEnvelope

from .executed_run import ARMS_V162

RECEIPT_V3_SCHEMA = "mini-agi-v16.5-evidence-receipt-v3"
BUNDLE_V3_SCHEMA = "mini-agi-v16.5-evaluation-bundle-v1"
PREDICTION_BLOCKS = ("hidden", "retention", "security",
                     "retention_delayed")


def _canon_line(rec: Mapping[str, Any]) -> str:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def prediction_record(task_id: str, block: str, input_text: str,
                      output_text: str, generated_tokens: int,
                      max_new_tokens: int, status: str = "ok") -> dict:
    if block not in PREDICTION_BLOCKS:
        raise ValueError(f"unknown prediction block: {block}")
    if int(generated_tokens) < 0:
        raise ValueError("generated_tokens must be non-negative")
    if int(max_new_tokens) <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not task_id:
        raise ValueError("task id required")
    return {"id": str(task_id), "block": block,
            "input": str(input_text),
            "input_sha256": sha256_bytes(str(input_text).encode()),
            "output": str(output_text),
            "generated_tokens": int(generated_tokens),
            "decoding": {"mode": "greedy",
                         "max_new_tokens": int(max_new_tokens)},
            "status": str(status)}


def predictions_canonical_bytes(records: Iterable[Mapping]) -> bytes:
    return ("\n".join(_canon_line(r) for r in records) + "\n").encode()


def predictions_digest_of(records: Iterable[Mapping]) -> str:
    return sha256_bytes(predictions_canonical_bytes(records))


def write_predictions(records: Iterable[Mapping], path) -> str:
    """Write canonical JSONL, fsync, return the predictions digest."""
    data = predictions_canonical_bytes(records)
    p = Path(path)
    with p.open("wb") as f:
        f.write(data)
        f.flush()
        import os
        os.fsync(f.fileno())
    return sha256_bytes(data)


def read_predictions(path) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def input_manifest_digest(records: Iterable[Mapping]) -> str:
    """Digest over [{id, block, input_sha256}] — binds WHICH inputs were
    evaluated without copying corpus content into the manifest."""
    entries = [{"id": str(r["id"]), "block": str(r["block"]),
                "input_sha256": str(r["input_sha256"])}
               for r in records]
    return digest(entries)


def evaluation_bundle(campaign_digest: str, protocol_digest: str,
                      seed: int, arm: str, cell: str,
                      input_manifest_d: str, predictions_d: str,
                      metrics_reported: Mapping) -> dict:
    """The unsigned worker-side bundle body the receipt binds."""
    return {"schema": BUNDLE_V3_SCHEMA,
            "campaign_digest": campaign_digest,
            "protocol_digest": protocol_digest,
            "seed": int(seed), "arm": arm, "cell": cell,
            "input_manifest_digest": input_manifest_d,
            "predictions_digest": predictions_d,
            "metrics_reported": dict(metrics_reported)}


@dataclass(frozen=True)
class EvidenceReceiptV3:
    """Signed binding of identity context -> evidence bundle ->
    canonical predictions. Metrics are intentionally NOT receipt
    fields — they are recomputed independently at qualification."""
    campaign_digest: str
    protocol_digest: str
    seed: int
    arm: str
    model_digest: str
    tokenizer_digest: str
    adapter_digest: str
    state_digest: str
    environment_digest: str
    evaluation_dataset_digest: str
    evaluator_digest: str
    input_manifest_digest: str
    predictions_digest: str
    evaluation_bundle_digest: str
    signer_key_id: str
    signature_b64: str
    schema: str = RECEIPT_V3_SCHEMA

    def __post_init__(self):
        for x in (self.campaign_digest, self.protocol_digest,
                  self.model_digest, self.tokenizer_digest,
                  self.adapter_digest, self.state_digest,
                  self.environment_digest, self.evaluation_dataset_digest,
                  self.evaluator_digest, self.input_manifest_digest,
                  self.predictions_digest, self.evaluation_bundle_digest):
            validate_digest(x)
        if self.arm not in ARMS_V162:
            raise ValueError(f"arm must be one of {sorted(ARMS_V162)}")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if not self.signer_key_id or not self.signature_b64:
            raise ValueError("signed evidence receipt required")

    @property
    def body(self) -> dict:
        return {"schema": self.schema,
                "campaign_digest": self.campaign_digest,
                "protocol_digest": self.protocol_digest,
                "seed": int(self.seed), "arm": self.arm,
                "model_digest": self.model_digest,
                "tokenizer_digest": self.tokenizer_digest,
                "adapter_digest": self.adapter_digest,
                "state_digest": self.state_digest,
                "environment_digest": self.environment_digest,
                "evaluation_dataset_digest": self.evaluation_dataset_digest,
                "evaluator_digest": self.evaluator_digest,
                "input_manifest_digest": self.input_manifest_digest,
                "predictions_digest": self.predictions_digest,
                "evaluation_bundle_digest": self.evaluation_bundle_digest,
                "signer_key_id": self.signer_key_id}

    @property
    def digest(self) -> str:
        return digest(self)

    @classmethod
    def sign(cls, *, signer: Ed25519Signer, campaign_digest: str,
             protocol_digest: str, seed: int, arm: str,
             model_digest: str, tokenizer_digest: str,
             adapter_digest: str, state_digest: str,
             environment_digest: str, evaluation_dataset_digest: str,
             evaluator_digest: str, input_manifest_digest: str,
             predictions_digest: str,
             evaluation_bundle_digest: str) -> "EvidenceReceiptV3":
        env = signer.sign({"schema": RECEIPT_V3_SCHEMA,
                           "campaign_digest": campaign_digest,
                           "protocol_digest": protocol_digest,
                           "seed": int(seed), "arm": arm,
                           "model_digest": model_digest,
                           "tokenizer_digest": tokenizer_digest,
                           "adapter_digest": adapter_digest,
                           "state_digest": state_digest,
                           "environment_digest": environment_digest,
                           "evaluation_dataset_digest":
                               evaluation_dataset_digest,
                           "evaluator_digest": evaluator_digest,
                           "input_manifest_digest": input_manifest_digest,
                           "predictions_digest": predictions_digest,
                           "evaluation_bundle_digest":
                               evaluation_bundle_digest,
                           "signer_key_id": signer.key_id})
        return cls(campaign_digest=campaign_digest,
                   protocol_digest=protocol_digest, seed=int(seed),
                   arm=arm, model_digest=model_digest,
                   tokenizer_digest=tokenizer_digest,
                   adapter_digest=adapter_digest,
                   state_digest=state_digest,
                   environment_digest=environment_digest,
                   evaluation_dataset_digest=evaluation_dataset_digest,
                   evaluator_digest=evaluator_digest,
                   input_manifest_digest=input_manifest_digest,
                   predictions_digest=predictions_digest,
                   evaluation_bundle_digest=evaluation_bundle_digest,
                   signer_key_id=env.key_id,
                   signature_b64=env.signature_b64)

    def verify(self, verifier) -> bool:
        return verifier.verify(
            self.body, SignedEnvelope(self.signer_key_id,
                                      self.signature_b64))


def load_verified_seed_result_v3(run_dir, seed: int,
                                 campaign_digest: str,
                                 arms: Iterable[str],
                                 verifier) -> dict | None:
    """v3 verified resume: a seed dir is evidence iff its atomic commit
    is intact — COMMIT_MANIFEST lists every published file with matching
    digests, every recorded v3 receipt verifies under the witness key
    and binds this campaign/seed, and each cell's predictions recompute
    to the receipt-bound predictions/input-manifest/bundle digests.

    Fail-closed: anything inconsistent discards the seed for
    re-execution — old V162 receipts do NOT qualify under v3.
    """
    run_dir = Path(run_dir)
    seed_result_path = run_dir / "SEED_RESULT.json"
    manifest_path = run_dir / "COMMIT_MANIFEST.json"
    if not ((run_dir / "COMPLETE").is_file()
            and seed_result_path.is_file()
            and manifest_path.is_file()):
        return None
    try:
        sd = json.loads(seed_result_path.read_text())
        if int(sd.get("seed", -1)) != int(seed):
            return None
        man = json.loads(manifest_path.read_text())
        files = dict(man.get("files") or {})
        if int(man.get("seed", -1)) != int(seed) \
                or man.get("campaign_digest") != campaign_digest:
            return None
        if not files:
            return None
        for rel, recorded in files.items():
            f = run_dir / rel
            if not f.is_file() or sha256_bytes(f.read_bytes()) != recorded:
                return None
        rec_digests = dict(sd.get("receipt_digests") or {})
        arms = tuple(arms)
        if not set(rec_digests) >= set(arms):
            return None
        for name, recorded_digest in rec_digests.items():
            doc = json.loads((run_dir / f"{name}.json").read_text())
            r = EvidenceReceiptV3(**doc["receipt"])
            if r.campaign_digest != campaign_digest or r.arm not in arms \
                    or int(r.seed) != int(seed):
                return None
            if r.digest != recorded_digest or not r.verify(verifier):
                return None
            preds = read_predictions(run_dir / f"PREDICTIONS-{name}.jsonl")
            if predictions_digest_of(preds) != r.predictions_digest:
                return None
            if input_manifest_digest(preds) != r.input_manifest_digest:
                return None
            if digest(doc.get("bundle") or {}) \
                    != r.evaluation_bundle_digest:
                return None
        return sd
    except Exception:
        return None
