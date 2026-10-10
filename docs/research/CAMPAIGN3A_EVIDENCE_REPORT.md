# Campaign 3A Evidence Report — v16.4.5 Packaging

The v16.4.4 source archive mixed production source with Campaign 3A
outputs, and 246 evidence files were unmanifested. In v16.4.5 the
evidence is a separately verified artifact.

## Contents

`mini-AGI-Campaign3A-Evidence.zip` contains the complete
`colab-evidence/` tree exactly as it existed at v16.4.5 freeze:

- `campaign3a/` — per-seed arm results (L2/L3/L4/L5 layer outputs),
  `AUTHORITY_LEDGER.jsonl`, receipts, predictions, result files.
- `campaign3a-*.tgz` — raw seed/final/latest archives as received
  (13 binary archives; gitignored, untracked, but part of the
  experiment record and therefore manifested in the evidence
  artifact).

## Integrity

- `EVIDENCE_MANIFEST.json` — sha256 over every file in the archive
  (schema_version 1, same format family as `SOURCE_MANIFEST.json`).
- `EVIDENCE_SIGNATURE.bin` — Ed25519 signature over the canonical
  manifest, under the same pinned release key.
- `EVIDENCE_PROVENANCE.json` — records the source archive it was
  split from (`mini-AGI-V1-v16.4.4-Verified-Runtime-Closure.zip`,
  sha256 `ad5993bd…e7d5`) and the separation repair (SEC-403).
- `RELEASE_PUBLIC_KEY.pem` — the release public key for verification.

## Scientific record

The Campaign 3A result stands: **REFUSED**. Reported neural gain over
grounded replay +0.00219 vs required +0.0200 (~11% of threshold);
6/10 positive-transfer seeds. The negative result is part of the
research record and is preserved unmodified — the v16.5.0 learning-
quality investigation starts from this evidence.
