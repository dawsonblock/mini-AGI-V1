# v4.0.3-r1 — Holdout reclassification

`configs/holdout/campaign4a_holdout.jsonl` shipped in the source ZIP
**with expected answers**, and the research process that writes code in
this repository can read it. A sealed evaluation cannot use a dataset
whose answers the subject process may have memorized into training,
retrieval or prompt material. Removing it from a *later* ZIP does not
restore secrecy — exposure is a one-way door.

## Decision

- `campaign4a_holdout` is **reclassified: public development /
  regression benchmark.** It stays in the tree and may be used for
  development iteration, rescore checks and the dev matrix.
- It is **never** a fresh final-confirmation surface. No release may
  cite results on this dataset as sealed evidence.

## What replaces it for final confirmation

Fresh-confirmation evidence must come from an evaluator-custody set:

- generated **after** the candidate is frozen, under the evaluator's
  control (the `final_holdout` lane already binds only a digest to the
  runner — see `seal_final_holdout.py` and
  `SEALED_CONFIRMATION_PROTOCOL.md`);
- answers held where the research process has no filesystem or API
  reach — a signed dataset hash binds the dataset, it does not hide it;
- submissions rate-limited under a predetermined protocol so the
  holdout cannot be adaptively overfit;
- every result binds candidate digest + dataset commitment + protocol
  + evaluator identity — exactly the fields `independent-evaluation-v1`
  already carries, so no new receipt machinery is needed.

## Note on `within_holdout`

The within-holdout generalization set ships in the corpus; its answers
are consumed only inside the evaluator's runtime (the operator's Colab
session), never by the learning-agent process — statistical
independence on unseen inputs plus evaluator-side custody, not a
third-party attestation. A hardened v4.1.x could put its answers behind
the same seal as `final_holdout` if a second operator runs
confirmation.
