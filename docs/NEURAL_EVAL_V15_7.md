# v15.7 NeuralEval1 — governed real-weight evaluation harness

v15.7 closes the experiment-identity gap between the governed adapter pipeline and a real QW3 A0/A1 run. It does **not** attach or claim a real-weight improvement result from this build environment.

## Invariant

A0 and A1 may differ in their authorized adapter state. They must not differ in foundation-model bytes, tokenizer identity, KVMem archive, retrieval policy, SkillIR policy, or runtime binary. Every arm must report measured runtime closure before any scored generation is accepted.

`QW3NeuralArm` re-verifies the governed runtime before every generation and records a content-addressed `NeuralRuntimeEvidence` object binding the epoch, served-manifest digest, aggregate artifact root, adapter-set root, native-adapter root, measured runtime-state digest, and output digest.

`SealedNeuralAdapterEvaluator` consumes each fresh task once, runs both governed arms, stores the runtime evidence in CAS, and feeds the paired scores plus retention/security outcomes into the existing `FrozenBaselineGateV145`. A PASS is therefore evidence about one adapter delta under a frozen production identity rather than a loose comparison between two unrelated servers.

## What is proven by the release tests

- unmeasured QW3 runtimes are rejected;
- foundation/tokenizer/KVMem/retrieval/skill/runtime drift invalidates an A0/A1 experiment;
- a nonzero candidate adapter root without a native adapter bundle is rejected;
- every A0/A1 output is bound to runtime-state evidence and fresh-task provenance;
- positive paired transfer with adequate retention and zero security regressions can PASS;
- any security regression BLOCKs the experiment.

The PASS test uses deterministic test doubles for model behavior. It validates the harness and authority semantics, not neural learning quality.

## Remaining proof

Run the harness against two real QW3 processes over the same real model: A0 without the candidate adapter and A1 with the exact qualified native adapter. Use sealed unseen tasks, an independent retention battery, and security regressions. Publish the raw runtime evidence, task commitments/consumption receipts, and baseline report. Only then is a real-weight improvement claim warranted.
