# mini-AGI 14.1.0-alpha5 — Authority-Converged, Evidence-Bound Continual Runtime

Alpha5 is a contraction-and-proof release built on alpha4. It does **not** add a second promotion path. Alpha4's exact-content promotion and atomic learned-state/runtime activation remain authoritative. Alpha5 closes four higher-level gaps around that transaction: evaluation provenance, serving-state closure, epistemic invalidation, and empirical learning proof.

## Canonical API

New production code imports `minagi.v15`. That namespace intentionally exports one `GovernedContinualRuntime`, one `GovernanceDB`, one `RuntimeManifest`, and one `StateEpochRegistry`. Historical V141-V144 symbols remain under `minagi.v14` for compatibility and migration only.

## Authority chain

```text
Experience
  -> signed episode verification
  -> signed evidence-stage chain / derived evidence strength
  -> LearningProposal
  -> CandidateManifest
  -> BuildManifest + exact mutation targets
  -> provenance-bound raw EvaluationCases
       fresh_hidden -> signed FreshTaskConsumptionReceipt
       falsification/negative_control -> preregistered FalsificationCaseCommitment
  -> signed EvaluationBundle
  -> mechanical independent Qualification
  -> exact RuntimeManifest + ArtifactRoot
  -> exact one-use PromotionAuthorization
  -> alpha4 atomic learned-state + RuntimeManifest commit
  -> StateEpoch LOCAL_COMMIT
  -> signed EXTERNAL_WITNESS
  -> signed ATTESTATION
  -> SERVABLE
  -> request-scoped epoch lease
```

## New alpha5 invariants

1. `fresh_hidden` cannot be asserted by the evaluator without a valid signed fresh-task consumption receipt.
2. `falsification` and `negative_control` cannot be asserted without a candidate/case/category/task-bound preregistration commitment.
3. The StateEpoch becomes servable only after local atomic activation, external witness, and attestation are distinct recorded states.
4. A request lease pins one servable StateEpoch for its lifetime; retiring an epoch does not rewrite an in-flight request's execution context.
5. Evidence dependencies are explicit. Revocation is terminal for the revoked object and transitively quarantines derived beliefs, skills, runtime artifacts, or epochs registered as descendants.
6. Trajectory skill induction is proposal-only, requires repeated independently verified support over distinct inputs, and blocks contradictory verified trajectories.
7. Frozen-foundation A0/A1 experiments require the same foundation-model digest and production identity in both arms. Positive transfer is gated by paired bootstrap lower bounds, retention, and zero security regressions.

## Apple Silicon donor integration

Only the clean, non-authoritative performance helpers from RC12.3 were imported:

- conservative unified-memory/context planner;
- deterministic context-budget selection;
- guarded llama.cpp optional-flag negotiation;
- OpenAI-compatible streaming TTFT/decode benchmark;
- MLX conversion helper that inspects installed CLI support before execution;
- local Apple performance qualification evidence generator that explicitly does not self-authorize promotion.

These tools remain measurement/configuration helpers. They do not alter governance or promotion authority.

## What alpha5 proves

The included tests prove the implemented software invariants: provenance checks, authority bindings, atomicity inherited from alpha4, StateEpoch lifecycle constraints, request leases, transitive invalidation, contradiction blocking, and frozen-baseline identity checks.

They do **not** prove AGI, autonomous RSI, long-horizon stability, independent reproduction, production safety, or real-weight Apple/NVIDIA qualification.
