# EGAI v3 RC8.2 architecture

## Plane 0 — Evidence
Append-only signed history with provenance and explicit eligibility classes. Evidence records describe what was recorded; they are not synonymous with truth. Simulation/replay/model inference cannot silently become factual belief support.

## Plane 1 — Cognition
A frozen foundation model is combined with retrieval, verified procedures, tools and a budgeted controller. `FrozenModelGuard` checks the exact model identity during sequential experiments.

## Plane 2 — Learning
Learning is allowed inside an isolated research/sandbox state. RC8.2 includes verified corrective-feedback mining into ephemeral `LearnedProcedure` objects, retrieval, abstraction helpers and transfer-ring evaluation. The learning plane can export a candidate artifact but cannot promote it.

## Plane 3 — Authority
Persistent changes cross a role-scoped trust boundary:

`LearningProposal -> CandidateManifest -> signed BuildManifest -> signed BenchmarkSpec -> signed ResultBundle -> signed EvaluationBundle -> signed QualificationRecord -> signed PromotionDecision -> RuntimeManifest`

A valid Ed25519 signature is insufficient by itself. The signer key must also be authorized for the exact role.

## Benchmark closure
Benchmark cases are stored as immutable CAS artifacts. A signed `BenchmarkSpec` binds future, retention and security set digests. The runner loads those exact artifacts; callers cannot substitute an in-memory case list after preregistration.

## Frozen-weight research loop

`experience -> verified feedback -> sandbox procedure -> retrieval -> future task`

At checkpoints the held-out evaluation pool is queried without sending its outcomes into `observe()`. Exact case/input overlap with the experience stream is rejected before execution.

## Deferred neural plasticity
Adapters, new neural modules, shared-weight consolidation and architecture mutation remain out of scope. RC8.2 exists to establish whether useful forward transfer occurs before any of those mechanisms are allowed.
