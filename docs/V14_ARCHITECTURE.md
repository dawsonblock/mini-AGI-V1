# mini-AGI v14 architecture

## Design rule

The system is built around one hard invariant: **the component that discovers an improvement does not have authority to persist it**.

The control path is divided into cognitive, research/learning, independent evaluation/qualification, promotion, and persistence responsibilities. Execution acceleration is orthogonal to semantic learning.

## Two-axis governance

A learning proposal has two coordinates.

### Permanence

- L0 working context
- L1 raw evidence
- L2 episodic memory
- L3 semantic belief
- L4 reusable skill
- L5 routing/composition
- L6 isolated neural memory
- L7 shared adapter
- L8 new module
- L9 foundation consolidation
- L10 architecture change

### Evidence strength

- E0 unverified
- E1 internally consistent
- E2 independently verified
- E3 replicated
- E4 fresh/OOD validated
- E5 independently reproduced

The default policy maps increasing permanence to increasing evidence requirements and keeps L7–L10 proposal-only.

## Improvement cycle

1. Inference produces an experience.
2. Evidence is stored without granting it truth authority.
3. Independent verification creates signed receipts.
4. Belief, skill or hypothesis mechanisms derive candidate abstractions.
5. Dream/replay research may search candidate improvement policies but cannot promote.
6. The plasticity layer chooses the least permanent feasible intervention.
7. A content-bound candidate is registered.
8. A preregistered falsification plan attacks it with negative controls, retention, security, transfer and OOD cases.
9. Fresh hidden-task authority provides restart-safe one-shot evaluation tasks.
10. Independent qualification recomputes policy-relevant metrics from evidence.
11. Promotion authority signs authorization.
12. Persistence/activation requires that authorization and emits an auditable runtime transition.

## Execution-memory separation

KVMem/HYPIC artifacts accelerate reconstruction of model execution state. They are not evidence, beliefs or semantic memories. Their validity is tied to model/runtime/kernel/adapter identity and can fail closed to exact replay.

## Adapter policy

The learner may build adapter candidates, but the learning worker cannot qualify or activate them. v14's default policy leaves shared L7 adapter activation disabled. The hardened execution registry already supports signed promotion and transition receipts and is the sole registry authority used by the compatibility layer.

## Fresh-task authority

`DurableFreshTaskAuthority` stores commitments and lease consumption in SQLite with WAL and FULL synchronous durability. Hidden tasks follow SEALED -> LEASED -> CONSUMED -> CLOSED. A process restart does not make a consumed task fresh again.

## Falsification

A valid `FalsificationPlan` must be preregistered and contain at least negative-control, retention and security test classes. OOD, adversarial, ablation and counterexample cases can be added per permanence level.

## Dream/replay research

`DreamPolicyResearcher` can rank replay-derived improvement policies. It has no signing, persistence, activation or promotion interface. Its output is a proposal for the normal candidate pipeline.

## Replay scheduling

`ModelTimeClock` tracks cumulative parameter displacement. `AdaptiveReplayScheduler` combines displacement, estimated forgetting risk and memory importance instead of assuming fixed replay every N optimizer steps.

## State machine

Production-directed candidates must move through:

`REGISTERED -> BUILT -> EVALUATED -> QUALIFIED -> AUTHORIZED -> ACTIVE`

Privileged transitions require an authorization digest. Skipping directly to ACTIVE is rejected at the state-store layer.
