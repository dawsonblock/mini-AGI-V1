# Frozen-Foundation 1,000-Experience Experiment

This is the first falsification experiment for the EGAI v2 architecture.

## Question

Can the deployed system improve on sealed future tasks while the foundation-model bytes remain exactly unchanged?

## Conditions

Run the same task stream under four conditions:

- **A — Frozen:** foundation model only.
- **B — Evidence:** frozen model + evidence retrieval.
- **C — Skills:** B + promoted beliefs, procedures, skills, and composition.
- **D — Dream:** C + grounded replay optimization of routing/exploration/learning policies.

No adapter training, new neural modules, shared-weight consolidation, or architecture modification is allowed.

## Task family

The recommended first domain is repository/debugging work with machine-verifiable outcomes. Candidate tasks should have objective checks such as unit tests, compilation, deterministic static analysis, or exact artifact checks. Do not use the sealed future set as training, replay, abstraction, skill-generation, or qualification input.

## Checkpoints

Evaluate at:

```text
0, 50, 100, 250, 500, 1000 experiences
```

The `FrozenFoundationProtocol` rejects a checkpoint if the base-model digest differs from the experiment spec.

## Primary metrics

- future-task success rate;
- old-task retention;
- calibration error;
- tokens/task;
- tool calls/task;
- wall time/task;
- repeated-error rate;
- skill reuse rate;
- compute cost;
- capacity growth;
- FTE.

## Primary hypothesis

```text
P_future(E_1000) > P_future(E_0)
```

subject to:

```text
base model unchanged
old-task retention drop <= epsilon
security regression = 0
unauthorized persistent writes = 0
provenance closure = 100%
```

## Attribution

A vs B estimates retrieval/evidence value.

B vs C estimates procedural abstraction/skill value beyond retrieval.

C vs D estimates whether replay-world policy optimization improves how accumulated knowledge is deployed.

The experiment should report all pairwise differences with confidence intervals rather than only the best final curve.

## Leakage controls

- sealed future-task identifiers/digests are unavailable to the learning plane;
- qualification worlds are one-shot resources;
- audit worlds are never used during development;
- replay policies receive prefix-only views;
- simulation outputs remain tagged simulation and never count as observations.

## Interpretation

A successful result demonstrates system-level non-parametric learning under a frozen foundation. It does not by itself establish general continual intelligence or prove that neural plasticity is unnecessary.
