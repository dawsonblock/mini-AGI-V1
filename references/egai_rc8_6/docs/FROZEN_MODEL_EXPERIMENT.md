# Frozen-model continual-learning experiment protocol

1. Hash and freeze the base model artifacts.
2. Freeze the benchmark task pools before the learner sees any experience.
3. Reject exact overlap between experience inputs/cases/IDs and hidden evaluation pools.
4. Evaluate the raw frozen baseline at episode 0.
5. Feed only experience cases to the learner's `observe()` path.
6. At preregistered checkpoints, evaluate without learning from evaluation outcomes.
7. Assert the base-model digest is unchanged at every checkpoint.
8. Report paired future-task gain and bootstrap confidence intervals.
9. Independently report retention and security performance.
10. Export learned procedures as an artifact only after the experiment; persistent deployment still requires the Authority Plane.

A positive experiment should show a replicated positive future-task effect whose confidence interval does not depend on a few leaked or repeated tasks, while retention/security constraints remain satisfied.
