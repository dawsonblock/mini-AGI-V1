# RC8.2 release notes

RC8.2 adds the experimental layer required to test the core EGAI hypothesis rather than merely represent it.

Major changes:
1. Frozen model backends and immutable model guard.
2. Sequential hidden-future continual-learning harness.
3. Verified-feedback skill induction from repaired failures.
4. Benchmark leakage checks and paired-bootstrap uncertainty.
5. Preregistered benchmark case binding in the signed authority runner.
6. Runner-owned scorer registry.
7. JSONL dataset tooling and local-model CLI support.
8. Neural plasticity remains disabled by configuration and design.

This release does not claim that continual learning has been solved. The synthetic test demonstrates that the system can improve through verified procedure acquisition while the model identity remains fixed; real evidence requires a preregistered benchmark with a real frozen model.
