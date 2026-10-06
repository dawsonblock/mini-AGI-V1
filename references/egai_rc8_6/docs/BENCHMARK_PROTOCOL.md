# Frozen-weight continual-learning benchmark

1. Freeze and hash the foundation model files/config/tokenizer.
2. Pre-register exact future/retention/security case artifacts and scorer ID.
3. Keep the experience stream physically/logically separate from evaluation sets.
4. Reject exact IDs, exact inputs, exact cases and high-overlap near duplicates across train/evaluation.
5. Evaluate fixed checkpoints such as 0, 10, 50, 100, 250, 500, 1000.
6. Never pass future labels/results to `SandboxAdaptiveAgent.observe()`.
7. Learn procedures only from externally verified repairs/demonstrations.
8. Compare frozen baseline and adaptive agent on the same frozen model digest.
9. Report future success, retention, security, paired-bootstrap intervals, resource use and FTE.
10. The Authority Plane signs the benchmark, runner results, independent evaluation, qualification and promotion decision.
11. Neural adaptation remains disabled until the frozen-weight experiment passes.
