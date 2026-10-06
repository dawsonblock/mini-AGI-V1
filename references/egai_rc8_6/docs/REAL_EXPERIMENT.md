# Running the first real experiment

1. Build a frozen model manifest for the exact local model files.
2. Create an experience JSONL and a disjoint hidden evaluation JSONL.
3. Use deterministic decoding (`temperature=0`) and a fixed backend configuration.
4. Run `egai research-run ...`.
5. Preserve the generated experiment manifest digest, journal, result signature and model digest.
6. Repeat across multiple seeds/task families before making a continual-learning claim.

A result is interesting only when future-task gain is positive with a confidence interval above zero, retention/security constraints hold, leakage checks pass, and the model digest is unchanged.
