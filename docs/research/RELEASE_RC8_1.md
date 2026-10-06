# RC8.1 acceptance criteria

1. No public learning/skill registry method can directly promote/deploy a candidate.
2. Promotion requires valid evaluator, qualifier and promotion signatures.
3. Candidate/build/evaluation/qualification/decision digests form a closed chain.
4. Promotion decision nonce is single-use.
5. Factual beliefs exclude simulation, replay and model inference by default.
6. Raw benchmark results, not candidate-supplied metrics, produce EvaluationBundle metrics.
7. Security regression is a hard reject independent of future-task gain.
8. CAS rejects malformed digests and verifies bytes on retrieval.
9. Replay sees only its signed visible prefix.
10. Improvement history is signed and hash chained.
