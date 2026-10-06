# RC8.6 — End-to-End Governed Learning Integration

RC8.6 removes research shortcuts that could invalidate a real continual-learning claim.

## Major changes
- New production experiment path: Task -> Evidence -> Model Attempt -> Independent Repair Verification -> Evidence -> Skill Synthesis -> Hidden Evaluation.
- `procedure_hint` and `feedback_verified` remain only for backwards-compatible legacy tests; RC8.6 governed orchestration does not consume them.
- Procedures are synthesized from repeated, independently verified repair trajectories.
- Every learned procedure carries evidence ancestry and verification receipt digests.
- Evidence ledger appends are atomic under `BEGIN IMMEDIATE`; SQLite uses WAL + FULL synchronous mode.
- CAS writes use unique temp files and atomic publication.
- Cognitive execution passes through the controller and records action traces.
- Belief snapshots are compiled from eligible evidence and bound into experiment results.
- Learned procedures are converted to SkillManifest + Hypothesis + Abstraction + TransferEvaluation + LearningProposal artifacts.
- Research qualification and production promotion are separate. `GovernedPromotionPipeline` runs the full Candidate -> Build -> Benchmark -> Result -> Evaluation -> Qualification -> Promotion -> Runtime chain.
- Signed HTTP runtime attestation primitives bind an externally served model to a model digest and runtime digest.
- Neural weight updates remain disabled.
