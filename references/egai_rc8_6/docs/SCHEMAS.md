# RC8.2 durable and experimental objects

## Durable authority objects
- `EvidenceRecord`
- `SkillManifest`
- `LearningProposal`
- `CandidateManifest`
- `BuildManifest` (builder signed)
- `BenchmarkSpec` (benchmark registrar signed)
- `ResultBundle` (runner signed)
- `EvaluationBundle` (evaluator signed)
- `QualificationRecord` (qualifier signed)
- `PromotionDecision` (promotion authority signed)
- `RuntimeManifest`
- `ImprovementRecord`

## Sandbox/research objects
- `VerifiedEpisode`
- `LearnedProcedure`
- `AgentResponse`
- `TaskCase`
- `SequentialPoint`
- `SequentialReport`
- `Hypothesis`
- `AbstractionCandidate`
- `Counterexample`
- `TransferEvaluation`
- `CapabilityGap`

Sandbox objects are not production authority. They become persistent only after being packaged into a candidate artifact and crossing the Authority Plane.
