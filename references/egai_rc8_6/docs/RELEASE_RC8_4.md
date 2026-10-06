# RC8.4 — Real Model Runner + Experiment Reliability

RC8.4 hardens the RC8.3 experimental harness for real frozen-model trials.

## Major changes
- Signed verification receipts now authorize learning from repairs.
- Model identity includes runtime executable/family and generation configuration.
- OpenAI-compatible local HTTP and llama.cpp CLI backends bind deterministic generation settings.
- Experiment closure binds manifest, model, backend, environment, source tree, journal head and final skill snapshot.
- Skill induction requires distinct verified inputs and records verifier/receipt provenance.
- Failed skill usage is recorded and reduces retrieval reliability.
- Sequential reports expose verified-repair and skill-failure counts.
- Qualification can require closure and minimum independently verified repair evidence.
- Fault-injection primitives added for backend/runtime reliability tests.

Neural weight updates remain intentionally disabled.
