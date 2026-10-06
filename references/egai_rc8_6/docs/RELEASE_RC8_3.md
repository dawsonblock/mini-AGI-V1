# RC8.3 — Real Frozen-Model Experiment Harness

RC8.3 turns the RC8.2 sequential-learning prototype into a preregistered, auditable experiment.

New in this release:
- Signed ExperimentManifest binding model digest, experience stream, hidden evaluation set, checkpoints, scorer, seed, leakage policy and learner configuration.
- Signed append-only ExperimentJournal with checkpoint records.
- ExperimentOrchestrator that refuses model/dataset substitutions.
- Signed ExperimentResult and independent ContinualLearningQualifier.
- Baseline future/retention/security scores recorded at each checkpoint.
- Export of final learned-skill snapshot into content-addressed storage.
- Stronger normalized/character-shingle leakage checks.
- Research-mode HTTP backend rejects stochastic temperature settings.
- CLI `research-run` path for local frozen models.

Still deliberately excluded: LoRA training, dynamic neural expansion, recurrent latent-state training and architecture self-modification.
