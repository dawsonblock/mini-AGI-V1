# RC8.2 frozen-model experiment protocol

## Primary hypothesis
A frozen model plus governed non-parametric learning will improve on unseen future tasks as verified experience accumulates without material retention loss.

## Pre-registration
Before a run, freeze:
- model manifest/digest,
- experience stream,
- future-task set,
- retention set,
- security set,
- scorer ID,
- checkpoints,
- qualification thresholds,
- random seeds.

## Required arms
A. Frozen model only.
B. Frozen + evidence/retrieval.
C. Frozen + verified learned procedures.
D. Frozen + cognitive routing (when enabled).
E. Frozen + replay/meta-policy (later).

## Primary endpoint
Future-task success AUC and checkpoint gain vs the frozen arm.

## Hard constraints
- model digest constant,
- no evaluation examples passed to observe(),
- leakage report clean,
- forgetting <= configured epsilon,
- no security regression,
- provenance closure = 100%,
- resource budget satisfied.

## Interpretation
A positive synthetic run proves only the machinery is capable of retaining and reusing verified procedures. A real research claim requires multiple seeds, held-out task families, confidence intervals and preregistered evaluation.

## Backend identity caveat
For the strongest frozen-weight claim, prefer the direct `llama_cpp_cli` backend or another backend that hashes the exact model files it executes. The HTTP backend binds a declared local model manifest but cannot, by itself, prove that an independently managed server did not swap its underlying weights during the run.
