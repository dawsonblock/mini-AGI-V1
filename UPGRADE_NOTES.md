# v16.1 Upgrade Notes

v16.1 is an additive convergence release built on the v15.8 execution trunk.

Key changes:

1. Added RC14.7 executed-run, preregistration, qualification, and independent-reproduction modules.
2. Added RC12.5 grounded replay and live-canary authority modules.
3. Added Dream-RSI as a proposal-side subsystem.
4. Added selected v16 plasticity execution, replicated Dream statistics, isolated solver, and convergence-control modules without adopting v16 RuntimeClosure2.
5. Added `minagi.v161` hardening: content-addressed evaluator identities, real dataset membership/disjointness verification, signed run receipts, converged evidence references, and physical RuntimeClosureV161.
6. Added Colab/CUDA execution via PyTorch, Transformers, PEFT, Accelerate, Safetensors, and optional bitsandbytes.
7. Added persistent content-addressed Colab/Drive storage, seed-level campaign resume, dataset freezing, result export, environment diagnostics, smoke and Qwen configs, and a Colab notebook.
8. Replaced inherited release metadata with a new versioned source manifest and assembly signature.

The legacy v15.8 APIs remain available so the existing regression suite continues to pass.
