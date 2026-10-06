# EGAI v3 RC8.6 — End-to-End Governed Learning Integration

RC8.4 is the hardened frozen-model continual-learning experiment harness. It preserves the Evidence, Cognition, Learning, and independent Authority planes while making the real-model experiment substantially harder to fake or accidentally invalidate.

## What changed from RC8.3

- **Signed repair verification receipts.** Experience data can suggest a repair, but the real orchestrated path only learns after an independent verifier signs a receipt bound to the exact case, attempted output, repair, expected output, score, and verifier key.
- **Stronger frozen-model identity.** Model identity can bind weight files, tokenizer/config digests, runtime executable/family, and generation configuration. llama.cpp executable changes or generation-seed changes therefore change identity.
- **Deterministic local runner settings.** OpenAI-compatible local servers are pinned to temperature=0, top_p=1 and an explicit seed in research mode. llama.cpp receives explicit seed/temperature/token limits.
- **Experiment closure.** Every completed run binds the preregistration manifest, model digest, backend digest, environment digest, source-tree digest, journal head, and final skill snapshot into one closure object stored in CAS.
- **Stricter skill induction.** The production research path requires support from distinct verified inputs and records verifier IDs and verification-receipt digests inside the learned procedure.
- **Negative skill evidence.** When a retrieved skill participates in a failed attempt, the sandbox skill memory records the failure and lowers its future reliability.
- **Stricter qualification.** A qualifier may require experiment closure and a minimum count of independently verified repairs in addition to forward gain, positive confidence interval, retention, and security constraints.
- **Fault-injection primitives and additional adversarial tests.**

## Scientific invariant

During the frozen-model experiment:

```text
model_digest(E_0) == model_digest(E_N)
backend_digest(E_0) == backend_digest(E_N)
```

Improvement must come from evidence, verified procedures, retrieval, and governed policy state—not neural weight mutation.

## Validation

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python scripts/run_preregistered_synthetic.py
PYTHONPATH=. python -m compileall -q egai tests scripts
```

## Local llama.cpp

```json
{
  "backend": "llama_cpp_cli",
  "executable": "/path/to/llama-cli",
  "model_path": "/path/to/model.gguf",
  "args": [],
  "seed": 0,
  "temperature": 0,
  "max_tokens": 2048,
  "timeout": 180
}
```

## OpenAI-compatible local server

Create an immutable model manifest first, then configure the endpoint:

```json
{
  "backend": "http",
  "endpoint": "http://127.0.0.1:8000",
  "served_model": "qwen-local",
  "model_manifest": "qwen-manifest.json",
  "temperature": 0,
  "top_p": 1.0,
  "seed": 0,
  "max_tokens": 2048,
  "runtime_fingerprint": "pin-to-your-local-server-build"
}
```

Run the preregistered experiment:

```bash
egai research-run \
  --experience examples/experience.jsonl \
  --evaluation examples/evaluation.jsonl \
  --backend-config backend.json \
  --root ./egai-research-run \
  --checkpoints 0,10,50,100,250,500,1000
```

## Still deliberately disabled

LoRA/adapters, dynamic neural expansion, shared-weight consolidation, recurrent latent reasoning, and architecture self-modification remain disabled. They should only be unlocked after a real frozen-weight experiment demonstrates statistically credible forward transfer on hidden future tasks without unacceptable retention or security regression.


## RC8.5 additions

See `docs/RELEASE_RC8_5.md`. Replay integrity, negative transfer, skill lifecycle retirement, and stricter statistical qualification are now first-class release evidence.


## RC8.6 additions
See `docs/RELEASE_RC8_6.md`. The governed experiment path now uses authoritative evidence, synthesized procedures, cognitive traces, belief compilation, evidence-backed learning proposals, and an explicit research-to-promotion bridge.
