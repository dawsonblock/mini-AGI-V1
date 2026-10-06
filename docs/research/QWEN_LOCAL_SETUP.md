# Using a local Qwen-class frozen model

RC8.2 does not download a model automatically. It accepts a local frozen model through either of two adapters.

## OpenAI-compatible server

1. Serve the local model with your preferred local runtime.
2. Compute/record the exact immutable model artifact digests.
3. Construct `FrozenModelIdentity` using those digests.
4. Instantiate `OpenAICompatibleFrozenModel` with the local `/v1/chat/completions` endpoint.

The endpoint name is not treated as model identity. The artifact manifest is.

## Apple Silicon / MLX

Install the optional dependency:

```bash
python -m pip install -e '.[mlx]'
```

Then point `MLXLMFrozenModel` at the local model directory. RC8.2 hashes relevant model/config/tokenizer files and uses that closure as the immutable model identity.

For large model directories, hashing is streamed; model files are not loaded wholesale merely to compute identity.

## Research caution

Changing quantization, tokenizer, config, weights or model shards changes the model closure and therefore constitutes a different frozen baseline. Do not compare sequential checkpoints across different model digests as if the base model were unchanged.
