# macOS / Apple Silicon profile

This release can run on Apple Silicon Macs in a **functional local continual-memory mode**.

The original QW3 execution engine was written around NVIDIA CUDA. macOS does not provide CUDA, so the custom native QW3 CUDA kernels, CUDA KVMem page movement, and model-specific CUDA Gated-DeltaNet reconstruction are not translated to Metal in this release. Instead, the Mac profile separates the system into:

- **Model execution:** current `llama.cpp` Metal or MLX.
- **Persistent continual memory:** KVContinual SQLite memory store.
- **Experience logging:** local JSONL episode log.
- **Qualification / promotion:** the same digest-bound adapter registry and admin control plane.
- **Native QW3 host tools:** buildable with CUDA disabled for inspection, archive utilities, policies, and host tests.

This is intentional. It produces a working Mac system without pretending CUDA kernels can be recompiled as Metal.

## Recommended backend

Use `llama.cpp`/Metal first. Current llama.cpp supports Qwen3.5 Gated-DeltaNet models on Apple Silicon and exposes an OpenAI-compatible server. MLX is available as an alternative and current MLX-LM contains Qwen3.5 model code.

## 1. Install

```bash
cd kvmem-qw3-coherent-continual-rc10.5-macos
./macos/qw3-mac install
./macos/qw3-mac doctor
```

The installer creates `.venv-mac`, installs the continual-control Python package, and attempts to install both MLX-LM and Homebrew `llama.cpp`.

## 2A. Run with llama.cpp / Metal

Use a local GGUF:

```bash
export QW3_MODEL="$HOME/models/Qwen3.5-9B-Q4_K_M.gguf"
export QW3_MAC_CONTEXT=16384
./macos/qw3-mac backend-llama
```

Or use a Hugging Face model accepted by your current `llama-server` build:

```bash
export QW3_HF_MODEL='<repo-or-repo:file>'
./macos/qw3-mac backend-llama
```

Keep the backend bound to `127.0.0.1` unless you intentionally configure authentication and network exposure.

## 2B. Run with MLX

In another terminal:

```bash
export QW3_MLX_MODEL='<current MLX Qwen3.5 model or local MLX model>'
./macos/qw3-mac backend-mlx
```

The Mac profile deliberately does not pin a community model ID because quantizations and best available conversions change quickly.

## 3. Start the continual-memory gateway

In another terminal:

```bash
./macos/qw3-mac gateway
```

The gateway listens on:

```text
http://127.0.0.1:8088/v1
```

It forwards inference to the backend on `127.0.0.1:8080`, injects relevant persistent memories, and logs each episode.

Test it:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"local-qwen",
    "messages":[{"role":"user","content":"Explain why recurrent memory can forget old state."}],
    "max_tokens":128
  }'
```

The endpoint supports ordinary non-streaming and pass-through streaming requests. Unknown OpenAI-compatible fields such as `tools`, `top_p`, and `stop` are forwarded to the local backend.

## 4. Add persistent memory

```bash
curl http://127.0.0.1:8088/memory \
  -H 'Content-Type: application/json' \
  -d '{
    "text":"The current local service endpoint is port 9000.",
    "memory_type":"fact",
    "confidence":0.95,
    "novelty":0.9,
    "future_utility":0.9,
    "importance":0.8,
    "verified":true
  }'
```

The database defaults to `data/memory.sqlite3`. Episodes default to `data/episodes.jsonl`.

## 5. Train a candidate LoRA with MLX

Prepare:

```text
data/training/train.jsonl
```

using MLX-LM's chat/completion JSONL format. Then:

```bash
export QW3_MLX_MODEL='<the MLX base/quantized model>'
export QW3_TRAIN_ITERS=200
export QW3_TRAIN_LAYERS=4
./macos/qw3-mac train-lora
```

This uses LoRA/QLoRA rather than rewriting the foundation weights. Treat trained adapters as **candidates** until they pass the project's evaluation/qualification path.

For large models, start with a small number of trainable layers, batch size 1, short examples, and a quantized base. Qwen3.5 training support on MLX has changed rapidly during 2026; re-run qualification after every MLX upgrade.

## 6. Build the native host tools on macOS

```bash
./macos/qw3-mac build-host
```

This builds and tests the C++ host components with CUDA disabled. It does **not** turn the QW3 CUDA inference engine into a Metal engine.

## Feature matrix

| Capability | Linux + NVIDIA | Apple Silicon Mac |
|---|---|---|
| Native QW3 CUDA inference | Yes | No |
| Qwen local inference | Native QW3 | llama.cpp Metal / MLX |
| Native CUDA KVMem paging | Yes | No |
| Custom `(T,Z)` GDN CUDA capture | Research/integration boundary | No |
| Exact upstream model execution | QW3 selected replay path | Backend dependent |
| Persistent temporal memory | Yes | Yes |
| Memory retrieval injection | Yes | Yes |
| Episode logging | Yes | Yes |
| Candidate adapter registry | Yes | Yes |
| Qualification / rollback | Yes | Yes |
| LoRA training | PEFT path | MLX LoRA/QLoRA path |

## Memory sizing

Unified memory is shared by the model, Metal/MLX working buffers, KV/recurrent state, macOS, and other applications. Do not size a model from its weight file alone.

As a practical starting point, use smaller 4-bit models on 16 GB systems, 9B-class models on 24 GB systems, and consider 27B 4-bit models only when there is sufficient unified-memory headroom for your requested context. Use `./macos/qw3-mac doctor` to see installed memory.

## Known limitations

1. The custom QW3 KVMem CUDA algorithms are **not** executed by Metal/MLX in this Mac profile.
2. Memory retrieval on the gateway uses a deterministic local lexical fallback. Replace it with embeddings/FAISS when you have enough memories to need semantic indexing.
3. Streaming is passed through, but the gateway records request/memory provenance rather than reconstructing full assistant text from backend-specific SSE extensions.
4. Adapter application semantics differ between llama.cpp and MLX. Promotion should restart/reload the selected backend adapter rather than assuming an execution cache survives an adapter change.
5. Metal/MLX behavior changes quickly. Pin versions once you have a qualified local configuration.

## Target for a true native Metal KVMem port

A full Metal-equivalent of the CUDA runtime still requires model-specific work:

1. Metal Gated-DeltaNet `(T_C, Z_C)` capture.
2. Metal attention-KV block relocation.
3. unified-memory tier semantics replacing CUDA host/device copies.
4. seam/suffix replay kernels.
5. exact replay oracle parity against the same model weights.
6. adapter-aware cache invalidation.

Until those are implemented and benchmarked, this Mac profile keeps the execution-cache boundary explicit.


## RC10.5 Gated-DeltaNet correctness oracle

Before a native Metal KVMem implementation is attempted, RC10.5 now includes an independent CPU/fp64 implementation of the actual QW3 Gated-DeltaNet state update. Run:

```bash
./macos/qw3-mac gdn-check
```

The check compares direct token-by-token recurrence against a composed block `(T_C,Z_C)` summary and verifies that `Z_C` equals the zero-start final state. This validates the fixed-input recurrence algebra only; arbitrary segment relocation in the full deep network can still change hidden inputs and therefore still requires seam/suffix/exact-replay repair.

## RC10.4 controlled learning cycle

RC10.4 stores every non-streaming interaction in `data/experience.sqlite3` in addition to the diagnostic JSONL log. A stored interaction is **not** automatically eligible for training. Verify useful/correct episodes explicitly:

```bash
curl -X POST http://127.0.0.1:8088/learning/episodes/<episode-id>/verify \
  -H 'Content-Type: application/json' \
  -d '{"verified":true,"importance":0.9}'
```

Inspect the training-ready queue:

```bash
curl 'http://127.0.0.1:8088/learning/ready?limit=100'
```

Then run a complete offline candidate-training cycle:

```bash
export QW3_MLX_MODEL='<your qualified MLX base model>'
export QW3_BASE_MODEL_DIGEST='sha256:<immutable-base-model-digest>'
./macos/qw3-mac learning-cycle
```

The cycle exports only verified episodes, trains an MLX LoRA/QLoRA adapter, hashes the complete adapter directory, and registers it as a candidate. It **does not promote** the adapter. Evaluation and qualification still have to pass before the protected admin API can promote it.

## RC10.4 adapter routing with llama.cpp

Current llama.cpp servers can preload LoRA adapters and apply a configured set per request. To preload an adapter:

```bash
export QW3_LLAMA_LORA='/absolute/path/to/adapter.gguf'
export QW3_LLAMA_LORA_HOT_SWITCH=1
./macos/qw3-mac backend-llama
```

Create `runtime/configs-macos/adapters.json` from `adapters.example.json`, using the adapter IDs reported by llama.cpp. The gateway performs deterministic top-1 routing and emits the request-local `lora` selection. If no router entry clears the score threshold, no adapter is selected.

## Additional Mac commands

```bash
./macos/qw3-mac bootstrap   # install + doctor + initialize directories
./macos/qw3-mac verify      # verify the Python/control-plane environment
./macos/qw3-mac test        # Python tests + native host tests
./macos/qw3-mac benchmark   # small gateway latency smoke benchmark
./macos/qw3-mac gdn-check   # validate CPU GDN block-composition algebra
```
