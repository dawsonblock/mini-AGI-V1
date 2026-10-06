# RC10.4 — Mac continual-learning integration

RC10.4 closes the largest practical gap in the previous Mac profile: interactions can now move through a controlled, persistent learning pipeline without granting the training worker authority to mutate production.

## Runtime flow

```text
chat request
   -> temporal memory retrieval
   -> llama.cpp/Metal or MLX inference
   -> JSONL diagnostic episode
   -> SQLite ExperienceStore
   -> explicit verification
   -> training-ready queue
   -> MLX dataset export
   -> candidate LoRA/QLoRA training
   -> immutable candidate registration
   -> evaluation + qualification
   -> protected promotion
   -> cache policy = REPLAY_REQUIRED
```

## Authority separation

The gateway may record experiences. The offline learning worker may export verified experiences, train adapters, register candidates, and bind evaluation results. It cannot promote a candidate. Promotion remains in the admin control API and requires the admin token.

## Adapter artifacts

MLX adapters are directories rather than a single weight file. RC10.4 therefore extends the registry with deterministic directory hashing. The tree digest binds every relative filename and each file digest. Mutation after qualification invalidates promotion.

## Adapter routing

`continual/configs-macos/adapters.json` may describe already-loaded llama.cpp adapters. For llama.cpp backends, the gateway can select one adapter and pass a request-local `lora` configuration. This does not merge adapters into the base model and does not change the cache-compatibility rule: any promoted adapter-generation change still requires neural execution caches to be replayed or invalidated.

## What remains research work

RC10.4 does not implement native Metal HYPIC/KVMem. It also does not automatically decide that a user correction is verified truth. Verification is explicit because automatically training on unverified model/user text would create a self-poisoning loop.
