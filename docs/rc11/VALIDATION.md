# RC11 Validation

The release validation entrypoint is:

```bash
./scripts/rc11/validate_release.sh
```

It runs four independent suites so failures are attributable to the correct validity domain:

1. RC10.5 operational continual/runtime regression suite.
2. RC11 execution-memory and integrity suite.
3. mini-AGI v6 authority/control reference suite.
4. mini-AGI v4.3 RC10/Qwen oracle reference suite.

The RC11 execution suite includes adversarial regression tests for:

- source ID rewrite attempts;
- tokenizer-sensitive source digest binding;
- source/artifact alias mismatch;
- adapter mutation after qualification;
- signed promotion receipt verification;
- partial-dimension split-half RoPE relocation;
- weighted replay uniqueness;
- dense T/Z storage budget classification;
- candidate-ID traversal and manifest path escape;
- symlink substitution;
- non-finite signed receipt values;
- signed rollback after cache-namespace tampering;
- active production-state metadata tampering;
- stale production-snapshot recovery from the signed ledger;
- external monotonic-floor/tail-anchor enforcement;
- candidate-continuity and registry-identity tampering;
- qualification-record rewrite attempts;
- promoted cache-namespace mutation;
- required qualification-bundle model/tokenizer/execution identity binding;
- qualification bundle schema-v2 tampering and runtime-build binding.

## Still required on Apple Silicon hardware

This archive cannot claim a qualified Metal HYPIC path until the following are measured on the target Mac with real Qwen weights:

- real pre/post-RoPE tensor equivalence;
- recurrent state equivalence against exact execution;
- causal-convolution seam repair equivalence;
- full hybrid-layer differential traces;
- logit KL / top-k agreement;
- task-level quality deltas;
- latency / TTFT / throughput;
- hardware, execution-identity, runtime-build and kernel-build binding in signed qualification receipts.

Exact replay remains the authority until those gates pass.

## Release-tree integrity

`validate_release.sh` also runs `scripts/rc11/verify_release.py`. The verifier checks every shipped file (except the manifest itself), rejects additions/deletions/byte changes, and recomputes the deterministic release-tree SHA-256. Regenerate only with `scripts/rc11/build_release_manifest.py` after an intentional source change.
