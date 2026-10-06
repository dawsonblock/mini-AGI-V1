"""mini-AGI v5.0 research stack.

The v3-compatible path remains available for existing byte checkpoints.  New
v4 model families can bind a ByteLevel-BPE tokenizer, keep a materially larger
stable dense semantic path beside sparse paged experts, search the expert pool
hierarchically, consolidate validated expert behaviour back into the stable
path, quarantine live-learning text into shadow candidates, execute explicit
capability-scoped code actions, score intermediate reasoning with an external
process verifier, and replay recorded discovery trees when improving search
policies.

The trusted state machinery remains separate from those research features:
immutable checkpoint generations, single-writer authority, per-expert AdamW
clocks, functional regression tests, and provenance-aware memory are the
mechanisms that decide what state is allowed to persist.
"""

__all__ = [
    "tokenizer", "model", "decode", "pool", "expert_index", "recur", "stream",
    "corpus", "store", "checkpointing", "authority", "paged_adamw",
    "memory", "optim_groups", "functional_eval", "growth_lab", "consolidation",
    "shadow_live", "candidate_pipeline", "stats", "segmented", "provenance", "codeact", "verifier", "dream", "report",
    "expert_store", "expert_cache", "expert_grad_store", "graph_epoch",
    "paged_autograd", "external_adamw", "virtual_pool", "full_bptt",
]

__version__ = "6.0.0"
