"""
Write a fresh weights directory: one file per expert, nothing trained yet.

The directory IS the model, so starting from scratch means writing the
directory rather than initialising an object and hoping something saves it
later. Every expert file holds its weights beside a pair of zeroed Adam
moments, which is what the paged pool expects to find when it loads one.

This used to be a script in tools/, which meant a checkout without tools/
could not create a model at all, and the documented first step was a file
that might not be there. Training calls `create` itself now when it is
pointed at a directory that does not exist, so there is no first step.

Every default comes from config.yaml. Nothing here decides model shape.
"""

import os
import shutil

import torch


def create(out, seed=0, verbose=True, force=False, **over):
    """
    Build `out` from config.yaml, returning the config that was written.

    Keyword overrides take the same names the settings do - experts,
    resident, d_ff, depth, d_model, trunk_d_ff, n_head, block, max_steps,
    top_k - and a None is ignored, so a caller can forward unset CLI
    arguments without special-casing each one.
    """
    from dataclasses import asdict

    from minagi import store
    from minagi.config import load, get
    from minagi.recur import RecurConfig, RecurCoder

    c = load()
    over = {k: v for k, v in over.items() if v is not None}

    def pick(name, *path, default=None):
        if name in over:
            return over[name]
        for p in path:
            v = get(c, p, None)
            if v is not None:
                return v
        return default

    experts    = int(pick("experts", "pool.experts", default=32))
    resident   = int(pick("resident", "pool.resident", default=32))
    d_ff       = int(pick("d_ff", "pool.width", "pool.d_ff", default=1024))
    depth      = int(pick("depth", "pool.depth", default=1))
    d_model    = int(pick("d_model", "model.d_model", default=512))
    trunk_d_ff = int(pick("trunk_d_ff", "model.d_ff", default=1408))
    n_head     = int(pick("n_head", "model.n_head", default=8))
    block      = int(pick("block", "model.context_end", "model.context",
                          default=16384))
    max_steps  = int(pick("max_steps", "model.max_steps", default=6))
    top_k      = int(pick("top_k", "pool.top_k", default=8))
    n_prelude  = int(pick("n_prelude", "model.n_prelude", default=2))
    n_recur    = int(pick("n_recur", "model.n_recur", default=1))
    n_coda     = int(pick("n_coda", "model.n_coda", default=0))
    min_steps  = int(pick("min_steps", "model.min_steps", default=1))
    train_steps_mean = float(pick("train_steps_mean", "model.train_steps_mean", default=0.0))
    bptt_window = int(pick("bptt_window", "model.bptt_window", default=max_steps))
    ponder_beta = float(pick("ponder_beta", "model.ponder_beta", default=0.01))
    halt_prior = float(pick("halt_prior", "model.halt_prior", default=0.1))
    halt_thresh = float(pick("halt_thresh", "model.halt_thresh", default=0.9))
    halt_freeze = bool(pick("halt_freeze", "model.halt_freeze", default=False))
    rope_theta = float(pick("rope_theta", "model.rope_theta", default=10000.0))
    vocab_size = int(pick("vocab_size", "model.vocab_size", default=265))
    tokenizer_path = pick("tokenizer_path", "model.tokenizer_path", default=None)
    pool_hierarchical_index = bool(pick("pool_hierarchical_index", "pool.hierarchical_index", default=False))
    pool_index_group_size = int(pick("pool_index_group_size", "pool.index_group_size", default=64))
    pool_index_top_groups = int(pick("pool_index_top_groups", "pool.index_top_groups", default=4))
    pool_index_refresh = int(pick("pool_index_refresh", "pool.index_refresh", default=128))
    pool_index_max_candidates = int(pick("pool_index_max_candidates", "pool.index_max_candidates", default=512))
    pool_index_audit_every = int(pick("pool_index_audit_every", "pool.index_audit_every", default=256))
    pool_index_min_recall = float(pick("pool_index_min_recall", "pool.index_min_recall", default=0.95))
    pool_index_fallback_calls = int(pick("pool_index_fallback_calls", "pool.index_fallback_calls", default=16))
    pool_index_strategy = str(pick("pool_index_strategy", "pool.index_strategy", default="kmeans"))
    pool_index_kmeans_iters = int(pick("pool_index_kmeans_iters", "pool.index_kmeans_iters", default=6))

    if os.path.exists(out):
        if not force:
            raise FileExistsError(f"{out} exists; pass force=True to replace it")
        shutil.rmtree(out)

    tok_spec = {"kind": "byte", "vocab_size": 265}
    if tokenizer_path:
        # A tokenizer is architecture, not runtime configuration.  Bind an
        # immutable copy into the fresh model before the first checkpoint is
        # committed so every generation carries the same token-ID mapping.
        os.makedirs(out, exist_ok=True)
        from minagi.tokenizer import copy_tokenizer_into_weights
        tok_spec = copy_tokenizer_into_weights(tokenizer_path, out)
        vocab_size = int(tok_spec["vocab_size"])
    elif vocab_size != 265:
        raise ValueError("a non-byte vocab_size requires tokenizer_path")

    cfg = RecurConfig(vocab_size=vocab_size, d_model=d_model, n_head=n_head,
                      d_ff=trunk_d_ff, rope_theta=rope_theta,
                      n_prelude=n_prelude, n_recur=n_recur, n_coda=n_coda,
                      max_steps=max_steps, min_steps=min_steps, block=block,
                      train_steps_mean=train_steps_mean, bptt_window=bptt_window,
                      ponder_beta=ponder_beta, halt_prior=halt_prior,
                      halt_thresh=halt_thresh, halt_freeze=halt_freeze,
                      use_pool=True,
                      pool_experts=experts, pool_d_ff=d_ff,
                      pool_depth=depth, pool_top_k=top_k,
                      pool_hierarchical_index=pool_hierarchical_index,
                      pool_index_group_size=pool_index_group_size,
                      pool_index_top_groups=pool_index_top_groups,
                      pool_index_refresh=pool_index_refresh,
                      pool_index_max_candidates=pool_index_max_candidates,
                      pool_index_audit_every=pool_index_audit_every,
                      pool_index_min_recall=pool_index_min_recall,
                      pool_index_fallback_calls=pool_index_fallback_calls,
                      pool_index_strategy=pool_index_strategy,
                      pool_index_kmeans_iters=pool_index_kmeans_iters,
                      pool_dense_residual=bool(pick("pool_dense_residual", "pool.dense_residual", default=False)),
                      # Router rows belong to EXPERTS, not to VRAM slots, so
                      # this is one row per expert in the pool - not the
                      # card's slots. Sizing it to `resident` writes a router
                      # too small to address most of the pool, and the model
                      # then refuses to load. Growth adds a row per newborn.
                      pool_max=experts)
    torch.manual_seed(seed)
    m = RecurCoder(cfg)
    d = asdict(cfg)
    d["pool_resident"] = resident
    d["tokenizer"] = tok_spec
    store.save(m, out, step=-1, val=None, opt=None, cfg=d, verbose=verbose)

    if verbose:
        per = depth * 3 * d_model * d_ff
        print(f"\n  {experts} experts x {per:,} parameters "
              f"({d_ff} hidden units, depth {depth})")
        print(f"  pool {experts * per / 1e6:.2f}M, resident {resident} = "
              f"{resident * per / 1e6:.2f}M in VRAM")
        print(f"  total {m.n_params() / 1e6:.2f}M")
    return d
