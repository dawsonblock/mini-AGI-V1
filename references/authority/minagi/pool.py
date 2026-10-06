#!/usr/bin/env python3
"""
A shared, self-growing expert pool with no assigned specialities.

One pool, every domain in one mixture, no labels anywhere. Soft top-k routing
distributes capability across experts by itself, and a character can combine
fragments from several. The cost is real and worth stating: capabilities share
parameters, so they CAN interfere. What holds forgetting off is the trunk
learning rate, kept at a fraction of the experts' (`training.trunk_lr_mult`) -
not the pool, and not replay. Measured, not assumed: see `runs/cl/`.

Because it is not provable, it is made falsifiable instead. `superposition()`
asks whether each domain lights up a disjoint set of experts (specialisation -
the design failed) or whether domains share experts heavily (superposition -
the design worked).

GROWTH WITHOUT INTERVENTION
`maybe_grow()` runs inside the training loop. Nobody decides "now add chess".
The pool grows when it is saturated - every expert carrying load and routing
entropy near maximum - and NOT merely when the loss is flat, because a plateau
with idle experts means the optimiser is stuck, not that the model is full.
New experts arrive gated to almost zero and are pruned away if they never contribute.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .paged_adamw import mark_rowwise
from torch.utils.checkpoint import checkpoint

from .precision import compute_dtype

class Expert(nn.Module):
    """
    The unit that gets paged: `depth` SwiGLU blocks, residual on each other.

    At depth 1 - the default, and what every saved model so far contains - it
    is a single block and keeps its projections under the names w1, w3, w2,
    which is what the expert files hold. Deeper experts keep the extra blocks
    under blocks[i], so depth can change without renaming anything a depth-1
    model was saved with.

    Depth and width buy parameters at the same price in VRAM: width 2048 at
    depth 1 and width 1024 at depth 2 are both 3.15M and both cost the same
    slot. What differs is whether an expert can compose two transformations or
    only one.
    """

    def __init__(self, d_model, d_ff, depth=1):
        super().__init__()
        self.depth = depth
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({"w1": nn.Linear(d_model, d_ff, bias=False),
                           "w3": nn.Linear(d_model, d_ff, bias=False),
                           "w2": nn.Linear(d_ff, d_model, bias=False)})
            for _ in range(depth - 1)])

    def forward(self, x):
        y = self.w2(F.silu(self.w1(x)) * self.w3(x))
        if not self.blocks:
            return y
        x = x + y
        for b in self.blocks:
            x = x + b["w2"](F.silu(b["w1"](x)) * b["w3"](x))
        return x



def expert_levels(e):
    """
    The (w1, w3, w2) triples inside one expert, outermost first.

    The first block always lives directly on the expert under w1/w3/w2, which
    is what every saved model contains; any further blocks live in `blocks`.
    Reading both here means depth can change without renaming the tensors a
    shallower model was saved with.
    """
    lv = [(e.w1, e.w3, e.w2)]
    for b in getattr(e, "blocks", []) or []:
        lv.append((b["w1"], b["w3"], b["w2"]))
    return lv


class SharedPool(nn.Module):
    """One pool of experts, reachable from every depth, with no fixed size."""

    def __init__(self, d_model, n_experts=64, d_ff=192, max_experts=1024,
                 depth=1):
        super().__init__()
        self.d_model, self.d_ff, self.max_experts = d_model, d_ff, max_experts
        self.depth = depth
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff, depth) for _ in range(n_experts)])
        # per-expert gate: a newly grown expert starts at zero and therefore
        # changes nothing until it earns its way up
        self.gate = mark_rowwise(nn.Parameter(torch.ones(n_experts)))
        self.register_buffer("use", torch.zeros(n_experts), persistent=False)
        self.register_buffer("age", torch.zeros(n_experts), persistent=False)
        # `age` counts call-site invocations, so it advances once per
        # block-application per micro-batch - far faster than the step counter
        # and on a scale that depends on the depth actually run. It cannot be
        # compared against a step count. `born` records the training step an
        # expert was created at, which can be.
        self.register_buffer("born", torch.zeros(n_experts), persistent=False)
        # |gate| at the previous prune check, so a gate that is small but
        # still climbing can be told apart from one that never moved
        self.register_buffer("gate_seen", torch.zeros(n_experts),
                             persistent=False)
        self.grow_events = []
        self.pressure = 0.0        # EMA of routing mass lost to top-k
        self.want_k = 0.0          # EMA of experts covering 90% of the mass
        # v2 names distinguish true full-pool demand from any resident-card
        # routing statistics. SharedPool routes the full pool, so they track
        # the same observation here.
        self.capacity_pressure = 0.0
        self.capacity_want_k = 0.0
        self.route_top_k = 1
        self.admission_mode = "full_pool"
        self._stack_cache = None            # invalidated on growth/pruning

    def stacked(self):
        """
        All expert weights as three contiguous tensors, for batched dispatch.

        Dispatching expert by expert costs one small matmul per expert per call
        site, which is experts x sites kernel launches per forward and dominates
        everything else at this pool size. Stacked, the whole pool runs as three
        batched matmuls regardless of how many experts there are.
        """
        n = len(self.experts)
        # Only cache when gradients are off. A stacked tensor is a node in the
        # autograd graph, so reusing one after backward() has freed that graph
        # would cut the experts off from their gradient and leave only the gate
        # receiving one. Under no_grad there is no graph to go stale.
        if (not torch.is_grad_enabled() and self._stack_cache is not None
                and self._stack_cache[0] == n):
            return self._stack_cache[1]
        # an expert's projection may have been wrapped by a test-time adapter;
        # the underlying Linear is what stacks
        def W(mod):
            return mod.weight if hasattr(mod, "weight") else mod.base.weight
        # An expert is one SwiGLU by default and several when it has depth.
        # Returned as a list of levels so the dispatch can apply them in turn;
        # a depth-1 pool returns a single level and takes the same path it
        # always did.
        depth = len(expert_levels(self.experts[0]))
        levels = []
        for d in range(depth):
            lv = [expert_levels(e)[d] for e in self.experts]
            levels.append((torch.stack([W(a) for a, _, _ in lv]),   # [E,dff,d]
                           torch.stack([W(b) for _, b, _ in lv]),
                           torch.stack([W(c) for _, _, c in lv])))  # [E,d,dff]
        if not torch.is_grad_enabled():
            self._stack_cache = (n, levels)
        return levels

    def invalidate(self):
        self._stack_cache = None

    def n_experts(self):
        return len(self.experts)

    def n_routable(self):
        """How many experts a token may choose between. Every one of them here;
        a paged pool answers with its resident set instead."""
        return len(self.experts)

    def router_rows(self):
        """Rows the per-token router needs. Growth happens into this headroom."""
        return self.max_experts

    def routable_gate(self):
        return self.gate

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def add_experts(self, k, seed_from=None, device=None, step=0,
                    birth_gate=0.0):
        """
        Grow the pool by k.

        `birth_gate` is the scale a newcomer starts at, and it is here because
        AutoGrow passes it to whichever pool it is holding - PagedPool needs
        it, since an expert born at exactly zero can never be chosen there at
        all. This pool has always started them at zero and still does unless
        told otherwise, so the default changes nothing; what it buys is that
        one grower can drive either pool.
        """
        device = device or self.gate.device
        for _ in range(k):
            if len(self.experts) >= self.max_experts:
                break
            e = Expert(self.d_model, self.d_ff,
                       getattr(self, "depth", 1)).to(device)
            if seed_from is not None:
                for pn, po in zip(e.parameters(),
                                  self.experts[seed_from].parameters()):
                    pn.copy_(po + 0.02 * torch.randn_like(po))
            self.experts.append(e)
        n = len(self.experts)
        g = torch.full((n,), float(birth_gate), device=device)
        g[:self.gate.numel()] = self.gate.data
        old_gate = self.gate
        self.gate = mark_rowwise(nn.Parameter(g))          # newcomers at birth_gate
        # The training loop resynchronises optimiser parameter ownership after
        # growth.  Preserve an explicit lineage edge so rowwise Adam state for
        # all pre-existing gates survives that Parameter replacement.
        self.gate._minagi_state_parent = old_gate
        self.gate._minagi_state_grow = n - old_gate.numel()
        self.invalidate()
        self.use = torch.cat([self.use,
                              torch.zeros(n - self.use.numel(),
                                          device=self.use.device)])
        self.age = torch.cat([self.age,
                              torch.zeros(n - self.age.numel(),
                                          device=self.age.device)])
        self.born = torch.cat([self.born,
                               torch.full((n - self.born.numel(),), float(step),
                                          device=self.born.device)])
        self.gate_seen = torch.cat([self.gate_seen,
                                    torch.zeros(n - self.gate_seen.numel(),
                                                device=self.gate_seen.device)])
        return n

    @torch.no_grad()
    def prune(self, step, survival=None, min_gate=0.005, min_age=8000,
              min_delta=5e-4, protect=0):
        """
        Remove experts that were grown and never contributed.

        `survival` is the window, and is the name every caller uses because
        PagedPool.prune takes it. It means the same thing here that `min_age`
        always did - how long an expert gets before it is judged - and it is
        accepted so that a caller does not have to know which pool it holds.

        THE TEST DIFFERS FROM PagedPool'S, and deliberately. There, an expert
        that is not resident does not train, so its gate cannot move and a
        gate test would condemn exactly the experts being starved of the card;
        residency is the only honest signal. Here there is no card. Every
        expert is resident and trains on every step, so the reverse holds -
        see below.

        Contribution is read off the GATE, not off routing traffic. Routing
        traffic cannot answer this question: the load-balancing auxiliary loss
        pushes utilisation toward uniform on purpose, so every expert - live or
        dead - receives roughly its 1/n share and no `use` threshold can ever
        separate them. The gate is under no such pressure. It is trained only
        by whether the expert helps, and a new expert starts at exactly zero,
        so a gate still at zero after a fair number of steps is the model
        saying it found no use for that expert.

        Two conditions guard against pruning something that is still learning:
        the expert must be old enough in TRAINING STEPS, and its |gate| must
        not have climbed measurably since the last check. A gate that is tiny
        but rising is working its way in and is left alone.

        `protect` keeps the first N experts (the ones the pool started with)
        out of pruning entirely - they carry the trunk's learned capability and
        are not speculative additions.
        """
        if survival is not None:
            min_age = survival
        n = len(self.experts)
        g = self.gate.data.abs()
        keep = []
        for i in range(n):
            if i < protect:
                keep.append(i)
                continue
            old_enough = (step - float(self.born[i])) > min_age
            silent = float(g[i]) < min_gate
            flat = (float(g[i]) - float(self.gate_seen[i])) < min_delta
            if not (old_enough and silent and flat):
                keep.append(i)
        self.gate_seen = g.clone()
        if len(keep) == n:
            return 0
        self.experts = nn.ModuleList([self.experts[i] for i in keep])
        old_gate = self.gate
        self.gate = mark_rowwise(nn.Parameter(self.gate.data[keep].clone()))
        self.gate._minagi_state_parent = old_gate
        self.gate._minagi_state_index = torch.tensor(keep, dtype=torch.long,
                                                     device=old_gate.device)
        self.use = self.use[keep].clone()
        self.age = self.age[keep].clone()
        self.born = self.born[keep].clone()
        self.gate_seen = self.gate_seen[keep].clone()
        self.invalidate()
        return n - len(keep)

    def saturation(self):
        """How fully the existing pool is being used, in [0,1] per measure."""
        u = self.use / self.use.sum().clamp_min(1)
        n = u.numel()
        ideal = 1.0 / n
        ent = float(-(u.clamp_min(1e-9) * u.clamp_min(1e-9).log()).sum())
        # idle is a GATE question, not a routing question - see prune()
        return {"experts": n,
                "idle": int((self.gate.data.abs() < 0.01).sum()),
                "idle_by_routing": int((u < 0.1 * ideal).sum()),
                "entropy_frac": ent / max(math.log(n), 1e-9),
                "peak_over_ideal": float(u.max()) / ideal,
                "pressure": float(self.pressure),
                "want_k": float(self.want_k)}


class PooledMLP(nn.Module):
    """Routes into the shared pool. Nothing here names a domain."""

    def __init__(self, pool, d_model, top_k=4, site=0, z_weight=1e-3,
                 grad_checkpoint=True, capacity_factor=1.5,
                 hierarchical_index=False, index_group_size=64,
                 index_top_groups=4, index_refresh=128,
                 index_max_candidates=512, index_audit_every=256,
                 index_min_recall=0.95, index_fallback_calls=16,
                 index_strategy="contiguous", index_kmeans_iters=6):
        super().__init__()
        self._pool = [pool]
        self.top_k, self.site, self.z_weight = top_k, site, z_weight
        # Growth decisions need the actual routing width rather than guessing
        # it from the resident card size. Multiple sites share a pool, so keep
        # the maximum observed top-k.
        pool.route_top_k = max(int(getattr(pool, "route_top_k", 1)), int(top_k))
        # One row per EXPERT, always - never one row per VRAM slot.
        #
        # While a forward has room on the card, every character ranks the
        # whole pool with these rows (that is how a forward asks for experts -
        # see PagedPool.admit), and every character weighs the ones on the
        # card with the same rows. A row that meant "slot 5" would mean a
        # different expert from one forward to the next, and the router
        # could not express "this kind of character wants that expert" - the
        # one thing it exists to say. The cost is d_model per expert, a few
        # megabytes at a thousand experts.
        width = (pool.router_rows() if hasattr(pool, "router_rows")
                 else pool.max_experts)
        self.router = nn.Linear(d_model, width, bias=False)
        mark_rowwise(self.router.weight)
        nn.init.normal_(self.router.weight, 0.0, 0.01)
        self.depth_emb = nn.Parameter(torch.zeros(d_model))
        self.aux = torch.zeros(())
        self.last_route = None
        self.last_weight = None
        self.record_weights = False
        # Batched dispatch pads every expert to a capacity buffer, and autograd
        # retains one per call site - gigabytes across the depth this model
        # runs. Recomputing the expert pass during backward trades ~30% more
        # compute for most of that memory, and is still far faster than
        # dispatching expert by expert.
        self.grad_checkpoint = grad_checkpoint
        # The largest share of a batch any one expert may take, as a multiple
        # of its fair share. Assignments past it are dropped. Without a bound
        # the padded rectangle scales with the worst imbalance rather than with
        # the work - see the dispatch below.
        self.capacity_factor = float(capacity_factor or 0.0)
        self.dropped = 0
        # ...and the denominator, so the counter reads as a share of the
        # work rather than as a number nobody can scale.
        self.routed = 0
        self.expert_index = None
        if hierarchical_index:
            from .expert_index import HierarchicalExpertIndex
            self.expert_index = HierarchicalExpertIndex(
                group_size=index_group_size, top_groups=index_top_groups,
                refresh_every=index_refresh, max_candidates=index_max_candidates,
                audit_every=index_audit_every, min_recall=index_min_recall,
                fallback_calls=index_fallback_calls, strategy=index_strategy,
                kmeans_iters=index_kmeans_iters)
        self.index_last_coverage = 1.0
        self.index_last_candidates = 0

    @property
    def pool(self):
        return self._pool[0]

    # Block.forward passes `active` only to an MLP that says it takes it
    takes_active = True

    def forward(self, x, active=None):
        """
        `active` [B, T] marks the characters still being computed; the rest
        have halted, and route nowhere. They ask for no experts, take no
        expert's capacity, and count in none of the statistics or the
        balancing loss - the pool sees exactly what writing would ask of it.
        """
        if active is None or bool(active.all()):
            return self._route(x)
        B, T, D = x.shape
        m = active.reshape(-1)
        out = torch.zeros(B * T, D, device=x.device, dtype=x.dtype)
        if bool(m.any()):
            out[m] = self._route(x.reshape(-1, D)[m].unsqueeze(0))[0]
        return out.view(B, T, D)

    def _route(self, x):
        # see capture_routes() at the bottom of this file
        B, T, D = x.shape
        p = self.pool
        # how many this token may choose between. For a resident pool that is
        # every expert; for a paged one it is the slots of the card, and the
        # indices are into the slots rather than the whole pool.
        n = p.n_routable() if hasattr(p, "n_routable") else p.n_experts()
        flat = x.reshape(-1, D)
        rows = p.resident_rows() if hasattr(p, "resident_rows") else None
        # THE TEMPERATURE OF EXPERT SELECTION - pool.select_temperature, not
        # anything about the text. 0 takes each character's top_k; above 0
        # each character DRAWS its top_k without replacement, in proportion to
        # p^(1/T), and the card is drawn the same way (PagedPool._draw).
        temp = float(getattr(p, "select_temperature", 0.0) or 0.0)
        route_coverage = 1.0
        sparse_route_ids = None  # [tokens,candidates] global UIDs in v6 indexed virtual mode
        if rows is None:
            # v5.1 virtual routing can use the hierarchical index per token.
            # Candidate selection is no-grad, but every selected row is then
            # scored by the real trainable router, so router/state gradients
            # remain ordinary autograd.  Exact recall audits fail safe to the
            # full scan.  The dense logit shell below is only an interface to
            # the established routing/statistics code; expensive O(E*d) dot
            # products are avoided for experts outside the candidate set.
            use_vindex = (getattr(p, "external_autograd", False)
                          and self.expert_index is not None
                          and n > self.expert_index.group_size)
            if use_vindex:
                inp = flat + self.depth_emb
                replay_candidates = (hasattr(p, "routing_replay")
                                     and p.routing_replay()
                                     and getattr(p, "routing_next_kind", lambda: None)() == "candidates")
                if replay_candidates:
                    cand = p.replay_candidates(self.site, flat.device)
                    coverage = torch.ones(flat.shape[0], device=flat.device)
                else:
                    cand, coverage = self.expert_index.token_candidates(
                        inp, self.router.weight, n, min_candidates=self.top_k)
                    if cand is not None and hasattr(p, "candidate_tape"):
                        cand = p.candidate_tape(self.site, cand)
                if cand is not None:
                    valid = cand >= 0
                    safe = cand.clamp_min(0)
                    rw = self.router.weight[safe]              # [N,C,D]
                    cz = (rw * inp.unsqueeze(1)).sum(-1).float()
                    cz = cz.masked_fill(~valid, float("-inf"))
                    # v6 keeps candidate coordinates sparse end-to-end.  v5.1
                    # scattered into an [tokens, experts] dense shell here,
                    # which removed O(E*d) router matmuls but retained O(T*E)
                    # memory.  ``sparse_route_ids`` maps each candidate column
                    # back to its stable global expert UID.
                    logits = cz
                    sparse_route_ids = cand
                    route_coverage = float(coverage.float().mean())
                    self.index_last_coverage = route_coverage
                    self.index_last_candidates = int(valid.sum(-1).float().mean())
                else:
                    logits = self.router(inp)[:, :n].float()
                    self.index_last_coverage = 1.0
                    self.index_last_candidates = n
            else:
                logits = self.router(flat + self.depth_emb)[:, :n].float()
                self.index_last_coverage = 1.0
                self.index_last_candidates = n
        else:
            if p.admitting():
                # THE SELECTION RULE, while the forward has room on the card.
                # Every character ranks the WHOLE pool and asks for its
                # top_k; each request carries the probability the router gave
                # it, and the most-asked-for experts are admitted until the
                # card is full. No gradient: admission decides what is
                # reachable, the routing below decides weights.
                with torch.no_grad():
                    E = p.router_rows()
                    mode = getattr(p, "admission_mode", "prefix_causal")
                    all_in = flat + self.depth_emb
                    if mode == "window_vote":
                        # Compatibility only: this is O(tokens * experts) and
                        # permits future-token influence on admission.
                        admit_in = all_in
                    elif mode == "causal_prefix_vote":
                        # v3: use a bounded prefix rather than only the first
                        # token. RecurCoder masks LM loss before the last token
                        # of this prefix, so every scored target is strictly
                        # after all states that were allowed to vote. This is
                        # causal while giving the card enough semantic evidence
                        # to distinguish, e.g., prose from code inside a chunk.
                        k = max(1, min(T, int(getattr(
                            p, "admission_prefix_tokens", 64) or 1)))
                        admit_in = all_in.view(B, T, D)[:, :k].reshape(-1, D)
                    else:
                        # v2 compatibility: full-pool scoring is O(batch *
                        # experts), independent of sequence length, but the
                        # first token is often too little evidence.
                        first = torch.arange(B, device=flat.device) * T
                        admit_in = all_in[first]
                    cand_ids = None
                    coverage = 1.0
                    if self.expert_index is not None and E > self.expert_index.group_size:
                        cand_ids, admit_z, coverage = self.expert_index.score(
                            admit_in, self.router.weight, E,
                            min_candidates=max(self.top_k, int(getattr(p, "resident", self.top_k))))
                        self.index_last_coverage = float(coverage)
                        self.index_last_candidates = int(cand_ids.numel())
                    else:
                        admit_z = F.linear(admit_in, self.router.weight[:E]).float()
                        self.index_last_coverage = 1.0
                        self.index_last_candidates = E

                    def requested(scores):
                        q = F.softmax(scores, -1)
                        if temp > 0:
                            local = q.float().sum(0)
                        else:
                            tq = torch.topk(q, min(self.top_k, scores.shape[-1]), dim=-1)
                            local = torch.zeros(scores.shape[-1], device=q.device)
                            local.index_add_(0, tq.indices.reshape(-1),
                                             tq.values.reshape(-1).float())
                        if cand_ids is None:
                            return local
                        mass = torch.zeros(E, device=q.device)
                        mass.index_add_(0, cand_ids, local)
                        return mass
                    # CAUSAL ADMISSION. The legacy implementation aggregated
                    # requests from every position in the window. v3 either
                    # uses one prefix-visible state (v2 compatibility) or a
                    # bounded causal prefix whose early LM losses are masked.
                    # In both causal modes, a later suffix cannot change the
                    # expert set used by any scored earlier target.
                    mass = requested(admit_z)

                    # Capacity demand must be measured over the FULL expert
                    # population, before residency truncates the distribution.
                    # AutoGrow consumes these EMAs; using resident-only logits
                    # would confuse a full card with a full model.
                    qfull = F.softmax(admit_z, -1)
                    kk = min(self.top_k, qfull.shape[-1])
                    kept_local = torch.topk(qfull, kk, dim=-1).values.sum(-1).mean()
                    # With hierarchical retrieval, selected groups represent
                    # only ``coverage`` of coarse probability mass.  Treat the
                    # omitted mass as discarded rather than pretending the
                    # candidate softmax was the whole pool.
                    kept_full = float(kept_local) * float(coverage)
                    sf = torch.sort(qfull, dim=-1, descending=True).values
                    want_local = float((sf.cumsum(-1) < 0.90).sum(-1).float().mean() + 1)
                    want_full = min(float(E), want_local / max(float(coverage), 1e-3))
                    p.capacity_pressure = (
                        0.9 * float(getattr(p, "capacity_pressure", 0.0))
                        + 0.1 * (1.0 - kept_full))
                    p.capacity_want_k = (
                        0.9 * float(getattr(p, "capacity_want_k", 0.0))
                        + 0.1 * float(want_full))
                # THE BALANCE TERM (pool.balance), once per forward that
                # trains: the router pays for the probability it puts on each
                # expert in proportion to that expert's share of recent
                # admissions - the Switch Transformer's balancing term, with
                # the share taken over the last ~1,000 training forwards
                # instead of this one, so a text may still want few experts as
                # long as the texts between them want them all. 0 when usage is
                # even. Only the router's rows learn from it: the characters'
                # states are detached, so it cannot bend what the trunk computes.
                if (float(getattr(p, "balance", 0.0) or 0.0) > 0
                        and self.training and torch.is_grad_enabled()
                        and p.balance_term() is None):
                    # Balance is a per-text pressure, so sampling the same
                    # causal admission states is sufficient and avoids
                    # reintroducing O(tokens * experts) work in training.
                    if cand_ids is None:
                        zg = F.linear(admit_in.detach(), self.router.weight[:E]).float()
                        P = F.softmax(zg, -1).mean(0)
                        use = p.usage_share().to(P.device)
                        p.note_balance(p.balance * (E * (use * P).sum() - 1.0))
                    else:
                        # Keep admission sublinear: balance only within the
                        # retrieved candidate set, using the real router rows so
                        # gradients still update the expert-selection policy.
                        zg = F.linear(admit_in.detach(), self.router.weight[cand_ids]).float()
                        P = F.softmax(zg, -1).mean(0)
                        use = p.usage_share().to(P.device)[cand_ids]
                        use = use / use.sum().clamp_min(1e-9)
                        C = max(1, int(cand_ids.numel()))
                        p.note_balance(p.balance * (C * (use * P).sum() - 1.0))
                p.admit(mass, draw=self.top_k)
                rows = p.resident_rows()
            # only the rows belonging to the experts in VRAM, in slot order,
            # so column j of the logits is slot j and row rows[j] is its expert
            w = self.router.weight[rows]                      # [n, d_model]
            logits = F.linear(flat + self.depth_emb, w).float()
            # a character whose request was not admitted takes its best
            # admitted expert: slots holding anything else are out of reach
            logits = logits.masked_fill(~p.admitted_mask(), float("-inf"))
        probs = F.softmax(logits, dim=-1)
        k = min(self.top_k, n)
        replay_topk = (hasattr(p, "routing_replay") and p.routing_replay()
                       and getattr(p, "routing_next_kind", lambda: None)() == "topk")
        if sparse_route_ids is not None:
            # ``probs`` lives in per-token candidate coordinates.  Top-k IDs
            # recorded on the routing tape remain global logical expert UIDs.
            k = min(k, probs.shape[-1])
            if replay_topk:
                idx = p.replay_topk(self.site, probs.device)
                if tuple(idx.shape) != (probs.shape[0], k):
                    raise RuntimeError(
                        f"routing replay top-k shape {tuple(idx.shape)} does not match "
                        f"expected {(probs.shape[0], k)}")
                match = sparse_route_ids.unsqueeze(-1).eq(idx.unsqueeze(1))
                if not bool(match.any(1).all()):
                    raise RuntimeError("routing replay expert is absent from recorded candidate set")
                pos = match.float().argmax(1)
                w = probs.gather(1, pos)
            else:
                if temp > 0:
                    keys = logits.detach()
                    gum = -torch.log(-torch.log(
                        torch.rand_like(keys).clamp_(1e-12, 1 - 1e-7)))
                    pos = torch.topk(keys / temp + gum, k, dim=-1).indices
                    w = probs.gather(1, pos)
                else:
                    w, pos = torch.topk(probs, k, dim=-1)
                idx = sparse_route_ids.gather(1, pos)
                if bool((idx < 0).any()):
                    raise RuntimeError("masked candidate selected during sparse routing")
                if hasattr(p, "route_tape"):
                    idx = p.route_tape(self.site, idx)
                    match = sparse_route_ids.unsqueeze(-1).eq(idx.unsqueeze(1))
                    if not bool(match.any(1).all()):
                        raise RuntimeError("routing tape expert is absent from candidate set")
                    pos = match.float().argmax(1)
                    w = probs.gather(1, pos)
        else:
            if replay_topk:
                # Do not rerun stochastic Gumbel selection during activation
                # recomputation: even if the tape later overwrote the ids, the
                # fresh RNG draw would perturb subsequent stochastic operations.
                idx = p.replay_topk(self.site, probs.device)
                if tuple(idx.shape) != (probs.shape[0], k):
                    raise RuntimeError(
                        f"routing replay top-k shape {tuple(idx.shape)} does not match "
                        f"expected {(probs.shape[0], k)}")
                w = probs.gather(1, idx)
            else:
                if temp > 0:
                    keys = logits.detach()
                    gum = -torch.log(-torch.log(
                        torch.rand_like(keys).clamp_(1e-12, 1 - 1e-7)))
                    idx = torch.topk(keys / temp + gum, k, dim=-1).indices
                    w = probs.gather(1, idx)
                else:
                    w, idx = torch.topk(probs, k, dim=-1)
                if hasattr(p, "route_tape"):
                    idx = p.route_tape(self.site, idx)
                    w = probs.gather(1, idx)
        # the share of the router's distribution that top-k actually captures,
        # measured BEFORE normalisation - afterwards it sums to 1 by
        # construction and carries no information
        with torch.no_grad():
            kept = float(w.sum(-1).mean()) * float(route_coverage)
            # How many experts the router actually wants: the smallest number
            # covering 90% of its probability mass. If that exceeds k, the
            # router is being forced to discard experts it would have used, and
            # the pool is genuinely too small. Unlike raw discarded mass this
            # is not confounded with an untrained, spread-out router - a uniform
            # router wants nearly all of them, but so does a well-trained one
            # that needs them, and the demand is what matters.
            srt = torch.sort(probs, dim=-1, descending=True).values
            cum = srt.cumsum(-1)
            want = (cum < 0.90).sum(-1).float().mean() + 1
            if route_coverage < 0.999999:
                want = torch.as_tensor(
                    min(float(n), float(want) / max(route_coverage, 1e-3)),
                    device=probs.device)
        w = w / w.sum(-1, keepdim=True)
        g = p.routable_gate() if hasattr(p, "routable_gate") else p.gate
        w = w * g[idx].to(w.dtype)
        if self.record_weights:
            # What each chosen expert actually contributes, AFTER the gate.
            # last_route records how often a slot was picked, which is 1/k for
            # every pick and so says nothing about weight - and the weight is
            # the whole point, because an expert gated to zero is chosen just
            # as often as one gated to 0.9 and adds nothing. Off by default;
            # tools/capture_routing.py turns it on.
            with torch.no_grad():
                # O(E), not O(tokens*E): aggregate sparse chosen weights by UID.
                total = torch.zeros(n, device=w.device, dtype=torch.float32)
                total.index_add_(0, idx.reshape(-1), w.float().reshape(-1))
                self.last_weight = total / max(1, idx.shape[0])
                one = torch.zeros(n, device=w.device, dtype=torch.float32)
                one.index_add_(0, idx[-1], w[-1].float())
                self.last_weight_one = one

        if _ROUTES is not None:
            # which EXPERTS this call-site invocation picked, in the order the
            # invocations happen - so a caller can reconstruct depth by depth
            # what ran for each character
            _ROUTES.append(((rows[idx] if rows is not None else idx)
                            .detach().cpu(),
                            w.detach().float().cpu()))

        with torch.no_grad():
            hit = torch.bincount(idx.reshape(-1), minlength=n).float()
            if hasattr(p, "note_use"):
                p.note_use(hit)          # slots map back to experts
            else:
                p.use += hit
                p.age += 1
            self.last_route = hit / hit.sum().clamp_min(1)
            # How much probability mass top-k had to discard. If the router
            # wants to spread across more experts than k, the pool is too
            # small to express what it is trying to do - that is capacity
            # pressure, and it is visible without waiting for a plateau.
            p.pressure = 0.9 * float(p.pressure) + 0.1 * (1.0 - kept)
            p.want_k = 0.9 * float(p.want_k) + 0.1 * float(want)
            if rows is None:
                p.capacity_pressure = p.pressure
                p.capacity_want_k = p.want_k
        frac = torch.bincount(idx[:, 0], minlength=n).float() / max(1, idx.shape[0])
        if sparse_route_ids is not None:
            # Mean router probability in global UID coordinates without an
            # [tokens, experts] materialization.
            p_sum = torch.zeros(n, device=probs.device, dtype=probs.dtype)
            valid = sparse_route_ids >= 0
            p_sum.index_add_(0, sparse_route_ids[valid], probs[valid])
            p_mean = p_sum / max(1, probs.shape[0])
        else:
            p_mean = probs.mean(0)
        self.aux = ((frac * p_mean).sum() * n
                    + self.z_weight * torch.logsumexp(logits, -1).pow(2).mean())

        # capacity-based batched dispatch (Switch-Transformer style): every
        # expert gets a fixed-size slot buffer, so the whole pool runs as three
        # batched matmuls instead of one small matmul per expert.
        N = flat.shape[0]
        flat_e = idx.reshape(-1)                       # [N*k]
        flat_w = w.reshape(-1).to(flat.dtype)
        tok = torch.arange(N, device=flat.device).repeat_interleave(k)

        order = torch.argsort(flat_e)
        e_sorted, t_sorted, w_sorted = flat_e[order], tok[order], flat_w[order]
        counts = torch.bincount(e_sorted, minlength=n)
        cap = int(counts.max().item()) if n else 0
        if cap == 0:
            return torch.zeros_like(x)

        # position of each assignment within its own expert's slot buffer
        starts = torch.cumsum(counts, 0) - counts
        slot = torch.arange(e_sorted.numel(), device=flat.device) - starts[e_sorted]

        # v5 virtual paging: logical expert identity is independent of the GPU
        # cache slot.  We can therefore dispatch every token to its causally
        # selected expert in one untruncated graph.  The custom autograd op
        # reloads UID@digest during backward and accumulates external dW by UID.
        # Apply the same Switch-style capacity guard before crossing that
        # boundary; no expert weights are materialised as nn.Parameters.
        if hasattr(p, "dispatch_external"):
            self.routed += int(e_sorted.numel())
            if n and self.capacity_factor:
                limit = max(1, int(math.ceil(
                    self.capacity_factor * e_sorted.numel() / n)))
                if cap > limit:
                    keep = slot < limit
                    self.dropped += int((~keep).sum())
                    e_sorted, t_sorted = e_sorted[keep], t_sorted[keep]
                    w_sorted, slot = w_sorted[keep], slot[keep]
            out = p.dispatch_external(flat, e_sorted, t_sorted, w_sorted)
            return out.view(B, T, D)

        # A disk-backed pool loads only the experts this batch selected. The
        # resident pool stacks everything, which is fine while it fits.
        if hasattr(p, "stacked_subset"):
            present = torch.unique(idx).tolist()
            remap = torch.full((n,), -1, dtype=torch.long, device=flat.device)
            for slot_i, e in enumerate(present):
                remap[e] = slot_i
            e_sorted = remap[e_sorted]
            n = len(present)
            counts = torch.bincount(e_sorted, minlength=n)
            cap = int(counts.max().item()) if n else 0
            starts = torch.cumsum(counts, 0) - counts
            slot = torch.arange(e_sorted.numel(),
                                device=flat.device) - starts[e_sorted]
            levels = p.stacked_subset(present)
            if isinstance(levels, tuple) and len(levels) == 3:
                levels = [levels]              # a paged pool returns one level
        else:
            levels = p.stacked()

        # CAPACITY. The dispatch buffer is a padded rectangle [n, cap, D], so
        # without a bound `cap` is whatever the busiest expert happened to
        # receive and the memory tracks the worst imbalance rather than the
        # work - one expert taking 44% of a window is enough to cost an order
        # of magnitude more memory for the same arithmetic, and an OOM.
        #
        # So every expert takes at most `capacity_factor` times its fair share
        # and assignments past that are DROPPED, which is what Switch does. A
        # dropped assignment costs a character one of its top_k experts, not
        # the character itself; the load-balancing term above is what keeps it
        # rare. Constant dropping means the router is collapsing, which is
        # worth seeing rather than paying for - hence the counters, reported on
        # the progress line as a share of `routed`.
        self.routed += int(e_sorted.numel())
        if n and self.capacity_factor:
            limit = max(1, int(math.ceil(
                self.capacity_factor * e_sorted.numel() / n)))
            if cap > limit:
                keep = slot < limit
                self.dropped += int((~keep).sum())
                e_sorted, t_sorted = e_sorted[keep], t_sorted[keep]
                w_sorted, slot = w_sorted[keep], slot[keep]
                cap = limit

        def run(src, *ws):
            lv = [(ws[i], ws[i + 1], ws[i + 2]) for i in range(0, len(ws), 3)]
            # The slot buffer, and everything computed from it, in the COMPUTE
            # dtype rather than the residual stream's. The residual stays fp32
            # by design; the dispatch does not need to, and halving this buffer
            # is the difference between fitting on the card and not. Autocast
            # does not do it for us: the weights are cast TO buf's dtype a few
            # lines below, so the matmul runs at whatever buf is.
            dt = compute_dtype()
            if src.device.type != "cuda":
                dt = src.dtype
            buf = torch.zeros(n, cap, D, device=src.device, dtype=dt)
            buf[e_sorted, slot] = src[t_sorted].to(dt)
            if len(lv) == 1:
                W1, W3, W2 = lv[0]
                h = F.silu(torch.bmm(buf, W1.transpose(1, 2).to(buf.dtype))) * \
                    torch.bmm(buf, W3.transpose(1, 2).to(buf.dtype))
                y = torch.bmm(h, W2.transpose(1, 2).to(buf.dtype))  # [n,cap,D]
                return y[e_sorted, slot].to(src.dtype)
            # deeper experts stack residually, so an expert of any depth can
            # still be added to the mixture without changing its scale
            x = buf
            for W1, W3, W2 in lv:
                h = F.silu(torch.bmm(x, W1.transpose(1, 2).to(x.dtype))) * \
                    torch.bmm(x, W3.transpose(1, 2).to(x.dtype))
                x = x + torch.bmm(h, W2.transpose(1, 2).to(x.dtype))
            return x[e_sorted, slot].to(src.dtype)

        flat_w = tuple(t for lv_ in levels for t in lv_)
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            gathered = checkpoint(run, flat, *flat_w, use_reentrant=False)
        else:
            gathered = run(flat, *flat_w)

        out = torch.zeros_like(flat)
        out.index_add_(0, t_sorted, gathered * w_sorted.unsqueeze(-1))
        return out.view(B, T, D)


def _mem_frac():
    """
    Peak fraction of the GPU this process actually needs. 0.0 without a GPU.

    Deliberately NOT mem_get_info(): that reports the caching allocator's
    reserved pool, which stays near 100% once the run is warm whether or not
    there is real room, so a brake reading it would refuse growth forever.
    Peak *allocated* is the honest number - it is what has to fit.
    """
    if not torch.cuda.is_available():
        return 0.0
    total = torch.cuda.get_device_properties(0).total_memory
    return torch.cuda.max_memory_allocated() / max(total, 1)


class AutoGrow:
    """
    Let the pool find its own size by trying, not by reading its own statistics.

    Growth is speculative and regular: add a few experts, gated near zero so
    nothing already working is damaged, and let pruning remove the ones that
    never contribute. The pool ratchets toward the size the data needs, and the
    cost of guessing wrong is a few experts that get deleted again.

    The mechanism, in the order it runs:

      Every `growth.every_chars` characters the reader asks this class whether
      to grow. It adds `growth.k` experts if all of these hold, and otherwise
      says in one line which one refused:

        ROOM    the pool is under its disk ceiling and VRAM is not nearly
                spent. An expert file is ~25 MB with its Adam moments, so the
                disk ceiling is what decides how large the model may ever get.

        USED    the capacity already added is being asked for. Measured on
                STALENESS, never on routing share: the load-balancing loss
                forces routing toward uniform on purpose, so every expert -
                live or dead - receives roughly its 1/n share and a
                routing-based idle count reports zero forever.

        KEPT    the previous cohort survived its trial. Counted from `born`,
                so it names the experts it is about.

        FITS    no more than `growth.max_in_flight` experts are inside their
                trial at once. Neither usage brake can see a newborn, so
                without this cap there is a whole trial window in which
                nothing can refuse.

      A fifth condition lives in the reader, because the pool cannot see it
      from the inside: whether train and held-out have separated, which is what
      memorising looks like. See `growth.max_gap`.

      A new expert is born at a small gate - `growth.birth_gate` - with its
      parents' router rows averaged, so it scores every character with the
      same average of their scores. It is on TRIAL for
      `prune.survival_chars`, during which prune cannot touch it; at the end
      of the trial prune deletes it unless it is still being admitted. The
      gate scales what it contributes, never whether it is asked for.
    """

    def __init__(self, grow_k=8, max_experts=1024, dying_frac_max=0.25,
                 keep_ratio_min=0.35, mem_frac_max=0.85, max_disk_gb=0.0,
                 max_in_flight=0, birth_gate=0.001, recent_mult=4.0,
                 pressure_min=0.08, want_k_ratio_min=1.15,
                 plateau_checks=3, plateau_delta=0.01, demand_hits=2,
                 qualification_checks=3, qualification_gain_min=0.002):
        self.grow_k, self.max_experts = grow_k, max_experts
        self.dying_frac_max = dying_frac_max
        self.max_in_flight = max_in_flight
        self.keep_ratio_min = keep_ratio_min
        self.mem_frac_max = mem_frac_max
        self.max_disk_gb = max_disk_gb
        self.birth_gate = birth_gate
        # v2: growth must be justified by persistent full-pool routing demand
        # AND a held-out plateau. This prevents load-balancing from creating a
        # self-fulfilling add/use/survive/add loop while the model is still
        # improving with the capacity it already has.
        self.pressure_min = float(pressure_min)
        self.want_k_ratio_min = float(want_k_ratio_min)
        self.plateau_checks = max(1, int(plateau_checks))
        self.plateau_delta = float(plateau_delta)
        self.demand_hits_required = max(1, int(demand_hits))
        self._val_history = []
        self._demand_hits = 0
        # v3 promotion gate. A new cohort is a hypothesis, not evidence of
        # useful capacity. No further cohort may be born until held-out loss
        # beats the pre-growth baseline by a configured minimum. This does not
        # delete a useful-but-slow expert; it simply prevents an unvalidated
        # growth cascade. Staleness pruning remains the rejection mechanism.
        self.qualification_checks = max(1, int(qualification_checks))
        self.qualification_gain_min = max(0.0, float(qualification_gain_min))
        self._pending_growth = None
        self.last_qualification = None
        # how far past its trial an expert still counts as "recent"
        self.recent_mult = recent_mult
        self.keep_ratio = 1.0
        self.log = []

    def _in_flight(self, pool, model_step):
        """Experts that have been added and whose trial has not ended."""
        born = getattr(pool, "born", None)
        if born is None:
            return 0
        trial = float(getattr(pool, "trial", 0) or 0)
        if trial <= 0:
            return 0
        age = model_step - born
        return int(((born > 0) & (age < trial)).sum())

    def _earning(self, pool, model_step):
        """
        Are the experts added recently being used? Read from the pool
        as it is now, not from anything remembered.

        The question is asked of experts that have FINISHED their trial and
        are still young - old enough to have had their fair turn, recent
        enough that their fate says something about whether the pool still
        needs more. `born` records the step an expert joined and `last_seen`
        records when anything last wanted it, so both are properties of the
        network at this instant and nothing has to be carried between
        decisions.

        "Earning" means being asked for, not holding a gate above a bar. The
        gate says how loudly an expert speaks when it is chosen, which is
        near-uninformative about whether it will be chosen again - see
        PagedPool.dying().

        Returns 1.0 when there is nothing to judge: a pool of originals has
        not yet failed at anything.
        """
        born = getattr(pool, "born", None)
        if born is None:
            return 1.0
        trial = float(getattr(pool, "trial", 0) or 0) or 1600.0
        age = model_step - born
        judged = (born > 0) & (age >= trial) & (age < trial * self.recent_mult)
        n = int(judged.sum())
        if n == 0:
            return 1.0
        # The same staleness prune deletes on and the brake calls dying. One
        # definition of "nothing wants this", read from the pool.
        d = pool.dying() if hasattr(pool, "dying") else None
        if d is None:
            return 1.0
        at = float(getattr(pool, "dying_at", 0.75))
        return float((d[judged[:d.numel()]] < at).sum()) / float(n)

    def _evidence(self, val_loss, pool):
        """Return (demand, plateau, diagnostics) for a growth decision."""
        pressure = float(getattr(pool, "capacity_pressure",
                                 getattr(pool, "pressure", 0.0)) or 0.0)
        want_k = float(getattr(pool, "capacity_want_k",
                               getattr(pool, "want_k", 0.0)) or 0.0)
        top_k = max(1, int(getattr(pool, "route_top_k", 1) or 1))
        demand_now = (pressure >= self.pressure_min and
                      want_k >= top_k * self.want_k_ratio_min)
        self._demand_hits = self._demand_hits + 1 if demand_now else 0
        demand = self._demand_hits >= self.demand_hits_required

        self._val_history.append(float(val_loss))
        if len(self._val_history) > self.plateau_checks:
            del self._val_history[:-self.plateau_checks]
        if len(self._val_history) < self.plateau_checks:
            plateau = False
        else:
            span = max(self._val_history) - min(self._val_history)
            improvement = self._val_history[0] - self._val_history[-1]
            plateau = (span <= self.plateau_delta * 2.0 and
                       improvement <= self.plateau_delta)
        return demand, plateau, {
            "pressure": pressure, "want_k": want_k, "top_k": top_k,
            "demand_hits": self._demand_hits,
            "plateau_points": len(self._val_history),
        }

    def _qualification(self, val_loss, pool, model_step):
        p = self._pending_growth
        if p is None:
            return True, {"state": "clear", "retired": 0}
        p["checks"] += 1
        gain = float(p["baseline"]) - float(val_loss)
        p["best_gain"] = max(float(p.get("best_gain", float("-inf"))), gain)
        if gain >= self.qualification_gain_min:
            rec = {"state": "promoted", "born_step": p["step"],
                   "checks": p["checks"], "gain": gain,
                   "best_gain": p["best_gain"],
                   "required_gain": self.qualification_gain_min,
                   "from": p["from"], "to": p["to"], "retired": 0}
            self.last_qualification = rec
            self._pending_growth = None
            return True, rec
        if p["checks"] >= self.qualification_checks:
            retired = 0
            uids = list(p.get("uids") or [])
            if uids and hasattr(pool, "retire_uids"):
                retired = int(pool.retire_uids(uids, step=model_step,
                                               reason="failed_growth_qualification"))
            rec = {"state": "rejected", "born_step": p["step"],
                   "checks": p["checks"], "gain": gain,
                   "best_gain": p["best_gain"],
                   "required_gain": self.qualification_gain_min,
                   "from": p["from"], "to": p["to"],
                   "uids": uids, "retired": retired}
            self.last_qualification = rec
            self._pending_growth = None
            # The rejected cohort cannot leave stale evidence priming another
            # immediate birth. Require fresh demand and a fresh plateau.
            self._demand_hits = 0
            self._val_history = []
            return False, rec
        return False, {"state": "trial", "born_step": p["step"],
                       "checks": p["checks"], "gain": gain,
                       "best_gain": p["best_gain"],
                       "required_gain": self.qualification_gain_min,
                       "from": p["from"], "to": p["to"],
                       "uids": list(p.get("uids") or []), "retired": 0}

    def step(self, val_loss, pool, model_step):
        # Qualification may roll back a failed cohort, so it must run before
        # measuring the current pool size/saturation for this decision.
        qualification_clear, qualification = self._qualification(
            val_loss, pool, model_step)
        s = pool.saturation()
        n = s["experts"]
        idle_frac = s["idle"] / max(n, 1)
        self.keep_ratio = self._earning(pool, model_step)
        mem_frac = _mem_frac()
        demand, plateau, evidence = self._evidence(val_loss, pool)

        # Growth happens only when every question says yes.
        #   ROOM    is there space for it, on the card and on the disk
        #   USED    is the capacity already added being asked for
        #   KEPT    did the previous cohort survive its trial
        #   FITS    is there room inside the in-flight cap
        # One more - have train and held-out separated - is asked by the
        # reader, which is the only thing that sees both. See `growth.max_gap`.
        disk = getattr(pool, "disk_bytes", None)
        want = disk(self.grow_k) / 1e9 if disk else 0.0
        # How many are still inside their trial. Neither usage brake can see
        # these - a newborn reads dying 0, and keep_ratio judges only experts
        # whose trial has ENDED - so this cap is what stops a whole trial
        # window passing with nothing able to refuse. It is also what makes
        # keep_ratio_min bind: the next cohort cannot arrive until this one
        # has been judged.
        in_flight = self._in_flight(pool, model_step)
        fits = (not self.max_in_flight
                or in_flight + self.grow_k <= self.max_in_flight)
        room = (n + self.grow_k <= self.max_experts
                and mem_frac <= self.mem_frac_max
                and fits
                and (not self.max_disk_gb or want <= self.max_disk_gb))
        used = idle_frac <= self.dying_frac_max
        kept = self.keep_ratio >= self.keep_ratio_min

        rec = {"step": model_step, "val": round(float(val_loss), 4),
               "experts": n, "idle": s["idle"], "mem": round(mem_frac, 3),
               "in_flight": in_flight,
               "disk_gb": round(disk(0) / 1e9, 2) if disk else 0.0,
               "keep_ratio": round(self.keep_ratio, 3), "grew": 0,
               "pressure": round(evidence["pressure"], 4),
               "want_k": round(evidence["want_k"], 2),
               "route_top_k": evidence["top_k"],
               "demand_hits": evidence["demand_hits"],
               "plateau": bool(plateau),
               "qualification": qualification,
               "retired": int(qualification.get("retired", 0))}

        if room and used and kept and demand and plateau and qualification_clear:
            k = self.grow_k
            pool.add_experts(k, seed_from=int(pool.use.argmax()),
                             step=model_step, birth_gate=self.birth_gate)
            pool.grow_events.append({"step": model_step, "to": pool.n_experts()})
            uids = []
            puid = getattr(pool, "uid", None)
            if puid is not None and int(getattr(puid, "numel", lambda: 0)()) >= k:
                uids = [int(x) for x in puid[-k:].detach().cpu().tolist()]
            self._pending_growth = {"step": int(model_step),
                                    "baseline": float(val_loss),
                                    "checks": 0, "best_gain": float("-inf"),
                                    "from": int(n), "to": int(pool.n_experts()),
                                    "uids": uids}
            rec["grew"] = k
            rec["experts"] = pool.n_experts()      # count AFTER the addition
            rec["reason"] = (f"validated capacity pressure {evidence['pressure']:.3f}, "
                             f"want_k {evidence['want_k']:.1f}/{evidence['top_k']}, "
                             f"held-out plateau, {s['idle']}/{n} dying, recent "
                             f"additions earning {self.keep_ratio:.2f}")
        elif not qualification_clear:
            if qualification["state"] == "rejected":
                rec["reason"] = (f"previous growth cohort rejected after "
                                 f"{qualification['checks']} checks; rolled back "
                                 f"{qualification.get('retired', 0)} expert(s). "
                                 f"Fresh demand evidence is required.")
            else:
                rec["reason"] = (f"previous growth cohort is {qualification['state']}: "
                                 f"best held-out gain {qualification.get('best_gain', qualification.get('gain', 0.0)):.4f}, "
                                 f"need {self.qualification_gain_min:.4f} before another cohort")
        elif not demand:
            rec["reason"] = (f"no persistent capacity demand: pressure "
                             f"{evidence['pressure']:.3f} (need {self.pressure_min:.3f}), "
                             f"want_k {evidence['want_k']:.1f} for top-{evidence['top_k']} "
                             f"({evidence['demand_hits']}/{self.demand_hits_required} checks)")
        elif not plateau:
            rec["reason"] = (f"held-out is not on a stable plateau across "
                             f"{self.plateau_checks} checks; use existing capacity first")
        elif not fits:
            rec["reason"] = (f"{in_flight} experts are still inside their "
                             f"trial, of {self.max_in_flight} allowed at once "
                             f"- the last additions have not been judged yet")
        elif not room:
            rec["reason"] = (f"no room: {n} experts, "
                             f"{rec['disk_gb']:.1f} GB on disk"
                             f"{' of %.1f' % self.max_disk_gb if self.max_disk_gb else ''}"
                             f", VRAM {100*mem_frac:.0f}%")
        elif not used:
            rec["reason"] = (f"{s['idle']}/{n} experts have gone "
                             f"{100*pool.dying_at:.0f}% of the way to the prune "
                             f"line unaddressed - capacity already added is "
                             f"not being asked for")
        else:
            rec["reason"] = (f"only {self.keep_ratio:.2f} of the experts "
                             f"added recently earned a gate - pool has found "
                             f"its size")
        self.log.append(rec)
        return rec


@torch.no_grad()
def superposition(model, batches_by_domain, device):
    """
    Are capabilities localised or shared?

    For each domain, record which experts fire. Then compare the distributions.
    Disjoint sets mean the pool specialised by domain and the design failed;
    heavy overlap means fragments are shared and combined, which is the point.
    """
    sites = [m for m in model.modules() if isinstance(m, PooledMLP)]
    if not sites:
        return None
    pool = sites[0].pool
    n = pool.n_experts()
    dist = {}
    model.eval()
    for name, batches in batches_by_domain.items():
        acc = torch.zeros(n, device=device)
        for x, y in batches:
            for s in sites:
                s.last_route = None
            model(x, y)
            for s in sites:
                if s.last_route is not None:
                    acc[:s.last_route.numel()] += s.last_route
        dist[name] = (acc / acc.sum().clamp_min(1)).cpu()
    model.train()

    names = list(dist)
    out = {"n_experts": n, "domains": names, "usage": {k: v.tolist()
                                                       for k, v in dist.items()}}
    # pairwise overlap: 1 - Jensen-Shannon distance, and shared-expert count
    pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p, q = dist[names[i]], dist[names[j]]
            m = 0.5 * (p + q)

            def kl(a, b):
                a = a.clamp_min(1e-9); b = b.clamp_min(1e-9)
                return float((a * (a / b).log()).sum())
            js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
            active_p = set((p > 0.2 / n).nonzero().flatten().tolist())
            active_q = set((q > 0.2 / n).nonzero().flatten().tolist())
            inter = len(active_p & active_q)
            union = len(active_p | active_q) or 1
            pairs[f"{names[i]}|{names[j]}"] = {
                "js_divergence": round(js, 4),
                "jaccard": round(inter / union, 4),
                "shared_experts": inter}
    out["pairs"] = pairs
    j = [v["jaccard"] for v in pairs.values()]
    out["mean_jaccard"] = round(sum(j) / max(len(j), 1), 4)
    out["verdict"] = ("superposition - domains share most experts"
                      if out["mean_jaccard"] > 0.6 else
                      "partially shared" if out["mean_jaccard"] > 0.3 else
                      "SPECIALISED - domains use disjoint experts")
    return out


# ----------------------------------------------------------------------------
# introspection
# ----------------------------------------------------------------------------
_ROUTES = None


class capture_routes:
    """
    Record which experts every call site picks, for one forward.

        with capture_routes() as picks:
            model(idx)
        # picks == [(ids[N,k], weights[N,k]), ...] one entry per invocation,
        # in the order the invocations happened - prelude first, then each
        # latent pass, so the index into the list is the depth.

    The ids are EXPERT ids, not slot numbers, so they stay comparable across
    a swap: the same expert keeps the same id whichever slot it lands in.
    """

    def __enter__(self):
        global _ROUTES
        _ROUTES = []
        return _ROUTES

    def __exit__(self, *exc):
        global _ROUTES
        _ROUTES = None
        return False
