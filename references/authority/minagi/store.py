#!/usr/bin/env python3
"""
The weights directory IS the model.

Everything the model works with is a separate file. There is no monolithic
checkpoint; this directory is what training reads at the start of a session,
what it writes at every evaluation, and what inference loads from.

    weights/
      manifest.json          what exists, its shape, and where it came from
      core.npz               embeddings, attention, norms, halting head
      routers.npz            one entry per call site - which experts it reaches
      optim.npz              Adam moments for the trunk and the routers
      experts/               one file per expert: w1, w3, w2 and that
        e00000.npz           expert's own Adam moments
        e00001.npz
        ...

Only the experts are split out, because an expert is the unit that gets paged,
grown and pruned - a new expert really is a new file. Everything else is always
resident together and gains nothing from being separate, so it is bundled.

When the model grows an expert, a new file appears in experts/. When it prunes
one, that file is deleted. `ls weights/experts | wc -l` is the parameter count
in the most literal sense available.

Writes are atomic: every file is written to a .tmp and renamed, so an
interrupted save cannot leave a half-written weight behind.
"""

import hashlib
import json
import os
import sys

import numpy as np
import torch

from .precision import pack_bf16, unpack_bf16
from .integrity import (fsync_dir, verify_commit, verify_sidecar,
                        write_commit, write_sidecar)
from .checkpointing import snapshot_workspace, recover_workspace, resolve_checkpoint
from .paged_adamw import is_paged_expert_param, is_rowwise

CORE, ROUTERS, EXPERTS, OPTIM = "core", "routers", "experts", "optim"


def _json_digest(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _stamp_manifest(manifest, core, routers, opt=None):
    """Bind a checkpoint to its model shape, configuration, and optimiser ABI."""
    manifest["schema_version"] = 4
    manifest["optimizer_implementation"] = (
        "PagedAdamW/1" if opt is not None and opt.__class__.__name__ == "PagedAdamW"
        else (opt.__class__.__name__ if opt is not None else None))
    manifest["config_sha256"] = _json_digest(manifest.get("cfg") or {})
    shape_doc = {
        "d_model": manifest.get("d_model"),
        "d_ff": manifest.get("d_ff"),
        "n_experts": manifest.get("n_experts"),
        "paged": bool(manifest.get("paged", False)),
        "paging_runtime": manifest.get("paging_runtime"),
        "core": {k: list(v.shape) for k, v in sorted(core.items())},
        "routers": {k: list(v.shape) for k, v in sorted(routers.items())},
    }
    manifest["architecture_sha256"] = _json_digest(shape_doc)
    return manifest


def _is_expert(key):
    return ".experts." in key and key.startswith("pool.")


def _is_router(key):
    return key.endswith("router.weight") or key.endswith("depth_emb")


def expert_index(key):
    """pool.experts.12.w1.weight -> 12"""
    parts = key.split(".")
    return int(parts[parts.index("experts") + 1])


def expert_leaf(key):
    """
    What to call this tensor inside the expert's own file.

        pool.experts.12.w1.weight           -> w1
        pool.experts.12.blocks.0.w1.weight  -> b0_w1

    Taking the second-to-last name instead would call both of those `w1`, and
    a deeper expert would quietly save only its first block.
    """
    parts = key.split(".")
    i = parts.index("experts") + 2          # past the index
    tail = parts[i:-1]                      # drop the trailing "weight"
    if tail[0] == "blocks":
        return f"b{tail[1]}_{tail[2]}"
    return tail[0]


def save(model, path, step=None, val=None, opt=None, cfg=None, verbose=False,
         extra=None):
    """Write the whole model out: bundles for the resident parts, one file per expert."""
    os.makedirs(os.path.join(path, EXPERTS), exist_ok=True)

    # A paged pool keeps its experts as files already - it has no
    # pool.experts.N tensors to gather, only the VRAM slots. Grouping its
    # state_dict by the usual pattern finds nothing, writes an empty expert
    # list, and then deletes every file the manifest does not mention, which
    # is all of them. So a paged pool is asked to flush itself instead.
    pool = getattr(model, "pool", None)
    if pool is not None and hasattr(pool, "tiers"):
        return _save_paged(model, pool, path, step, val, opt, cfg, verbose,
                           extra)

    sd = model.state_dict()

    by_expert, core, routers = {}, {}, {}
    for k, v in sd.items():
        if _is_expert(k):
            by_expert.setdefault(expert_index(k), {})[expert_leaf(k)] = v
        elif _is_router(k):
            routers[k] = v.detach().cpu().numpy()
        else:
            core[k] = v.detach().cpu().numpy()

    total = _savez(os.path.join(path, "core.npz"), core)
    total += _savez(os.path.join(path, "routers.npz"), routers)

    entries = []
    gate = sd.get("pool.gate")
    moments = _expert_moments(opt, model) if opt is not None else {}
    for i in sorted(by_expert):
        parts = by_expert[i]
        arrays = {k: v.detach().cpu().numpy() for k, v in parts.items()}
        # an expert's Adam moments belong to the expert, so they live in the
        # expert's own file. A file that holds only weights cannot be paged
        # back in and resumed - it comes back with no history, or worse, with
        # whatever history the slot it lands in happened to have.
        arrays.update(moments.get(i, {}))
        n = _savez(os.path.join(path, EXPERTS, "e%05d.npz" % i), arrays)
        total += n
        params = sum(a.size for k, a in arrays.items()
                     if not k.endswith(("_m", "_v", "_t")))
        entries.append({"id": i, "file": "e%05d.npz" % i,
                        "params": int(params), "bytes": int(n),
                        "moments": any(k.endswith("_m") for k in arrays),
                        "gate": float(gate[i]) if gate is not None
                        and i < gate.numel() else 1.0})
    live = {e["file"] for e in entries}
    ed = os.path.join(path, EXPERTS)
    removed = 0
    for f in os.listdir(ed):
        if f.endswith((".npy", ".npz")) and f not in live:
            os.remove(os.path.join(ed, f))
            removed += 1

    if opt is not None:
        total += _save_optim(opt, model, path)

    d_ff = d_model = None
    for i in sorted(by_expert)[:1]:
        d_ff, d_model = by_expert[i]["w1"].shape
    manifest = {
        "step": step, "val": val, "cfg": cfg,
        "n_experts": len(entries), "d_model": d_model, "d_ff": d_ff,
        "core_tensors": sorted(core), "router_tensors": sorted(routers),
        "experts": entries, "total_bytes": int(total),
        "removed_expert_files": removed,
    }
    _stamp_manifest(manifest, core, routers, opt=opt)
    tmp = os.path.join(path, "manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    manifest_path = os.path.join(path, "manifest.json")
    os.replace(tmp, manifest_path)
    fsync_dir(path)
    write_sidecar(manifest_path)
    commit_files = ["manifest.json", "core.npz", "routers.npz", "optim.npz"]
    if os.path.exists(os.path.join(path, "tokenizer.json")):
        commit_files.append("tokenizer.json")
    commit_files += [f"{EXPERTS}/{e['file']}" for e in entries]
    write_commit(path, commit_files, meta={"step": step, "val": val,
                                           "paged": False, "workspace": True,
                                           "architecture_sha256": manifest["architecture_sha256"],
                                           "config_sha256": manifest["config_sha256"],
                                           "optimizer": manifest["optimizer_implementation"]})
    snapshot_workspace(path, step=step, val=val)
    if verbose:
        n_files = len(entries) + 3
        print(f"  weights/: {len(entries)} experts + 3 bundles = {n_files} files, "
              f"{total/1e6:.1f} MB"
              + (f", {removed} expert files removed" if removed else ""))
    return len(entries), total


def _expert_moments(opt, model):
    """Adam's two moments per expert, keyed by expert id."""
    name_of = {id(p): n for n, p in model.named_parameters()}
    out = {}
    for group in opt.param_groups:
        for p in group["params"]:
            n = name_of.get(id(p))
            if n is None or not _is_expert(n):
                continue
            st = opt.state.get(p)
            if not st or "exp_avg" not in st:
                continue
            i, leaf = expert_index(n), expert_leaf(n)
            out.setdefault(i, {})[leaf + "_m"] = st["exp_avg"].cpu().numpy()
            out[i][leaf + "_v"] = st["exp_avg_sq"].cpu().numpy()
            t = st.get("step")
            if t is not None:
                out[i][leaf + "_t"] = np.asarray(
                    float(t.item() if torch.is_tensor(t) else t), dtype=np.float32)
    return out


def _save_paged(model, pool, path, step, val, opt, cfg, verbose, extra=None):
    """
    Save a model whose experts are already on disk.

    The expert files are the pool, so saving means parking whatever is
    resident - with its optimiser moments - and writing the manifest around
    what is there. Nothing is gathered and nothing is deleted.
    """
    pool.flush()
    core, routers = {}, {}
    for k, v in model.state_dict().items():
        if k.startswith("pool."):
            if k.endswith("gate"):
                routers[k] = v.detach().cpu().numpy()
            continue                      # w1/w3/w2 are slots, not experts
        (routers if _is_router(k) else core)[k] = v.detach().cpu().numpy()
    total = _savez(os.path.join(path, "core.npz"), core)
    total += _savez(os.path.join(path, "routers.npz"), routers)
    if opt is not None:
        total += _save_optim(opt, model, path)
    ed = os.path.join(path, EXPERTS)
    entries = []
    gate = pool.gate.detach().cpu()
    # An expert's FILE is named by its uid, not by its position. Pruning does
    # not renumber the directory, so the two diverge as soon as anything is
    # deleted, and building the name from the position would skip every expert
    # whose file no longer matches its index.
    uid = getattr(pool, "uid", None)
    missing = []
    for i in range(pool.n_experts()):
        u = int(uid[i]) if uid is not None and i < uid.numel() else i
        f = "e%05d.npz" % u
        p = os.path.join(ed, f)
        if not os.path.exists(p):
            missing.append(u)
            continue
        n = os.path.getsize(p)
        total += n
        entries.append({"id": u, "file": f, "bytes": int(n),
                        "params": pool.d_ff * pool.d_model * 3,
                        "moments": True,
                        "gate": float(gate[i]) if i < gate.numel() else 1.0})
    if missing:
        raise RuntimeError(
            f"refusing incoherent checkpoint: {len(missing)} active experts "
            f"have no file on disk ({missing[:6]}...) while the router/gate "
            f"still names {pool.n_experts()} experts")
    d = dict(cfg or {})
    d["pool_resident"] = pool.resident
    # Record the ceiling as at least what the pool actually holds. Growth is
    # not capped in the read path, so a stale `pool_max` from an older config
    # leaves the routers a row short of the experts on disk and the directory
    # cannot be loaded back without widening it by hand.
    d["pool_max"] = max(int(d.get("pool_max") or 0), len(entries))
    # Which experts have ever been resident, and when each was admitted.
    # Whether each expert has ever been on the card: a record, carried so a
    # resumed run still knows it.
    if hasattr(pool, "telemetry"):
        # Everything the pool knows about its own experts. Rebuilt from
        # nothing every run otherwise, so a long run would end able to say
        # what the model is and nothing about how it got there.
        tel = pool.telemetry()
        manifest_tel = tel
        d["pool_ever"] = tel["ever"]
    else:
        manifest_tel = None
    manifest = {"step": step, "val": val, "cfg": d,
                "telemetry": manifest_tel,
                # The pool's own count, NOT len(entries). The router
                # tensors were just written with one row per expert, and
                # loading sizes them from this number - so a file count that
                # disagrees produces a shape mismatch rather than a warning.
                "n_experts": int(pool.n_experts()), "d_model": pool.d_model,
                "d_ff": pool.d_ff, "paged": True,
                "paging_runtime": ("virtual_autograd_v1"
                                   if getattr(pool, "external_autograd", False)
                                   else "resident_slots_v4"),
                "core_tensors": sorted(core), "router_tensors": sorted(routers),
                "experts": entries, "total_bytes": int(total)}
    manifest.update(extra or {})       # where the reader had got to
    _stamp_manifest(manifest, core, routers, opt=opt)
    tmp = os.path.join(path, "manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    manifest_path = os.path.join(path, "manifest.json")
    os.replace(tmp, manifest_path)
    fsync_dir(path)
    write_sidecar(manifest_path)
    commit_files = ["manifest.json", "core.npz", "routers.npz", "optim.npz"]
    if os.path.exists(os.path.join(path, "tokenizer.json")):
        commit_files.append("tokenizer.json")
    commit_files += [f"{EXPERTS}/{e['file']}" for e in entries]
    write_commit(path, commit_files, meta={"step": step, "val": val,
                                           "paged": True, "experts": len(entries),
                                           "workspace": True,
                                           "architecture_sha256": manifest["architecture_sha256"],
                                           "config_sha256": manifest["config_sha256"],
                                           "optimizer": manifest["optimizer_implementation"]})
    snapshot_workspace(path, step=step, val=val)
    if verbose:
        print(f"  weights/: {len(entries)} experts + 3 bundles, "
              f"{total/1e6:.1f} MB (paged - experts were already on disk)")
    return len(entries), total


def _savez(path, arrays):
    """Atomic bundle write."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    fsync_dir(os.path.dirname(path) or ".")
    write_sidecar(path)
    return sum(a.nbytes for a in arrays.values())


def _save_optim(opt, model, path):
    """Persist optimiser state not owned by paged expert files.

    Dense parameters store a scalar ``|t``. Axis-0 rowwise parameters (router
    rows and gates) store ``|rt`` so growth/prune/reactivation preserves each
    expert-selection row's own Adam clock. Resident paged expert tensors are
    omitted entirely: their m/v/t live in each expert file.
    """
    name_of = {id(p): n for n, p in model.named_parameters()}
    out = {}
    for group in opt.param_groups:
        for p in group["params"]:
            if is_paged_expert_param(p):
                continue
            st = opt.state.get(p)
            if not st or "exp_avg" not in st:
                continue
            n = name_of.get(id(p))
            if n is None or _is_expert(n):
                continue
            out[n + "|m"] = pack_bf16(st["exp_avg"].detach().cpu())
            out[n + "|v"] = pack_bf16(st["exp_avg_sq"].detach().cpu())
            if st.get("row_step") is not None:
                out[n + "|rt"] = st["row_step"].detach().cpu().to(torch.float32).numpy()
            elif st.get("step") is not None:
                v = st["step"]
                out[n + "|t"] = np.asarray(float(v.item() if torch.is_tensor(v) else v))
    return _savez(os.path.join(path, "optim.npz"), out) if out else 0


def load(model, path, opt=None, device=None, strict=False, verbose=False):
    """Rebuild tensors from the latest committed generation when available."""
    device = device or next(model.parameters()).device
    # A crash can leave the mutable paging workspace newer than its formal
    # checkpoint. Restore it to LAST before loading instead of accepting a
    # self-consistent subset of mixed expert/core generations.
    # ``recover_workspace`` is already a no-op when there is no LAST pointer
    # (including immutable generation directories).  If LAST exists but is
    # corrupt, recovery must fail closed; silently loading a mixed mutable
    # workspace defeats the generation protocol.
    recover_workspace(path)
    verify_commit(path, required=False)
    verify_sidecar(os.path.join(path, "manifest.json"), required=False)
    with open(os.path.join(path, "manifest.json")) as f:
        man = json.load(f)
    sd = {}
    core = np.load(os.path.join(path, "core.npz"))
    routers = np.load(os.path.join(path, "routers.npz"))
    for k in core.files:
        sd[k] = torch.from_numpy(core[k])
    for k in routers.files:
        sd[k] = torch.from_numpy(routers[k])
    d_model, d_ff = man["d_model"], man["d_ff"]
    n1 = d_ff * d_model
    for e in man["experts"]:
        f = os.path.join(path, EXPERTS, e["file"])
        i = e["id"]
        if f.endswith(".npz"):
            z = np.load(f)
            for leaf in z.files:
                if leaf.endswith(("_m", "_v", "_t")):
                    continue                       # optimiser state, not weights
                if leaf.startswith("b") and "_" in leaf:
                    b, nm = leaf[1:].split("_", 1)
                    key = f"pool.experts.{i}.blocks.{b}.{nm}.weight"
                else:
                    key = f"pool.experts.{i}.{leaf}.weight"
                sd[key] = torch.from_numpy(z[leaf])
        else:
            # the older flat layout, before moments moved into the file
            flat = torch.from_numpy(np.load(f))
            sd[f"pool.experts.{i}.w1.weight"] = flat[:n1].view(d_ff, d_model)
            sd[f"pool.experts.{i}.w3.weight"] = flat[n1:2 * n1].view(d_ff, d_model)
            sd[f"pool.experts.{i}.w2.weight"] = flat[2 * n1:].view(d_model, d_ff)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if opt is not None:
        _load_optim(opt, model, path)
    if verbose:
        print(f"  loaded weights/ at step {man.get('step')}: "
              f"{man['n_experts']} experts, {man['total_bytes']/1e6:.1f} MB"
              + (f", {len(missing)} tensors not in the directory" if missing else ""))
    return man, missing, unexpected


def _load_optim(opt, model, path):
    """Restore optimiser state, including independent clocks for rowwise state."""
    _load_expert_moments(opt, model, path)
    f = os.path.join(path, "optim.npz")
    if not os.path.exists(f):
        return
    z = np.load(f)
    name_of = {id(p): n for n, p in model.named_parameters()}
    ensure = getattr(opt, "ensure_state", None)
    for group in opt.param_groups:
        for p in group["params"]:
            if is_paged_expert_param(p):
                continue
            n = name_of.get(id(p))
            if n is None or (n + "|m") not in z.files:
                continue
            if (n + "|v") not in z.files:
                raise RuntimeError(f"optimizer state is incomplete for {n}: missing second moment")
            m = unpack_bf16(z[n + "|m"])
            v = unpack_bf16(z[n + "|v"])
            if m.shape != p.shape or v.shape != p.shape:
                raise RuntimeError(
                    f"optimizer state shape mismatch for {n}: "
                    f"m={tuple(m.shape)} v={tuple(v.shape)} param={tuple(p.shape)}")
            st = ensure(p) if ensure is not None else opt.state[p]
            st["exp_avg"] = m.to(device=p.device, dtype=p.dtype)
            st["exp_avg_sq"] = v.to(device=p.device, dtype=p.dtype)
            if is_rowwise(p):
                if (n + "|rt") in z.files:
                    rt = torch.from_numpy(np.asarray(z[n + "|rt"], dtype=np.float32))
                    if tuple(rt.shape) == (p.shape[0],):
                        st["row_step"] = rt.to(p.device)
                elif (n + "|t") in z.files:
                    # v3 migration: old AdamW had one clock for the whole
                    # parameter. It is the best available estimate for every row.
                    st["row_step"] = torch.full((p.shape[0],), float(z[n + "|t"]),
                                                dtype=torch.float32, device=p.device)
            elif (n + "|t") in z.files:
                st["step"] = torch.tensor(float(z[n + "|t"]), device=p.device)


def _load_expert_moments(opt, model, path):
    name_of = {id(p): n for n, p in model.named_parameters()}
    cache = {}
    for group in opt.param_groups:
        for p in group["params"]:
            n = name_of.get(id(p))
            if n is None or not _is_expert(n):
                continue
            i, leaf = expert_index(n), expert_leaf(n)
            if i not in cache:
                f = os.path.join(path, EXPERTS, "e%05d.npz" % i)
                cache[i] = np.load(f) if os.path.exists(f) else None
            z = cache[i]
            if z is None or (leaf + "_m") not in z.files:
                continue
            m = unpack_bf16(z[leaf + "_m"])
            v = unpack_bf16(z[leaf + "_v"])
            if m.shape != p.shape:
                continue
            ensure = getattr(opt, "ensure_state", None)
            st = ensure(p) if ensure is not None else opt.state[p]
            st["exp_avg"] = m.to(device=p.device, dtype=p.dtype)
            st["exp_avg_sq"] = v.to(device=p.device, dtype=p.dtype)
            tk = leaf + "_t"
            if tk in z.files:
                st["step"] = torch.tensor(float(z[tk]), dtype=torch.float32,
                                          device=p.device)


def summarise(path):
    with open(os.path.join(path, "manifest.json")) as f:
        man = json.load(f)
    ne = len([f for f in os.listdir(os.path.join(path, EXPERTS))
              if f.endswith((".npy", ".npz"))])
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            tot += os.path.getsize(os.path.join(root, f))
    n_files = sum(len(fs) for _, _, fs in os.walk(path))
    print(f"{path}/  —  step {man.get('step')}, val {man.get('val')}")
    print(f"  experts/   {ne:>5} files   one per expert - the unit that is "
          f"paged, grown and pruned")
    print(f"  core.npz         1 file    {len(man['core_tensors'])} tensors: "
          f"embeddings, attention, norms")
    print(f"  routers.npz      1 file    {len(man['router_tensors'])} tensors: "
          f"one per call site")
    print(f"  optim.npz        1 file    Adam moments")
    print(f"  manifest.json    1 file    the index")
    print(f"  {n_files} files total, {tot/1e6:.1f} MB on disk")
    gates = [e["gate"] for e in man["experts"]]
    live = sum(1 for g in gates if abs(g) > 0.01)
    print(f"  {live} experts carrying weight, {len(gates)-live} gated to rest")
    print()
    for line in param_table(count_params(path)):
        print("  " + line)
    return 0


def count_params(path):
    """
    How many parameters the model on disk has, counted from its files.

    The README's table is generated from this and the scaling chart reads its
    sizes from it, so the three cannot disagree, and none of them can go stale
    while the pool grows and prunes:

        core         the trunk - embeddings, attention, norms, adapter,
                     halting head - from core.npz
        routers      one row per expert at each call site, its depth
                     embedding, and the experts' gates, from routers.npz
        experts      every expert in the manifest, at the size it was written
        resident     the part that is ever on the card at once: the core, the
                     routers and `pool_resident` experts
        dense_equiv  what one byte costs to compute, in the 6ND sense: two
                     prelude blocks, then attention, top_k of the resident
                     experts and a ranking of the pool on each recurrent step,
                     counted at train_steps_mean steps - the dense model that
                     costs the same per byte, which is what the scaling chart
                     is drawn against
    """
    with open(os.path.join(path, "manifest.json")) as f:
        man = json.load(f)

    def size(name):
        with np.load(os.path.join(path, name)) as z:
            return sum(int(np.prod(z[k].shape)) for k in z.files)
    c = man.get("cfg", {})
    d = int(c.get("d_model", man.get("d_model", 512)))
    dff = int(c.get("d_ff", man.get("d_ff", 1408)))
    pff = int(c.get("pool_d_ff", 2048))
    depth = int(c.get("pool_depth", 1) or 1)
    k = int(c.get("pool_top_k", 8))
    steps = float(c.get("train_steps_mean", 12.8))
    n = len(man["experts"])
    experts = sum(int(e["params"]) for e in man["experts"])
    per = experts // n if n else depth * 3 * d * pff
    core, routers = size("core.npz"), size("routers.npz")
    resident = int(c.get("pool_resident", 32))
    with np.load(os.path.join(path, "core.npz")) as z:
        vocab = int(z["tok_emb.weight"].shape[0]) if "tok_emb.weight" in z.files else 265
    prelude = 4 * d * d + 3 * d * dff             # attention + feed-forward
    dense = (2 * prelude + steps * (4 * d * d + k * per + n * d)
             + d * 2 * d + vocab * d)
    return {"step": man.get("step"), "n_experts": n, "per_expert": per,
            "core": core, "routers": routers, "experts": experts,
            "total": core + routers + experts, "resident": resident,
            "resident_params": core + routers + min(resident, n) * per,
            "dense_equiv": dense, "top_k": k, "steps": steps,
            "max_steps": int(c.get("max_steps", 24)), "d_model": d,
            "pool_d_ff": pff, "pool_depth": depth}


def param_table(p):
    """count_params as the table `python3 -m minagi.store` prints and the
    README carries, decimal points aligned."""
    def m(v):
        s = f"{v / 1e6:.2f}" if v < 10e6 else f"{v / 1e6:.1f}"
        i, f_ = s.split(".")
        return f"{i:>4}.{f_}M".ljust(10)
    shape = (f"{p['pool_depth']} x " if p["pool_depth"] > 1 else "") + \
        f"3 x {p['d_model']} x {p['pool_d_ff']}"
    return [line.rstrip() for line in (
            f"core     {m(p['core'])}embeddings, attention, norms, adapter, "
            f"halting head",
            f"routers  {m(p['routers'])}one row per expert at each call site, "
            f"depth embedding, gates",
            f"experts  {m(p['experts'])}{p['n_experts']} x "
            f"{p['per_expert'] / 1e6:.2f}M each  ({shape})",
            "-" * 22,
            f"total    {m(p['total'])}")]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="weights")
    sys.exit(summarise(ap.parse_args().path))


# ----------------------------------------------------------------------------
# keeping the live model, and being able to get back
# ----------------------------------------------------------------------------

def best_val(path):
    """
    Validation loss of the state on disk, or inf if there is none.

    The directory holds the best state the model has reached, not the most
    recent one: `save` is called only when a run improves on it. A second copy
    kept aside as a snapshot would double a figure that is already 236 MB and
    grows with the pool, and it would be a copy of exactly what is already
    here.

    That makes the directory the thing to fall back to. A run that diverges
    reloads it and carries on, which costs the steps since the last
    improvement and nothing more.
    """
    # resolve_checkpoint already falls back to the workspace when BEST does
    # not exist. A malformed BEST pointer is corruption and must not be hidden.
    best = resolve_checkpoint(path, "BEST")
    f = os.path.join(best, "manifest.json")
    if not os.path.exists(f):
        return float("inf")
    with open(f) as fh:
        v = json.load(fh).get("val")
    return float(v) if v is not None else float("inf")
