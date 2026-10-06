"""v5 virtual-memory expert pool.

Unlike PagedPool, this module has no trainable expert slot Parameters.  The
router selects logical UIDs directly.  ExpertStore is durable parameter memory,
ExpertCache is merely a performance cache, and PagedExpertFunction bridges the
external weights into one untruncated autograd graph.
"""
from __future__ import annotations
import math
import os
import numpy as np
import torch
import torch.nn as nn

from .expert_store import ExpertStore
from .expert_cache import ExpertCache
from .expert_grad_store import ExpertGradStore
from .external_adamw import ExternalExpertAdamW
from .graph_epoch import GraphEpoch
from .paged_autograd import paged_expert
from .paged_adamw import mark_rowwise
from .integrity import fsync_dir, write_sidecar, verify_sidecar


class VirtualPagedPool(nn.Module):
    external_autograd = True
    is_external_paged = True
    admission_mode = "virtual_full"

    def __init__(self, path, d_model, d_ff, n_experts, resident=16,
                 ram_capacity=256, device="cpu", max_experts=1_000_000,
                 read_only=False, prefetch_workers=4, pin_host=True):
        super().__init__()
        self.path = str(path)
        self.d_model, self.d_ff = int(d_model), int(d_ff)
        self.resident = max(1, int(resident))  # cache capacity, not semantic cap
        self.max_experts = int(max_experts) if max_experts is not None else None
        self.read_only = bool(read_only)
        self.store = ExpertStore(path, d_model, d_ff, read_only=read_only)
        # `tiers` keeps store.save's paged checkpoint path backward compatible.
        self.tiers = self.store
        self.cache = ExpertCache(
            self.store, capacity=self.resident, device=device,
            host_capacity=max(self.resident, int(ram_capacity or self.resident)),
            prefetch_workers=prefetch_workers, pin_host=pin_host)
        self.grad_store = ExpertGradStore(device="cpu")
        self.expert_optim = ExternalExpertAdamW(self.store, self.grad_store, self.cache)
        self._n = int(n_experts)
        self.gate = mark_rowwise(nn.Parameter(torch.ones(self._n, device=device)))
        for name in ("use", "age", "born", "gate_seen", "last_seen", "recent"):
            self.register_buffer(name, torch.zeros(self._n, device=device), persistent=False)
        self.register_buffer("ever", torch.zeros(self._n, dtype=torch.bool, device=device), persistent=False)
        self.register_buffer("admits", torch.zeros(self._n, device=device), persistent=False)
        uids = self.store.uids()
        if len(uids) < self._n:
            # Fresh stores are allowed only when their files already exist; create.py
            # writes them before this runtime is selected.
            uids = list(range(self._n))
        self.register_buffer("uid", torch.tensor(uids[:self._n], dtype=torch.long, device=device), persistent=False)
        self.next_uid = (max(uids) + 1) if uids else self._n
        self._sites = []
        self._opt = None
        self._epoch: GraphEpoch | None = None
        self.loads = self.swaps = 0
        self.segments = 0
        self.trial = 0
        self.now = 0
        self.dying_at = 0.75
        self.usage_steps = 1000.0
        self.balance = 0.0
        self.select_temperature = 0.0
        self.route_top_k = 1
        self.pressure = self.want_k = 0.0
        self.capacity_pressure = self.capacity_want_k = 0.0
        self.grow_events = []
        self._balance = None
        self._trains = False
        self._last_cache_loads = 0

    def n_experts(self): return self._n
    def n_routable(self): return self._n
    def router_rows(self): return self._n
    def n_resident(self): return self.resident
    def routable_gate(self): return self.gate[:self._n]
    def n_params(self): return self._n * 3 * self.d_model * self.d_ff + self.gate.numel()
    def vram_params(self): return self.resident * 3 * self.d_model * self.d_ff + self.gate.numel()
    def disk_bytes(self, extra=0): return (self._n + int(extra)) * 3 * self.d_model * self.d_ff * 8

    def attach_sites(self, model):
        from .pool import PooledMLP
        self._sites = [m for m in model.modules() if isinstance(m, PooledMLP)]
        return len(self._sites)

    def attach_optimiser(self, opt):
        self._opt = opt

    def _carry(self, old_p, new_p, idx=None, grow=0):
        """Carry rowwise optimiser state across gate/router shape changes."""
        opt = getattr(self, "_opt", None)
        if opt is None:
            return
        st = opt.state.pop(old_p, None)
        if not st:
            return
        out = {}
        for k, v in st.items():
            if torch.is_tensor(v) and v.dim() and v.shape[0] == old_p.shape[0]:
                if idx is not None:
                    v = v[idx.to(v.device)].clone()
                elif grow:
                    pad = torch.zeros((int(grow),) + tuple(v.shape[1:]),
                                      dtype=v.dtype, device=v.device)
                    v = torch.cat([v, pad])
            out[k] = v
        opt.state[new_p] = out

    def begin_text(self, explore=False):
        self.segments += 1
        self.begin_forward(explore)

    def begin_forward(self, explore=False):
        self._balance = None
        self._trains = bool(explore)
        if self._trains:
            self.recent.mul_(1.0 - 1.0 / self.usage_steps)

    def usage_share(self):
        r = self.recent[:self._n].float()
        s = r.sum()
        return r / s.clamp_min(1e-9) if float(s) > 0 else torch.full_like(r, 1.0 / max(1, self._n))

    def note_balance(self, term):
        self._balance = term if self._balance is None else self._balance + term

    def balance_term(self): return self._balance

    @torch.no_grad()
    def note_use(self, hit):
        h = hit.to(self.use.device, self.use.dtype)
        self.use[:h.numel()] += h
        self.age[:self._n] += 1
        active = torch.nonzero(h > 0, as_tuple=False).flatten()
        if active.numel():
            self.last_seen[active] = float(self.segments)
            self.ever[active] = True
            self.admits[active] += 1
            if self._trains:
                self.recent[active] += 1.0 / self.usage_steps

    def open_graph(self, training=True):
        if self._epoch is not None and self._epoch.state not in ("closed", "aborted"):
            raise RuntimeError("another graph epoch is already active")
        self.grad_store.clear()
        self._epoch = GraphEpoch(self.store, training=bool(training))
        return self._epoch

    @property
    def graph_epoch(self): return self._epoch

    def route_tape(self, site, idx):
        if self._epoch is None:
            return idx
        return self._epoch.routing.choose(site, idx)

    def candidate_tape(self, site, idx):
        if self._epoch is None:
            return idx
        return self._epoch.routing.choose_candidates(site, idx)

    def routing_replay(self):
        return bool(self._epoch is not None and self._epoch.routing.replay)

    def routing_next_kind(self):
        return None if self._epoch is None else self._epoch.routing.next_kind()

    def replay_candidates(self, site, device):
        if self._epoch is None:
            raise RuntimeError("no graph epoch for candidate replay")
        return self._epoch.routing.replay_next("candidates", site, device=device)

    def replay_topk(self, site, device):
        if self._epoch is None:
            raise RuntimeError("no graph epoch for top-k replay")
        return self._epoch.routing.replay_next("topk", site, device=device)

    def dispatch_external(self, flat, expert_ids, token_ids, weights):
        training_graph = self.training and torch.is_grad_enabled()
        implicit = False
        if self._epoch is None or self._epoch.state in ("closed", "aborted"):
            if training_graph:
                raise RuntimeError(
                    "VirtualPagedPool training requires an open GraphEpoch; use FullBPTTStepper or pool.open_graph()")
            self.open_graph(training=False)
            implicit = True
        out = torch.zeros_like(flat)
        # Expert IDs are logical pool positions; files are stable UIDs.  Pin
        # every expert version before starting asynchronous host prefetch so a
        # graph cannot observe a mixture of versions if storage changes while
        # I/O is in flight.
        positions = torch.unique(expert_ids, sorted=True).tolist()
        prefetch = []
        for pos in positions:
            uid = int(self.uid[int(pos)])
            prefetch.append((uid, self._epoch.pin(uid)))
        self.cache.prefetch(prefetch)
        for pos in positions:
            mask = expert_ids == int(pos)
            if not bool(mask.any()):
                continue
            tid = token_ids[mask]
            uid = int(self.uid[int(pos)])
            y = paged_expert(flat[tid], uid, self._epoch, self.cache, self.grad_store)
            out = out.index_add(0, tid, y * weights[mask].unsqueeze(-1).to(y.dtype))
        now = self.cache.loads
        self.loads += max(0, now - self._last_cache_loads)
        self._last_cache_loads = now
        if implicit:
            self._epoch.state = "backward_done"  # read-only graph has no backward barrier
            self._epoch.close()
        return out

    def finish_backward(self):
        if self._epoch is None:
            raise RuntimeError("no active graph epoch")
        self._epoch.finish_backward()

    def external_step(self, *, lr, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8):
        if self.read_only:
            raise RuntimeError("read-only virtual pool cannot step")
        if self._epoch is None:
            raise RuntimeError("no active graph epoch")
        self.expert_optim.betas = tuple(betas)
        self.expert_optim.eps = float(eps)
        changed = self.expert_optim.step(self._epoch, lr=lr, weight_decay=weight_decay)
        self._epoch.close()
        return changed

    def abort_graph(self):
        if self._epoch is not None:
            self._epoch.abort()
        self.grad_store.clear()

    def flush(self):
        # External updates are committed atomically inside external_step.
        if self._epoch is not None and self._epoch.state not in ("closed", "aborted") and self._epoch.training:
            raise RuntimeError("cannot checkpoint while a training graph epoch is active")

    def dying(self):
        z = torch.zeros(self._n, device=self.gate.device)
        survival = float(self.trial or 0); step = float(self.now or 0)
        if self._n == 0 or survival <= 0 or step <= 0: return z
        per_step = self.segments / max(step, 1.0); window = survival * per_step
        if window <= 0: return z
        born_seg = self.born[:self._n] * per_step
        since = torch.minimum(torch.full_like(z, float(self.segments)) - self.last_seen[:self._n],
                              torch.full_like(z, float(self.segments)) - born_seg)
        young = (step - self.born[:self._n]) < survival
        out = (since / window).clamp_min(0)
        out[young] = 0
        return out

    def saturation(self):
        u = self.use[:self._n] / self.use[:self._n].sum().clamp_min(1)
        n = max(1, self._n); ideal = 1.0 / n
        ent = float(-(u.clamp_min(1e-9) * u.clamp_min(1e-9).log()).sum()) if self._n else 0.0
        return {"experts": self._n, "idle": int((self.dying() >= self.dying_at).sum()),
                "idle_by_routing": int((u < 0.1 * ideal).sum()) if self._n else 0,
                "entropy_frac": ent / max(math.log(n), 1e-9),
                "peak_over_ideal": float(u.max()) / ideal if self._n else 0.0,
                "pressure": float(self.pressure), "want_k": float(self.want_k)}

    @torch.no_grad()
    def add_experts(self, k, seed_from=None, device=None, step=0,
                    birth_gate=0.001, **_):
        if self._epoch is not None and self._epoch.state not in ("closed", "aborted"):
            raise RuntimeError("cannot grow pool during an active graph")
        k = max(0, int(k))
        if self.max_experts is not None:
            k = min(k, max(0, self.max_experts - self._n))
        if k == 0: return 0
        old_n = self._n
        parent_positions = list(range(old_n))
        for j in range(k):
            uid = self.next_uid; self.next_uid += 1
            if parent_positions:
                # Unit-level recombination from up to 16 trained experts.
                pick = parent_positions if len(parent_positions) <= 16 else torch.randperm(len(parent_positions))[:16].tolist()
                states = [self.store.load(int(self.uid[p])) for p in pick]
                src = torch.randint(0, len(states), (self.d_ff,))
                w1 = torch.empty(self.d_ff, self.d_model)
                w3 = torch.empty_like(w1); w2 = torch.empty(self.d_model, self.d_ff)
                for r, si in enumerate(src.tolist()):
                    w1[r] = states[si]["w1"][r]; w3[r] = states[si]["w3"][r]; w2[:, r] = states[si]["w2"][:, r]
                self.store.create(uid, from_state={"w1": w1, "w3": w3, "w2": w2})
            else:
                self.store.create(uid, seed=uid)
            self.uid = torch.cat([self.uid, torch.tensor([uid], device=self.uid.device)])
        self._n += k
        old_gate = self.gate
        self.gate = mark_rowwise(nn.Parameter(torch.cat([
            old_gate.data, torch.full((k,), float(birth_gate), device=old_gate.device)])))
        self.gate._minagi_state_parent = old_gate; self.gate._minagi_state_grow = k
        self._carry(old_gate, self.gate, grow=k)
        for nm in ("use", "age", "born", "gate_seen", "last_seen", "recent", "admits"):
            v = getattr(self, nm)
            fill = torch.full((k,), float(step) if nm == "born" else 0.0, dtype=v.dtype, device=v.device)
            setattr(self, nm, torch.cat([v, fill]))
        self.ever = torch.cat([self.ever, torch.zeros(k, dtype=torch.bool, device=self.ever.device)])
        for site in self._sites:
            w = site.router.weight
            fresh = nn.Linear(w.shape[1], self._n, bias=False).to(w.device)
            fresh.weight.data[:old_n].copy_(w.data[:old_n])
            if old_n:
                parent = int(seed_from) if seed_from is not None and 0 <= int(seed_from) < old_n else int(self.use[:old_n].argmax())
                fresh.weight.data[old_n:].copy_(w.data[parent].unsqueeze(0).expand(k, -1))
            else:
                nn.init.normal_(fresh.weight.data, 0.0, 0.01)
            mark_rowwise(fresh.weight)
            fresh.weight._minagi_state_parent = w; fresh.weight._minagi_state_grow = k
            self._carry(w, fresh.weight, grow=k)
            site.router = fresh
        return k

    def _archive_optimizer_row(self, meta, prefix, param, row):
        opt = getattr(self, "_opt", None)
        if opt is None:
            return
        st = opt.state.get(param)
        if not st or "exp_avg" not in st:
            return
        meta[prefix + "_m"] = st["exp_avg"][row].detach().cpu().numpy()
        meta[prefix + "_v"] = st["exp_avg_sq"][row].detach().cpu().numpy()
        if st.get("row_step") is not None:
            meta[prefix + "_t"] = np.asarray(float(st["row_step"][row]), dtype=np.float32)
        elif st.get("step") is not None:
            t = st["step"]
            meta[prefix + "_t"] = np.asarray(float(t.item() if torch.is_tensor(t) else t), dtype=np.float32)

    def _restore_optimizer_row(self, meta, prefix, param, row):
        opt = getattr(self, "_opt", None)
        if opt is None or meta is None:
            return
        ensure = getattr(opt, "ensure_state", None)
        st = ensure(param) if ensure is not None else opt.state[param]
        if prefix + "_m" in meta.files and "exp_avg" in st:
            st["exp_avg"][row].copy_(torch.from_numpy(meta[prefix + "_m"]).to(
                st["exp_avg"].device, st["exp_avg"].dtype))
            st["exp_avg_sq"][row].copy_(torch.from_numpy(meta[prefix + "_v"]).to(
                st["exp_avg_sq"].device, st["exp_avg_sq"].dtype))
        if prefix + "_t" in meta.files:
            t = float(meta[prefix + "_t"])
            if st.get("row_step") is not None:
                st["row_step"][row] = t
            elif st.get("step") is not None:
                st["step"] = torch.tensor(t, device=param.device)

    @torch.no_grad()
    def retire_uids(self, uids, step=0, reason="manual"):
        # v5 keeps logical files immutable by identity; retirement is router/gate
        # compaction.  Files are preserved under experts/archive for reversibility.
        wanted = {int(u) for u in uids}
        keep = [i for i in range(self._n) if int(self.uid[i]) not in wanted]
        retired_count = self._n - len(keep)
        if retired_count == 0: return 0
        if not keep: raise RuntimeError("refusing to retire every active expert")
        if self._epoch is not None and self._epoch.state not in ("closed", "aborted"):
            raise RuntimeError("cannot retire experts during an active graph")
        archive = os.path.join(self.path, "archive"); os.makedirs(archive, exist_ok=True)
        for i in range(self._n):
            uid = int(self.uid[i])
            if uid not in wanted:
                continue
            meta = {
                "uid": np.asarray(uid, dtype=np.int64),
                "gate": np.asarray(float(self.gate.data[i]), dtype=np.float32),
                "born": np.asarray(float(self.born[i]), dtype=np.float64),
                "last_seen": np.asarray(float(self.last_seen[i]), dtype=np.float64),
                "admits": np.asarray(float(self.admits[i]), dtype=np.float64),
                "retired_step": np.asarray(float(step), dtype=np.float64),
                "reason": np.asarray(str(reason)),
            }
            self._archive_optimizer_row(meta, "gate_opt", self.gate, i)
            for si, site in enumerate(self._sites):
                meta[f"router_{si}"] = site.router.weight.data[i].detach().cpu().numpy()
                self._archive_optimizer_row(meta, f"router_{si}_opt", site.router.weight, i)
            mt = os.path.join(archive, f"e{uid:05d}.meta.tmp.npz")
            mp = os.path.join(archive, f"e{uid:05d}.meta.npz")
            np.savez(mt, **meta); os.replace(mt, mp); fsync_dir(archive); write_sidecar(mp)
            src = self.store.file(uid)
            if os.path.exists(src):
                dst = os.path.join(archive, os.path.basename(src))
                os.replace(src, dst)
                if os.path.exists(src + ".sha256"):
                    os.replace(src + ".sha256", dst + ".sha256")
                fsync_dir(archive); fsync_dir(self.path)
        idx = torch.tensor(keep, dtype=torch.long, device=self.gate.device)
        old_gate = self.gate
        self.gate = mark_rowwise(nn.Parameter(old_gate.data[idx].clone()))
        self.gate._minagi_state_parent = old_gate; self.gate._minagi_state_index = idx
        self._carry(old_gate, self.gate, idx=idx)
        for nm in ("use", "age", "born", "gate_seen", "last_seen", "recent", "admits", "ever", "uid"):
            setattr(self, nm, getattr(self, nm)[idx].clone())
        for site in self._sites:
            w = site.router.weight
            fresh = nn.Linear(w.shape[1], len(keep), bias=False).to(w.device)
            fresh.weight.data.copy_(w.data[idx]); mark_rowwise(fresh.weight)
            fresh.weight._minagi_state_parent = w; fresh.weight._minagi_state_index = idx
            self._carry(w, fresh.weight, idx=idx)
            site.router = fresh
        self._n = len(keep); self.cache.clear()
        return retired_count

    def archived_uids(self):
        archive = os.path.join(self.path, "archive")
        if not os.path.isdir(archive):
            return []
        out = []
        for name in os.listdir(archive):
            if name.startswith("e") and name.endswith(".npz") and ".meta." not in name:
                try:
                    out.append(int(name[1:-4]))
                except ValueError:
                    pass
        return sorted(set(out))

    @torch.no_grad()
    def reactivate_archived(self, uid, step=0):
        uid = int(uid)
        if uid in {int(x) for x in self.uid.detach().cpu().tolist()}:
            return False
        if self._epoch is not None and self._epoch.state not in ("closed", "aborted"):
            raise RuntimeError("cannot reactivate expert during an active graph")
        if self.max_experts is not None and self._n >= self.max_experts:
            raise RuntimeError("active expert ceiling reached")
        archive = os.path.join(self.path, "archive")
        src = os.path.join(archive, f"e{uid:05d}.npz")
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        dst = self.store.file(uid)
        os.replace(src, dst)
        if os.path.exists(src + ".sha256"):
            os.replace(src + ".sha256", dst + ".sha256")
        fsync_dir(archive); fsync_dir(self.path)
        mp = os.path.join(archive, f"e{uid:05d}.meta.npz")
        meta = None
        if os.path.exists(mp):
            verify_sidecar(mp, required=False); meta = np.load(mp)
        old_n = self._n
        g = float(meta["gate"]) if meta is not None and "gate" in meta.files else 0.001
        old_gate = self.gate
        self.gate = mark_rowwise(nn.Parameter(torch.cat([old_gate.data, old_gate.data.new_tensor([g])])))
        self.gate._minagi_state_parent = old_gate; self.gate._minagi_state_grow = 1
        self._carry(old_gate, self.gate, grow=1)
        self._restore_optimizer_row(meta, "gate_opt", self.gate, old_n)
        def add(t, value=0.0):
            return torch.cat([t, t.new_tensor([value])])
        self.use = add(self.use); self.age = add(self.age)
        self.born = add(self.born, float(step)); self.gate_seen = add(self.gate_seen, abs(g))
        self.last_seen = add(self.last_seen, float(self.segments)); self.recent = add(self.recent)
        adm = float(meta["admits"]) if meta is not None and "admits" in meta.files else 0.0
        self.admits = add(self.admits, adm)
        self.ever = torch.cat([self.ever, self.ever.new_tensor([True])])
        self.uid = torch.cat([self.uid, self.uid.new_tensor([uid])])
        self._n += 1; self.next_uid = max(self.next_uid, uid + 1)
        for si, site in enumerate(self._sites):
            old_r = site.router.weight
            if meta is not None and f"router_{si}" in meta.files:
                row = torch.from_numpy(meta[f"router_{si}"]).to(old_r.device, old_r.dtype)
            else:
                row = torch.randn(self.d_model, device=old_r.device, dtype=old_r.dtype) * 0.01
            fresh = nn.Linear(self.d_model, self._n, bias=False).to(old_r.device)
            mark_rowwise(fresh.weight)
            fresh.weight.data.copy_(torch.cat([old_r.data, row.view(1, -1)], 0))
            fresh.weight._minagi_state_parent = old_r; fresh.weight._minagi_state_grow = 1
            self._carry(old_r, fresh.weight, grow=1)
            site.router = fresh
            self._restore_optimizer_row(meta, f"router_{si}_opt", fresh.weight, self._n - 1)
        self.cache.invalidate_uid(uid)
        return True

    def prune(self, step, survival=8600, protect=0):
        self.trial = int(survival); self.now = int(step)
        d = self.dying()
        gone = [int(self.uid[i]) for i in range(protect, self._n) if float(d[i]) >= 1.0]
        if len(gone) >= self._n: gone = gone[:-1]
        return self.retire_uids(gone, step=step, reason="stale") if gone else 0

    def telemetry(self):
        return {"use": self.use.detach().cpu().tolist(), "age": self.age.detach().cpu().tolist(),
                "born": self.born.detach().cpu().tolist(), "gate_seen": self.gate_seen.detach().cpu().tolist(),
                "last_seen": self.last_seen.detach().cpu().tolist(), "recent": self.recent.detach().cpu().tolist(),
                "admits": self.admits.detach().cpu().tolist(), "ever": self.ever.detach().cpu().tolist(),
                "uid": self.uid.detach().cpu().tolist(), "next_uid": self.next_uid,
                "segments": self.segments, "cache": self.cache.report(), "runtime": "virtual_autograd_v1"}

    def load_telemetry(self, t):
        if not t: return
        for nm in ("use", "age", "born", "gate_seen", "last_seen", "recent", "admits"):
            vals = t.get(nm)
            if vals:
                b = getattr(self, nm); k = min(len(vals), b.numel()); b[:k] = torch.tensor(vals[:k], dtype=b.dtype, device=b.device)
        vals = t.get("ever")
        if vals:
            k = min(len(vals), self.ever.numel()); self.ever[:k] = torch.tensor(vals[:k], dtype=torch.bool, device=self.ever.device)
        vals = t.get("uid")
        if vals:
            k = min(len(vals), self.uid.numel()); self.uid[:k] = torch.tensor(vals[:k], dtype=torch.long, device=self.uid.device)
        self.next_uid = int(t.get("next_uid") or (int(self.uid.max()) + 1 if self.uid.numel() else 0))
        self.segments = int(t.get("segments") or 0)
