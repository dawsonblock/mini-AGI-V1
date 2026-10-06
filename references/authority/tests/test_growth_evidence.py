import torch

from minagi.pool import AutoGrow


class FakePool:
    def __init__(self):
        self._n = 8
        self.capacity_pressure = 0.2
        self.capacity_want_k = 3.0
        self.route_top_k = 2
        self.use = torch.ones(self._n)
        self.born = torch.zeros(self._n)
        self.trial = 10
        self.now = 100
        self.dying_at = 0.75
        self.grow_events = []
        self.uid = torch.arange(self._n, dtype=torch.long)
        self.next_uid = self._n

    def n_experts(self):
        return self._n

    def saturation(self):
        return {"experts": self._n, "idle": 0}

    def dying(self):
        return torch.zeros(self._n)

    def disk_bytes(self, extra=0):
        return (self._n + extra) * 1024

    def add_experts(self, k, **kwargs):
        fresh = torch.arange(self.next_uid, self.next_uid + k, dtype=torch.long)
        self.next_uid += k
        self._n += k
        self.uid = torch.cat([self.uid, fresh])
        self.use = torch.ones(self._n)
        self.born = torch.cat([self.born, torch.full((k,), float(kwargs.get("step", 0)))])

    def retire_uids(self, uids, **kwargs):
        wanted = set(map(int, uids))
        keep = [i for i, u in enumerate(self.uid.tolist()) if int(u) not in wanted]
        removed = self._n - len(keep)
        self.uid = self.uid[keep]
        self.born = self.born[keep]
        self.use = torch.ones(len(keep))
        self._n = len(keep)
        return removed


def test_growth_waits_for_plateau_and_persistent_demand():
    pool = FakePool()
    g = AutoGrow(grow_k=1, max_experts=16, pressure_min=0.1,
                 want_k_ratio_min=1.1, plateau_checks=3,
                 plateau_delta=0.01, demand_hits=2)
    # clear improvement: do not grow despite routing pressure
    assert g.step(1.00, pool, 10)["grew"] == 0
    assert g.step(0.90, pool, 20)["grew"] == 0
    assert g.step(0.80, pool, 30)["grew"] == 0
    # stable held-out values establish the plateau
    assert g.step(0.805, pool, 40)["grew"] == 0
    rec = g.step(0.804, pool, 50)
    assert rec["grew"] == 1
    assert rec["plateau"] is True


def test_growth_refuses_without_full_pool_demand():
    pool = FakePool()
    pool.capacity_pressure = 0.01
    g = AutoGrow(grow_k=1, max_experts=16, pressure_min=0.1,
                 plateau_checks=2, demand_hits=1)
    g.step(1.0, pool, 10)
    rec = g.step(1.0, pool, 20)
    assert rec["grew"] == 0
    assert "no persistent capacity demand" in rec["reason"]


def test_new_growth_cohort_must_improve_heldout_before_more_growth():
    pool = FakePool()
    g = AutoGrow(grow_k=1, max_experts=20, pressure_min=0.1,
                 want_k_ratio_min=1.1, plateau_checks=2,
                 plateau_delta=0.01, demand_hits=1,
                 qualification_checks=2, qualification_gain_min=0.01)
    g.step(1.0, pool, 10)
    first = g.step(1.0, pool, 20)
    assert first["grew"] == 1
    blocked = g.step(0.999, pool, 30)
    assert blocked["grew"] == 0
    assert blocked["qualification"]["state"] in {"trial", "rejected"}
    # Once the cohort causes a measurable held-out improvement, the promotion
    # gate clears. Other growth brakes may still refuse, which is correct.
    cleared = g.step(0.98, pool, 40)
    assert cleared["qualification"]["state"] == "promoted"


def test_failed_growth_cohort_rolls_back_and_clears_pending():
    pool = FakePool()
    g = AutoGrow(grow_k=1, max_experts=20, pressure_min=0.1,
                 want_k_ratio_min=1.1, plateau_checks=2,
                 plateau_delta=0.01, demand_hits=1,
                 qualification_checks=2, qualification_gain_min=0.05)
    g.step(1.0, pool, 10)
    born = g.step(1.0, pool, 20)
    assert born["grew"] == 1 and pool.n_experts() == 9
    g.step(0.999, pool, 30)
    rejected = g.step(0.998, pool, 40)
    assert rejected["qualification"]["state"] == "rejected"
    assert rejected["retired"] == 1
    assert pool.n_experts() == 8
    assert g._pending_growth is None
