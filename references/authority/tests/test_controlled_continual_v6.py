import copy
from pathlib import Path

import torch
import torch.nn as nn

from minagi.continual import (
    AdapterBank, AdapterMetadata, LowRankAdapter,
    FrozenBackboneAdapterLM, FastAdapterTrainer,
    save_adapter_artifact, load_adapter_artifact,
    ControlledContinualManager,
)
from minagi.candidate_pipeline import QualificationPolicy, Ed25519PromotionSigner


class _TinyFrozenLM(FrozenBackboneAdapterLM):
    def __init__(self, vocab=17, d=8):
        backbone = nn.Embedding(vocab, d)
        head = nn.Linear(d, vocab, bias=False)
        super().__init__(backbone, d, head)

    def frozen_forward(self, input_ids, **kwargs):
        h = self.backbone(input_ids)
        return self.output_projection(h), h


def test_adapter_training_cannot_mutate_frozen_base():
    torch.manual_seed(0)
    lm = _TinyFrozenLM()
    before_backbone = copy.deepcopy(lm.backbone.state_dict())
    before_head = copy.deepcopy(lm.output_projection.state_dict())
    ad = LowRankAdapter(8, rank=4, alpha=4)
    x = torch.tensor([[1,2,3,4,5]])
    result = FastAdapterTrainer(lm, lr=1e-2).train_batch(ad, x, x, steps=2)
    assert result.losses and result.update_norms[-1] > 0
    for k, v in before_backbone.items():
        assert torch.equal(v, lm.backbone.state_dict()[k])
    for k, v in before_head.items():
        assert torch.equal(v, lm.output_projection.state_dict()[k])
    assert all(not p.requires_grad for p in lm.backbone.parameters())
    assert all(not p.requires_grad for p in lm.output_projection.parameters())


def test_safe_npz_adapter_artifact_roundtrip(tmp_path, monkeypatch):
    torch.manual_seed(1)
    ad = LowRankAdapter(8, rank=3, alpha=6)
    with torch.no_grad():
        ad.up.weight.normal_()
    meta = AdapterMetadata.new("skill", 8, 3, 6)
    c = torch.arange(8).float()
    save_adapter_artifact(tmp_path / "a", ad, meta, centroid=c, extra={"source":"test"})
    # The loader must not rely on torch.load/pickle at all.
    monkeypatch.setattr(torch, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("torch.load called")))
    got, m2, c2, extra = load_adapter_artifact(tmp_path / "a")
    assert m2.id == meta.id and extra["source"] == "test"
    assert torch.equal(c2, c)
    for k, v in ad.state_dict().items():
        assert torch.equal(v, got.state_dict()[k])


def test_adapter_router_can_abstain_on_ambiguous_specialists():
    bank = AdapterBank(4, top_k=2, min_similarity=0.2, min_margin=0.05)
    a = bank.create("a", centroid=torch.tensor([1.,0,0,0]))
    b = bank.create("b", centroid=torch.tensor([0.99,0.01,0,0]))
    q = torch.tensor([[1.,0,0,0]])
    assert bank.route(q) == []  # almost-tied specialists => abstain
    routes = bank.route(q, allow_ambiguous=True)
    assert {r.adapter_id for r in routes} == {a,b}


def test_controlled_manager_qualifies_signs_and_verifies_adapter(tmp_path):
    mgr = ControlledContinualManager(tmp_path / "state")
    ad = LowRankAdapter(6, rank=2, alpha=2)
    with torch.no_grad():
        ad.up.weight.fill_(0.1)
    meta = AdapterMetadata.new("qualified-skill", 6, 2, 2)
    cand = mgr.stage_adapter(ad, meta, base_generation="g42", centroid=torch.ones(6))
    metrics = {
        "functional_accuracy": 0.91,
        "heldout_regression": 0.0,
        "old_domain_regression": 0.0,
        "checks": {"rollback": True},
    }
    mgr.evaluate_adapter(cand, metrics)
    policy = QualificationPolicy(min_functional_accuracy=0.9,
                                 max_heldout_regression=0.0,
                                 max_old_domain_regression=0.01,
                                 require_checks=("rollback",))
    rec = mgr.qualify_adapter(cand, metrics, policy)
    assert rec["state"] == "qualified"
    signer = Ed25519PromotionSigner.generate("v6-test")
    active = mgr.promote_adapter(cand, signer)
    assert active["candidate_id"] == cand.candidate_id
    got, got_meta, centroid, _ = mgr.load_active_adapter(signer.verifier())
    assert got_meta.id == meta.id
    assert centroid.shape == (6,)
    assert torch.allclose(got.up.weight, ad.up.weight)
