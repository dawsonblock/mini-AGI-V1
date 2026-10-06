import json
import torch

from minagi.create import create
from minagi.full_bptt import FullBPTTStepper
from minagi.optim_groups import adamw_groups
from minagi.paged_adamw import build_adamw
from minagi import store as weights_store


def _tiny(path):
    create(str(path), force=True, verbose=False, experts=3, resident=1,
           d_ff=12, d_model=8, trunk_d_ff=16, n_head=2, block=32,
           max_steps=2, top_k=2, n_prelude=1, n_recur=1, n_coda=0,
           bptt_window=2)
    import train
    return train.build_paged(str(path), torch.device("cpu"), resident=1,
                             ram_capacity=2)


def test_fresh_checkpoint_loads_v5_virtual_runtime_and_trains_one_graph(tmp_path):
    model, cfg, pool, man = _tiny(tmp_path / "weights")
    assert pool.__class__.__name__ == "VirtualPagedPool"
    opt = build_adamw(adamw_groups(model, 1e-3, 0.2, 0.0),
                      lr=1e-3, betas=(0.9, 0.95))
    stepper = FullBPTTStepper(model, opt, clip=1.0, expert_lr=1e-3)
    g = torch.Generator().manual_seed(4)
    x = torch.randint(0, cfg.vocab_size, (1, 8), generator=g)
    y = torch.randint(0, cfg.vocab_size, (1, 8), generator=g)
    report = stepper.step(x, y)
    assert report.experts_updated > 0
    assert report.loss > 0
    assert pool.graph_epoch.state == "closed"


def test_saved_v5_checkpoint_records_virtual_runtime(tmp_path):
    root = tmp_path / "weights"
    model, cfg, pool, man = _tiny(root)
    opt = build_adamw(adamw_groups(model, 1e-3, 0.2, 0.0),
                      lr=1e-3, betas=(0.9, 0.95))
    weights_store.save(model, str(root), step=0, val=1.0, opt=opt,
                       cfg=man["cfg"], verbose=False)
    saved = json.loads((root / "manifest.json").read_text())
    assert saved["paging_runtime"] == "virtual_autograd_v1"
    assert saved["paged"] is True
