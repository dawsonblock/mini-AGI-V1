import torch

from minagi.recur import RecurCoder, RecurConfig


def test_prefix_vote_masks_targets_that_help_choose_card():
    torch.manual_seed(0)
    cfg = RecurConfig(vocab_size=16, d_model=16, n_head=4, d_ff=32,
                      block=8, use_pool=True, pool_experts=4,
                      pool_d_ff=24, pool_top_k=2, pool_max=4,
                      n_prelude=1, n_recur=1, n_coda=0, max_steps=1,
                      min_steps=1)
    m = RecurCoder(cfg)
    m.pool.admission_mode = "causal_prefix_vote"
    m.pool.admission_prefix_tokens = 3
    m.eval()
    x = torch.tensor([[1, 2, 3, 4, 5]])
    y1 = torch.tensor([[2, 3, 4, 5, 6]])
    y2 = y1.clone()
    y2[:, :2] = torch.tensor([[9, 10]])
    with torch.no_grad():
        _, l1 = m(x, y1)
        _, l2 = m(x, y2)
    torch.testing.assert_close(l1, l2, rtol=0, atol=1e-7)
