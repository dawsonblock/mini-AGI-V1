from types import SimpleNamespace
import torch
import torch.nn as nn

from minagi.v4 import (
    TokenBlock, standard_rc10_cases, ExactFullModelOracle,
    FullModelQualificationHarness, ExactModelResult, compare_model_results,
)


class TinyCausalModel(nn.Module):
    def __init__(self, vocab=32, hidden=6):
        super().__init__()
        torch.manual_seed(8)
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, use_cache=False, output_hidden_states=False):
        x = self.embed(input_ids)
        # causal history so deletion/reorder changes the result
        h = torch.cumsum(x, dim=1)
        logits = self.proj(h)
        hidden = (x, h) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hidden)


def _blocks():
    return [TokenBlock(chr(ord('A') + i), (i + 1, i + 9)) for i in range(7)]


def test_standard_case_generator_covers_required_reassembly_classes():
    cases = standard_rc10_cases(_blocks())
    kinds = {x.kind.value for x in cases}
    assert kinds == {
        "prefix_control", "deletion", "sparse_retrieval", "reorder",
        "long_jump", "many_fragments", "changed_predecessor", "duplicate_retrieval",
    }
    sparse = next(x for x in cases if x.kind.value == "sparse_retrieval")
    assert sparse.block_ids == ("A", "D", "F", "G")


def test_exact_full_model_oracle_and_harness_zero_when_runner_is_exact(tmp_path):
    model = TinyCausalModel()
    oracle = ExactFullModelOracle(model)
    cases = standard_rc10_cases(_blocks())
    harness = FullModelQualificationHarness(oracle, lambda case: oracle.run(case.tokens))
    records = harness.run(cases)
    assert all(r.divergence.first_token_logit_kl == 0.0 for r in records)
    assert all(r.divergence.top1_agree for r in records)
    assert all(r.divergence.prompt_top1_agreement == 1.0 for r in records)
    out = tmp_path / "qual.jsonl"
    harness.write_jsonl(out, records)
    assert len(out.read_text().strip().splitlines()) == len(records)


def test_full_model_comparison_detects_context_conditioned_drift():
    model = TinyCausalModel()
    oracle = ExactFullModelOracle(model)
    exact = oracle.run((1, 2, 3, 4))
    bad = ExactModelResult(
        exact.tokens,
        exact.logits + torch.linspace(0, 0.3, exact.logits.shape[-1]),
        tuple(h + 0.05 for h in exact.hidden_states),
    )
    div = compare_model_results(exact, bad)
    assert div.first_token_logit_kl > 0
    assert div.layer_hidden[-1].relative_l2 > 0
