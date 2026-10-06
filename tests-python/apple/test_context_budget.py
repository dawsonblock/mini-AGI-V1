from kvcontinual.continual.macos.context_budget import approximate_tokens, select_by_token_budget


def test_approximate_tokens_is_deterministic():
    assert approximate_tokens("abcdefgh") == 2
    assert approximate_tokens("") == 0


def test_budget_selection_never_exceeds_budget():
    items = ["a" * 8, "b" * 20, "c" * 8]
    s = select_by_token_budget(items, render=lambda x: x, budget_tokens=4, max_items=10)
    assert s.estimated_tokens <= 4
    assert list(s.selected) == [items[0], items[2]]
