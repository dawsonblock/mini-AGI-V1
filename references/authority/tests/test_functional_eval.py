from minagi.functional_eval import EvalTask, score


def test_numeric_scorer_uses_answer_number():
    t = EvalTask("x", "", "numeric", 80)
    assert score(t, "The answer is 80")["ok"]
    assert not score(t, "The answer is 60")["ok"]


def test_python_ast_rejects_dangerous_imports():
    t = EvalTask("x", "", "python_ast")
    assert score(t, "```python\ndef f(x):\n    return x + 1\n```")["ok"]
    assert not score(t, "import subprocess\ndef f(x): return x")["ok"]


def test_python_tests_measure_behavior_not_syntax():
    spec = {"entrypoint": "largest_number", "cases": [
        {"args": [[1, 5, 3]], "expected": 5},
        {"args": [[-4, -1, -8]], "expected": -1},
    ]}
    t = EvalTask("x", "", "python_tests", spec)
    good = "def largest_number(xs):\n    return max(xs)"
    bad = "def largest_number(xs):\n    return 7"
    assert score(t, good)["ok"]
    assert not score(t, bad)["ok"]
