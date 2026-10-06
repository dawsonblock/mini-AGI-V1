import json
from pathlib import Path

def test_v41_regression_suite_is_substantial_and_valid_jsonl():
    p=Path(__file__).parents[1]/'benchmarks'/'functional_v41.jsonl'
    rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    assert len(rows) >= 250
    assert len({x['id'] for x in rows}) == len(rows)
    assert {'numeric','python_tests','exact'} <= {x['scorer'] for x in rows}
