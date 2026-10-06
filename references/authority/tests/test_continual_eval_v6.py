from minagi.continual_eval import continual_metrics, temporal_fact_accuracy


def test_continual_metrics_expose_gain_and_forgetting():
    m=continual_metrics(
        {"old-a":0.9,"old-b":0.8},{"old-a":0.89,"old-b":0.7},
        {"new-a":0.2},{"new-a":0.75},retention_tolerance=0.02)
    assert m.new_capability_gain > 0.5
    assert m.old_domain_regression > 0
    assert m.forgetting_max >= 0.1-1e-9
    assert m.retained_fraction == 0.5


def test_temporal_fact_accuracy():
    cases=[{"query":"capital","as_of":1,"expected":"A"},
           {"query":"capital","as_of":2,"expected":"B"}]
    got=temporal_fact_accuracy(cases,lambda q,t: "A" if t==1 else "B")
    assert got=={"n":2,"accuracy":1.0}
