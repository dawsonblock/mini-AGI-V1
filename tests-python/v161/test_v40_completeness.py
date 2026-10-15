"""V4-15 — experiment matrix completeness auditing."""
from minagi.v161.experiment_matrix import (COMPLETE, INCOMPLETE,
                                           INCONSISTENT, CellIdentity,
                                           MatrixSpec, audit_matrix,
                                           required_cells)


def _spec(**kw):
    d = dict(campaign_id="c1", seeds=(0, 1), arms=("L1", "L6"))
    d.update(kw)
    return MatrixSpec(**d)


def test_required_cells_includes_per_arm_extras():
    s = _spec(per_arm_cells={"L6": ("L6_delayed",)})
    assert required_cells(s, "L6") == ("L6", "L6_delayed")
    assert required_cells(s, "L1") == ("L1",)


def test_complete_matrix():
    s = _spec()
    cells = {(0, "L1"): {}, (0, "L6"): {}, (1, "L1"): {}, (1, "L6"): {}}
    rep = audit_matrix(s, cells)
    assert rep["status"] == COMPLETE
    assert rep["qualified_comparison"] is True
    assert rep["n_cells_present"] == rep["n_cells_required"] == 4


def test_missing_cell_is_incomplete():
    s = _spec()
    cells = {(0, "L1"): {}, (0, "L6"): {}, (1, "L1"): {}}
    rep = audit_matrix(s, cells)
    assert rep["status"] == INCOMPLETE
    assert rep["missing"] == [{"seed": 1, "cell": "L6"}]
    assert rep["qualified_comparison"] is False


def test_identity_conflict_is_inconsistent():
    s = _spec()
    cells = {
        (0, "L1"): {"model_digest": "sha256:aaa"},
        (0, "L6"): {"model_digest": "sha256:bbb"},
        (1, "L1"): {"model_digest": "sha256:aaa"},
        (1, "L6"): {"model_digest": "sha256:aaa"},
    }
    rep = audit_matrix(s, cells)
    assert rep["status"] == INCONSISTENT
    fields = [v["field"] for v in rep["identity_violations"]]
    assert "model_digest" in fields


def test_expected_identity_binding():
    s = _spec(expected_identity=CellIdentity(model_digest="sha256:good"))
    cells = {(0, "L1"): {"model_digest": "sha256:good"},
             (0, "L6"): {"model_digest": "sha256:good"},
             (1, "L1"): {"model_digest": "sha256:good"},
             (1, "L6"): {"model_digest": "sha256:evil"}}
    rep = audit_matrix(s, cells)
    assert rep["status"] == INCONSISTENT
    assert any(v.get("found") == "sha256:evil"
               for v in rep["identity_violations"])


def test_unbound_fields_check_consistency_only():
    s = _spec()
    cells = {(0, "L1"): {"environment_digest": "e1"},
             (0, "L6"): {"environment_digest": "e1"},
             (1, "L1"): {"environment_digest": "e2"},  # differs
             (1, "L6"): {"environment_digest": "e2"}}
    rep = audit_matrix(s, cells)
    assert rep["status"] == INCONSISTENT  # env conflict, none bound


def test_attribute_shaped_cells():
    class R:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    s = _spec()
    cells = {(s_, a): R(model_digest="m", tokenizer_digest="t",
                        evaluator_digest="e", environment_digest="env",
                        dataset_digest="d")
             for s_ in (0, 1) for a in ("L1", "L6")}
    assert audit_matrix(s, cells)["status"] == COMPLETE
