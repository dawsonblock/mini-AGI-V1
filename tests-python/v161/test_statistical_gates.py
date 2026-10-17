"""v4.0.3-r2 Phase 4 — statistical qualification policy.

The versioned statistical_policy.yaml governs the analyzer's
statistical gate: CI-based acquisition (LCB > delta_min), retention
non-inferiority (UCB <= declared max), and fail-closed
NOT_EVALUABLE on insufficient coverage. These tests pin the
policy-override path and the insufficiency semantics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(Path(__file__).parent))

from test_independent_eval_receipt import _authority, build_run  # noqa: E402
from test_matrix_false_positive import (  # noqa: E402
    ARM, WHO_PASS, _analyze, _corpus, _issue, _rid, _security_dir,
    _trust_root, _write_cfg, _write_manifest)


def _policy(tmp_path, name="statistical_policy.yaml", **overrides):
    """Write a statistical_policy.yaml whose gates: block carries the
    given overrides and return its path."""
    doc = {"schema": "mini-agi-v4.0.3-statistical-policy-v1",
           "version": 1,
           "gates": {"within_holdout_min_delta": 0.15,
                     "retention_max_destruction_rate": 0.10,
                     "statistical_min_family_clusters": 2,
                     "statistical_min_seeds": 1,
                     "bootstrap_resamples": 2000, "ci_alpha": 0.05}}
    doc["gates"].update(overrides)
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc))
    return p


def _setup(tmp_path, *, seeds=(0,), signers=None,
           who_rows=WHO_PASS, ret_rows=None):
    """One evaluable candidate cell per declared seed (receipted via
    the shared trust root) and return the _analyze kwargs."""
    signers = signers or _authority("evaluation")[2]
    trust = _trust_root(tmp_path, signers)
    ikw = {"who_rows": who_rows}
    if ret_rows is not None:
        ikw["ret_rows"] = ret_rows
    for s in seeds:
        d = build_run(tmp_path / "runs", run_id=_rid(seed=s), **ikw)
        _issue(d, signers["evaluation"], tmp_path, **ikw)
    manifest = _write_manifest(tmp_path, [ARM], seeds=seeds)
    cfg = _write_cfg(tmp_path)
    security = _security_dir(
        tmp_path, {_rid(seed=s): ("REFUSE", "REFUSE") for s in seeds})
    return {"manifest": manifest, "cfg": cfg, "security": security,
            "trust": trust, "who_rows": who_rows,
            **({"ret_rows": ret_rows} if ret_rows is not None else {})}


def test_statistical_policy_overrides_confirmation_gates(tmp_path):
    """The versioned policy's delta_min wins — an arm passing at the
    confirmation default 0.15 fails when the policy preregisters a
    larger meaningful improvement."""
    reg, ver, signers = _authority("evaluation")
    kw = _setup(tmp_path, signers=signers)
    loose = _policy(tmp_path, "sp_loose.yaml",
                    within_holdout_min_delta=0.15)
    strict = _policy(tmp_path, "sp_strict.yaml",
                     within_holdout_min_delta=0.60)
    _, rep_loose = _analyze(
        tmp_path, **kw, extra_args=["--statistical-policy", str(loose)])
    _, rep_strict = _analyze(
        tmp_path, **kw, extra_args=["--statistical-policy",
                                    str(strict)])
    st_loose = rep_loose["arms_report"][ARM]["gate_detail"][
        "statistical"]
    st_strict = rep_strict["arms_report"][ARM]["gate_detail"][
        "statistical"]
    assert st_loose["status"] == "PASS", st_loose
    assert st_strict["status"] == "FAIL", st_strict
    assert rep_strict["matrix_status"] == "COMPLETE_WITH_FAILURES"
    assert rep_loose["matrix_status"] == "QUALIFIED"


def test_min_independent_seeds_is_enforced(tmp_path):
    """One seed's evidence can be evaluable yet statistically
    insufficient — independent replication is a coverage requirement."""
    reg, ver, signers = _authority("evaluation")
    kw = _setup(tmp_path, seeds=(0,), signers=signers)
    pol = _policy(tmp_path, statistical_min_seeds=2)
    _, rep = _analyze(tmp_path, **kw,
                      extra_args=["--statistical-policy", str(pol)])
    stat = rep["arms_report"][ARM]["gate_detail"]["statistical"]
    assert stat["status"] == "NOT_EVALUABLE"
    assert "seed" in (stat.get("reason") or "")
    assert rep["arms_report"][ARM]["experiment_status"] != "QUALIFIED"


def test_insufficient_family_coverage_is_unevaluable(tmp_path):
    """A one-cluster bootstrap is degenerate — NOT_EVALUABLE, not a
    narrow fabricated CI."""
    reg, ver, signers = _authority("evaluation")
    one_fam = (("w1", "famA", "G", 1, 1), ("w2", "famA", "H", 0, 1))
    kw = _setup(tmp_path, signers=signers, who_rows=one_fam)
    pol = _policy(tmp_path)
    _, rep = _analyze(tmp_path, **kw,
                      extra_args=["--statistical-policy", str(pol)])
    stat = rep["arms_report"][ARM]["gate_detail"]["statistical"]
    assert stat["status"] == "NOT_EVALUABLE"
    assert "family" in (stat.get("reason") or "") or \
        "cluster" in (stat.get("reason") or "")


def test_missing_retention_pairs_is_unevaluable(tmp_path):
    """No frozen-correct retention pairs → undefined interval →
    NOT_EVALUABLE (never a zero-destruction pass)."""
    reg, ver, signers = _authority("evaluation")
    # all retention rows frozen-WRONG → no destruction interval exists
    bad_ret = tuple((f"r{i}", "famR", f"R{i}", 0, 0) for i in range(10))
    kw = _setup(tmp_path, signers=signers, ret_rows=bad_ret)
    pol = _policy(tmp_path)
    _, rep = _analyze(tmp_path, **kw,
                      extra_args=["--statistical-policy", str(pol)])
    detail = rep["arms_report"][ARM]["gate_detail"]
    unevaluable = {k: g["status"] for k, g in detail.items()
                   if g["status"] == "NOT_EVALUABLE"}
    assert "G6_retention" in unevaluable or \
        "statistical" in unevaluable, detail
    assert rep["arms_report"][ARM]["experiment_status"] != "QUALIFIED"


def test_report_names_the_governing_policy(tmp_path):
    reg, ver, signers = _authority("evaluation")
    kw = _setup(tmp_path, signers=signers)
    pol = _policy(tmp_path)
    _, rep = _analyze(tmp_path, **kw,
                      extra_args=["--statistical-policy", str(pol)])
    assert rep["statistical_policy"]["path"].endswith(
        "statistical_policy.yaml")
    assert rep["statistical_policy"]["schema"] == \
        "mini-agi-v4.0.3-statistical-policy-v1"
