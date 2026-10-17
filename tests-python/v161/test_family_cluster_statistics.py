"""3B Step 15 — family-aware statistics.

Repeated rows generated from the same template are one cluster, never
independent evidence. These tests pin the family analysis (per-family
deltas, positive-family fraction, worst family, leave-one-family-out)
and the property that clustered uncertainty is not narrower than the
row-level bootstrap.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.stats import (bootstrap_ci, cluster_bootstrap_ci,  # noqa: E402
                               family_delta_analysis)


def test_family_analysis_reports_family_effects():
    pairs = [0.0, 0.0, 1.0, 1.0]           # two families, split outcomes
    fams = ["fam-a", "fam-a", "fam-b", "fam-b"]
    out = family_delta_analysis(pairs, fams, arm_scores=[1, 1, 1, 0])
    assert out["n_families"] == 2
    assert out["family_mean_delta"] == {"fam-a": 0.0, "fam-b": 1.0}
    assert out["family_row_counts"] == {"fam-a": 2, "fam-b": 2}
    assert out["families_positive"] == 1
    assert out["fraction_families_positive"] == 0.5
    assert out["worst_family"] == "fam-a"
    assert out["worst_family_delta"] == 0.0
    # leave-one-family-out means recompute over the remaining families
    assert out["leave_one_family_out_mean"] == {"fam-a": 1.0, "fam-b": 0.0}
    assert out["arm_strict_accuracy_by_family"] == {"fam-a": 1.0,
                                                    "fam-b": 0.5}


def test_single_family_leave_one_out_is_undefined():
    out = family_delta_analysis([1.0, 0.0], ["only", "only"])
    assert out["n_families"] == 1
    assert out["leave_one_family_out_mean"] == {"only": None}
    assert out["worst_family"] == "only"


def test_family_analysis_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        family_delta_analysis([1.0], ["a", "b"])
    with pytest.raises(ValueError):
        family_delta_analysis([1.0], ["a"], arm_scores=[1.0, 0.0])


def test_empty_family_analysis():
    assert family_delta_analysis([], []) == {"n_families": 0}


def test_clustered_uncertainty_is_not_narrower_than_row_level():
    """The family-clustered CI must not claim more precision than the
    row-level bootstrap for a clustered design (Step 15: CIs must
    account for repeated tasks)."""
    samples = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    clusters = ["fam-a", "fam-a", "fam-a", "fam-b", "fam-b", "fam-b"]
    row_ci = bootstrap_ci(samples, resamples=2000, alpha=0.05)
    clu_ci = cluster_bootstrap_ci(samples, clusters, resamples=2000,
                                  alpha=0.05)
    row_width = row_ci["upper"] - row_ci["lower"]
    clu_width = clu_ci["upper"] - clu_ci["lower"]
    assert clu_width >= row_width
    assert clu_ci["n_clusters"] == 2
