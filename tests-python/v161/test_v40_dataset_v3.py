"""V4-05/V4-09 — DatasetPartitionSetV3 compositional split and the
tightened pairwise disjointness checks."""
import pytest

from minagi.v161.dataset_manifest import (DatasetMember,
                                          DatasetMembershipManifest,
                                          DatasetPartitionSet,
                                          DatasetPartitionSetV3)


def _m(sid, fam):
    return DatasetMember(sid, fam, "sha256:" + "1" * 64)


def _parts(comp=None):
    kw = {}
    if comp is not None:
        kw["compositional"] = comp
    return DatasetPartitionSetV3(
        DatasetMembershipManifest("train", (_m("t1", "fam-t"),)),
        DatasetMembershipManifest("validation", (_m("v1", "fam-t"),)),
        DatasetMembershipManifest("hidden", (_m("h1", "fam-h"),)),
        DatasetMembershipManifest("retention", (_m("r1", "fam-r"),)),
        DatasetMembershipManifest("security", (_m("s1", "fam-s"),)),
        require_family_disjoint_hidden=True, **kw)


def test_v3_accepts_disjoint_compositional():
    p = _parts(DatasetMembershipManifest(
        "compositional", (_m("c1", "fam-t-x"),)))
    p.assert_disjoint()  # no raise


def test_v3_rejects_compositional_sample_overlap():
    with pytest.raises(PermissionError, match="compositional"):
        _parts(DatasetMembershipManifest(
            "compositional", (_m("t1", "fam-c"),)))


def test_v3_rejects_compositional_family_overlap_with_hidden():
    with pytest.raises(PermissionError, match="compositional"):
        _parts(DatasetMembershipManifest(
            "compositional", (_m("c1", "fam-h"),)))


def test_v1_tightened_pairwise_checks():
    # train/retention sample overlap is now a leak, not silence —
    # the check runs at construction (assert_disjoint in __post_init__)
    with pytest.raises(PermissionError, match="leakage"):
        DatasetPartitionSet(
            DatasetMembershipManifest("train", (_m("x", "ft"),)),
            DatasetMembershipManifest("validation", (_m("v", "ft"),)),
            DatasetMembershipManifest("hidden", (_m("h", "fh"),)),
            DatasetMembershipManifest("retention", (_m("x", "fr"),)),
            require_family_disjoint_hidden=True)


def test_v1_train_validation_family_overlap_still_allowed():
    # within-family generalization requires shared families
    p = DatasetPartitionSet(
        DatasetMembershipManifest("train", (_m("t", "f"),)),
        DatasetMembershipManifest("validation", (_m("v", "f"),)),
        DatasetMembershipManifest("hidden", (_m("h", "fh"),)),
        require_family_disjoint_hidden=True)
    p.assert_disjoint()


def test_v3_digest_differs_from_v1():
    v1 = DatasetPartitionSet(
        DatasetMembershipManifest("train", (_m("t", "ft"),)),
        DatasetMembershipManifest("validation", (_m("v", "fv"),)),
        DatasetMembershipManifest("hidden", (_m("h", "fh"),)),
        require_family_disjoint_hidden=True)
    v3 = DatasetPartitionSetV3(
        DatasetMembershipManifest("train", (_m("t", "ft"),)),
        DatasetMembershipManifest("validation", (_m("v", "fv"),)),
        DatasetMembershipManifest("hidden", (_m("h", "fh"),)),
        require_family_disjoint_hidden=True)
    assert v1.digest != v3.digest  # different schema => different binding
