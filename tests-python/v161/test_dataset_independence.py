"""v4.1.0 — dataset independence and within-family holdout invariants.

Covers the new partition set (``within_holdout`` families must be a
subset of train, sample-disjoint everywhere, family-disjoint from
hidden/compositional/final_holdout) and the generated campaign4a
corpus: byte-identical train rows vs campaign3e, word-level
disjointness across the within-family surfaces, partition sizes, and
exact (prompt, expected) leakage across splits.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "scripts"))

from egai.common.canonical import sha256_bytes  # noqa: E402
from minagi.v161.dataset_manifest import (  # noqa: E402
    DatasetMember, DatasetMembershipManifest, DatasetPartitionSetV4)


def _member(rid, family):
    payload = json.dumps({"id": rid, "family": family},
                         sort_keys=True).encode()
    return DatasetMember(rid, family, sha256_bytes(payload))


def _part(name, pairs):
    return DatasetMembershipManifest(
        name, tuple(_member(rid, fam) for rid, fam in pairs))


def _base_set(**kw):
    args = dict(
        train=_part("train", [("t1", "famA"), ("t2", "famB")]),
        validation=_part("validation", [("v1", "famA"), ("v2", "famB")]),
        hidden=_part("hidden", [("h1", "famX"), ("h2", "famY")]),
        retention=_part("retention", [("r1", "retA")]),
        security=_part("security", [("s1", "secA")]),
        compositional=_part("compositional", [("c1", "compA")]),
        final_holdout=_part("final_holdout", [("o1", "holdA")]),
        require_family_disjoint_hidden=True)
    args.update(kw)
    return DatasetPartitionSetV4(**args)


def test_within_holdout_must_be_subset_of_train_families():
    with pytest.raises(PermissionError, match="untrained families"):
        _base_set(within_holdout=_part(
            "within_holdout", [("w1", "famZ")]))


def test_within_holdout_accepts_trained_families():
    ps = _base_set(within_holdout=_part(
        "within_holdout", [("w1", "famA"), ("w2", "famB")]))
    assert ps.proof()["within_holdout_digest"]


def test_within_holdout_sample_overlap_rejected():
    with pytest.raises(PermissionError, match="sample leakage"):
        _base_set(within_holdout=_part(
            "within_holdout", [("v1", "famA")]))   # validation id reused


def test_within_holdout_family_overlap_with_hidden_rejected():
    with pytest.raises(PermissionError, match="family leakage"):
        _base_set(within_holdout=_part(
            "within_holdout", [("w1", "famX")]))   # hidden family


def test_v3_digest_stability_v4_field_absent():
    """A set built without within_holdout must still behave like V3."""
    ps = _base_set()
    ps.assert_disjoint()
    assert ps.within_holdout is None


def test_rehearsal_family_overlap_with_measured_surface_rejected():
    """Rehearsal is train-side: a rehearsal row in a measured family
    would contaminate that measurement, so it must fail closed."""
    with pytest.raises(PermissionError, match="rehearsal family leakage"):
        _base_set(rehearsal=_part("rehearsal", [("rh1", "famX")]))  # hidden
    with pytest.raises(PermissionError, match="rehearsal family leakage"):
        _base_set(rehearsal=_part("rehearsal", [("rh2", "famA")]))  # train


def test_rehearsal_sample_overlap_rejected():
    with pytest.raises(PermissionError, match="rehearsal sample leakage"):
        _base_set(rehearsal=_part("rehearsal", [("t1", "famR")]))


def test_rehearsal_distinct_families_accepted():
    ps = _base_set(rehearsal=_part(
        "rehearsal", [("rh1", "famR"), ("rh2", "famR")]))
    assert ps.proof()["rehearsal_digest"]


# ---------- generated corpus (campaign4a) ----------------------------

@pytest.fixture(scope="module")
def corpus():
    import gen_campaign4a
    return gen_campaign4a.build()


def _words(group):
    return {str(r["prompt"]).split("\n")[1].split(": ", 1)[1]
            for r in group}


def test_campaign4a_partition_sizes(corpus):
    rows, holdout = corpus
    from collections import Counter
    c = Counter(r["split"] for r in rows)
    assert c["train"] == 80
    assert c["validation"] == 8 * 24
    assert c["within_holdout"] == 8 * 32
    assert c["hidden"] == 10 * 24
    assert c["compositional"] == 8 * 12
    assert c["retention"] == 108
    assert c["security"] >= 30
    assert c["rehearsal"] == 4 * 16 + 16
    assert len(holdout) == 5 * 20


def test_campaign4a_rehearsal_disjoint_from_measured(corpus):
    """Rehearsal rows must carry capabilities that are neither trained
    nor measured anywhere — otherwise the preservation arms would be
    rehearsing a surface we score."""
    rows, _ = corpus
    measured = [r for r in rows
                if r["split"] in ("train", "validation",
                                  "within_holdout", "hidden",
                                  "compositional", "retention",
                                  "security")]
    reh = [r for r in rows if r["split"] == "rehearsal"]
    assert reh, "corpus must carry a rehearsal pool for arms C/E"
    measured_fams = {r["family"] for r in measured}
    assert {r["family"] for r in reh}.isdisjoint(measured_fams)
    # the runner's default pool filter admits exactly the train-side
    # whitelist — rehearsal must be in it, and nothing measured may be
    src = (ROOT / "scripts" / "positive_control_fit.py").read_text()
    import re as _re
    m = _re.search(r'r\["split"\] in \(([^)]*)\)', src)
    pool = set(_re.findall(r'"(\w+)"', m.group(1)))
    assert "rehearsal" in pool
    assert pool.isdisjoint({"hidden", "within_holdout", "retention",
                            "security", "compositional",
                            "final_holdout"})


def test_campaign4a_train_matches_campaign3e(corpus):
    """The learning condition is unchanged: every 4a train row must be
    byte-identical to the campaign3e train row with the same id."""
    rows, _ = corpus
    e3 = {r["id"]: r for r in (json.loads(ln) for ln in
          (ROOT / "configs" / "campaign3e_tasks.jsonl")
          .read_text().splitlines() if ln.strip())
          if r["split"] == "train"}
    ours = {r["id"]: r for r in rows if r["split"] == "train"}
    assert ours == e3


def test_campaign4a_within_holdout_invariants(corpus):
    rows, _ = corpus
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    who = [r for r in rows if r["split"] == "within_holdout"]

    train_fams = {r["family"] for r in train}
    assert {r["family"] for r in who} <= train_fams
    assert {r["family"] for r in val} <= train_fams

    # input-word disjointness: within-family means unseen INPUTS,
    # and the confirmation surface is input-disjoint from the dev set
    assert not (_words(who) & _words(train))
    assert not (_words(val) & _words(train))
    assert not (_words(who) & _words(val))

    # instruction-surface coverage: a nontrivial share of confirmation
    # rows restate the function in the variant-B phrasing
    n_b = sum(1 for r in who if r.get("variant") == "B")
    assert 0.15 <= n_b / len(who) <= 0.35


def test_campaign4a_no_exact_pair_leakage(corpus):
    rows, holdout = corpus
    seen: dict[str, str] = {}
    for r in rows + holdout:
        key = json.dumps([r["prompt"], r["expected"]], sort_keys=True)
        if key in seen:
            assert seen[key] == r["split"], (
                f"(prompt, expected) pair crosses splits "
                f"{seen[key]!r}/{r['split']!r} at {r['id']}")
        seen[key] = r["split"]


def test_campaign4a_full_check_passes(corpus):
    import gen_campaign4a
    res = gen_campaign4a.check(*corpus)
    assert res["within_holdout_families_ok"]
