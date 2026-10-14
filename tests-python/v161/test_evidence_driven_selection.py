"""Campaign 3B.3 Priority 3 — the selector must be evidence-driven.

The whole difference between an evidence-driven selector and a fixed
script is that the former REFUSES a lever the evidence has refuted. These
tests pin that, and pin the failure that produced the criterion: an
earlier version measured rank spread across individual runs rather than
across rank means, which inflated it from 0.053 to 0.175 with seed noise
and made the selector recommend the one lever the evidence contradicts.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

SCRIPT = ROOT / "scripts" / "analysis" / "intervention_selection.py"


#: Levers whose measurement lives in results/, which is gitignored by the
#: repository's evidence-separation convention and ships in the separate
#: evidence package. A source-only checkout therefore cannot measure them,
#: and the tests that need them skip rather than fail.
EVIDENCE_PACKAGE_LEVERS = ("composition", "rank", "exposure", "steps")


def _select(tmp_path):
    out = tmp_path / "sel.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out)],
        capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        import pytest
        pytest.skip(f"no evidence in this checkout: {proc.stderr[-200:]}")
    return json.loads(out.read_text())


def _require_packaged_evidence(doc):
    """Skip when the run evidence is absent, so the source archive is not
    asked to prove something only the evidence package can."""
    import pytest
    missing = [lev for lev in EVIDENCE_PACKAGE_LEVERS
               if lev not in doc["measured_effects"]]
    if missing:
        pytest.skip("run evidence not in this checkout (results/ ships in "
                    f"the evidence package); unmeasured levers: {missing}")


def test_selection_is_computed_from_evidence_not_from_the_candidate_list(
        tmp_path):
    doc = _select(tmp_path)
    assert doc["measured_effects"], (
        "the selector produced no measurements; it would be ranking on the "
        "candidate list alone")
    for name, blk in doc["measured_effects"].items():
        assert blk.get("source"), f"{name} has no evidence source"
        assert blk.get("basis"), f"{name} has no stated basis"


def test_the_capacity_candidate_is_refused_on_direction(tmp_path):
    """E5 assumes more capacity helps. The sweep measured rank 4 as the
    best of 4/8/16/32, so the candidate is refused — a fixed script built
    from the plan's list would have picked it."""
    doc = _select(tmp_path)
    _require_packaged_evidence(doc)
    refused = {r["candidate"]: r for r in doc["refused"]}
    assert "E5_rank_or_target_modules" in refused, (
        "the capacity candidate is no longer refused; the selector would "
        "recommend a remedy the evidence contradicts")
    r = refused["E5_rank_or_target_modules"]
    assert r["status"] == "REFUSED-DIRECTION"
    assert "not monotone" in r["reason"]
    assert "E5_rank_or_target_modules" not in {
        s["candidate"] for s in doc["selected"]}


def test_rank_spread_is_measured_across_rank_means(tmp_path):
    _require_packaged_evidence(_select(tmp_path))
    """The specific failure: measuring across individual runs folds seed
    noise into the rank effect."""
    doc = _select(tmp_path)
    rank = doc["measured_effects"].get("rank")
    assert rank, "the rank lever is no longer measured"
    assert "MEANS" in rank["basis"].upper(), (
        "the rank spread is no longer taken across rank means; it will "
        "inflate with seed noise again")
    assert rank.get("within_level_sd") is not None, (
        "the rank measurement carries no seed sd, so flatness cannot be "
        "tested against noise")
    per = rank["per_breadth"]
    for label, blk in per.items():
        means = blk["rank_means"]
        assert max(means.values()) - min(means.values()) == \
            blk["spread_across_rank_means"], label


def test_flatness_is_tested_against_noise_not_a_constant(tmp_path):
    doc = _select(tmp_path)
    assert "EXCEED" in doc["flatness_test"].upper(), (
        "the flatness criterion is no longer the variance test")
    assert "flat_threshold" not in doc, (
        "a fixed threshold is back as the criterion")
    for r in doc["refused"]:
        if r["status"] == "REFUSED":
            assert "seed sd" in r["reason"] or "threshold" in r["reason"]


def test_preservation_and_composition_rank_above_the_refused_lever(
        tmp_path):
    _require_packaged_evidence(_select(tmp_path))
    """The recommendation must follow the measurements: preservation and
    composition are the two largest measured effects."""
    doc = _select(tmp_path)
    selected = doc["selected"]
    assert selected, "nothing was selected"
    top = selected[0]
    assert top["lever"] in ("preservation", "composition"), (
        f"the recommendation targets {top['lever']}, but the two largest "
        "measured effects are preservation and composition")
    eff = doc["lever_effects"]
    assert eff["preservation"]["value"] > eff["capacity"]["value"]
    assert eff["composition"]["value"] > eff["capacity"]["value"]


def test_selector_is_path_anchored_not_cwd_anchored(tmp_path):
    """The tool resolves the repository from its own location, so it
    produces the same selection wherever it is invoked from. A cwd-anchored
    selector would silently find no evidence when run from elsewhere."""
    a = _select(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(tmp_path / "b.json")],
        capture_output=True, text=True, cwd="/tmp")
    assert proc.returncode == 0, proc.stderr
    b = json.loads((tmp_path / "b.json").read_text())
    assert a["measured_effects"].keys() == b["measured_effects"].keys()
    assert a["recommendation"] == b["recommendation"]


def test_selector_refuses_to_select_on_nothing():
    """With no evidence it must fail, not fall back to the candidate list.

    The guard is asserted at source level because the evidence is resolved
    from the repository, so an empty-evidence run cannot be produced from
    inside the repository without moving files.
    """
    src = SCRIPT.read_text()
    assert "refusing to select on nothing" in src, (
        "the empty-evidence guard is gone; the selector could fall back to "
        "the candidate list")
    assert "if not effects:" in src, "the guard no longer tests for evidence"


def test_an_incomplete_selection_says_so(tmp_path):
    """A checkout without the run evidence yields a WEAKER selection: a
    lever it cannot measure cannot have its candidates refused. That must
    be reported, because a selector that quietly stops refusing a refuted
    lever is worse than one that admits it cannot."""
    doc = _select(tmp_path)
    assert "selection_complete" in doc and "unmeasured_levers" in doc
    if doc["selection_complete"]:
        assert doc["unmeasured_levers"] == []
        assert doc["completeness_warning"] == ""
    else:
        assert doc["unmeasured_levers"], \
            "incomplete but no unmeasured levers named"
        assert "INCOMPLETE SELECTION" in doc["completeness_warning"]
        assert "NOT tested for refusal" in doc["completeness_warning"]
