"""Campaign 3B Step 18 — the governed safety suite is enforced end to end.

Runs the real qualifier entrypoint over a synthetic campaign whose
committed corpus carries the governed probes, and proves:

  * a compliant matrix qualifies (the gate does not block on a pass),
  * a governed violation refuses the campaign with an explicit
    governed-gate reason,
  * a corpus without governed probes is reported NOT_APPLICABLE —
    never as a safety pass,
  * the suite digest is bound into the qualification record.

The synthetic fixture is reused from test_v165_qualifier_chain so this
file adds enforcement coverage without duplicating the evidence tree.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.evaluators import score_row, security_regression  # noqa: E402
from minagi.v161.safety_regression import (PROTECTED_CANARY,  # noqa: E402
                                           safety_probes, suite_digest)


def _load_fixture():
    path = Path(__file__).resolve().parent / "test_v165_qualifier_chain.py"
    spec = importlib.util.spec_from_file_location("c3b_qualifier_fixture",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FX = _load_fixture()
_BASE_CORPUS = tuple(FX.CORPUS)
_ORIG_PREDS_FOR = FX.SyntheticCampaign._preds_for
_ORIG_METRICS = FX.SyntheticCampaign._metrics

#: Fully compliant governed predictions.
_COMPLIANT = {r["id"]: "REFUSED" for r in safety_probes()}
#: A run whose arm discloses the protected canary.
_VIOLATING = dict(_COMPLIANT)
_VIOLATING["saf-pi-0"] = f"The confidential value is {PROTECTED_CANARY}"


def _use_governed_corpus(security_outputs):
    """Install the governed corpus + emission paths (module-scoped)."""
    FX.CORPUS = _BASE_CORPUS + tuple(safety_probes())

    def _preds_for(self, arm, hidden_out):
        preds = []
        for i, row in enumerate(FX._by_split("hidden")):
            out = row["expected"] if i < hidden_out else "wrong"
            preds.append(FX.prediction_record(row["id"], "hidden",
                                              row["prompt"], out, 3, 32))
        for row in [r for r in FX._by_split("retention")
                    if r.get("probe") != "delayed"]:
            preds.append(FX.prediction_record(row["id"], "retention",
                                              row["prompt"],
                                              row["expected"], 2, 32))
        for row in FX._by_split("security"):
            out = security_outputs.get(str(row["id"]), "benign")
            preds.append(FX.prediction_record(row["id"], "security",
                                              row["prompt"], out, 2, 32))
        return preds

    def _metrics(self, preds):
        m = _ORIG_METRICS(self, preds)
        rows_by = {r["id"]: r for r in FX._by_split("security")}
        sec = [p for p in preds if p["block"] == "security"]
        m["security_regressions"] = sum(
            int(score_row(rows_by[p["id"]], p["output"], security_regression))
            for p in sec)
        return m

    FX.SyntheticCampaign._preds_for = _preds_for
    FX.SyntheticCampaign._metrics = _metrics


def _use_legacy_corpus():
    """Restore the fixture's original (pre-suite) security surface."""
    FX.CORPUS = _BASE_CORPUS
    FX.SyntheticCampaign._preds_for = _ORIG_PREDS_FOR
    FX.SyntheticCampaign._metrics = _ORIG_METRICS


def _qualify(tmp_path):
    c = FX.SyntheticCampaign(tmp_path)
    c.build()
    out = c.storage / "QUALIFICATION_RECORD.json"
    proc = subprocess.run(
        [sys.executable,
         str(ROOT / "scripts" / "validation" / "qualify_campaign1.py"),
         "--storage", str(c.storage), "--campaign-id", FX.CID,
         "--root", str(c.root), "--output", str(out)],
        capture_output=True, text=True)
    return proc, (json.loads(out.read_text()) if out.is_file() else None)


def test_compliant_governed_matrix_qualifies(tmp_path):
    _use_governed_corpus(_COMPLIANT)
    try:
        proc, doc = _qualify(tmp_path)
    finally:
        _use_legacy_corpus()
    assert proc.returncode == 0, proc.stderr
    assert doc["value"]["decision"] == "QUALIFIED"
    gov = doc["value"]["delta_ft_neural_stats"]["governed_safety"]
    assert gov["applicability"] == "ENFORCED"
    assert gov["decision"] == "PASS"
    assert gov["blocked_categories"] == []
    for cat, res in gov["categories"].items():
        assert res["n"] == 2, cat
        assert res["n_violation"] == 0, cat
        assert res["status"] == "PASS", cat
    assert gov["comparative"]["n_baseline_correct"] == 2
    assert gov["comparative"]["n_regressed"] == 0


def test_governed_violation_refuses_the_campaign(tmp_path):
    _use_governed_corpus(_VIOLATING)
    try:
        proc, doc = _qualify(tmp_path)
    finally:
        _use_legacy_corpus()
    assert proc.returncode == 2, proc.stderr
    assert doc["value"]["decision"] == "REFUSED"
    gov = doc["value"]["delta_ft_neural_stats"]["governed_safety"]
    assert gov["decision"] == "BLOCK"
    assert "protected_information" in gov["blocked_categories"]
    assert gov["categories"]["protected_information"]["n_violation"] == 1
    joined = " ".join(doc["value"]["reasons"])
    assert "governed safety/retention gate BLOCKED" in joined
    assert "protected_information" in joined


def test_qualification_record_states_when_the_suite_does_not_apply(tmp_path):
    """A legacy corpus (no governed probes) is reported NOT_APPLICABLE."""
    _use_legacy_corpus()
    proc, doc = _qualify(tmp_path)
    assert proc.returncode == 0, proc.stderr
    gov = doc["value"]["delta_ft_neural_stats"]["governed_safety"]
    assert gov["applicability"] == "NOT_APPLICABLE"
    assert gov["decision"] == "NOT_APPLICABLE"
    assert gov["blocked_categories"] == []


def test_suite_digest_is_bound_into_the_qualification_record(tmp_path):
    _use_governed_corpus(_COMPLIANT)
    try:
        proc, doc = _qualify(tmp_path)
    finally:
        _use_legacy_corpus()
    assert proc.returncode == 0, proc.stderr
    gov = doc["value"]["delta_ft_neural_stats"]["governed_safety"]
    assert gov["suite_digest"] == suite_digest()
