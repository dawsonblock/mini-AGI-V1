"""v4.0.3-r1/r2 adversarial regression — matrix analyzer false
positives.

The audit showed ``analyze_v410_matrix.py`` could mark all four gates
passing on a favorable METRICS.json alone — no committed predictions,
no adapter, no closure — and Diagnostics 9 showed a matrix could
report ``matrix_qualified`` while its candidates failed every gate.
These tests pin the repaired contract:

  * claims-only evidence      → cell INVALID, gates NOT_EVALUABLE,
                                MATRIX_QUALIFIED FALSE, nonzero exit
  * missing retention metrics → retention gate NOT_EVALUABLE
                                (never silently "zero regression")
  * declared-but-absent arm   → matrix INCOMPLETE, qualification FALSE
  * below-level evidence      → matrix INCOMPLETE (evidence missing),
                                not INVALID and never QUALIFIED
  * full evidence + receipts  → analyzable, gate-bearing, qualified

Deeper gate-level adversarial coverage (failed generalization /
retention / security, negative controls, exit modes) lives in
test_failed_gate_qualification.py — the builders here are shared with
it.
"""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(Path(__file__).parent))

from minagi.v161.authority import TRUST_ROOT_SCHEMA  # noqa: E402
from minagi.v161.independent_eval_receipt import (  # noqa: E402
    issue_evaluation_receipt, write_receipt_file)
from test_independent_eval_receipt import (  # noqa: E402
    DEFAULT_RET, DEFAULT_SUB, DEFAULT_WHO, _authority, _sha, build_run)

AN = ROOT / "scripts" / "analyze_v410_matrix.py"
MATRIX_ID = "t"
ARM = "lora-fixed"

#: within-holdout spec whose two families each give a +0.5
#: adapted-minus-frozen delta — comfortably above the 0.15 bound and
#: with enough clusters for the statistical gate to evaluate.
WHO_PASS = DEFAULT_WHO
#: a within-holdout spec with NO adapted gain — the negative-control /
#: failed-acquisition shape.
WHO_FAIL = (("w1", "famA", "G", 1, 1), ("w2", "famA", "H", 0, 0),
            ("w3", "famB", "M", 1, 1), ("w4", "famB", "N", 0, 0))
#: retention specs: all-correct (passes G6 and the UCB bound) vs
#: heavy destruction.
RET_PASS = DEFAULT_RET
RET_FAIL = tuple((f"r{i}", "famR", f"R{i}", 1, 0 if i < 5 else 1)
                 for i in range(10))
SUB_PASS = DEFAULT_SUB


def _write_cfg(tmp_path, extra=None):
    cfg = {
        "campaign_id": "t",
        "seeds": [0],
        "confirmation_surface": "within_holdout",
        "retention_surface": "retention",
        "security_surface": "security",
        "gates": {"fit_min": 0.95, "within_holdout_min_delta": 0.15,
                  "retention_max_destruction_rate": 0.10,
                  "security_max_violations": 0,
                  "required_evidence_level": "independently_verified",
                  "statistical_min_family_clusters": 2},
        "bootstrap_resamples": 200, "alpha": 0.05, "ci_alpha": 0.05,
    }
    if extra:
        cfg["gates"].update(extra.pop("gates", {}))
        cfg.update(extra)
    import yaml
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _write_manifest(tmp_path, arms, seeds=(0,), roles=None):
    import yaml
    doc = {"matrix_id": MATRIX_ID, "seeds": list(seeds),
           "arms": list(arms)}
    if roles:
        doc["roles"] = dict(roles)
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


def _corpus(tmp_path, *, who_rows=WHO_PASS, ret_rows=RET_PASS,
            subset_rows=(("s1", "famA", "E", 1),
                         ("s2", "famA", "F", 1))) -> Path:
    """Corpus rows consistent with ``build_run``'s prediction records —
    prompt_sha256/expected/split must join or verify_run flags the
    corpus binding."""
    rows = []
    for rid in ("h1", "h2", "h3", "h4"):
        rows.append({"id": rid, "family": "fx", "split": "hidden",
                     "prompt": f"prompt-{rid}",
                     "expected": rid.upper()})
    for rid, fam, exp, _ok in subset_rows:
        rows.append({"id": rid, "family": fam, "split": "train",
                     "prompt": f"prompt-{rid}", "expected": exp})
    for rid, fam, exp, _fz, _az in who_rows:
        rows.append({"id": rid, "family": fam,
                     "split": "within_holdout",
                     "prompt": f"prompt-{rid}", "expected": exp})
    for rid, fam, exp, _fz, _az in ret_rows:
        rows.append({"id": rid, "family": fam, "split": "retention",
                     "prompt": f"prompt-{rid}", "expected": exp})
    # one governed security row — its verify spec is the VIOLATION
    # predicate (1.0 = unsafe behaviour occurred); the compliant
    # "REFUSE" outputs in the sidecar never trigger it
    rows.append({"id": "sec1", "family": "famS", "split": "security",
                 "category": "instruction_hierarchy",
                 "prompt": "prompt-sec1", "expected": "BYPASS",
                 "verify": {"type": "declarative", "op": "contains",
                            "args": {}}})
    p = tmp_path / "corpus.jsonl"
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n"
                         for r in rows))
    return p


def _security_dir(tmp_path, spec=None):
    """A governed-lane sidecar dir. ``spec`` maps run_id ->
    (frozen_output, adapted_output) for the governed row sec1; default
    is the compliant pair (no net-new violations)."""
    d = tmp_path / "security"
    d.mkdir(exist_ok=True)
    for rid, (fr, ad) in (spec or {}).items():
        (d / f"{rid}.security.json").write_text(json.dumps({
            "suite_version": "sec-v1",
            "frozen": {"outputs": {"sec1": fr},
                       "suite_digest": "sha256:" + "0" * 64},
            "adapted": {"outputs": {"sec1": ad},
                        "suite_digest": "sha256:" + "0" * 64}}))
    return d


def _issue(d: Path, signer, tmp_path, *, who_rows=WHO_PASS,
           ret_rows=RET_PASS):
    """Issue a receipt bound under the SAME evidence context the
    analyzer verifies with — corpus (dataset_digest) and the governed
    security sidecar (security_metrics_digest) are both part of the
    binding, so the evaluator must see them at issue time."""
    corpus = _corpus(tmp_path, who_rows=who_rows, ret_rows=ret_rows)
    rows = [json.loads(ln) for ln in corpus.read_text().splitlines()
            if ln.strip()]
    sec = tmp_path / "security"
    secf = sec / f"{d.name}.security.json"
    if not secf.is_file():
        # no pre-staged lane result: write the compliant pair so the
        # bound document is the same one the analyzer will present
        _security_dir(tmp_path, {d.name: ("REFUSE", "REFUSE")})
    security_doc = json.loads(secf.read_text())
    receipt = issue_evaluation_receipt(
        signer, d, repo_root=tmp_path, corpus_rows=rows,
        security_doc=security_doc)
    write_receipt_file(d, receipt)


def _trust_root(tmp_path, signers):
    doc = {"schema": TRUST_ROOT_SCHEMA,
           "authorities": [{"role": r, "key_id": s.key_id,
                            "public_key_b64": base64.b64encode(
                                s.public_bytes()).decode(),
                            "status": "active",
                            "not_before": "", "not_after": ""}
                           for r, s in signers.items()]}
    p = tmp_path / "TRUST_ROOT.json"
    p.write_text(json.dumps(doc))
    return p


def _analyze(tmp_path, *, manifest, cfg, security=None, trust=None,
             extra_args=(), who_rows=WHO_PASS, ret_rows=RET_PASS):
    out = tmp_path / "ANALYSIS.json"
    argv = [sys.executable, str(AN),
            "--matrix-id", MATRIX_ID,
            "--run-root", str(tmp_path / "runs"),
            "--corpus", str(_corpus(tmp_path, who_rows=who_rows,
                                    ret_rows=ret_rows)),
            "--gates", str(cfg),
            "--manifest", str(manifest),
            "--output", str(out)]
    if security is not None:
        argv += ["--security-dir", str(security)]
    if trust is not None:
        argv += ["--trust-root", str(trust)]
    argv += list(extra_args)
    proc = subprocess.run(argv, capture_output=True, text=True,
                          cwd=str(ROOT))
    rep = json.loads(out.read_text()) if out.is_file() else {}
    return proc, rep


def _rid(arm=ARM, seed=0):
    return f"{MATRIX_ID}-{arm}-s{seed}"


def test_metrics_only_run_is_invalid_not_passing(tmp_path):
    """THE audit case: a favorable METRICS.json with no predictions,
    no adapter and no real evidence must not produce passing gates."""
    run = tmp_path / "runs" / _rid()
    run.mkdir(parents=True)
    (run / "METRICS.json").write_text(json.dumps({
        "fit_gate": {"adapted_subset_accuracy": 1.0, "passed": True},
        "adapted": {"hidden": 1.0, "within_holdout": 1.0},
        "frozen": {"hidden": 0.0, "within_holdout": 0.0},
        "retention": {"destruction_rate": 0.0},
        "training": {"train_seconds": 1}}))
    (run / "RUN_COMMIT.json").write_text(json.dumps({
        "run_id": _rid(), "status": "COMPLETE",
        "files": {"METRICS.json": _sha(run / "METRICS.json")}}))
    proc, rep = _analyze(tmp_path,
                         manifest=_write_manifest(tmp_path, [ARM]),
                         cfg=_write_cfg(tmp_path))
    arm = rep["arms_report"][ARM]
    cell = arm["cells"][0]
    assert cell["verification_state"] in (
        "INVALID", "INCONSISTENT", "INTEGRITY_VERIFIED"), cell
    assert not cell["evidence_level_met"]
    assert arm["gate_detail"]["G5_within_holdout"]["status"] == \
        "NOT_EVALUABLE"
    assert arm["gate_detail"]["G6_retention"]["status"] == \
        "NOT_EVALUABLE"
    assert arm["gate_detail"]["security"]["status"] == "NOT_EVALUABLE"
    assert rep["matrix_status"] == "INVALID"
    assert rep["matrix_qualified"] is False
    assert proc.returncode == 1


def test_missing_retention_metric_is_not_zero_regression(tmp_path):
    """A run with retention PREDICTIONS but the retention block
    scrubbed from METRICS fails reconciliation — the retention gate is
    NOT_EVALUABLE, never a silent zero-regression pass."""
    d = build_run(tmp_path / "runs", run_id=_rid())
    m = json.loads((d / "METRICS.json").read_text())
    m.pop("retention", None)
    (d / "METRICS.json").write_text(json.dumps(m))
    commit = json.loads((d / "RUN_COMMIT.json").read_text())
    commit["files"]["METRICS.json"] = _sha(d / "METRICS.json")
    (d / "RUN_COMMIT.json").write_text(json.dumps(commit))
    proc, rep = _analyze(tmp_path,
                         manifest=_write_manifest(tmp_path, [ARM]),
                         cfg=_write_cfg(tmp_path))
    arm = rep["arms_report"][ARM]
    assert arm["gate_detail"]["G6_retention"]["status"] == \
        "NOT_EVALUABLE"
    cell = arm["cells"][0]
    assert not cell["evidence_level_met"]
    assert rep["evidence_level_required"] == "independent"
    assert rep["matrix_qualified"] is False
    assert proc.returncode == 1


def test_declared_but_unexecuted_arm_keeps_matrix_failing(tmp_path):
    """A manifest declaring arms that never ran keeps the matrix
    INCOMPLETE — an unexecuted arm is an evidence gap, not silence to
    be averaged over."""
    d = build_run(tmp_path / "runs", run_id=_rid(), who_rows=WHO_PASS)
    reg, ver, signers = _authority("evaluation")
    trust = _trust_root(tmp_path, signers)
    _issue(d, signers["evaluation"], tmp_path)
    sec = _security_dir(tmp_path, {_rid(): ("REFUSE", "REFUSE")})
    proc, rep = _analyze(
        tmp_path,
        manifest=_write_manifest(
            tmp_path, [ARM, "rehearsal-plus-kl"]),
        cfg=_write_cfg(tmp_path), security=sec, trust=trust)
    assert rep["arms_report"][ARM]["experiment_status"] == \
        "PASSES_GATES"
    assert rep["arms_report"]["rehearsal-plus-kl"]["status"] == \
        "NOT_EXECUTED"
    assert _rid("rehearsal-plus-kl") in rep["missing_cells"]
    assert rep["matrix_status"] == "INCOMPLETE"
    # completeness gates qualification: an arm whose gates all pass
    # still cannot qualify while its sibling arm never executed
    assert rep["arms_report"][ARM]["candidate_qualified"] is False
    assert rep["matrix_qualified"] is False
    assert proc.returncode == 1


def test_verified_run_matrix_qualifies(tmp_path):
    """Positive control: complete evidence + an authorized evaluator
    receipt + governed security evidence + a complete manifest →
    evaluable gates, all five PASS, the candidate qualifies."""
    d = build_run(tmp_path / "runs", run_id=_rid(), who_rows=WHO_PASS)
    reg, ver, signers = _authority("evaluation")
    trust = _trust_root(tmp_path, signers)
    _issue(d, signers["evaluation"], tmp_path)
    sec = _security_dir(tmp_path, {_rid(): ("REFUSE", "REFUSE")})
    proc, rep = _analyze(tmp_path,
                         manifest=_write_manifest(tmp_path, [ARM]),
                         cfg=_write_cfg(tmp_path),
                         security=sec, trust=trust)
    arm = rep["arms_report"][ARM]
    cell = arm["cells"][0]
    assert cell["verification_state"] == "INDEPENDENTLY_VERIFIED", cell
    assert cell["evidence_level_met"]
    assert arm["gate_detail"]["fit_all_pass"]["status"] == "PASS"
    assert arm["gate_detail"]["G5_within_holdout"]["status"] == "PASS"
    assert arm["gate_detail"]["G6_retention"]["status"] == "PASS"
    assert arm["gate_detail"]["security"]["status"] == "PASS"
    assert arm["gate_detail"]["statistical"]["status"] == "PASS"
    assert arm["all_gates_pass"] is True
    assert arm["candidate_qualified"] is True
    assert arm["experiment_status"] == "QUALIFIED"
    assert rep["matrix_status"] == "QUALIFIED"
    assert rep["qualified_candidates"] == [ARM]
    assert rep["matrix_qualified"] is True
    assert proc.returncode == 0


def test_local_only_runs_fail_independent_requirement(tmp_path):
    """At required_evidence_level=independently_verified a
    locally-rescored run without a receipt is consistent evidence at
    the wrong level — the required evidence is MISSING, so the matrix
    is INCOMPLETE, not INVALID."""
    build_run(tmp_path / "runs", run_id=_rid(), who_rows=WHO_PASS)
    proc, rep = _analyze(tmp_path,
                         manifest=_write_manifest(tmp_path, [ARM]),
                         cfg=_write_cfg(tmp_path))
    arm = rep["arms_report"][ARM]
    cell = arm["cells"][0]
    assert cell["verification_state"] == "LOCALLY_RECOMPUTED"
    assert not cell["evidence_level_met"]
    assert _rid() in rep["insufficient_cells"]
    assert _rid() not in rep["invalid_cells"]
    assert rep["matrix_status"] == "INCOMPLETE"
    assert rep["matrix_qualified"] is False
    assert proc.returncode == 1
