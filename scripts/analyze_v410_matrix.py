#!/usr/bin/env python3
"""v4.1.0 confirmatory analysis — gate evaluation for the intervention
matrix (G5 generalization, G6 retention, security), v4.0.3-r1 evidence
closure, v4.0.3-r2 qualification semantics.

Reads committed run directories produced by capability_matrix.py across
the preregistered seed set, joins prediction records to the corpus for
family labels, and evaluates the preregistered gates declared in
configs/capability_acquisition/confirmation.yaml:

  * fit_min                       — adapted subset (fit-gate) accuracy
  * within_holdout_min_delta      — G5: adapted-frozen on the
                                    INDEPENDENT within-family surface,
                                    per-seed mean over family clusters
  * retention_max_destruction_rate— G6: share of previously-correct
                                    retention probes destroyed
  * security_max_violations       — frozen-vs-adapted security
                                    regressions
  * required_evidence_level       — the evidence level every cell must
                                    reach (integrity|local|independent|
                                    qualified)

EVIDENCE CLOSURE (v4.0.3-r1) — before any gate evaluates a cell, the
cell's evidence is closed:

  * a valid committed run inventory (RUN_COMMIT.json + hashes);
  * exact model/adapter identity bindings, scorer identity, dataset
    and protocol digests;
  * every expected prediction file present with complete, unique ids;
  * every gate input INDEPENDENTLY RESCORED from committed predictions
    (runner-asserted METRICS fields are never consumed);
  * paired per-row security evidence derived from the governed lane
    sidecar;
  * the configured evidence level: ``independent`` requires a signed
    independent-evaluation-v1 receipt from an evaluation-role
    authority, ``qualified`` additionally requires a signed
    qualification decision.

A cell that fails closure is INVALID — its gates are NOT_EVALUABLE,
never PASS. Each metric has three states: measured, explicitly
optional/not-applicable, or missing required evidence; a gate that
needs a missing value is NOT_EVALUABLE. Missing retention evidence is
NOT a zero regression, and an undefined confidence interval is not
passing statistical evidence.

MATRIX STATUS vs CANDIDATE QUALIFICATION (v4.0.3-r2) — completeness of
the research record and qualification of a candidate are DISTINCT:

  INCOMPLETE              required cells never executed, or present
                          cells lack the required evidence level
                          (e.g. no independent receipt)
  INVALID                 evidence, authorization or experiment
                          identity is inconsistent — failed closure,
                          refused receipt/qualification documents,
                          cross-cell protocol identity mismatch
  COMPLETE                every required record present and verified;
                          no candidate arm has a failed required gate
                          (a matrix of only controls ends here)
  COMPLETE_WITH_FAILURES  every required record present and verified;
                          >=1 candidate arm failed a required gate
  QUALIFIED               >=1 candidate arm satisfies every applicable
                          scientific and authorization gate

  matrix_complete  = all required runs present + verified at the
                     required evidence level + consistent protocol
                     identities
  candidate_qualified[arm] = matrix_complete AND all the arm's cells
                     authorized at the required level AND fit, G5, G6,
                     security AND statistical gates all PASS

  Arm roles come from the manifest (``roles: {arm: role}`` or
  ``arms: [{name, role}, ...]``) or --control-arms. Roles:
  ``intervention`` (a candidate — the default), ``baseline`` and
  ``negative_control`` (controls: their evidence must verify, but a
  failed acquisition gate is EXPECTED evidence, not a qualification
  failure — a negative control that passes is flagged as a
  control-integrity anomaly, not a success).

EXIT MODES — default is strict completeness:
  default / --strict-completeness   exit nonzero unless the matrix is
                                    COMPLETE, COMPLETE_WITH_FAILURES
                                    or QUALIFIED (a negative result is
                                    a legitimate finding; missing or
                                    inconsistent evidence is not)
  --informational                   report-only: exit 0 whenever the
                                    analysis produced a report
  --require-qualified [ARM]         promotion check: exit nonzero
                                    unless a candidate qualifies (the
                                    named arm, if given; otherwise any
                                    candidate)

Usage:
    python scripts/analyze_v410_matrix.py \
        --matrix-id v410-4a --run-root results/campaign4a/matrix/runs \
        --corpus configs/campaign4a_tasks.jsonl \
        --gates configs/capability_acquisition/confirmation.yaml \
        --manifest configs/capability_acquisition/matrix_manifest.yaml \
        --trust-root path/to/TRUST_ROOT.json \
        --security-dir results/campaign4a/security
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.qualified_outcome import (  # noqa: E402
    QUALIFICATION_FILE, QUALIFIED, STATE_INDEPENDENT, STATE_LOCAL,
    STATE_QUALIFIED, STATE_RANK, derive_qualified_outcome)
from minagi.v161.independent_eval_receipt import (  # noqa: E402
    RECEIPT_FILE)

#: Evidence levels a matrix cell can be required to reach.
LEVEL_RANK = {"integrity": 1, "local": 2, "independent": 3,
              "qualified": 4}
_LEVEL_STATE = {1: "INTEGRITY_VERIFIED", 2: STATE_LOCAL,
                3: STATE_INDEPENDENT, 4: STATE_QUALIFIED}
_LEVEL_ALIASES = {
    "integrity_verified": "integrity",
    "locally_recomputed": "local",
    "independently_verified": "independent",
    "evaluation_verified": "independent",   # deprecated v4.0.3 name
}


def _level_of(name: str) -> str | None:
    n = str(name).strip().lower()
    return n if n in LEVEL_RANK else _LEVEL_ALIASES.get(n)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines()
            if ln.strip()]


def _family_map(corpus: Path) -> dict[str, str]:
    return {str(r["id"]): str(r["family"])
            for r in _load_jsonl(corpus)}


def _governed_security_rows(corpus: Path) -> list[dict]:
    """Security-partition rows carrying a governed probe (a declared
    safety category + declarative verify spec). Smoke lures carry no
    category and are not governed evidence — they are excluded."""
    from minagi.v161.safety_regression import (
        SECURITY_PARTITION_CATEGORIES, category_of)
    out = []
    for r in _load_jsonl(corpus):
        if r.get("split") != "security":
            continue
        if category_of(r) in SECURITY_PARTITION_CATEGORIES:
            out.append(r)
    return out


def _security_block(lane: dict | None,
                    governed_rows: list[dict]) -> dict:
    """Derive the paired security regression from the sidecar's stored
    per-row outputs — the gate quantity is recomputed from evidence,
    never taken from the lane's rollup fields. A "regression" is a
    governed probe the ADAPTED model violates that the FROZEN model
    did not violate on the same row; the frozen baseline's own
    violations are the reference point, not a failing grade.

    Fail-closed: absent sidecar, missing outputs, or any governed row
    lacking a paired output leaves the block ``evaluated: False``."""
    from minagi.v161.safety_regression import score_probe
    if not isinstance(lane, dict) or not governed_rows:
        return {"evaluated": False}
    fr_out = (lane.get("frozen") or {}).get("outputs")
    ad_out = (lane.get("adapted") or {}).get("outputs")
    if not isinstance(fr_out, dict) or not isinstance(ad_out, dict):
        return {"evaluated": False}
    regressed, repaired, missing = [], [], []
    frozen_viol = adapted_viol = 0
    for row in governed_rows:
        rid = str(row["id"])
        if rid not in fr_out or rid not in ad_out:
            missing.append(rid)
            continue
        fv = score_probe(row, str(fr_out[rid])) >= 1.0
        av = score_probe(row, str(ad_out[rid])) >= 1.0
        frozen_viol += int(fv)
        adapted_viol += int(av)
        if av and not fv:
            regressed.append(rid)
        if fv and not av:
            repaired.append(rid)
    if missing:
        return {"evaluated": False,
                "unpaired_rows": sorted(missing),
                "n_governed_rows": len(governed_rows)}
    return {"evaluated": True,
            "regressions": len(regressed),
            "regressed_rows": sorted(regressed),
            "repaired_rows": sorted(repaired),
            "adapted_violations": adapted_viol,
            "frozen_violations": frozen_viol,
            "n_governed_rows": len(governed_rows),
            "regression_basis": "paired_per_row",
            "suite_version": lane.get("suite_version"),
            "suite_digest": (lane.get("adapted") or {}).get("suite_digest")}


def _correct_map(run: Path, fname: str) -> dict[str, int]:
    """id -> 0/1 correctness, INDEPENDENTLY rescored from the committed
    prediction file under the strict oracle (exact-match-v2) — the
    records carry `output` (raw generation), not a precomputed score."""
    from minagi.v161.evaluators import exact_match_strict
    p = run / fname
    if not p.is_file():
        return {}
    out = {}
    for r in _load_jsonl(p):
        pred = str(r.get("output", r.get("prediction", "")))
        exp = str(r.get("expected", ""))
        out[str(r["id"])] = int(exact_match_strict(pred, exp))
    return out


def _by_family(scores: dict[str, int],
               fams: dict[str, str]) -> dict[str, list[int]]:
    per: dict[str, list[int]] = {}
    for rid, s in scores.items():
        per.setdefault(fams.get(rid, "?"), []).append(s)
    return per


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _destruction_rate(frozen: dict[str, int],
                      adapted: dict[str, int]) -> float | None:
    """Share of previously-correct answers the adapted model destroyed,
    rescored per id. None when there is nothing to measure — a missing
    denominator is not a zero regression."""
    if not frozen or not adapted:
        return None
    n_fc = sum(1 for v in frozen.values() if v == 1)
    if not n_fc:
        return None
    destroyed = sum(1 for rid, v in frozen.items()
                    if v == 1 and adapted.get(rid) == 0)
    return destroyed / n_fc


def cluster_bootstrap_delta(
        frozen: dict[str, list[int]],
        adapted: dict[str, list[int]], *, resamples: int, alpha: float,
        rng: random.Random) -> dict:
    """Bootstrap the adapted-minus-frozen within-family accuracy by
    resampling FAMILIES (clusters) with replacement."""
    fams = sorted(set(frozen) | set(adapted))
    if not fams:
        return {"delta": None, "ci_low": None, "ci_high": None}

    def mean_acc(m: dict[str, list[int]], keys) -> float:
        vals = [s for k in keys for s in m.get(k, [])]
        return sum(vals) / len(vals) if vals else 0.0

    point = mean_acc(adapted, fams) - mean_acc(frozen, fams)
    boots = []
    for _ in range(resamples):
        keys = [rng.choice(fams) for _ in fams]
        boots.append(mean_acc(adapted, keys) - mean_acc(frozen, keys))
    boots.sort()
    lo = boots[int((alpha / 2) * len(boots))]
    hi = boots[min(len(boots) - 1,
                   int((1 - alpha / 2) * len(boots)) - 1)]
    return {"delta": round(point, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "resamples": resamples, "clusters": len(fams)}


def bootstrap_destruction_rate(
        pairs: list[tuple[int, int]], *,
        resamples: int, alpha: float, rng: random.Random) -> dict:
    """Item-level bootstrap of the retention-destruction rate over
    paired (frozen, adapted) correctness on previously-correct rows —
    retention probes carry no family clustering, so rows are resampled
    directly. Returns the point estimate and a two-sided interval; the
    non-inferiority check uses the UPPER bound (the destruction rate
    must stay under the declared maximum even at the pessimistic end
    of its uncertainty)."""
    fc = [p for p in pairs if p[0] == 1]
    if not fc:
        return {"rate": None, "ci_low": None, "ci_high": None, "n": 0}

    def rate(ps) -> float:
        return sum(1 for _, a in ps if a == 0) / len(ps)

    point = rate(fc)
    boots = sorted(rate([rng.choice(fc) for _ in fc])
                   for _ in range(resamples))
    lo = boots[int((alpha / 2) * len(boots))]
    hi = boots[min(len(boots) - 1,
                   int((1 - alpha / 2) * len(boots)) - 1)]
    return {"rate": round(point, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "resamples": resamples, "n": len(fc)}


def run_summary(run: Path, fams: dict[str, str],
                security_dir: Path | None = None,
                governed_rows: list[dict] | None = None, *,
                repo_root=None, corpus_rows=None,
                verifier=None, authority=None) -> dict | None:
    """Evidence-closed per-cell summary.

    Returns None only when the run directory does not exist at all
    (a MISSING cell). Everything else — committed inventory, rescore
    reconciliation, receipt/qualification authority — comes back in
    ``evidence``. Gate inputs are rescore-derived; a cell that did not
    reach LOCALLY_RECOMPUTED carries no evaluable gate value."""
    run = Path(run)
    if not run.is_dir():
        return None
    m = {}
    mpath = run / "METRICS.json"
    if mpath.is_file():
        try:
            m = json.loads(mpath.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            m = {}
    # The governed security lane writes a sidecar report (it is not part
    # of the run's commit inventory — scoring happens after commit and
    # must not mutate committed evidence). The paired regression is
    # DERIVED here from the sidecar's stored outputs — never taken from
    # the lane's rollup fields, and a missing/unpaired block fails
    # closed, never silently passes. When the evaluator bound the same
    # sidecar into the receipt's security_metrics_digest, the document
    # is presented to the binding check as well.
    lane = None
    if security_dir is not None:
        sp = security_dir / f"{run.name}.security.json"
        if sp.is_file():
            try:
                lane = json.loads(sp.read_text())
            except (OSError, json.JSONDecodeError):
                lane = None
    sec = _security_block(lane, governed_rows or []) \
        if security_dir is not None else {"evaluated": False}
    oc = derive_qualified_outcome(run, verifier=verifier,
                                  authority=authority,
                                  repo_root=repo_root,
                                  corpus_rows=corpus_rows,
                                  security_doc=lane)
    fr_who = _correct_map(run, "PREDICTIONS-FROZEN-WITHIN_HOLDOUT.jsonl")
    ad_who = _correct_map(run, "PREDICTIONS-ADAPTED-WITHIN_HOLDOUT.jsonl")
    fr_ret = _correct_map(run, "PREDICTIONS-FROZEN-RETENTION.jsonl")
    ad_ret = _correct_map(run, "PREDICTIONS-ADAPTED-RETENTION.jsonl")
    subset = _correct_map(run, "PREDICTIONS-SUBSET.jsonl")
    who_delta = None
    if fr_who and ad_who:
        who_delta = (sum(ad_who.values()) / len(ad_who)
                     - sum(fr_who.values()) / len(fr_who))
    # Post-commit evidence that is PRESENT but refused is not merely
    # absent evidence — a claimed attestation that fails verification
    # makes the cell's authorization inconsistent (INVALID), whereas a
    # decided REJECTED qualification is verified negative evidence.
    receipt_refused = (
        (run / RECEIPT_FILE).is_file()
        and oc.evaluation_receipt_digest is None)
    qualification_decision = "none"
    qualification_invalid = False
    qpath = run / QUALIFICATION_FILE
    if qpath.is_file():
        try:
            qdoc = json.loads(qpath.read_text())
            dec = str(qdoc.get("decision") or "")
            if oc.qualification_status == QUALIFIED:
                qualification_decision = "QUALIFIED"
            elif dec == "REJECTED":
                qualification_decision = "REJECTED"
            else:
                qualification_invalid = True
        except (json.JSONDecodeError, UnicodeDecodeError):
            qualification_invalid = True
    # Protocol identities — from CONFIG.json, which the commit
    # inventory hash-verifies; the matrix requires these consistent
    # across every cell.
    cfg = {}
    cpath = run / "CONFIG.json"
    if cpath.is_file():
        try:
            cfg = json.loads(cpath.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            cfg = {}
    val = m.get("validation") or {}
    who = m.get("within_holdout") or {}
    ret = m.get("retention") or {}
    return {
        "run_id": run.name,
        "verification_state": oc.verification_state,
        "evaluation_status": oc.evaluation_status,
        "qualification_status": oc.qualification_status,
        "qualification_decision": qualification_decision,
        "qualification_invalid": qualification_invalid,
        "receipt_refused": receipt_refused,
        "evidence_reasons": list(oc.reasons[:8]),
        "identities": {
            "model_id": str(cfg.get("model_id") or oc.model_id),
            "model_revision": str(cfg.get("model_revision")
                                  or oc.model_revision),
            "score_oracle": str(cfg.get("score_oracle")
                                or oc.score_oracle),
            "corpus": str(cfg.get("corpus") or ""),
        },
        # ---- rescore-derived gate inputs (never METRICS-asserted) ---
        "fit_subset_rescore": _mean(subset.values()),
        "within_holdout_delta_rescore": who_delta,
        "retention_destruction_rescore":
            _destruction_rate(fr_ret, ad_ret),
        "retention_n_frozen_correct":
            sum(1 for v in fr_ret.values() if v == 1) if fr_ret else None,
        "security": sec if isinstance(sec, dict) else {},
        # ---- reported metrics, audit only — gates never read these ---
        "reported": {
            "fit_gate_passed":
                bool((m.get("fit_gate") or {}).get("passed")),
            "within_holdout": {"frozen": who.get("frozen"),
                               "adapted": who.get("adapted")},
            "retention": {"destruction_rate":
                          ret.get("destruction_rate")},
            "validation": {"frozen": val.get("frozen"),
                           "adapted": val.get("adapted")},
        },
        "train_seconds": (m.get("training") or {}).get("train_seconds"),
        "_families": {"frozen_who": _by_family(fr_who, fams),
                      "adapted_who": _by_family(ad_who, fams)},
        "_retention_pairs": (fr_ret, ad_ret)}


def _gate(value, op: str, bound, *, evaluable: bool) -> dict:
    """Three-state gate verdict. ``value`` None means the measurement
    was never taken — that is missing evidence, NOT a zero and NOT a
    pass."""
    if not evaluable:
        return {"status": "NOT_EVALUABLE", "value": None}
    if value is None:
        return {"status": "NOT_EVALUABLE", "value": None}
    ok = value >= bound if op == "min" else value <= bound
    return {"status": "PASS" if ok else "FAIL", "value": value}


def _bool_of(status: str):
    return {"PASS": True, "FAIL": False}.get(status)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-id", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gates",
                    default="configs/capability_acquisition/"
                            "confirmation.yaml")
    ap.add_argument("--manifest", default=None,
                    help="experiment manifest (YAML or JSON) declaring "
                         "the expected seeds x arms; when given it "
                         "overrides --seeds/--arms")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--arms", default="fixed,rehearsal,kl,both,shuffled")
    ap.add_argument("--security-dir", default=None,
                    help="dir of <run_id>.security.json sidecars from "
                         "run_security_lane.py; absent => security gate "
                         "stays NOT_EVALUABLE (fail-closed)")
    ap.add_argument("--evidence-level",
                    choices=sorted(LEVEL_RANK), default=None,
                    help="minimum evidence level every cell must reach "
                         "(default: gates.required_evidence_level, "
                         "else 'local')")
    ap.add_argument("--trust-root", default=None,
                    help="TRUST_ROOT.json — required for evidence "
                         "levels independent/qualified (receipt and "
                         "qualification signatures verify against it)")
    ap.add_argument("--screened-arms", default="",
                    help="comma-separated arm names declared screened/"
                         "deferred by design (dev-lane only — never for "
                         "the confirmation lane); screened cells are "
                         "labelled SCREENED, not silently treated as "
                         "run")
    ap.add_argument("--screened-reason", default=None,
                    help="mandatory rationale when --screened-arms is "
                         "used; recorded verbatim in the report")
    ap.add_argument("--control-arms", default="",
                    help="comma-separated arms declared negative "
                         "controls — their acquisition-gate failures "
                         "are expected evidence, never candidate "
                         "failures (manifest ``roles:`` overrides)")
    ap.add_argument("--informational", action="store_true",
                    help="report-only mode: exit 0 whenever the "
                         "analysis produced a report, even when the "
                         "matrix is incomplete/invalid")
    ap.add_argument("--strict-completeness", action="store_true",
                    help="(default) exit nonzero unless every declared "
                         "cell is present, consistent and verified at "
                         "the required evidence level")
    ap.add_argument("--require-qualified", nargs="?", const="__any__",
                    default=None, metavar="ARM",
                    help="promotion check: exit nonzero unless a "
                         "candidate qualifies — the named arm if "
                         "given, otherwise any candidate arm; implies "
                         "--strict-completeness")
    ap.add_argument("--statistical-policy", default=None,
                    help="statistical_policy.yaml — the versioned "
                         "statistical contract; its gates: block "
                         "overrides confirmation.yaml's statistical "
                         "keys so method+thresholds live in one place")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import yaml
    gcfg = yaml.safe_load((ROOT / args.gates).read_text())
    gates = gcfg.get("gates") or {}
    stat_policy = None
    if args.statistical_policy:
        stat_policy = yaml.safe_load(
            (ROOT / args.statistical_policy).read_text())
        # the versioned policy wins for statistical keys — confirmation
        # cannot silently relax the preregistered statistical contract
        gates.update(stat_policy.get("gates") or {})
    corpus_path = (ROOT / args.corpus
                   if not Path(args.corpus).is_absolute()
                   else Path(args.corpus))
    fams = _family_map(corpus_path)
    governed_rows = _governed_security_rows(corpus_path)
    corpus_rows = _load_jsonl(corpus_path)
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = ROOT / run_root

    # ---- expected matrix shape: manifest wins over CLI declarations --
    manifest_doc = None
    if args.manifest:
        mpath = Path(args.manifest)
        if not mpath.is_absolute():
            mpath = ROOT / mpath
        manifest_doc = yaml.safe_load(mpath.read_text())
        if not isinstance(manifest_doc, dict):
            print("experiment manifest is not a mapping",
                  file=sys.stderr)
            return 2
        if manifest_doc.get("matrix_id") \
                and str(manifest_doc["matrix_id"]) != args.matrix_id:
            print(f"manifest matrix_id "
                  f"{manifest_doc['matrix_id']!r} != --matrix-id "
                  f"{args.matrix_id!r}", file=sys.stderr)
            return 2
        seeds = [int(s) for s in manifest_doc["seeds"]]
        # Arms may be plain names or {name, role} mappings; a top-level
        # ``roles: {arm: role}`` map also works.
        roles = {str(k): str(v)
                 for k, v in (manifest_doc.get("roles") or {}).items()}
        arms = []
        for a in manifest_doc["arms"]:
            if isinstance(a, dict):
                arms.append(str(a["name"]))
                if a.get("role"):
                    roles[str(a["name"])] = str(a["role"])
            else:
                arms.append(str(a))
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        roles = {}
    for c in args.control_arms.split(","):
        c = c.strip()
        if c:
            roles.setdefault(c, "negative_control")
    unknown_role_arms = [a for a in roles if a not in arms]
    if unknown_role_arms:
        print(f"role declared for undeclared arms: "
              f"{sorted(unknown_role_arms)}", file=sys.stderr)
        return 2
    valid_roles = {"intervention", "baseline", "negative_control",
                   "control"}
    bad_roles = {a: r for a, r in roles.items()
                 if r not in valid_roles}
    if bad_roles:
        print(f"unknown arm roles {bad_roles} — allowed: "
              f"{sorted(valid_roles)}", file=sys.stderr)
        return 2
    role_of = {a: roles.get(a, "intervention") for a in arms}
    screened = {a.strip() for a in args.screened_arms.split(",")
                if a.strip()}
    if screened and not args.screened_reason:
        print("--screened-arms requires --screened-reason: a screened "
              "cell is only honest if the record says why",
              file=sys.stderr)
        return 2
    unknown_screened = screened - set(arms)
    if unknown_screened:
        print(f"screened arms not declared in --arms: "
              f"{sorted(unknown_screened)}", file=sys.stderr)
        return 2

    required_level = _level_of(
        args.evidence_level
        or gates.get("required_evidence_level") or "local")
    if required_level is None:
        print(f"unknown evidence level "
              f"{args.evidence_level or gates.get('required_evidence_level')!r}",
              file=sys.stderr)
        return 2
    required_rank = LEVEL_RANK[required_level]
    evaluable_rank = max(required_rank, LEVEL_RANK["local"])

    verifier = None
    authority = None
    if args.trust_root:
        from minagi.v161.evaluator_authority import load_trust_root
        authority, verifier = load_trust_root(
            (ROOT / args.trust_root) if not Path(
                args.trust_root).is_absolute()
            else args.trust_root)
    elif required_rank >= LEVEL_RANK["independent"]:
        print("note: --trust-root absent — receipts cannot verify; "
              f"cells cannot reach {required_level!r} and will be "
              "INSUFFICIENT_EVIDENCE", file=sys.stderr)

    rng = random.Random(20241010)
    sec_dir = (Path(args.security_dir) if args.security_dir else None)
    if sec_dir is not None and not sec_dir.is_absolute():
        sec_dir = ROOT / sec_dir

    matrix: dict[str, dict] = {}
    missing: list[str] = []
    invalid: list[str] = []
    insufficient: list[str] = []
    rejected: list[str] = []
    identity_sets: dict[tuple, list[str]] = {}
    for arm in arms:
        cells = []
        for seed in seeds:
            rid = f"{args.matrix_id}-{arm}-s{seed}"
            s = run_summary(run_root / rid, fams, sec_dir,
                            governed_rows, repo_root=ROOT,
                            corpus_rows=corpus_rows,
                            verifier=verifier, authority=authority)
            if s is None:
                missing.append(rid)
                continue
            rank = STATE_RANK.get(s["verification_state"], 0)
            s["evidence_rank"] = rank
            s["evidence_level_met"] = rank >= required_rank
            s["evaluable"] = rank >= evaluable_rank
            # INVALID = evidence/authorization/identity inconsistent:
            # anything that did not reach LOCALLY_RECOMPUTED (the
            # committed record contradicts itself — missing required
            # files, hash breaks, rescore mismatch), or a
            # present-but-refused receipt/qualification document (a
            # claimed attestation that does not verify is not merely
            # missing evidence).
            s["cell_invalid"] = (rank < STATE_RANK[STATE_LOCAL]
                                 or s["receipt_refused"]
                                 or s["qualification_invalid"])
            cells.append(s)
            if s["cell_invalid"]:
                invalid.append(rid)
            elif s["qualification_decision"] == "REJECTED":
                # a signed REJECTED decision is complete evidence with
                # a decided negative outcome — not missing evidence.
                rejected.append(rid)
            elif not s["evidence_level_met"]:
                insufficient.append(rid)
            ids = s.get("identities") or {}
            if ids.get("model_id"):
                key = (ids.get("model_id"), ids.get("model_revision"),
                       ids.get("score_oracle"), ids.get("corpus"))
                identity_sets.setdefault(key, []).append(rid)
        matrix[arm] = {"seeds": cells, "n_seeds": len(cells)}
    # Every cell must agree on the model/revision/oracle/corpus
    # identities — a matrix that mixes protocols is INVALID, not a
    # comparison.
    protocol_consistent = len(identity_sets) <= 1

    def _arm_of(rid: str) -> str:
        return rid[len(args.matrix_id) + 1:].rsplit("-s", 1)[0]
    unscreened_missing = [rid for rid in missing
                          if _arm_of(rid) not in screened]

    # Completeness of the RECORD is independent of outcomes: every
    # declared cell present, consistent, verified at the required
    # evidence level, on a single protocol identity.
    matrix_complete = (not missing and not invalid and not insufficient
                       and protocol_consistent)

    # ---- per-arm aggregate + gates ------------------------------------
    report: dict[str, dict] = {}
    qualified_arms: list[str] = []
    failed_arms: list[str] = []
    delta_min = float(gates.get("within_holdout_min_delta", 0.0))
    ret_max = float(gates.get("retention_max_destruction_rate", 0.0))
    min_clusters = int(gates.get("statistical_min_family_clusters", 2))
    min_seeds = int(gates.get("statistical_min_seeds", 1))
    for arm, blk in matrix.items():
        cells = blk["seeds"]
        role = role_of.get(arm, "intervention")
        is_candidate = role == "intervention"
        if not cells:
            report[arm] = (
                {"status": "SCREENED",
                 "role": role,
                 "reason": args.screened_reason,
                 "note": "declared screened/deferred by design — this "
                         "arm produced no evidence and counts for "
                         "nothing"}
                if arm in screened else
                {"status": "NOT_EXECUTED", "role": role})
            continue
        n_expected = len(seeds)
        n_missing = n_expected - len(cells)
        evaluable = all(c["evaluable"] for c in cells) \
            and n_missing == 0
        level_met = all(c["evidence_level_met"] for c in cells) \
            and n_missing == 0
        arm_invalid = any(c["cell_invalid"] for c in cells)
        arm_rejected = any(c["qualification_decision"] == "REJECTED"
                           for c in cells)

        # ---- rescore-derived per-seed gate inputs --------------------
        fit_vals = [c["fit_subset_rescore"] for c in cells]
        who_deltas = [c["within_holdout_delta_rescore"] for c in cells]
        ret_dest = [c["retention_destruction_rescore"] for c in cells]
        # pooled family-cluster bootstrap over all seeds' within_holdout
        fr: dict[str, list[int]] = {}
        ad: dict[str, list[int]] = {}
        for c in cells:
            for k, v in c["_families"]["frozen_who"].items():
                fr.setdefault(k, []).extend(v)
            for k, v in c["_families"]["adapted_who"].items():
                ad.setdefault(k, []).extend(v)
        boot = cluster_bootstrap_delta(
            fr, ad,
            resamples=int(gates.get("bootstrap_resamples", 2000)),
            alpha=float(gates.get("ci_alpha", 0.05)), rng=rng)
        # Retention uncertainty: pooled per-id bootstrap over the
        # frozen-correct retention pairs across this arm's seeds.
        ret_pairs = []
        for c in cells:
            fr_ret, ad_ret = c["_retention_pairs"]
            ret_pairs.extend((f, ad_ret[rid])
                             for rid, f in fr_ret.items()
                             if rid in ad_ret)
        ret_ci = bootstrap_destruction_rate(
            ret_pairs,
            resamples=int(gates.get("bootstrap_resamples", 2000)),
            alpha=float(gates.get("ci_alpha", 0.05)), rng=rng)
        # Security is DERIVED from the governed lane's stored per-row
        # outputs (paired adapted-vs-frozen rescoring), not from any
        # lane- or runner-asserted rollup — and a missing/unpaired
        # block fails closed, never silently passes.
        sec_vals = []
        sec_complete = evaluable
        for c in cells:
            sv = c["security"]
            v = (sv.get("regressions") if isinstance(sv, dict)
                 and sv.get("evaluated") else None)
            if v is None:
                sec_complete = False
            else:
                sec_vals.append(int(v))
        sec_viol = sum(sec_vals) if sec_complete and cells else None
        mean_delta = _mean(who_deltas)
        fit_min = float(gates.get("fit_min", 0.0))
        fit_ok = all(v is not None and v >= fit_min for v in fit_vals)

        # Statistical gate (Phase 4 hooks): acquisition requires the
        # cluster-bootstrap LCB to EXCEED the preregistered meaningful
        # delta — a point estimate alone cannot qualify an uncertain
        # experiment — and retention is non-inferiority: the UCB of the
        # destruction rate must stay inside the declared maximum.
        # A degenerate interval (too few family clusters, no
        # frozen-correct retention pairs) is NOT_EVALUABLE, never a
        # pass.
        acq_lcb_ok = (boot.get("ci_low") is not None
                      and boot.get("clusters", 0) >= min_clusters
                      and boot["ci_low"] > delta_min)
        ret_ucb_ok = (ret_ci.get("ci_high") is not None
                      and ret_ci["ci_high"] <= ret_max)
        if not evaluable:
            stat_status = "NOT_EVALUABLE"
            stat_reason = "cells not evaluable"
        elif len(cells) < min_seeds:
            stat_status = "NOT_EVALUABLE"
            stat_reason = (f"insufficient_independent_seeds: "
                           f"{len(cells)} seed(s) < {min_seeds} "
                           "required — one seed's interval is not "
                           "independent replication evidence")
        elif boot.get("clusters", 0) < min_clusters:
            stat_status = "NOT_EVALUABLE"
            stat_reason = (f"insufficient_family_coverage: "
                           f"{boot.get('clusters', 0)} cluster(s) < "
                           f"{min_clusters} required — a one-family "
                           "interval is not family-level evidence")
        elif ret_ci.get("ci_high") is None:
            stat_status = "NOT_EVALUABLE"
            stat_reason = ("retention destruction interval undefined "
                           "(no paired frozen-correct probes)")
        else:
            stat_status = "PASS" if (acq_lcb_ok and ret_ucb_ok) \
                else "FAIL"
            stat_reason = ""
        gate_detail = {
            "fit_all_pass": _gate(
                min((v for v in fit_vals if v is not None), default=None),
                "min", fit_min, evaluable=evaluable),
            "G5_within_holdout": _gate(
                mean_delta, "min",
                float(gates.get("within_holdout_min_delta", 0.0)),
                evaluable=evaluable),
            "G6_retention": _gate(
                max((v for v in ret_dest if v is not None), default=None),
                "max",
                float(gates.get("retention_max_destruction_rate", 0.0)),
                evaluable=evaluable
                and all(v is not None for v in ret_dest)),
            "security": _gate(
                sec_viol, "max",
                int(gates.get("security_max_violations", 0)),
                evaluable=sec_viol is not None),
            "statistical": {
                "status": stat_status,
                "reason": stat_reason or None,
                "acquisition_delta_lcb": boot.get("ci_low"),
                "acquisition_delta_min": delta_min,
                "acquisition_family_clusters": boot.get("clusters"),
                "min_family_clusters": min_clusters,
                "retention_destruction_ucb": ret_ci.get("ci_high"),
                "retention_destruction_max": ret_max,
                "retention_n_frozen_correct": ret_ci.get("n"),
            },
        }
        gates_bool = {k: _bool_of(g["status"])
                      for k, g in gate_detail.items()}
        all_pass = all(g["status"] == "PASS"
                       for g in gate_detail.values())

        # ---- arm status: completeness is orthogonal to outcomes -----
        if n_missing:
            cell_state = "INCOMPLETE"
        elif arm_invalid:
            cell_state = "INVALID"
        elif arm_rejected:
            # a signed REJECTED decision is a complete, verified record
            # with a decided negative outcome
            cell_state = "REJECTED"
        elif not level_met:
            cell_state = "INSUFFICIENT_EVIDENCE"
        elif not is_candidate:
            cell_state = "COMPLETE"
        elif all_pass and matrix_complete:
            cell_state = "QUALIFIED"
        elif all_pass:
            # every gate passed but the matrix is not complete — an
            # arm cannot qualify standing on a partial program
            cell_state = "PASSES_GATES"
        else:
            cell_state = "COMPLETE_WITH_FAILURES"

        # A negative control whose acquisition gate shows a
        # statistically supported gain is an integrity anomaly — a
        # leak or a false positive — never a "successful control".
        control_integrity = None
        if role in ("negative_control", "control") and level_met:
            control_integrity = (
                "unexpected_gain" if acq_lcb_ok else "clean")

        candidate_qualified = bool(
            is_candidate and matrix_complete and level_met
            and not arm_invalid and not arm_rejected and all_pass)
        if is_candidate:
            if candidate_qualified:
                qualified_arms.append(arm)
            elif cell_state in ("COMPLETE_WITH_FAILURES", "REJECTED",
                                "PASSES_GATES"):
                failed_arms.append(arm)

        verdict = {
            "experiment_status": cell_state,
            "role": role,
            "candidate_qualified": candidate_qualified,
            "control_integrity": control_integrity,
            "qualification_decision":
                ("REJECTED" if arm_rejected else
                 cells[0].get("qualification_decision")),
            "n_seeds": len(cells),
            "n_missing_cells": n_missing,
            "evidence_level_met": level_met,
            "within_holdout_delta_per_seed":
                [round(d, 4) if d is not None else None
                 for d in who_deltas],
            "within_holdout_delta_mean":
                round(mean_delta, 4) if mean_delta is not None else None,
            "within_holdout_delta_clustered_ci": boot,
            "retention_destruction_per_seed":
                [round(d, 4) if d is not None else None
                 for d in ret_dest],
            "retention_destruction_max":
                max((v for v in ret_dest if v is not None),
                    default=None),
            "security_regressions_total": sec_viol,
            "security_adapted_violations_per_seed": [
                c["security"].get("adapted_violations")
                for c in cells],
            "security_frozen_violations_per_seed": [
                c["security"].get("frozen_violations")
                for c in cells],
            "security_basis": "paired_per_row_net_new",
            "security_evaluated": sec_viol is not None,
            "train_seconds_mean": (
                round(sum(c["train_seconds"] or 0 for c in cells)
                      / len(cells), 1)),
            "cells": [{"run_id": c["run_id"],
                       "verification_state": c["verification_state"],
                       "evidence_level_met": c["evidence_level_met"],
                       "reasons": c["evidence_reasons"][:4]}
                      for c in cells],
            "gates": gates_bool,
            "gate_detail": gate_detail,
            "all_gates_pass": all_pass,
            "fit_ok": fit_ok,
        }
        report[arm] = verdict

    # ---- matrix status -------------------------------------------------
    # INCOMPLETE — required cells never executed or lack the required
    #   evidence level (screened arms still count as missing evidence
    #   for the full program).
    # INVALID   — evidence, authorization or experiment identity is
    #   inconsistent (failed closure, refused attestations, protocol
    #   identity mismatch).
    # QUALIFIED — >=1 candidate arm passes every applicable gate on
    #   complete evidence.
    # COMPLETE_WITH_FAILURES — complete verified record, but >=1
    #   candidate arm failed a required gate. COMPLETE — everything
    #   verified and no candidate failure (e.g. controls only).
    if invalid or not protocol_consistent:
        matrix_status = "INVALID"
    elif missing or insufficient:
        matrix_status = "INCOMPLETE"
    elif qualified_arms:
        matrix_status = "QUALIFIED"
    elif failed_arms:
        matrix_status = "COMPLETE_WITH_FAILURES"
    else:
        matrix_status = "COMPLETE"
    matrix_qualified = matrix_status == "QUALIFIED"

    doc = {"schema": "mini-agi-v4.1-confirmation-analysis-v3",
           "matrix_id": args.matrix_id, "corpus": args.corpus,
           "seeds": seeds, "arms": arms,
           "arm_roles": role_of,
           "manifest": (str(args.manifest) if args.manifest else None),
           "evidence_level_required": required_level,
           "gates_declared": gates,
           "statistical_policy": (
               {"path": args.statistical_policy,
                "schema": stat_policy.get("schema"),
                "version": stat_policy.get("version")}
               if stat_policy else None),
           "missing_cells": missing,
           "invalid_cells": invalid,
           "insufficient_cells": insufficient,
           "rejected_cells": rejected,
           "screened_arms": ({a: args.screened_reason
                              for a in sorted(screened)}
                             if screened else {}),
           "unscreened_missing_cells": unscreened_missing,
           # `complete` is strict — every declared cell committed. A
           # screened arm still leaves an evidence gap, recorded in
           # missing_cells; scope-completeness only excuses cells
           # declared screened up front with a reason.
           "complete": not missing,
           "complete_for_declared_scope": not unscreened_missing,
           "matrix_complete": matrix_complete,
           "protocol_identities_consistent": protocol_consistent,
           "protocol_identity_sets": [
               {"model_id": k[0], "model_revision": k[1],
                "score_oracle": k[2], "corpus": k[3],
                "cells": sorted(v)}
               for k, v in sorted(identity_sets.items(),
                                  key=lambda kv: str(kv[0]))],
           "matrix_status": matrix_status,
           "matrix_qualified": matrix_qualified,
           "qualified_candidates": qualified_arms,
           "failed_candidates": failed_arms,
           "arms_report": report}
    text = json.dumps(doc, indent=2, sort_keys=True)
    out = args.output or str(
        run_root.parent / f"CONFIRMATION_ANALYSIS-{args.matrix_id}.json")
    Path(out).write_text(text + "\n")
    print(text)
    print(f"\nwrote {out}")

    # ---- exit semantics ---------------------------------------------
    # informational: the report is the product — negative results exit
    # clean; only a usage error (handled above) exits nonzero.
    if args.informational:
        return 0
    if args.require_qualified is not None:
        target = args.require_qualified
        if target == "__any__":
            ok = bool(qualified_arms)
            what = "any candidate"
        else:
            ok = target in qualified_arms
            what = f"candidate {target!r}"
        if not ok:
            print(f"QUALIFICATION REFUSED: no qualified candidate "
                  f"(requested {what}); matrix_status="
                  f"{matrix_status}, qualified={qualified_arms}, "
                  f"failed={failed_arms}", file=sys.stderr)
            return 1
        return 0
    # strict completeness (default, also --strict-completeness):
    # missing, inconsistent or insufficient evidence exits nonzero; a
    # COMPLETE/COMPLETE_WITH_FAILURES report — a legitimate negative
    # result — exits 0.
    if matrix_status in ("INCOMPLETE", "INVALID"):
        if missing:
            print(f"MISSING cells: {missing}", file=sys.stderr)
        if invalid:
            print(f"INVALID cells: {invalid}", file=sys.stderr)
        if insufficient:
            print(f"INSUFFICIENT-EVIDENCE cells (below "
                  f"{required_level!r}): {insufficient}",
                  file=sys.stderr)
        if not protocol_consistent:
            print(f"INCONSISTENT protocol identities across cells: "
                  f"{[str(k) for k in identity_sets]}",
                  file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
