"""Executed-run evidence reconciliation (V4-27).

The v4.0.1 requalifier rescored only the three legacy prediction files
(PREDICTIONS-{FROZEN,SUBSET,ADAPTED}.jsonl — the hidden and train-subset
splits). The v4.0.1 runner additionally commits per-split prediction
files for the VALIDATION and COMPOSITIONAL surfaces, and those files
were never independently verified — a wrong number there would sail
through requalification.

This module verifies a whole run directory end to end:

1. RUN_COMMIT integrity — status plus every inventoried file's bytes.
2. Prediction-file coverage — the exact expected set exists
   (reconstructed from CONFIG.json's declared split row counts), each
   file carries the required number of unique task ids, and there are
   no missing, duplicated, or unexpected predictions. When the corpus
   named by CONFIG is resolvable, every prediction is also joined
   against its corpus row: the id must belong to the split, the
   prompt_sha256 must match the corpus prompt bytes, and the expected
   answer must match the corpus answer.
3. Strict rescore — EVERY prediction file is rescored with the
   authorized evaluator (``exact_match_strict`` / ``score_row``), and
   each recomputed number is compared against the corresponding
   METRICS.json claim: frozen/adapted accuracy per split, the
   within-family and untrained-family decomposition, the fit-gate
   accuracy, and the capability-destruction id lists.
4. Identity binding — commit↔config digest agreement, the config
   digest recomputed from its own body, adapter manifest ↔ adapter
   bytes ↔ training receipt artifact digest ↔ training-spec digest ↔
   declared model identity, contract digests, and the scorer identity
   the run was committed under.

Verdicts: VERIFIED (integrity + rescore + identity all agree),
INCONSISTENT (verification ran but something disagreed), FAILED_RUN
(commit status FAILED — evidence preserved, not a claim), INVALID
(integrity check failed or evidence missing).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from egai.common.canonical import digest, sha256_bytes
from minagi.v161.evaluators import exact_match_strict, score_row

VERIFIED = "VERIFIED"
INCONSISTENT = "INCONSISTENT"
FAILED_RUN = "FAILED_RUN"
INVALID = "INVALID"

TOL = 1e-9
AUTHORIZED_SCORER = "exact-match-v2"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines()
            if ln.strip()]


def adapter_artifact_digest_of(adapter_dir: Path) -> str:
    """Recompute ``peft_trainer.adapter_artifact_digest`` without
    importing the trainer (keeps the verifier dependency-light)."""
    d = Path(adapter_dir)
    cfg = d / "adapter_config.json"
    if not cfg.is_file():
        raise ValueError("adapter_config.json missing")
    rows = [("adapter_config.json", sha256_bytes(cfg.read_bytes()))]
    weights = sorted(p for p in d.glob("adapter_model*") if p.is_file())
    if not weights:
        raise ValueError("adapter_model weights missing")
    rows += [(p.name, sha256_bytes(p.read_bytes())) for p in weights]
    return digest(rows)


def expected_prediction_files(config: dict) -> dict[str, tuple[str, int | None]]:
    """The prediction-file set a run must contain, reconstructed from
    CONFIG.json's declared split row counts.

    Legacy runs (no validation_rows/compositional_rows fields) predate
    the per-split surfaces and are expected to carry only the three
    original files."""
    exp = {
        "PREDICTIONS-FROZEN.jsonl": ("hidden", config.get("hidden_rows")),
        "PREDICTIONS-ADAPTED.jsonl": ("hidden", config.get("hidden_rows")),
        "PREDICTIONS-SUBSET.jsonl": ("train-subset",
                                     config.get("subset_rows")),
    }
    if "validation_rows" in config:
        n = config.get("validation_rows")
        if n:
            exp["PREDICTIONS-FROZEN-VALIDATION.jsonl"] = ("validation", n)
            exp["PREDICTIONS-ADAPTED-VALIDATION.jsonl"] = ("validation", n)
    if "compositional_rows" in config:
        n = config.get("compositional_rows")
        if n:
            exp["PREDICTIONS-FROZEN-COMPOSITIONAL.jsonl"] = (
                "compositional", n)
            exp["PREDICTIONS-ADAPTED-COMPOSITIONAL.jsonl"] = (
                "compositional", n)
    # v4.1.0 surfaces — present only when the corpus declares them
    # (legacy configs never carry these keys, so expected sets for
    # committed runs stay byte-identical).
    for key, tag, split in (
            ("within_holdout_rows", "WITHIN_HOLDOUT", "within_holdout"),
            ("retention_rows", "RETENTION", "retention")):
        if key in config:
            n = config.get(key)
            if n:
                exp[f"PREDICTIONS-FROZEN-{tag}.jsonl"] = (split, n)
                exp[f"PREDICTIONS-ADAPTED-{tag}.jsonl"] = (split, n)
    return exp


def _rescore(rows: list[dict]) -> dict[str, float]:
    out = {}
    for r in rows:
        row = {"expected": r.get("expected", ""),
               "verify": r.get("verify")}
        out[str(r["id"])] = float(
            score_row(row, str(r.get("output", "")), exact_match_strict))
    return out


def _mean(vals) -> float | None:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else None


def verify_run(run_dir: Path, *,
               repo_root: Path | None = None,
               corpus_rows: list[dict] | None = None) -> dict:
    """Independently reconcile one committed run directory.

    ``repo_root`` lets the verifier resolve ``CONFIG.corpus`` for the
    corpus-join checks; ``corpus_rows`` may be supplied directly. No
    model is loaded — this is pure artifact verification."""
    run_dir = Path(run_dir)
    out = {"run_id": run_dir.name, "status": None, "checks": {},
           "rescore": {}, "reported": {}, "mismatches": []}
    mismatch = out["mismatches"].append

    commit_path = run_dir / "RUN_COMMIT.json"
    if not commit_path.is_file():
        out["status"] = INVALID
        out["checks"]["commit"] = "RUN_COMMIT.json missing"
        return out
    try:
        commit = json.loads(commit_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        out["status"] = INVALID
        out["checks"]["commit"] = f"RUN_COMMIT.json unparseable ({exc})"
        return out
    if not isinstance(commit, dict):
        out["status"] = INVALID
        out["checks"]["commit"] = "RUN_COMMIT.json is not an object"
        return out
    run_status = commit.get("status")
    out["checks"]["commit_status"] = run_status
    if commit.get("run_id") and commit["run_id"] != run_dir.name:
        mismatch(f"commit run_id {commit['run_id']!r} != directory "
                 f"{run_dir.name!r}")

    # --- file-hash inventory ------------------------------------------
    inventory = commit.get("files") or {}
    inv_bad = []
    for rel, want in inventory.items():
        fp = run_dir / rel
        if not fp.is_file():
            inv_bad.append(f"{rel}: missing")
            continue
        if sha256_file(fp) != want:
            inv_bad.append(f"{rel}: hash mismatch")
    out["checks"]["inventory"] = "ok" if not inv_bad else inv_bad
    extra = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*")
                   if p.is_file()
                   and str(p.relative_to(run_dir)) not in inventory
                   and p.name != "RUN_COMMIT.json")
    if run_status == "COMPLETE":
        out["checks"]["uncommitted_files"] = extra

    def _load(path: Path, label: str) -> dict:
        """A committed JSON file that cannot be parsed is evidence
        damage — recorded as a mismatch, never a verifier crash."""
        if not path.is_file():
            return {}
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            mismatch(f"{label} unparseable ({exc})")
            return {}
        if not isinstance(doc, dict):
            mismatch(f"{label} is not a JSON object")
            return {}
        return doc

    config_path = run_dir / "CONFIG.json"
    config = _load(config_path, "CONFIG.json")
    metrics_path = run_dir / "METRICS.json"
    reported = _load(metrics_path, "METRICS.json")
    if run_status == "COMPLETE":
        if not config_path.is_file():
            mismatch("CONFIG.json missing on a COMPLETE run")
        if not metrics_path.is_file():
            mismatch("METRICS.json missing on a COMPLETE run")

    # --- corpus join (when the declared corpus resolves) ---------------
    by_split: dict[str, dict[str, dict]] = {}
    if corpus_rows is None and repo_root and config.get("corpus"):
        cpath = Path(repo_root) / str(config["corpus"])
        if cpath.is_file():
            corpus_rows = _load_jsonl(cpath)
            out["checks"]["corpus"] = str(config["corpus"])
    if corpus_rows:
        for row in corpus_rows:
            by_split.setdefault(str(row["split"]), {})[str(row["id"])] = row

    # --- expected prediction files ------------------------------------
    split_scores: dict[str, dict[str, float]] = {}
    split_rows: dict[str, list[dict]] = {}
    if run_status == "COMPLETE" and config:
        exp = expected_prediction_files(config)
        for fname, (split, n_rows) in sorted(exp.items()):
            fp = run_dir / fname
            if not fp.is_file():
                mismatch(f"expected prediction file absent: {fname}")
                continue
            try:
                rows = _load_jsonl(fp)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                # an unparseable predictions file is evidence damage —
                # recorded, never a verifier crash
                mismatch(f"{fname}: unparseable ({exc})")
                continue
            split_rows[split] = split_rows.get(split, []) + rows
            scores = _rescore(rows)
            split_scores[fname] = scores
            ids = [str(r["id"]) for r in rows]
            dup = sorted(k for k, c in Counter(ids).items() if c > 1)
            if dup:
                mismatch(f"{fname}: duplicated task ids {dup[:5]}")
            if n_rows is not None and len(set(ids)) != n_rows:
                mismatch(f"{fname}: {len(set(ids))} unique ids, "
                         f"CONFIG declares {n_rows}")
            wrong_split = sorted({str(r.get("split")) for r in rows}
                                 - {split})
            if wrong_split:
                mismatch(f"{fname}: records carry split labels "
                         f"{wrong_split}, expected {split!r}")
            # corpus binding
            csplit = by_split.get(split if split != "train-subset"
                                  else "train")
            if csplit:
                ids_set = set(ids)
                missing_ids = sorted(set(csplit) - ids_set)
                unexpected_ids = sorted(ids_set - set(csplit))
                if missing_ids:
                    mismatch(f"{fname}: {len(missing_ids)} corpus "
                             f"{split} ids missing predictions "
                             f"(first: {missing_ids[:3]})")
                if unexpected_ids:
                    mismatch(f"{fname}: predictions for ids not in the "
                             f"corpus {split} split "
                             f"(first: {unexpected_ids[:3]})")
                for r in rows:
                    crow = csplit.get(str(r["id"]))
                    if crow is None:
                        continue
                    want_sha = hashlib.sha256(
                        str(crow["prompt"]).encode()).hexdigest()
                    if r.get("prompt_sha256") != want_sha:
                        mismatch(f"{fname}:{r['id']}: prompt_sha256 does "
                                 f"not match the corpus prompt")
                    if str(r.get("expected")) != str(crow["expected"]):
                        mismatch(f"{fname}:{r['id']}: expected answer "
                                 f"differs from the corpus")
            out["rescore"][fname] = {
                "n": len(rows), "unique_ids": len(set(ids)),
                "split_mean": _mean(scores.values())}

        on_disk = {p.name for p in run_dir.glob("PREDICTIONS-*.jsonl")}
        unexpected_files = sorted(on_disk - set(exp))
        if unexpected_files:
            mismatch(f"unexpected prediction files: {unexpected_files}")

    # --- recomputed metrics vs reported --------------------------------
    rep_adapted = reported.get("adapted") or {}
    rep_frozen = reported.get("frozen") or {}
    rep_val = reported.get("validation") or {}
    rep_who = reported.get("within_holdout") or {}
    rep_ret = reported.get("retention") or {}

    def cmp(fname, want, label):
        sc = split_scores.get(fname)
        if sc is None or want is None:
            return
        got = _mean(sc.values())
        if got is not None and abs(got - float(want)) > TOL:
            mismatch(f"{label}: reported {want} vs strict rescore {got}")

    # Required metric schema (v4.0.3): a COMPLETE run must report every
    # measurement its prediction surfaces make resolvable — an absent
    # required field is a mismatch, not a silently skipped check, and
    # every reported rate must be finite and inside [0, 1].
    def _req(fname, want, label):
        sc = split_scores.get(fname)
        if run_status != "COMPLETE" or not sc:
            return      # surface absent — the coverage pass ruled on it
        if want is None:
            mismatch(f"{label}: required metric absent")
            return
        try:
            v = float(want)
        except (TypeError, ValueError):
            mismatch(f"{label}: reported {want!r} is not numeric")
            return
        if not math.isfinite(v) or not (0.0 <= v <= 1.0):
            mismatch(f"{label}: reported {want} outside [0,1]")
            return
        got = _mean(sc.values())
        if got is not None and abs(got - v) > TOL:
            mismatch(f"{label}: reported {want} vs strict rescore {got}")

    _req("PREDICTIONS-FROZEN.jsonl", rep_frozen.get("hidden"),
         "frozen.hidden")
    _req("PREDICTIONS-ADAPTED.jsonl", rep_adapted.get("hidden"),
         "adapted.hidden")
    _req("PREDICTIONS-SUBSET.jsonl", rep_adapted.get("subset"),
         "adapted.subset")
    _req("PREDICTIONS-FROZEN-VALIDATION.jsonl",
         rep_frozen.get("validation"), "frozen.validation")
    _req("PREDICTIONS-ADAPTED-VALIDATION.jsonl",
         rep_adapted.get("validation"), "adapted.validation")
    _req("PREDICTIONS-FROZEN-COMPOSITIONAL.jsonl",
         rep_frozen.get("compositional"), "frozen.compositional")
    _req("PREDICTIONS-ADAPTED-COMPOSITIONAL.jsonl",
         rep_adapted.get("compositional"), "adapted.compositional")
    _req("PREDICTIONS-FROZEN-WITHIN_HOLDOUT.jsonl",
         rep_frozen.get("within_holdout"), "frozen.within_holdout")
    _req("PREDICTIONS-ADAPTED-WITHIN_HOLDOUT.jsonl",
         rep_adapted.get("within_holdout"), "adapted.within_holdout")
    _req("PREDICTIONS-FROZEN-RETENTION.jsonl",
         rep_frozen.get("retention"), "frozen.retention")
    _req("PREDICTIONS-ADAPTED-RETENTION.jsonl",
         rep_adapted.get("retention"), "adapted.retention")

    # fit_gate.adapted_subset_accuracy is a duplicate measurement of the
    # same subset surface — optional, but checked when present.
    cmp("PREDICTIONS-SUBSET.jsonl",
        (reported.get("fit_gate") or {}).get("adapted_subset_accuracy"),
        "fit_gate.adapted_subset_accuracy")

    # within-/untrained-family decomposition of within-family splits
    trained = set(config.get("families") or [])
    for fname, key, rep_block, label0 in (
            ("PREDICTIONS-FROZEN-VALIDATION.jsonl", "frozen",
             rep_val, "validation"),
            ("PREDICTIONS-ADAPTED-VALIDATION.jsonl", "adapted",
             rep_val, "validation"),
            ("PREDICTIONS-FROZEN-WITHIN_HOLDOUT.jsonl", "frozen",
             rep_who, "within_holdout"),
            ("PREDICTIONS-ADAPTED-WITHIN_HOLDOUT.jsonl", "adapted",
             rep_who, "within_holdout")):
        try:
            rows = _load_jsonl(run_dir / fname) \
                if (run_dir / fname).is_file() else []
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue        # already reported by the coverage pass
        scores = split_scores.get(fname)
        if not rows or not scores:
            continue
        wf = [scores[str(r["id"])] for r in rows
              if r.get("family") in trained]
        uf = [scores[str(r["id"])] for r in rows
              if r.get("family") not in trained]
        rep_wf = ((rep_block.get("within_family") or {}).get(key))
        rep_uf = ((rep_block.get("untrained_family") or {}).get(key))
        for label, got, want in (("within_family", _mean(wf), rep_wf),
                                 ("untrained_family", _mean(uf), rep_uf)):
            if want is None:
                continue
            if got is None or abs(got - float(want)) > TOL:
                mismatch(f"{label0}.{label}.{key}: reported {want} "
                         f"vs strict rescore {got}")

    # capability-destruction id lists on the hidden split — required
    # accounting on a COMPLETE run with hidden predictions (v4.0.3):
    # aggregate net accuracy must never conceal destroyed work, so the
    # destruction/acquisition counts are mandatory, not optional.
    f_hid = split_scores.get("PREDICTIONS-FROZEN.jsonl")
    a_hid = split_scores.get("PREDICTIONS-ADAPTED.jsonl")
    rep_cd = reported.get("capability_destruction") or {}
    if run_status == "COMPLETE" and f_hid and a_hid is not None:
        if not rep_cd:
            mismatch("capability_destruction: required accounting "
                     "absent on a COMPLETE run")
        else:
            for field in ("n_frozen_correct_hidden", "destroyed_ids",
                          "gained_ids"):
                if rep_cd.get(field) is None:
                    mismatch(f"capability_destruction.{field}: "
                             "required field absent")
            for n_field, ids_field in (("n_destroyed", "destroyed_ids"),
                                       ("n_gained", "gained_ids")):
                nv, iv = rep_cd.get(n_field), rep_cd.get(ids_field)
                if nv is not None and isinstance(iv, list) and \
                        int(nv) != len(iv):
                    mismatch(f"capability_destruction.{n_field}: "
                             f"{nv} != len({ids_field})")
    if f_hid is not None and a_hid is not None and rep_cd:
        destroyed = sorted(rid for rid in f_hid
                           if f_hid[rid] == 1.0 and a_hid.get(rid) == 0.0)
        gained = sorted(rid for rid in f_hid
                        if f_hid[rid] == 0.0 and a_hid.get(rid) == 1.0)
        if rep_cd.get("destroyed_ids") is not None and \
                sorted(rep_cd["destroyed_ids"]) != destroyed:
            mismatch("capability_destruction.destroyed_ids differ")
        if rep_cd.get("gained_ids") is not None and \
                sorted(rep_cd["gained_ids"]) != gained:
            mismatch("capability_destruction.gained_ids differ")
        n_fc = rep_cd.get("n_frozen_correct_hidden")
        if n_fc is not None and int(n_fc) != int(sum(f_hid.values())):
            mismatch(f"n_frozen_correct_hidden: reported {n_fc} vs "
                     f"rescore {int(sum(f_hid.values()))}")

    # retention destruction/acquisition accounting (v4.1.0 — Phase 5
    # quantities: the rate of frozen-correct answers lost, and the rate
    # of frozen-wrong answers gained, are checked separately so churn
    # can never hide behind a flat net score).
    f_ret = split_scores.get("PREDICTIONS-FROZEN-RETENTION.jsonl")
    a_ret = split_scores.get("PREDICTIONS-ADAPTED-RETENTION.jsonl")
    if f_ret is not None and a_ret is not None and rep_ret:
        r_destroyed = sorted(rid for rid in f_ret
                             if f_ret[rid] == 1.0
                             and a_ret.get(rid) == 0.0)
        r_gained = sorted(rid for rid in f_ret
                          if f_ret[rid] == 0.0
                          and a_ret.get(rid) == 1.0)
        if rep_ret.get("destroyed_ids") is not None and \
                sorted(rep_ret["destroyed_ids"]) != r_destroyed:
            mismatch("retention.destroyed_ids differ")
        if rep_ret.get("gained_ids") is not None and \
                sorted(rep_ret["gained_ids"]) != r_gained:
            mismatch("retention.gained_ids differ")
        n_fc = rep_ret.get("n_frozen_correct")
        if n_fc is not None and int(n_fc) != int(sum(f_ret.values())):
            mismatch(f"retention.n_frozen_correct: reported {n_fc} vs "
                     f"rescore {int(sum(f_ret.values()))}")
        n_wrong = len(f_ret) - int(sum(f_ret.values()))
        for label, want, got in (
                ("destruction_rate", rep_ret.get("destruction_rate"),
                 (len(r_destroyed) / int(sum(f_ret.values())))
                 if sum(f_ret.values()) else None),
                ("acquisition_rate", rep_ret.get("acquisition_rate"),
                 (len(r_gained) / n_wrong) if n_wrong else None)):
            if want is None and got is None:
                continue
            if want is None or got is None or \
                    abs(float(want) - float(got)) > TOL:
                mismatch(f"retention.{label}: reported {want} vs "
                         f"rescore {got}")

    # --- identity bindings ---------------------------------------------
    ident: list[str] = []
    if run_status == "COMPLETE":
        if not config.get("model_revision"):
            ident.append("model_revision absent — activation identity "
                         "cannot be established")
        if not config.get("score_oracle"):
            ident.append("score_oracle absent")
        elif config["score_oracle"] != AUTHORIZED_SCORER:
            ident.append(f"score_oracle {config['score_oracle']!r} != "
                         f"{AUTHORIZED_SCORER!r}")
        if config.get("config_digest") and \
                commit.get("config_digest") != config["config_digest"]:
            ident.append("RUN_COMMIT.config_digest != "
                         "CONFIG.config_digest")
        if config.get("config_digest"):
            body = {k: v for k, v in config.items()
                    if k not in ("config_digest", "attempt_id", "run_id")}
            if digest(body) != config["config_digest"]:
                ident.append("CONFIG.config_digest does not recompute "
                             "from the config body")
        aman_path = run_dir / "ADAPTER_MANIFEST.json"
        if aman_path.is_file():
            aman = _load(aman_path, "ADAPTER_MANIFEST.json")
            if aman.get("run_id") != run_dir.name:
                ident.append("ADAPTER_MANIFEST.run_id mismatch")
            for k in ("prompt_contract_digest",
                      "output_contract_digest"):
                if config.get(k) and aman.get(k) != config[k]:
                    ident.append(f"ADAPTER_MANIFEST.{k} != CONFIG")
            adir = run_dir / str(aman.get("adapter_dir", "adapter"))
            for rel, want in (aman.get("files") or {}).items():
                fp = adir / rel
                if not fp.is_file():
                    ident.append(f"adapter file missing: {rel}")
                elif sha256_file(fp) != want:
                    ident.append(f"adapter file hash mismatch: {rel}")
            if not aman.get("reload_verified"):
                ident.append("adapter reload was not verified")
        else:
            ident.append("ADAPTER_MANIFEST.json missing")
        # training receipt ↔ adapter artifact digest
        tr_path = run_dir / "TRAINING_RECEIPT.json"
        adir = run_dir / "adapter"
        if tr_path.is_file() and adir.is_dir():
            tr = _load(tr_path, "TRAINING_RECEIPT.json")
            try:
                actual = adapter_artifact_digest_of(adir)
            except Exception as exc:
                ident.append(f"adapter artifact digest unverifiable "
                             f"({exc})")
            else:
                if tr.get("adapter_artifact_digest") and \
                        tr["adapter_artifact_digest"] != actual:
                    ident.append("TRAINING_RECEIPT.adapter_artifact_"
                                 "digest != adapter bytes on disk")
        spec_path = adir / "TRAINING_SPEC.json"
        if spec_path.is_file() and config.get("lora_spec_digest"):
            if digest(_load(spec_path, "TRAINING_SPEC.json")) != \
                    config["lora_spec_digest"]:
                ident.append("adapter/TRAINING_SPEC.json digest != "
                             "CONFIG.lora_spec_digest")
        acfg = adir / "adapter_config.json"
        if acfg.is_file() and config.get("model_id"):
            bm = _load(acfg, "adapter_config.json").get(
                "base_model_name_or_path")
            if bm and bm != config["model_id"]:
                ident.append(f"adapter base model {bm!r} != CONFIG "
                             f"model_id {config['model_id']!r}")
    out["checks"]["identity"] = ident
    out["mismatches"].extend(f"identity: {m}" for m in ident)

    out["reported"] = {k: reported.get(k) for k in
                       ("adapted", "frozen", "fit_gate", "validation",
                        "compositional", "within_holdout", "retention")}

    if run_status != "COMPLETE":
        out["status"] = FAILED_RUN if run_status == "FAILED" else INVALID
    elif inv_bad:
        out["status"] = INVALID
    elif out["mismatches"]:
        out["status"] = INCONSISTENT
    else:
        out["status"] = VERIFIED
    return out
