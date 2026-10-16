"""SCI-402 / SCI-403 — fit-runner artifact isolation and contract integrity.

Two defects found by external review of the Campaign 3B.1 fit runner, both
reproduced before repair:

SCI-402  every run wrote its adapter to ``<output>.parent/adapter``, so runs
         sharing an output directory destroyed each other's adapters. Five of
         the six adapters from the first experiment were lost; only the JSON
         reports survived, which is exactly the reproducibility gap the review
         flagged.

SCI-403  ``--max-new-tokens`` was accepted independently of the frozen output
         contract, and generation used the CLI value while the contract
         recorded its own. A run could therefore generate with a budget the
         contract did not authenticate. The two agreed (both 16) in the runs
         that were actually executed, so no recorded result is invalid — but
         the mechanism was wrong and would have diverged silently.

The structural assertions below run without a model; the behavioural test
relies on the runner refusing an existing destination *before* it loads
anything, which keeps it fast and deterministic.
"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

SCRIPT = ROOT / "scripts" / "positive_control_fit.py"


def _tree():
    return ast.parse(SCRIPT.read_text())


def _source():
    return SCRIPT.read_text()


# ---------------------------------------------------------------- SCI-403

def _cli_flags():
    """Every literal passed as the first argument to add_argument."""
    flags = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call) \
                and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "add_argument" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                flags.append(first.value)
    return flags


def test_generation_budget_comes_from_the_contract_not_a_cli_flag():
    """No independent token budget may exist in the execution path."""
    flags = _cli_flags()
    assert flags, "flag extraction found nothing — the test is not testing"
    assert "--max-new-tokens" not in flags, (
        "an independent generation budget is back in the CLI; the frozen "
        "output contract must be the only source of max_new_tokens")
    src = _source()
    assert "contract.output.max_new_tokens" in src, (
        "generation must read the budget from the contract")


def test_effective_decoding_is_recorded_in_the_run():
    """The decoding configuration actually used must be in the evidence."""
    src = _source()
    assert "effective_decoding" in src
    assert '"max_new_tokens": max_new' in src
    assert '"source": "contract.output.max_new_tokens"' in src


# ---------------------------------------------------------------- SCI-402

def test_adapter_destination_is_run_scoped():
    """The adapter must not be derived from a caller-chosen output path."""
    src = _source()
    assert 'Path(args.output).parent / "adapter"' not in src, (
        "the shared-adapter destination is back; runs would clobber each "
        "other's artifacts")
    assert 'adapter_dir = staging / "adapter"' in src


def test_runs_are_staged_hashed_and_atomically_published():
    src = _source()
    assert "run_reservation" in src, \
        "runs must go through the atomic reservation lifecycle (V4-26)"
    assert "reservation.publish()" in src, "publish is not atomic"
    assert "RUN_COMMIT.json" in src and "dir_inventory" in src, \
        "the run must carry a hash inventory"
    assert "ADAPTER_MANIFEST.json" in src
    assert "reload_verified" in src, \
        "the adapter must be reload-verified before publication"


def test_no_path_can_delete_another_attempts_evidence():
    """The pre-v4.0.2 runner could ``rmtree`` a shared staging dir and a
    committed final dir during failure cleanup. Those calls are gone —
    no path in the runner may remove a directory at all."""
    src = _source()
    assert "shutil.rmtree" not in src, \
        "the runner must never delete directories"
    assert "os.replace" not in src, \
        "publication must be a non-replacing guarded rename"
    assert '".staging-"' not in src or '.staging-' not in src, (
        "staging names shared between attempts are back")


def test_failed_attempts_do_not_burn_the_canonical_name():
    """A FAILED record lives at ``<run_id>.FAILED-<attempt>``, freeing
    the canonical name for a later qualified attempt."""
    src = _source()
    assert "reservation.fail()" in src, \
        "failures must go through the ownership-bound reservation"


def test_a_failed_run_is_recorded_not_dropped():
    src = _source()
    assert '"status": "FAILED"' in src
    assert '"status": "COMPLETE"' in src


def test_existing_run_destination_is_refused(tmp_path):
    """Behavioural: the runner must refuse to overwrite a committed run.

    This fires before any model is loaded, so the test needs no weights.
    """
    run_root = tmp_path / "runs"
    (run_root / "taken").mkdir(parents=True)
    (run_root / "taken" / "RUN_COMMIT.json").write_text("{}\n")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--families", "pc-upper,pc-rev",
         "--steps", "1", "--run-root", str(run_root), "--run-id", "taken"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode != 0
    assert "refusing to overwrite" in (proc.stdout + proc.stderr)
    # the committed run is untouched
    assert (run_root / "taken" / "RUN_COMMIT.json").read_text() == "{}\n"


def test_unknown_families_are_refused_before_any_model_load():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--families", "not-a-family",
         "--steps", "1", "--run-root", "/tmp/c3b2-unused"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode != 0
    assert "unknown train families" in (proc.stdout + proc.stderr)


# ------------------------------------------------- Step 11 contrasts

def test_the_three_contrasts_are_mutually_exclusive():
    """--examples-per-family and --total-examples answer different causal
    questions; accepting both would make a run unattributable."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--families", "pc-upper,pc-rev",
         "--examples-per-family", "5", "--total-examples", "30",
         "--steps", "1", "--run-root", "/tmp/c3b2-unused"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode != 0
    assert "different contrasts" in (proc.stdout + proc.stderr)


def test_contrast_bounds_are_validated_before_any_model_load():
    for bad in (["--examples-per-family", "0"], ["--total-examples", "0"],
                ["--exposure", "0"]):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--families", "pc-upper,pc-rev",
             *bad, "--steps", "1", "--run-root", "/tmp/c3b2-unused"],
            capture_output=True, text=True, cwd=str(ROOT))
        assert proc.returncode != 0, bad
        assert "must be >= 1" in (proc.stdout + proc.stderr), bad


def test_contrast_allocation_is_recorded_in_the_run_config():
    """A run must state its assigned dataset, not just its score."""
    src = _source()
    for field in ("allocation", "examples_per_family", "training_examples",
                  "unique_scored_rows", "presentations_per_example",
                  "exposure_target"):
        assert f'"{field}"' in src, f"the run config omits {field}"
    # realised exposure must come from the receipt, not be assumed
    assert "sample_presentations" in src and "optimizer_updates" in src
    assert "exposure_actual" in src


def test_negative_control_is_deranged_over_the_trained_allocation():
    """A control must see the same example count as its matched condition,
    otherwise the contrast is confounded by the control itself."""
    src = _source()
    assert "shuffled_label_examples(train_rows, args.seed)" in src, (
        "the negative control must be deranged over train_rows, not the "
        "unique subset, or it sees fewer examples than its matched run")


# ----------------------------------------------------------------- v4.0.1

def test_within_family_and_compositional_splits_are_evaluated():
    """The campaign3e measurement surfaces must flow through the fit
    runner: a corpus that carries a validation or compositional split
    gets frozen/adapted predictions, strict scores, persisted per-split
    prediction files, and metric fields — otherwise the new corpus's
    within-family partition exists on disk but is never measured."""
    src = _source()
    # rows are collected from the corpus
    assert 'r["split"] == "validation"' in src
    assert 'r["split"] == "compositional"' in src
    # both models are evaluated over both splits (frozen BEFORE the
    # reference is freed — per_row(frozen, ...) must precede del frozen)
    for call in ("per_row(frozen, validation)",
                 "per_row(frozen, compositional)",
                 "per_row(model, validation)",
                 "per_row(model, compositional)"):
        assert call in src, f"missing evaluation call: {call}"
    # per-split prediction artifacts (f-string names: the writer builds
    # PREDICTIONS-{FROZEN,ADAPTED}-{VALIDATION,COMPOSITIONAL}.jsonl)
    assert 'f"PREDICTIONS-{name}-{split_name}.jsonl"' in src
    assert '("VALIDATION"' in src and '"COMPOSITIONAL"' in src
    for split_field in ('"frozen": _acc(f_val', '"adapted": _acc(a_val',
                        '"frozen": _acc(f_comp', '"adapted": _acc(a_comp'):
        assert split_field in src, f"metric missing: {split_field}"
    # the within-family decomposition is reported, not just pooled
    for field in ('"within_family"', '"untrained_family"',
                  '"within_family_rows"'):
        assert field in src, f"metrics omit {field}"


def test_frozen_evaluation_precedes_reference_freeing():
    """per_row(frozen, ...) calls must all sit before `del frozen` —
    a frozen-eval placed after it would raise at runtime instead of
    being caught by these source checks."""
    src = _source()
    free_at = src.index("del frozen")
    for call in ("per_row(frozen, subset)", "per_row(frozen, hidden)",
                 "per_row(frozen, validation)",
                 "per_row(frozen, compositional)",
                 "per_row(frozen, within_holdout)",
                 "per_row(frozen, retention)"):
        assert src.index(call) < free_at, (
            f"{call} appears after the frozen reference is freed")


# ----------------------------------------------------------------- v4.1.0

def test_within_holdout_and_retention_surfaces_are_evaluated():
    """The campaign4a confirmation + preservation surfaces must flow
    through the fit runner: rows collected, both models evaluated,
    per-split prediction files written, and the Phase-5
    destruction/acquisition rates reported on the retention set."""
    src = _source()
    assert 'r["split"] == "within_holdout"' in src
    assert 'r["split"] == "retention"' in src
    for call in ("per_row(frozen, within_holdout)",
                 "per_row(frozen, retention)",
                 "per_row(model, within_holdout)",
                 "per_row(model, retention)"):
        assert call in src, f"missing evaluation call: {call}"
    for tag in ('"WITHIN_HOLDOUT"', '"RETENTION"'):
        assert tag in src, f"prediction writer omits {tag}"
    for field in ('"within_holdout_rows"', '"retention_rows"',
                  '"destruction_rate"', '"acquisition_rate"'):
        assert field in src, f"metrics omit {field}"
    # the confirmation surface must NEVER feed the rehearsal pool —
    # the pool admits only whitelisted train-side splits
    # (train/validation/rehearsal), so within_holdout/retention rows
    # cannot contaminate training
    import re as _re
    m = _re.search(r'r\["split"\] in \(([^)]*)\)', src)
    assert m, "default rehearsal pool whitelist not found"
    allowed = set(_re.findall(r'"(\w+)"', m.group(1)))
    assert allowed <= {"train", "validation", "rehearsal"}, allowed
    assert allowed.isdisjoint(
        {"within_holdout", "retention", "hidden", "security",
         "compositional", "final_holdout"}), allowed
