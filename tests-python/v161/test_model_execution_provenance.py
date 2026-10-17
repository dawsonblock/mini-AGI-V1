"""v4.0.3-r2 Phase 2 — evaluator-owned model execution provenance.

Diagnostics 9: the evaluator authenticated predictions the RESEARCH
process generated — nothing established that the claimed
model+adapter produced them. These tests pin the repaired contract:

  * a ``model-execution-receipt-v1`` binds model+tokenizer+adapter
    bytes + evaluator tasks + generation config + outputs + scores;
  * ``simulated`` executions (mock loaders — test seams) are NEVER
    real-model confirmation;
  * wrong model identity, wrong/incompatible/missing adapter bytes,
    altered predictions, stale receipts, unauthorized signers and
    loading failures all refuse;
  * a real-model execution runs the actual artifacts (gated on the
    environment having the model — skipped otherwise).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(Path(__file__).parent))

from test_independent_eval_receipt import _authority  # noqa: E402

from minagi.v161.independent_model_executor import (  # noqa: E402
    execute_evaluation)
from minagi.v161.model_execution_receipt import (  # noqa: E402
    EXECUTION_SCHEMA, MODE_BASELINE, MODE_REAL, MODE_SIMULATED,
    issue_execution_receipt, verify_execution_receipt)


def _adapter_dir(tmp_path, *, base="model/m1") -> Path:
    d = tmp_path / "adapter"
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base}))
    (d / "adapter_model.bin").write_bytes(b"real-lora-bytes")
    return d


class _FakeTokenizer:
    """Minimal tokenizer surface for the simulated path — decode pops
    the scripted next output; eos id only satisfies _generate."""
    eos_token_id = 0

    def __init__(self, outputs):
        self.outputs = list(outputs)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1, 2, 3]}

    def decode(self, ids, skip_special_tokens=True):
        return self.outputs.pop(0)


class _FakeTensor:
    def __init__(self, ids):
        self.ids = ids

    def __getitem__(self, ix):
        if isinstance(ix, tuple):
            row, sl = ix
            return self.ids[row][sl]
        return self.ids[ix]


class _FakeModel:
    """A scripted generation surface — returns a fake tensor whose
    tail the fake tokenizer decodes."""

    def parameters(self):
        import torch
        return iter([torch.nn.Parameter(torch.zeros(1))])

    def generate(self, **kw):
        return _FakeTensor([[1, 2, 3, 9, 9]])


def _loader(outputs):
    """Injected loader — marks every execution 'simulated'."""
    import types
    from minagi.v161.prompt_contract import OutputContract
    contract = types.SimpleNamespace(
        output=OutputContract(), add_generation_prompt=True)

    def load(spec):
        return _FakeModel(), _FakeTokenizer(outputs), contract
    return load


TASKS = [{"id": "t1", "prompt": "p-one", "expected": "ONE"},
         {"id": "t2", "prompt": "p-two", "expected": "TWO"},
         {"id": "t3", "prompt": "p-three", "expected": "THREE"}]


def _execute_sim(tmp_path, signer, *, outputs=None, adapter=None,
                 tasks=TASKS):
    """Run a simulated execution (mock loader) and return the full
    result dict."""
    ad = adapter if adapter is not None else _adapter_dir(tmp_path)
    return execute_evaluation(
        signer, model_id="model/m1", model_revision="rev-1",
        tasks=tasks, adapter_dir=ad,
        generation_config={"max_new_tokens": 8},
        scoring_version="exact-match-v2",
        model_loader=_loader(outputs or ["ONE", "TWO", "X"]),
        out_dir=tmp_path / "exec-out")


# ---------- receipt issuance + verification ---------------------------


def test_simulated_execution_never_real_confirmation(tmp_path):
    """Mock-model receipts are valid signatures over mock outputs —
    and are refused for real-model confirmation by default."""
    reg, ver, signers = _authority("evaluation")
    res = _execute_sim(tmp_path, signers["evaluation"])
    rec = res["receipt"]
    assert rec["schema"] == EXECUTION_SCHEMA
    assert rec["execution_mode"] == MODE_SIMULATED
    assert rec["adapter_digest"]
    v = verify_execution_receipt(
        rec, verifier=ver, authority=reg,
        adapter_dir=tmp_path / "adapter",
        tasks=[{"id": t["id"],
                "prompt_sha256": hashlib.sha256(
                    t["prompt"].encode()).hexdigest(),
                "expected_sha256": hashlib.sha256(
                    t["expected"].encode()).hexdigest()}
               for t in TASKS],
        predictions_path=res["predictions_path"],
        scores=res["scores"],
        generation_config={"max_new_tokens": 8})
    assert not v["verified"]
    assert "simulated" in v["reason"]


def test_wrong_model_identity_refused(tmp_path):
    """A receipt attesting model/m1 cannot certify a run expecting
    model/other — the attested identity must match the claim."""
    reg, ver, signers = _authority("evaluation")
    res = _execute_sim(tmp_path, signers["evaluation"])
    v = verify_execution_receipt(
        res["receipt"], verifier=ver, authority=reg,
        expected_model={"model_id": "model/other",
                        "model_revision": "rev-1"},
        require_real=False)
    assert not v["verified"]
    assert "model_id" in v["reason"]


def test_wrong_adapter_hash_refused(tmp_path):
    """Tampering with adapter bytes after issuance breaks the bound
    adapter_digest — the receipt attested different bytes."""
    reg, ver, signers = _authority("evaluation")
    res = _execute_sim(tmp_path, signers["evaluation"])
    (tmp_path / "adapter" / "adapter_model.bin").write_bytes(
        b"DIFFERENT-weights")
    bound = [{"id": t["id"],
              "prompt_sha256": hashlib.sha256(
                  t["prompt"].encode()).hexdigest(),
              "expected_sha256": hashlib.sha256(
                  t["expected"].encode()).hexdigest()}
             for t in TASKS]
    v = verify_execution_receipt(
        res["receipt"], verifier=ver, authority=reg,
        adapter_dir=tmp_path / "adapter", tasks=bound,
        predictions_path=res["predictions_path"],
        scores=res["scores"],
        generation_config={"max_new_tokens": 8},
        require_real=False)
    assert not v["verified"]
    assert any("adapter_digest" in m for m in v["mismatches"])


def test_invalid_adapter_format_refused_at_issue(tmp_path):
    """An adapter dir without weights cannot produce an execution
    receipt at all."""
    reg, ver, signers = _authority("evaluation")
    bad = tmp_path / "bad-adapter"
    bad.mkdir()
    (bad / "adapter_config.json").write_text(json.dumps({}))
    with pytest.raises(ValueError):
        execute_evaluation(
            signers["evaluation"], model_id="model/m1",
            model_revision="rev-1", tasks=TASKS, adapter_dir=bad,
            generation_config={}, model_loader=_loader(["ONE"]))


def test_incompatible_adapter_refused_at_issue(tmp_path):
    """An adapter declaring a different base model is incompatible —
    the executor refuses before any inference."""
    reg, ver, signers = _authority("evaluation")
    bad = _adapter_dir(tmp_path / "w", base="model/other")
    with pytest.raises(ValueError, match="incompatible"):
        execute_evaluation(
            signers["evaluation"], model_id="model/m1",
            model_revision="rev-1", tasks=TASKS, adapter_dir=bad,
            generation_config={}, model_loader=_loader(["ONE"]))


def test_altered_predictions_break_binding(tmp_path):
    """Outputs the receipt binds are byte-exact: swapping the
    generated manifest after issuance is detected."""
    reg, ver, signers = _authority("evaluation")
    res = _execute_sim(tmp_path, signers["evaluation"])
    pp = res["predictions_path"]
    rows = [json.loads(ln) for ln in pp.read_text().splitlines()]
    rows[0]["output"] = "FORGED"
    pp.write_text("".join(json.dumps(r, sort_keys=True) + "\n"
                          for r in rows))
    bound = [{"id": t["id"],
              "prompt_sha256": hashlib.sha256(
                  t["prompt"].encode()).hexdigest(),
              "expected_sha256": hashlib.sha256(
                  t["expected"].encode()).hexdigest()}
             for t in TASKS]
    v = verify_execution_receipt(
        res["receipt"], verifier=ver, authority=reg,
        adapter_dir=tmp_path / "adapter", tasks=bound,
        predictions_path=pp, scores=res["scores"],
        generation_config={"max_new_tokens": 8},
        require_real=False)
    assert not v["verified"]
    assert any("outputs_digest" in m for m in v["mismatches"])


def test_stale_receipt_replay_refused(tmp_path):
    """A receipt from execution A does not attest execution B's
    artifacts — the bound digests differ."""
    reg, ver, signers = _authority("evaluation")
    res_a = _execute_sim(tmp_path / "a", signers["evaluation"])
    ad_b = _adapter_dir(tmp_path / "b")
    (ad_b / "adapter_model.bin").write_bytes(b"b-bytes")
    bound = [{"id": t["id"],
              "prompt_sha256": hashlib.sha256(
                  t["prompt"].encode()).hexdigest(),
              "expected_sha256": hashlib.sha256(
                  t["expected"].encode()).hexdigest()}
             for t in TASKS]
    v = verify_execution_receipt(
        res_a["receipt"], verifier=ver, authority=reg,
        adapter_dir=ad_b, tasks=bound,
        scores=res_a["scores"],
        generation_config={"max_new_tokens": 8},
        require_real=False)
    assert not v["verified"]
    assert v["mismatches"]


def test_unauthorized_evaluator_refused(tmp_path):
    """A signature from a non-evaluation role is not execution
    evidence — roles are separated."""
    reg, ver, signers = _authority("evaluation", "plan")
    res = _execute_sim(tmp_path, signers["plan"])
    v = verify_execution_receipt(
        res["receipt"], verifier=ver, authority=reg,
        adapter_dir=tmp_path / "adapter",
        require_real=False)
    assert not v["verified"]
    assert "not authorized" in v["reason"]


def test_model_loading_failure_means_no_receipt(tmp_path):
    """A model that cannot load produces no execution — the failure is
    the evidence, not an exception-free pass."""

    def boom(spec):
        raise RuntimeError("CUDA OOM loading model")
    reg, ver, signers = _authority("evaluation")
    with pytest.raises(RuntimeError, match="OOM"):
        execute_evaluation(
            signers["evaluation"], model_id="model/m1",
            model_revision="rev-1", tasks=TASKS,
            adapter_dir=_adapter_dir(tmp_path),
            generation_config={}, model_loader=boom)


def test_frozen_baseline_binds_no_adapter(tmp_path):
    """A declared baseline execution must carry adapter_digest=null —
    baseline means the bare model."""
    reg, ver, signers = _authority("evaluation")
    res = execute_evaluation(
        signers["evaluation"], model_id="model/m1",
        model_revision="rev-1", tasks=TASKS, adapter_dir=None,
        generation_config={"max_new_tokens": 4},
        model_loader=_loader(["ONE", "TWO", "THREE"]))
    assert res["execution_mode"] == MODE_SIMULATED
    assert res["receipt"]["adapter_digest"] is None
    # issuing a baseline-mode receipt WITH adapter bytes is refused
    with pytest.raises((ValueError, AssertionError)):
        issue_execution_receipt(
            signers["evaluation"], execution_id="x",
            execution_mode=MODE_BASELINE, model_id="m",
            model_revision="r", tokenizer_digest="t",
            adapter_dir=_adapter_dir(tmp_path / "z"),
            generation_config={})


# ---------- real-model path (environment-gated) -------------------------


def _real_model_available():
    try:
        import torch  # noqa
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if "Qwen2.5-0.5B-Instruct" in repo.repo_id:
                return True
    except Exception:
        return False
    return False


@pytest.mark.skipif(not _real_model_available(),
                    reason="Qwen2.5-0.5B-Instruct not in HF cache")
def test_real_model_execution(tmp_path):
    """The executor actually loads the frozen base + a real LoRA and
    produces scored outputs — the full provenance chain, gated on the
    model being cached locally."""
    reg, ver, signers = _authority("evaluation")
    src = (ROOT / "results" / "v410_dev" / "runs"
           / "v410-dev-fixed-s0" / "adapter")
    import shutil
    ad = tmp_path / "adapter"
    shutil.copytree(src, ad)
    for junk in ("TRAINING_RECEIPT.json", "TRAINING_SPEC.json",
                 "README.md"):
        (ad / junk).unlink(missing_ok=True)
    tasks = [{"id": "r1", "prompt": "Write the word OK",
              "expected": "OK"},
             {"id": "r2", "prompt": "Say hi", "expected": "hi"}]
    res = execute_evaluation(
        signers["evaluation"],
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_revision="7ae557604adf67be50417f59c2c2f167def9a775",
        tasks=tasks, adapter_dir=ad,
        generation_config={"max_new_tokens": 8},
        out_dir=tmp_path / "out")
    assert res["execution_mode"] == MODE_REAL
    assert res["n_tasks"] == 2
    assert res["receipt"]["adapter_digest"].startswith("sha256:")
    assert res["receipt"]["tokenizer_digest"].startswith("sha256:")
    bound = [{"id": t["id"],
              "prompt_sha256": hashlib.sha256(
                  t["prompt"].encode()).hexdigest(),
              "expected_sha256": hashlib.sha256(
                  t["expected"].encode()).hexdigest()}
             for t in tasks]
    v = verify_execution_receipt(
        res["receipt"], verifier=ver, authority=reg,
        adapter_dir=ad, tasks=bound,
        predictions_path=res["predictions_path"],
        scores=res["scores"],
        generation_config={"max_new_tokens": 8})
    assert v["verified"], v


@pytest.mark.skipif(not _real_model_available(),
                    reason="Qwen2.5-0.5B-Instruct not in HF cache")
def test_frozen_vs_adapted_are_distinct_executions(tmp_path):
    """Baseline and adapted executions produce separate receipts with
    different bound identities — the receipts can't be conflated."""
    reg, ver, signers = _authority("evaluation")
    import shutil
    ad = tmp_path / "adapter"
    shutil.copytree(ROOT / "results" / "v410_dev" / "runs"
                    / "v410-dev-fixed-s0" / "adapter", ad)
    for junk in ("TRAINING_RECEIPT.json", "TRAINING_SPEC.json",
                 "README.md"):
        (ad / junk).unlink(missing_ok=True)
    tasks = [{"id": "r1", "prompt": "Write the word OK",
              "expected": "OK"}]
    kw = dict(model_id="Qwen/Qwen2.5-0.5B-Instruct",
              model_revision=
              "7ae557604adf67be50417f59c2c2f167def9a775",
              tasks=tasks, generation_config={"max_new_tokens": 8})
    base_res = execute_evaluation(signers["evaluation"], **kw)
    ad_res = execute_evaluation(signers["evaluation"], adapter_dir=ad,
                                **kw)
    assert base_res["execution_mode"] == MODE_BASELINE
    assert ad_res["execution_mode"] == MODE_REAL
    assert base_res["receipt"]["adapter_digest"] is None
    assert ad_res["receipt"]["adapter_digest"]
    # frozen-vs-adapted measurements are distinct artifacts
    assert base_res["receipt"]["execution_id"] != \
        ad_res["receipt"]["execution_id"]
