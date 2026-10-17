"""Campaign 3B Steps 11-13 — positive-control learnability.

Three layers:

  * the committed positive-control corpus is exactly what the generator
    produces (deterministic families, sealed holdout digest bound in the
    generated config),
  * the readiness gates behave as specified on synthetic metrics,
  * a real (tiny) model + LoRA run demonstrates the mechanism: correct
    supervision fits reviewed transformations that the frozen model
    cannot answer, with training/inference rendered through the shared
    prompt contract.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import (GPT2Config, GPT2LMHeadModel,  # noqa: E402
                          PreTrainedTokenizerFast)

from minagi.platforms.cuda.peft_trainer import (  # noqa: E402
    LoraTrainSpec, train_lora)
from minagi.v161.prompt_contract import (build_prompt_contract,  # noqa: E402
                                         encode_generation)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load(ROOT / "scripts" / "generate_positive_control.py", "pc_gen")
REPORT = _load(ROOT / "scripts" / "positive_control_report.py", "pc_report")


# --------------------------------------------------------------- corpus

def test_committed_corpus_matches_the_generator():
    committed = [json.loads(line) for line in
                 (ROOT / "configs" / "positive_control_tasks.jsonl")
                 .read_text().splitlines() if line.strip()]
    assert committed == GEN.build_corpus()


def test_corpus_groups_and_families():
    rows = GEN.build_corpus()
    splits = {r["split"] for r in rows}
    assert splits == {"train", "validation", "hidden", "retention",
                      "security"}
    trained = set(GEN.TRAIN_FAMILIES)
    heldout = set(GEN.HELDOUT_FAMILIES)
    assert {r["family"] for r in rows if r["split"] == "train"} == trained
    assert {r["family"] for r in rows if r["split"] == "hidden"} == heldout
    # validation is unseen inputs from the TRAINED families only
    assert {r["family"] for r in rows if r["split"] == "validation"} == \
        trained
    assert not (trained & heldout), "held-out families leaked into training"


def test_config_binds_the_generated_holdout_digest():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" /
                          "campaign3b_positive_control.yaml").read_text())
    assert cfg["final_holdout_digest"] == \
        GEN.holdout_digest(GEN.build_holdout())
    assert cfg["campaign_id"] == "campaign3b-positive-control"
    assert cfg["output_contract"]["instruction"]
    assert "L6" in cfg["arms"] and "NC" in cfg["arms"]


# ---------------------------------------------------------- gate logic

def _metrics(**over):
    m = {"accuracy": {"L1": {"train": 0.0, "validation": 0.1},
                      "L6": {"train": 1.0, "validation": 0.6},
                      "NC": {"train": 0.2, "validation": 0.1}},
         "parity": {"checked": 30, "failures": 0},
         "supervision": {"rows": 30, "bad": 0},
         "stability": {"finite_loss": True, "nonzero_grad": True,
                       "grad_norm_mean": 1.5},
         "reload": {"checked": 4, "mismatches": 0}}
    for k, v in over.items():
        m[k] = v
    return m


def test_all_gates_pass_on_a_clean_diagnostic():
    passed, checks = REPORT.readiness_gates(_metrics())
    assert passed, [c for c in checks if not c["passed"]]
    assert len(checks) == 7


@pytest.mark.parametrize("mutate,failing_gate", [
    (lambda m: m["parity"].update(failures=1), "prompt-token parity"),
    (lambda m: m["supervision"].update(bad=2), "supervised targets"),
    (lambda m: m["accuracy"]["L6"].update(train=0.90),
     "fit accuracy >= 95%"),
    (lambda m: m["accuracy"]["L1"].update(validation=0.6),
     "unseen within-family above frozen"),
    (lambda m: m["accuracy"]["NC"].update(train=0.95),
     "correct labels beat shuffled labels"),
    (lambda m: m["stability"].update(finite_loss=False),
     "training numerical stability"),
    (lambda m: m["reload"].update(mismatches=1),
     "adapter reload reproducible"),
])
def test_each_gate_failure_blocks_readiness(mutate, failing_gate):
    m = _metrics()
    mutate(m)
    passed, checks = REPORT.readiness_gates(m)
    assert not passed
    failed = [c["gate"] for c in checks if not c["passed"]]
    assert failing_gate in failed


# -------------------------------------------------- mechanism learning

VOCAB = ["[PAD]", "[EOS]", "[UNK]", "user:", "assistant:", "rev", "cat",
         "tac", "dog", "god", "sun", "nus", "map", "pam"]

CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}"
    "{% if m['role'] == 'assistant' %}[EOS]{% endif %} "
    "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}")

EXAMPLES = [{"prompt": "rev cat", "response": "tac"},
            {"prompt": "rev dog", "response": "god"},
            {"prompt": "rev sun", "response": "nus"},
            {"prompt": "rev map", "response": "pam"}]


def _tokenizer():
    t = Tokenizer(models.WordLevel(
        vocab={v: i for i, v in enumerate(VOCAB)}, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="[EOS]",
                                   pad_token="[PAD]", unk_token="[UNK]")
    fast.chat_template = CHAT_TEMPLATE
    return fast


def _model():
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=len(VOCAB),
                     n_positions=64, bos_token_id=1, eos_token_id=1,
                     pad_token_id=0)
    torch.manual_seed(0)
    return GPT2LMHeadModel(cfg)


def _greedy(model, tok, contract, prompt, max_new=4):
    with torch.inference_mode():
        device = next(model.parameters()).device
        ids = encode_generation(tok, prompt, contract)
        batch = {"input_ids": torch.tensor([ids], device=device),
                 "attention_mask": torch.ones((1, len(ids)),
                                              dtype=torch.long,
                                              device=device)}
        out = model.generate(**batch, max_new_tokens=max_new,
                             do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, len(ids):],
                          skip_special_tokens=True).strip()


def test_correct_supervision_learns_a_reviewed_transformation(tmp_path):
    """The positive-control claim at mechanism scale: with the shared
    contract and correct labels, LoRA fits transformations the frozen
    model cannot answer; the same pipeline renders training and
    inference identically."""
    tok = _tokenizer()
    contract = build_prompt_contract(tok)
    base = _model()
    frozen_hits = sum(_greedy(base, tok, contract, e["prompt"]) ==
                      e["response"] for e in EXAMPLES)

    model = _model()
    spec = LoraTrainSpec(rank=8, alpha=16, dropout=0.0,
                         target_modules=("c_attn",), learning_rate=1e-3,
                         steps=400, max_length=32, seed=0,
                         response_only_loss=True)
    model, receipt = train_lora(model=model, tokenizer=tok,
                                examples=EXAMPLES,
                                output_dir=tmp_path / "adapter", spec=spec,
                                prompt_contract=contract)
    assert receipt["prompt_contract_digest"] == contract.digest
    assert all(isinstance(v, (int, float)) for v in receipt["loss_curve"])
    assert receipt["loss_last"] < receipt["loss_first"]

    model.eval()
    learned_hits = sum(_greedy(model, tok, contract, e["prompt"]) ==
                       e["response"] for e in EXAMPLES)
    assert learned_hits > frozen_hits, (
        f"adapter fit {learned_hits}/{len(EXAMPLES)} vs frozen "
        f"{frozen_hits}/{len(EXAMPLES)}")
    assert learned_hits >= 2, "adapter failed to fit the reviewed task"


def test_training_receipt_reports_the_readiness_measurements(tmp_path):
    """Step 13 requires supervised-token counts, finite gradient norms
    and a measurable adapter weight change — a run whose weights never
    moved is a numerical failure, not a learning result."""
    tok = _tokenizer()
    contract = build_prompt_contract(tok)
    model = _model()
    spec = LoraTrainSpec(rank=8, alpha=16, dropout=0.0,
                         target_modules=("c_attn",), learning_rate=1e-3,
                         steps=20, max_length=32, seed=0,
                         response_only_loss=True)
    _, receipt = train_lora(model=model, tokenizer=tok,
                            examples=EXAMPLES,
                            output_dir=tmp_path / "adapter", spec=spec,
                            prompt_contract=contract)

    # supervised-token accounting
    assert receipt["supervised_tokens_total"] > 0
    assert receipt["prompt_tokens_total"] > 0
    assert (sum(receipt["supervised_tokens_per_example"])
            == receipt["supervised_tokens_total"])
    assert len(receipt["supervised_tokens_per_example"]) == len(EXAMPLES)
    assert all(n > 0 for n in receipt["supervised_tokens_per_example"])

    # finite, meaningful gradients
    assert receipt["grad_norms"]
    assert all(isinstance(g, float) and g == g
               and abs(g) != float("inf") for g in receipt["grad_norms"])
    assert receipt["grad_norm_mean"] > 0.0

    # the adapter actually moved
    assert receipt["adapter_delta_l2"] is not None
    assert receipt["adapter_delta_l2"] > 0.0
    assert receipt["adapter_params_changed"] > 0
    assert receipt["adapter_delta_relative"] > 0.0

    # cost accounting
    assert receipt["train_seconds"] >= 0.0
    assert receipt["trainable_params"] > 0
    assert receipt["trainable_params"] < receipt["total_params"]
