"""v16.3.0 corrected LoRA trainer (REPAIR-019..024 + Phase-2.1).

Two input modes:

  texts=     legacy concatenated strings, every token supervised —
             retained for historical V162-V164 replay semantics only
  examples=  structured {"prompt","response"} records (v165+):
             the model's chat template renders the prompt boundary,
             prompt/padding tokens are masked to -100, and only the
             response (+EOS) carries supervised loss. Malformed
             examples — empty response, or response fully truncated
             by max_length — are rejected, never silently dropped.

v16.3.0 (Phase 2.1) — explicit training schedule. Every microbatch
carries exactly `microbatch_size` examples, every optimizer update
accumulates exactly `gradient_accumulation_steps` microbatches, so

    effective_batch_size = microbatch_size * gradient_accumulation_steps
    optimizer_updates    = steps
    sample_presentations = steps * effective_batch_size

The original trainer placed `gradient_accumulation_steps` examples in
each microbatch *and* accumulated over `gradient_accumulation_steps`
microbatches, silently squaring the effective batch size. The receipt
binds all schedule quantities, the actual sample order, per-update
gradient statistics, and the produced adapter artifact digest.

The training receipt records the full curve, the canonical data-order
digest, resolved optimizer, library versions, trainable parameter
counts, and wall-clock cost — the protocol binds the config; the
receipt binds what actually happened.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping, Sequence
import importlib.metadata
import json
import random
import time
from egai.common.canonical import digest, sha256_bytes

LABEL_IGNORE = -100


@dataclass(frozen=True)
class LoraTrainSpec:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("c_attn",)
    learning_rate: float = 2e-4
    steps: int = 20
    max_length: int = 128
    seed: int = 0
    optimizer: str = "adamw"
    gradient_accumulation_steps: int = 1
    microbatch_size: int = 1
    response_only_loss: bool = False
    malformed_policy: str = "reject"   # reject | fail (strict campaigns)
    schema: str = "mini-agi-v16.3-lora-train-spec-v2"

    def __post_init__(self):
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.microbatch_size < 1:
            raise ValueError("microbatch_size must be >= 1")
        if self.optimizer not in ("adamw", "adamw-torch", "adamw_fused",
                                  "sgd"):
            raise ValueError(f"unsupported optimizer: {self.optimizer}")
        if self.malformed_policy not in ("reject", "fail"):
            raise ValueError("malformed_policy must be reject|fail")

    @property
    def effective_batch_size(self) -> int:
        """Examples per optimizer update."""
        return self.microbatch_size * self.gradient_accumulation_steps

    @property
    def digest(self): return digest(self)


def example_digest(prompt: str, response: str) -> str:
    return sha256_bytes(json.dumps({"prompt": prompt, "response": response},
                                   sort_keys=True).encode())


def supervised_tokens(tokenizer, prompt: str, response: str,
                      max_length: int) -> dict:
    """Tokenize one {prompt, response} pair with prompt masked to -100.

    The chat template (when the tokenizer defines one) renders the
    prompt boundary exactly as inference does, so training sees the
    same surface generation will see.
    """
    if not prompt or not str(response).strip():
        raise ValueError("empty prompt or response")
    messages = [{"role": "user", "content": str(prompt)}]
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": str(response)}],
            tokenize=False)
        if not full_text.startswith(prompt_text):
            raise ValueError("chat template render is not prefix-consistent")
        response_text = full_text[len(prompt_text):]
    else:
        prompt_text = str(prompt)
        eos = getattr(tokenizer, "eos_token", "") or ""
        response_text = str(response) + eos
    p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    input_ids = list(p_ids) + list(r_ids)
    labels = [LABEL_IGNORE] * len(p_ids) + list(r_ids)
    if len(input_ids) > max_length:
        # keep the response tail: drop leading prompt context, never the
        # supervised suffix
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
    n_sup = sum(1 for t in labels if t != LABEL_IGNORE)
    if n_sup == 0:
        raise ValueError("response fully truncated by max_length — "
                         "no supervised tokens remain")
    return {"input_ids": input_ids, "labels": labels,
            "n_prompt": len(labels) - n_sup, "n_supervised": n_sup}


def _pad_batch(encoded, pad_token_id):
    width = max(len(e["input_ids"]) for e in encoded)
    input_ids, labels, attn = [], [], []
    for e in encoded:
        pad = width - len(e["input_ids"])
        input_ids.append(e["input_ids"] + [pad_token_id] * pad)
        labels.append(e["labels"] + [LABEL_IGNORE] * pad)
        attn.append([1] * len(e["input_ids"]) + [0] * pad)
    return input_ids, labels, attn


def _library_versions():
    out = {}
    for pkg in ("torch", "transformers", "peft"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except Exception:
            out[pkg] = "unknown"
    return out


def _make_optimizer(model, spec):
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    if spec.optimizer == "sgd":
        return torch.optim.SGD(params, lr=spec.learning_rate)
    fused = spec.optimizer == "adamw_fused" and torch.cuda.is_available()
    return torch.optim.AdamW(params, lr=spec.learning_rate, fused=fused)


def _grad_norm(model) -> float:
    """L2 norm of the currently accumulated parameter gradients —
    recorded once per optimizer update (gradient statistics)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().cpu())
    return total ** 0.5


def adapter_artifact_digest(adapter_dir) -> str:
    """Digest of the produced adapter weight+config bytes only.

    Stable across the later receipt/spec writes (those evidence files
    are covered by the directory closure digest in runtime manifests),
    so the training receipt can bind the exact artifact it produced
    before the receipt itself exists on disk."""
    d = Path(adapter_dir)
    rows = []
    cfg = d / "adapter_config.json"
    if not cfg.is_file():
        raise ValueError("adapter_config.json missing from output dir")
    rows.append(("adapter_config.json", sha256_bytes(cfg.read_bytes())))
    weights = sorted(p for p in d.glob("adapter_model*") if p.is_file())
    if not weights:
        raise ValueError("adapter_model weights missing from output dir")
    for p in weights:
        rows.append((p.name, sha256_bytes(p.read_bytes())))
    return digest(rows)


def train_lora(*, model, tokenizer, texts: Sequence[str] | None = None,
               examples: Sequence[Mapping] | None = None,
               output_dir: str | Path, spec: LoraTrainSpec):
    """Train a LoRA adapter and persist it.

    `examples` (v165+, structured {prompt,response} masking) takes
    precedence over legacy `texts`. Returns (model, receipt).
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if examples is None and not texts:
        raise ValueError("training texts or examples required")
    random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        pass
    cfg = LoraConfig(r=spec.rank, lora_alpha=spec.alpha,
                     lora_dropout=spec.dropout,
                     target_modules=list(spec.target_modules),
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.train()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    opt = _make_optimizer(model, spec)
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    gacc = spec.gradient_accumulation_steps
    mb = spec.microbatch_size

    losses, rejected = [], 0
    grad_norms: list[float] = []
    updates = 0
    started = time.time_ns()
    if examples is not None:
        mode = "response-masked" if spec.response_only_loss else \
            "concatenated"
        order_d = digest([example_digest(str(e["prompt"]),
                                         str(e["response"]))
                          for e in examples])
        encoded = []
        for e in examples:
            try:
                encoded.append(supervised_tokens(
                    tokenizer, str(e["prompt"]), str(e["response"]),
                    spec.max_length))
            except ValueError as exc:
                rejected += 1
                if spec.malformed_policy == "fail":
                    raise ValueError(
                        "malformed training example under strict "
                        f"malformed_policy='fail' — run invalidated: "
                        f"{exc}") from exc
        if not encoded:
            raise ValueError("every training example was rejected")
        rnd = random.Random(spec.seed)
        total_micro = spec.steps * gacc
        sampled: list[int] = []
        for micro in range(total_micro):
            idx = [rnd.randrange(len(encoded)) for _ in range(mb)]
            sampled.extend(idx)
            batch = [encoded[i] for i in idx]
            ids, labels, attn = _pad_batch(batch, pad_id)
            batch_t = {"input_ids": torch.tensor(ids, device=device),
                       "attention_mask": torch.tensor(attn, device=device),
                       "labels": torch.tensor(labels, device=device)}
            if spec.response_only_loss:
                out = model(**batch_t)
            else:  # concatenate prompt+response, supervise everything
                concat_labels = batch_t["input_ids"].clone()
                concat_labels[batch_t["attention_mask"] == 0] = LABEL_IGNORE
                out = model(input_ids=batch_t["input_ids"],
                            attention_mask=batch_t["attention_mask"],
                            labels=concat_labels)
            loss = out.loss / gacc
            loss.backward()
            losses.append(float(loss.detach().cpu()) * gacc)
            if (micro + 1) % gacc == 0 or micro == total_micro - 1:
                grad_norms.append(_grad_norm(model))
                opt.step()
                opt.zero_grad(set_to_none=True)
                updates += 1
        example_count = len(encoded)
        presentations = len(sampled)
        sample_order_digest = digest([int(i) for i in sampled])
    else:
        mode = "concatenated-legacy"
        order_d = digest([sha256_bytes(str(t).encode()) for t in texts])
        sampled = []
        for step in range(spec.steps):
            i = step % len(texts)
            sampled.append(i)
            batch = tokenizer(texts[i], return_tensors="pt",
                              truncation=True, max_length=spec.max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = LABEL_IGNORE
            out = model(**batch, labels=labels)
            out.loss.backward()
            grad_norms.append(_grad_norm(model))
            opt.step()
            opt.zero_grad(set_to_none=True)
            updates += 1
            losses.append(float(out.loss.detach().cpu()))
        example_count = len(texts)
        presentations = len(sampled)
        sample_order_digest = digest([int(i) for i in sampled])

    trainable = int(sum(p.numel() for p in model.parameters()
                        if p.requires_grad))
    total = int(sum(p.numel() for p in model.parameters()))
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(outdir, safe_serialization=True)
    artifact_digest = adapter_artifact_digest(outdir)
    receipt = {"schema": "mini-agi-v16.3-training-receipt-v2",
               "train_spec_digest": spec.digest,
               "supervision_mode": mode,
               "optimizer": spec.optimizer,
               "learning_rate": spec.learning_rate,
               "steps": spec.steps,
               "gradient_accumulation_steps": gacc,
               "microbatch_size": mb,
               "effective_batch_size": spec.effective_batch_size,
               "optimizer_updates": updates,
               "sample_presentations": presentations,
               "malformed_policy": spec.malformed_policy,
               "example_count": example_count,
               "rejected_examples": rejected,
               "data_order_digest": order_d,
               "sample_order_digest": sample_order_digest,
               "adapter_artifact_digest": artifact_digest,
               "loss_first": losses[0], "loss_last": losses[-1],
               "loss_curve": losses,
               "grad_norms": grad_norms,
               "grad_norm_mean": (sum(grad_norms) / len(grad_norms))
                                 if grad_norms else 0.0,
               "trainable_params": trainable, "total_params": total,
               "library_versions": _library_versions(),
               "started_ns": started, "finished_ns": time.time_ns()}
    (outdir / "TRAINING_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True))
    (outdir / "TRAINING_SPEC.json").write_text(
        json.dumps(asdict(spec), indent=2, sort_keys=True))
    return model, receipt
