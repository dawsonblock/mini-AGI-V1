"""v16.3.0 corrected LoRA trainer (REPAIR-019..024 + Phase-2.1).
v16.4.11 (Campaign 3B.3) adds verified rehearsal interleaving and
frozen-reference KL preservation.

Two input modes:

  texts=     legacy concatenated strings, every token supervised —
             retained for historical V162-V164 replay semantics only
  examples=  structured {"prompt","response"} records (v165+):
             the model's chat template renders the prompt boundary,
             prompt/padding tokens are masked to -100, and only the
             response (+EOS) carries supervised loss. Malformed
             examples — empty response, or response fully truncated
             by max_length — are rejected, never silently dropped.

Rehearsal (P2): when ``rehearsal_fraction > 0``, a deterministic share
of microbatches draws from ``rehearsal_examples`` instead of the task
stream — every Nth microbatch where N = round(1/fraction). Rehearsal
pairs carry the SAME response-only supervised loss as task pairs; the
corpus is the caller's verified-correct set, so gradient mass lands on
behavior the frozen model already had. The mechanism is different from
inference-time replay (campaign arm L7): L7 put replayed examples in
the PROMPT at generation; rehearsal puts them in the GRADIENT, which is
what counteracts destructive weight updates. Sampled presentations are
recorded per-source so the receipt binds exactly what was rehearsed.

Frozen-reference KL (P3, repaired v4.0): when ``kl_weight > 0``, every
``kl_every``-th microbatch additionally computes the mean next-token KL
divergence between the adapter's distribution and the FROZEN BASE
distribution (adapter disabled — the same weights the frozen eval arm
measured) and adds ``kl_weight * KL`` to the update.

The KL set is RESPONSE-AWARE and teacher-forced — the earlier
prompt-only KL anchored the distribution over positions that predict
prompt tokens and never touched the continuations the preservation term
exists to protect (defect V4-02). Two supply modes:

  kl_examples  verified {"prompt","response"} pairs — encoded through
               encode_supervised, prompt positions masked out, KL over
               the response (continuation) positions only;
  kl_prompts   bare prompt strings — the FROZEN model's own greedy
               continuation is generated once at setup and becomes the
               teacher-forced reference (self-distillation: preserve
               what the frozen model actually says on those prompts).

Both modes constrain the response-continuation distribution; the
receipt records which reference source was used, the KL set digest,
per-position counts, and the reference model's parameter digest.

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
import hashlib
import importlib.metadata
import json
import random
import time
from egai.common.canonical import digest, sha256_bytes
from minagi.v161.prompt_contract import (DEFAULT_OUTPUT_CONTRACT,
                                         PromptContract,
                                         build_prompt_contract,
                                         encode_generation,
                                         encode_supervised)

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
    # Campaign 3B.3 — rehearsal interleave. rehearsal_fraction of
    # microbatches draw from rehearsal_examples (verified-correct
    # pairs). A deterministic schedule: microbatch index where
    # micro % period == period - 1, period = round(1/fraction).
    rehearsal_fraction: float = 0.0
    # Frozen-reference KL preservation: every kl_every-th microbatch
    # adds kl_weight * mean KL(frozen || adapted) over kl_prompts.
    kl_weight: float = 0.0
    kl_every: int = 1
    schema: str = "mini-agi-v16.3-lora-train-spec-v3"

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
        if not (0.0 <= self.rehearsal_fraction <= 1.0):
            raise ValueError("rehearsal_fraction must be in [0, 1]")
        if self.kl_weight < 0.0:
            raise ValueError("kl_weight must be >= 0")
        if self.kl_every < 1:
            raise ValueError("kl_every must be >= 1")

    @property
    def effective_batch_size(self) -> int:
        """Examples per optimizer update."""
        return self.microbatch_size * self.gradient_accumulation_steps

    @property
    def rehearsal_period(self) -> int | None:
        """Every Nth microbatch is a rehearsal microbatch."""
        if self.rehearsal_fraction <= 0.0:
            return None
        return max(1, round(1.0 / self.rehearsal_fraction))

    @property
    def digest(self): return digest(self)


def example_digest(prompt: str, response: str) -> str:
    return sha256_bytes(json.dumps({"prompt": prompt, "response": response},
                                   sort_keys=True).encode())


def supervised_tokens(tokenizer, prompt: str, response: str,
                      max_length: int,
                      contract: PromptContract | None = None) -> dict:
    """Tokenize one {prompt, response} pair with prompt masked to -100.

    Delegates to the shared prompt contract (SCI-301): the prompt
    boundary is rendered exactly as ``run_campaign1.generate`` renders
    it, and the chat-template path additionally proves the prompt token
    sequence is a token-level prefix of the supervised sequence. When no
    contract is supplied, the historical default surface (no answer
    instruction) is built from the tokenizer.
    """
    if contract is None:
        contract = build_prompt_contract(tokenizer, DEFAULT_OUTPUT_CONTRACT)
    return encode_supervised(tokenizer, prompt, response, contract,
                             max_length)


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


def _adapter_param_vector(model):
    """Snapshot of the trainable parameters (the adapter surface).

    Recorded before training so the receipt can report how far the
    adapter actually moved — a run whose weights never change is a
    numerical failure, not a learning result.
    """
    return [p.detach().clone() for p in model.parameters()
            if p.requires_grad]


def _adapter_weight_delta(before, model) -> dict:
    """L2 distance and relative change of the adapter weights."""
    after = [p.detach() for p in model.parameters() if p.requires_grad]
    if not before or len(before) != len(after):
        return {"adapter_delta_l2": None, "adapter_delta_relative": None}
    import torch
    delta = sum(float((a - b).pow(2).sum().cpu())
                for a, b in zip(after, before)) ** 0.5
    base = sum(float(b.pow(2).sum().cpu()) for b in before) ** 0.5
    return {"adapter_delta_l2": delta,
            "adapter_delta_relative": (delta / base) if base else None,
            "adapter_params_changed": int(sum(
                1 for a, b in zip(after, before)
                if not torch.equal(a, b)))}


def _peak_memory_bytes() -> int | None:
    """Peak accelerator memory for this training run (None on CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:                                    # pragma: no cover
        return None
    return None


def _reference_weights_digest(model) -> str:
    """Digest of the frozen (non-trainable) parameter bytes — the
    identity of the reference model the KL term anchors to. Bound into
    the receipt so 'the frozen reference' is a measured artifact, not
    a name."""
    import torch
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if p.requires_grad:
            continue
        h.update(name.encode())
        t = p.detach().cpu()
        if not t.is_contiguous():
            t = t.contiguous()
        try:
            h.update(t.view(torch.uint8).numpy().tobytes())
        except (RuntimeError, TypeError):
            h.update(t.float().numpy().tobytes())
    return h.hexdigest()


def _frozen_kl(model, kl_encoded, kl_weight, gacc, pad_id):
    """Mean next-token KL(frozen || adapted) over RESPONSE positions.

    Teacher-forced KL preservation (V4-02): ``kl_encoded`` entries are
    encode_supervised outputs — input_ids plus labels in which prompt
    and padding tokens are masked to -100 and only the response
    (continuation) tokens carry supervision. Causal alignment is the
    standard LM shift: logits[:, t] predicts token t+1, so the mask is
    labels[:, 1:] != -100 evaluated over logits[:, :-1]. Prompt and
    padding positions contribute exactly nothing — the penalty is
    measured on the answer-generation behavior it exists to preserve.

    The reference is the SAME model with the adapter disabled — the
    frozen weights themselves, not a copy — so 'preservation' is
    measured against the exact distribution the frozen arm evaluated.
    The set is batched into one forward per side, so the KL term costs
    two forwards per scheduled microbatch rather than 2 x N. The
    log-softmax is computed in float32 for numerical stability under
    mixed-precision training.
    """
    import torch
    import torch.nn.functional as F
    seqs = [e for e in kl_encoded
            if len(e["input_ids"]) >= 2
            and any(t != LABEL_IGNORE for t in e["labels"])]
    if not seqs:
        return None
    width = max(len(e["input_ids"]) for e in seqs)
    ids = torch.tensor([e["input_ids"] + [pad_id] * (width - len(e["input_ids"]))
                        for e in seqs],
                       device=next(model.parameters()).device)
    attn = torch.tensor([[1] * len(e["input_ids"]) + [0] * (width - len(e["input_ids"]))
                         for e in seqs], device=ids.device)
    lab = torch.tensor([e["labels"] + [LABEL_IGNORE] * (width - len(e["labels"]))
                        for e in seqs], device=ids.device)
    # response-token mask, shifted to the predicting positions:
    # logits[:, t] -> token t+1, so supervise where labels[:, t+1] != -100
    pos_ok = (lab[:, 1:] != LABEL_IGNORE).float()
    if float(pos_ok.sum()) == 0.0:
        return None
    with torch.no_grad():
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                ref_logits = model(input_ids=ids,
                                   attention_mask=attn).logits
        else:                                    # pragma: no cover
            raise RuntimeError(
                "model exposes no disable_adapter() — cannot "
                "produce a frozen reference")
        ref_logp = F.log_softmax(ref_logits[:, :-1].float(), dim=-1)
    cur_logits = model(input_ids=ids, attention_mask=attn).logits
    cur_logp = F.log_softmax(cur_logits[:, :-1].float(), dim=-1)
    # KL(ref || cur) per position, weighted to response positions only
    kl_pos = (ref_logp.exp() * (ref_logp - cur_logp)).sum(dim=-1)
    kl = (kl_pos * pos_ok).sum() / pos_ok.sum().clamp(min=1.0)
    return kl * kl_weight / gacc


def _kl_reference_continuations(model, tokenizer, prompts, contract,
                                max_length):
    """Frozen model's own greedy continuations — the self-distillation
    reference for bare ``kl_prompts``.

    A prompt alone does not define the response distribution to
    preserve, so the adapter-disabled model generates its greedy answer
    once under the output contract; that continuation is then
    teacher-forced through encode_supervised exactly like a verified
    pair. EOS is handled by the output contract (stop_on_eos); the
    emitted tokens are decoded and re-encoded so the supervised surface
    is identical to a verified pair.
    """
    import torch
    pairs = []
    with torch.inference_mode():
        if hasattr(model, "disable_adapter"):
            ctx = model.disable_adapter()
        else:                                    # pragma: no cover
            raise RuntimeError(
                "model exposes no disable_adapter() — cannot "
                "produce a frozen reference")
        with ctx:
            for p in prompts:
                ids = encode_generation(tokenizer, str(p), contract)
                device = next(model.parameters()).device
                batch = {"input_ids": torch.tensor([ids], device=device),
                         "attention_mask": torch.ones(
                             (1, len(ids)), dtype=torch.long,
                             device=device)}
                out = model.generate(
                    **batch,
                    max_new_tokens=int(contract.output.max_new_tokens),
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
                cont = list(out[0, len(ids):])
                eos_id = tokenizer.eos_token_id
                if eos_id is not None and eos_id in cont:
                    cont = cont[:cont.index(eos_id)]
                resp = tokenizer.decode(
                    cont, skip_special_tokens=True).strip()
                if not resp:
                    # special-token-only continuations still carry a
                    # real distribution to preserve — keep the raw
                    # tokenization surface instead of dropping the row
                    resp = tokenizer.decode(
                        cont, skip_special_tokens=False).strip()
                if not resp:
                    continue
                pairs.append({"prompt": str(p), "response": resp,
                              "reference_source": "frozen_generation"})
    return pairs


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
               output_dir: str | Path, spec: LoraTrainSpec,
               prompt_contract: PromptContract | None = None,
               rehearsal_examples: Sequence[Mapping] | None = None,
               kl_prompts: Sequence[str] | None = None,
               kl_examples: Sequence[Mapping] | None = None):
    """Train a LoRA adapter and persist it.

    `examples` (v165+, structured {prompt,response} masking) takes
    precedence over legacy `texts`. The structured path encodes through
    `prompt_contract` (default: the historical no-instruction surface
    built from the tokenizer), so training renders the same input the
    runner's generation contract renders. Returns (model, receipt).

    `rehearsal_examples` — verified-correct {prompt,response} pairs
    interleaved into the gradient stream on the spec's rehearsal
    schedule (P2). `kl_examples` — verified {prompt,response} reference
    pairs teacher-forced through the frozen-reference KL term;
    `kl_prompts` — bare prompts whose FROZEN-model greedy continuations
    are derived at setup and teacher-forced the same way (P3, v4.0:
    response-aware KL — prompt-position-only KL no longer exists).
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if examples is None and not texts:
        raise ValueError("training texts or examples required")
    if spec.rehearsal_fraction > 0 and not rehearsal_examples:
        raise ValueError(
            "rehearsal_fraction > 0 requires rehearsal_examples — "
            "rehearsing nothing is not rehearsal")
    if spec.kl_weight > 0 and not kl_prompts and not kl_examples:
        raise ValueError(
            "kl_weight > 0 requires kl_examples (verified pairs) or "
            "kl_prompts (frozen continuations derived at setup) — a KL "
            "term without a reference set measures nothing")
    if prompt_contract is None:
        prompt_contract = build_prompt_contract(tokenizer,
                                                DEFAULT_OUTPUT_CONTRACT)
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
    rehearsal_period = spec.rehearsal_period

    kl_encoded = None
    reference_digest = None
    kl_prompts_digest = None
    kl_set_digest = None
    kl_set_size = 0
    kl_reference_sources: dict = {}
    rehearsal_corpus_digest = None
    rehearsal_corpus_size = 0
    if spec.kl_weight > 0:
        # V4-02: response-aware teacher-forced KL. Verified pairs are
        # encoded as-is; bare prompts get their reference continuation
        # from the FROZEN model's own greedy generation (adapter
        # disabled) — preserve what the base actually says.
        kl_pairs = [{"prompt": str(e["prompt"]),
                     "response": str(e["response"]),
                     "reference_source": str(
                         e.get("reference_source", "verified_pairs"))}
                    for e in (kl_examples or [])]
        if kl_prompts:
            kl_pairs += _kl_reference_continuations(
                model, tokenizer, kl_prompts, prompt_contract,
                spec.max_length)
        kl_set_digest = digest(
            [{"prompt": p["prompt"], "response": p["response"]}
             for p in kl_pairs])
        kl_reference_sources = {}
        for p in kl_pairs:
            kl_reference_sources[p["reference_source"]] = \
                kl_reference_sources.get(p["reference_source"], 0) + 1
        kl_encoded = []
        kl_rejected = 0
        for p in kl_pairs:
            try:
                kl_encoded.append(supervised_tokens(
                    tokenizer, p["prompt"], p["response"],
                    spec.max_length, contract=prompt_contract))
            except ValueError:
                kl_rejected += 1
        if not kl_encoded:
            raise ValueError(
                "every KL reference pair was rejected — the KL term "
                "has no measurable response surface")
        kl_prompts_digest = kl_set_digest
        kl_set_size = len(kl_encoded)
        reference_digest = _reference_weights_digest(model)

    losses, rejected = [], 0
    grad_norms: list[float] = []
    kl_trace: list[float] = []
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
                    spec.max_length, contract=prompt_contract))
            except ValueError as exc:
                rejected += 1
                if spec.malformed_policy == "fail":
                    raise ValueError(
                        "malformed training example under strict "
                        f"malformed_policy='fail' — run invalidated: "
                        f"{exc}") from exc
        if not encoded:
            raise ValueError("every training example was rejected")
        encoded_rehearsal: list = []
        rehearsal_rejected = 0
        if rehearsal_examples:
            rehearsal_corpus_digest = digest(
                [example_digest(str(e["prompt"]), str(e["response"]))
                 for e in rehearsal_examples])
            for e in rehearsal_examples:
                try:
                    encoded_rehearsal.append(supervised_tokens(
                        tokenizer, str(e["prompt"]), str(e["response"]),
                        spec.max_length, contract=prompt_contract))
                except ValueError as exc:
                    rehearsal_rejected += 1
                    if spec.malformed_policy == "fail":
                        raise ValueError(
                            "malformed rehearsal example — run "
                            f"invalidated: {exc}") from exc
            if rehearsal_period is not None and not encoded_rehearsal:
                raise ValueError(
                    "rehearsal scheduled but every rehearsal example "
                    "was rejected")
            rehearsal_corpus_size = len(encoded_rehearsal)
        rejected += rehearsal_rejected
        # Step 13: supervised-token accounting and the adapter's initial
        # parameter state, so a training run reports what it actually
        # supervised and how far the weights moved.
        supervised_token_counts = [int(e["n_supervised"]) for e in encoded]
        supervised_tokens_total = int(sum(supervised_token_counts))
        prompt_tokens_total = int(sum(int(e["n_prompt"]) for e in encoded))
        adapter_initial = _adapter_param_vector(model)
        rnd = random.Random(spec.seed)
        total_micro = spec.steps * gacc
        sampled: list[dict] = []
        for micro in range(total_micro):
            is_rehearsal = (rehearsal_period is not None
                            and micro % rehearsal_period
                            == rehearsal_period - 1)
            pool = encoded_rehearsal if is_rehearsal else encoded
            src = "rehearsal" if is_rehearsal else "task"
            idx = [rnd.randrange(len(pool)) for _ in range(mb)]
            sampled.extend({"src": src, "idx": i} for i in idx)
            batch = [pool[i] for i in idx]
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
            if kl_encoded is not None and micro % spec.kl_every == 0:
                kl_term = _frozen_kl(model, kl_encoded, spec.kl_weight,
                                     gacc, pad_id)
                if kl_term is not None:
                    loss = loss + kl_term
                    kl_trace.append(float(kl_term.detach().cpu()) * gacc)
            loss.backward()
            losses.append(float(loss.detach().cpu()) * gacc)
            if (micro + 1) % gacc == 0 or micro == total_micro - 1:
                grad_norms.append(_grad_norm(model))
                opt.step()
                opt.zero_grad(set_to_none=True)
                updates += 1
        example_count = len(encoded)
        presentations = len(sampled)
        rehearsal_presentations = sum(1 for s in sampled
                                      if s["src"] == "rehearsal")
        sample_order_digest = digest(
            [{"src": s["src"], "idx": int(s["idx"])} for s in sampled])
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
            loss = out.loss
            if kl_encoded is not None and step % spec.kl_every == 0:
                kl_term = _frozen_kl(model, kl_encoded, spec.kl_weight,
                                     1, pad_id)
                if kl_term is not None:
                    loss = loss + kl_term
                    kl_trace.append(float(kl_term.detach().cpu()))
            loss.backward()
            grad_norms.append(_grad_norm(model))
            opt.step()
            opt.zero_grad(set_to_none=True)
            updates += 1
            losses.append(float(loss.detach().cpu()))
        example_count = len(texts)
        presentations = len(sampled)
        rehearsal_presentations = 0
        sample_order_digest = digest(
            [{"src": "task", "idx": int(i)} for i in sampled])

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
               "rehearsal_presentations": rehearsal_presentations,
               "rehearsal_fraction_config": spec.rehearsal_fraction,
               "rehearsal_fraction_realized":
                   (rehearsal_presentations / presentations)
                   if presentations else 0.0,
               "rehearsal_period_microbatches": rehearsal_period,
               "rehearsal_corpus_digest": rehearsal_corpus_digest,
               "rehearsal_corpus_size": rehearsal_corpus_size,
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
    if spec.kl_weight > 0:
        receipt["kl_preservation"] = {            "kl_weight": spec.kl_weight,
            "kl_every": spec.kl_every,
            "n_kl_evaluations": len(kl_trace),
            "kl_term_mean": (sum(kl_trace) / len(kl_trace))
                            if kl_trace else None,
            # V4-02: the KL set is teacher-forced {prompt,response}
            # pairs and the divergence is masked to response
            # (continuation) token positions only.
            "kl_scope": "response_continuations",
            "kl_set_digest": kl_set_digest,
            "kl_set_size": kl_set_size,
            "kl_reference_sources": kl_reference_sources,
            "kl_prompts_digest": kl_prompts_digest,
            "kl_prompt_count": len(kl_prompts or []),
            "kl_example_count": len(kl_examples or []),
            # the frozen reference is the base weights with the adapter
            # disabled — its digest binds WHICH frozen model anchored
            # the penalty, satisfying P3's identity-binding requirement
            "reference_model": {
                "identity": getattr(getattr(model, "config", None),
                                    "_name_or_path", None)
                            or getattr(getattr(model, "config", None),
                                       "name_or_path", None),
                "weights_digest": reference_digest,
                "binding": "same-weights, adapter-disabled forward"}}
    if examples is not None:
        # SCI-301: the exact rendering contract this adapter was trained
        # under, so training/inference parity is auditable from the
        # artifact alone.
        receipt["prompt_contract_digest"] = prompt_contract.digest
        receipt["output_contract_digest"] = \
            prompt_contract.output_contract_digest
        # Step 13 readiness measurements: what was supervised, how far
        # the adapter moved, and what the run cost.
        receipt["supervised_tokens_total"] = supervised_tokens_total
        receipt["prompt_tokens_total"] = prompt_tokens_total
        receipt["supervised_tokens_per_example"] = supervised_token_counts
        receipt.update(_adapter_weight_delta(adapter_initial, model))
        peak = _peak_memory_bytes()
        if peak is not None:
            receipt["peak_accelerator_bytes"] = peak
        receipt["train_seconds"] = (receipt["finished_ns"]
                                    - receipt["started_ns"]) / 1e9
    (outdir / "TRAINING_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True))
    (outdir / "TRAINING_SPEC.json").write_text(
        json.dumps(asdict(spec), indent=2, sort_keys=True))
    return model, receipt


def train_family_adapters(*, model_loader, tokenizer,
                          family_examples: Mapping[str, Sequence[Mapping]],
                          output_dir: str | Path, spec: LoraTrainSpec,
                          prompt_contract: PromptContract | None = None,
                          on_adapter=None) -> dict[str, dict]:
    """Train one parameter-isolated adapter per task family (V4-17).

    Each family gets its own adapter directory under ``output_dir``,
    trained from a FRESH base model supplied by ``model_loader()`` — a
    fresh load per adapter is required, otherwise successive
    get_peft_model calls would stack adapters on top of each other and
    the family artifacts would not be independent. The same spec and
    prompt contract apply to every adapter, so the only experimental
    variable is the family the adapter saw.

    ``family_examples`` maps family_id -> [{prompt, response}, ...].
    ``on_adapter(family, receipt)`` is called after each adapter lands
    (useful for freeing the returned model between families).

    Returns {family_id: {"adapter_dir", "adapter_digest", "receipt"}} —
    the manifest a verified router needs to bind adapters to families.
    """
    results: dict[str, dict] = {}
    out_root = Path(output_dir)
    for family in sorted(family_examples):
        examples = family_examples[family]
        if not examples:
            raise ValueError(f"family {family!r} has no training examples")
        model = model_loader()
        adir = out_root / f"adapter-{family}"
        model, receipt = train_lora(
            model=model, tokenizer=tokenizer, examples=examples,
            output_dir=adir, spec=spec, prompt_contract=prompt_contract)
        results[family] = {
            "adapter_dir": str(adir),
            "adapter_digest": receipt["adapter_artifact_digest"],
            "receipt": receipt}
        if on_adapter is not None:
            on_adapter(family, receipt)
        del model
    manifest = {"schema": "mini-agi-v4.0-family-adapter-manifest-v1",
                "families": sorted(results),
                "train_spec_digest": spec.digest,
                "adapters": {f: {"dir": r["adapter_dir"],
                                 "digest": r["adapter_digest"]}
                             for f, r in results.items()}}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "FAMILY_ADAPTERS.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    return results
