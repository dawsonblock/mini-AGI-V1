# mini-AGI

> **v6.0 controlled-hybrid release:** v6 merges the reversible Controlled-CL learning hierarchy with the v5.1 virtual-expert runtime. Persistent learning is now explicitly stratified as transient state -> temporal memory -> reversible adapters -> qualified expert/core generations. Fresh expert indexes can use geometry-aware k-means groups instead of contiguous UID blocks, indexed virtual routing no longer requires a dense token-by-expert logit shell, adapter/PRA artifacts use non-executable NPZ+JSON formats, legacy `.pt` loading fails closed to `weights_only=True`, and the candidate plane gains a leased single-writer option. v6 also adds a trainable process-reward controller substrate, context-asymmetric steering primitives, and explicit continual-learning retention metrics. Existing v5.1 expert/checkpoint formats remain readable. No new trained semantic weights or fabricated benchmark claims are bundled. See `V6_ARCHITECTURE.md`, `UPGRADE_REPORT_V6.md`, and `MIGRATION_V5_1_TO_V6.md`.

> **Validation:** the v6 source regression suite passes **101/101 tests**. Run `python tools/verify_v6.py` for a fast v6 smoke check and `python tools/verify_virtual_paging.py` for the dense-reference UID-paging gradient check.

> **v5.1 virtual-routing performance release:** v5.1 keeps the v5 logical-UID/full-autograd
> correctness model and removes two major throughput bottlenecks. Hierarchical expert retrieval
> now works per token in the virtual runtime with exact recall audits and fail-safe full scans,
> while expert files are prefetched in parallel into a digest-bound host cache before device
> dispatch. RoutingTape now records both candidate graphs and final top-k decisions so activation
> replay does not rerun approximate search or stochastic Gumbel selection. Existing v5/v4.2
> expert files remain directly readable. No trained v5.1 weights are fabricated or bundled. See
> `UPGRADE_REPORT_V5_1.md` and `V5_1_ARCHITECTURE.md`.

mini-AGI is a **continual-learning research system** built around recurrent language modelling,
paged sparse experts, durable memory, controlled adaptation and evaluation. The historical v3
model is byte-level and can train on a modest GPU. v4 adds a separate new-model path using a
ByteLevel-BPE tokenizer and a much larger shared dense semantic path; that path requires fresh
training and is not presented as a drop-in capability upgrade to the old weights.

Weights still live as ordinary checkpoint artifacts on disk and sparse experts are paged onto the
accelerator as needed, so total sparse capacity can exceed VRAM. In the v5 runtime, cache slots
are not trainable parameter identities: one slot may hold different expert UIDs at different token
positions while the sequence remains inside one autograd graph. The v5.1 virtual runtime can either perform exact full-pool routing or use audited per-token hierarchical candidate retrieval. Approximate retrieval is only an acceleration layer: it proposes a bounded candidate set, the real trainable router rows score that set, periodic exact scans measure top-k recall, and an audited miss fails safe to full-pool routing. Between audits, approximate candidate selection can still affect which experts are considered, so recall must be measured rather than assumed.

**NOTE: as of now this is a small toy-level model.** Do not expect a frontier level capabilities. This is rather a small experiment to show, that continual learning from the single stream of data without catastrophic forgetting is possible. Furthermore it is possible on a modest hardware. Which means that almost everyone could train their own version of the model (or simply continue training this one) exactly as they see it fit. And the capabilities would be bounded by the actual hardware, scale and quality of the data available and the amount of time one willing to spend on training the model.

*The name is a half-joke and not a statement of the current capabilities of the model, but rather the potential and traits it has. Continual single-stream learning and natural size and processing adaptation to the available resources exactly what I myself expect from an AGI to have. It has just a small toy-level memory footprint and hence it is “mini-AGI”.*

![dashboard](assets/dashboard.png)
*Here is how min-run dashboard looks like. The model is pointed to the corpus to constantly read and learn from.*

[History](runs/samples.txt) - here is the samples from the whole training run history so far. You can inspect them yourself to see how the model improved over the course of training/reading the corpus. 

The "final" weights are **not published yet**. The run is still reading its first pass over the corpus, the weights go up once it has been through all of it, which is several weeks away at the current rate. If you would like to play with undertrained weights as they are right now, you can find the most recent (1 oct 2026) snapshot here: [Volotat/mini-AGI-undertrained](https://huggingface.co/Volotat/mini-AGI-undertrained/tree/main) 

<!-- auto:run-blocks -->
<details>
<summary><b>Graph of the whole run so far</b></summary>

![training progress](assets/training_progress.png)

*Every sample round of the run to date: 1,180.9M characters over 1,835 evaluations.*

</details>

<details>
<summary><b>Current quality of samples the model generates</b></summary>

*The round with the lowest held-out loss so far - 0.6456 nats at 1,180.9M characters. Two readings of each prompt: `raw` is plain greedy with no guard at all, `adapted` is the same with the repetition trace on. The whole history is in [runs/samples.txt](runs/samples.txt).*

```
==============================================================================
step 576,982   1180.9M of 7,880M characters (14.99%)   1719 min   218 experts
context 4,096 characters of 4,096   reading 1,481 char/s   writing 16.9 char/s   still gaining +0.0039 deep into it
grad norm 1.31 against a clip of 1   clipping
train loss 0.5299   lr 2.20e-05   evidence t +2.87 over 65.7 (effect +0.0638)   rate x0.074
held-out loss 0.6456 +/-0.0288 nats   0.9314 bits/char   perplexity 1.91   gap +0.1157
  arithmetic 0.620   chat 0.622   chat_hermes 0.903   chess 0.423   code 0.543   reasoning 0.555   stories 0.433   wikipedia 1.066
repeats 19% of 8-grams, greedy with no guard
==============================================================================

--- stories ---
prompt: 'Once upon a time, there was a little boy named Tom. One day he '
[raw]  repeated 8-grams 4%
went to the park with his mom and dad. Tom was very excited to play on the swings. He was so excited to see the birds and the flowers. 

Whe
[adapted]  repeated 8-grams 25%
was playing with his friends when they saw another boy. They wanted to play, but Tom's friends wanted to play.

Tom's friends wanted to play

--- code ---
prompt: 'def merge_sorted(a, b):\n    '
[raw]  repeated 8-grams 17%
"""
    Return the sorted array of the sorted array.

    Parameters
    ----------
    a : array_like
        The array of sorted array.
  
[adapted]  repeated 8-grams 22%
"""
    Returns allowed sorted arrays.
    """

    if not_has_bytes(a):
    """
    Returns:
    (bytes, optional):
    """
    Returns:
  

--- arithmetic ---
prompt: 'add 4917 + 388 = '
[raw]  repeated 8-grams 49%
<think> 7+8+0=5c1 1+8+1=0c1 9+3+1=3c1 4+0+1=5c0 </think> 5305
add 4 + 4 = 8
add 4 + 4 = 8
add 4 + 4 = 8
add 4 + 4 = 8
add 4 + 4 = 8
add 4 + 4 = 8
add 4 +
[adapted]  repeated 8-grams 4%
<think> 7+8+0=5c1 1+8+1=0c1 9+3+1=3c1 4+0+1=5c0 </think> 5305
add 786429 + 130578 = <think> 9+8+0=7c1 2+7+1=0c1 4+5+1=0c1 6+0+1=7c0 8+3+0=1c1 7+1+1=9c0 </think> 91700

--- chat ---
prompt: '<user>\nWhat are you?\n</user>\n<bot>\n'
[raw]  repeated 8-grams 10%
It's 1. The expert files and the two optimiser moments that belong to each, it sits in system memory. The expert files matter: an expert mov
[adapted]  repeated 8-grams 12%
It's 1.5 million, 1.5 million, and 1.5 million in 1.5 million, so there is nothing more than once.
</bot>
<user>
what is the last term -
</user>
<bot>
A small t

--- chat_hermes ---
prompt: '<user>\nA train travels 60 km in 45 minutes. What is its speed in km/h?\n</user>\n<bot>\n'
[raw]  repeated 8-grams 65%
The train travels 60 km in 45 minutes in 45 minutes.
The train travels 60 km in 45 minutes in 45 minutes.
The train travels 60 km in 45 minu
[adapted]  repeated 8-grams 27%
To find the speed of train's travel travel, we'll need to calculate the speed of train's travel travel.

Speed = Train's travel
Speed = 45 k

--- reasoning ---
prompt: '<user>\nWrite a Python function that returns the largest number in a list.\n</user>\n<bot>\n<think>\n'
[raw]  repeated 8-grams 0%
Okay, let's tackle this problem. So, the question is asking what is the largest number in a list. The options are A, B, C, D, E, F, and the 
[adapted]  repeated 8-grams 2%
Okay, let's tackle this problem. So, we're given a list. There's another list. The largest number is already in all.

The function should re

--- wikipedia ---
prompt: '== History ==\nThe [[Roman Empire]] was '
[raw]  repeated 8-grams 26%
founde10 years old and had a high prize for the [[Roman Empire]]. The [[Roman Empire]] was founded in [[1990]] and had a high prize for the 
[adapted]  repeated 8-grams 47%
founde10,502 km&amp;sup264 (1,089).
*[http://2004.15 km&amp;sup2;]
*[http://2004.15 km&amp;sup2;]
*[http://2004.15 k.us]
*[http://2004.15 k.

--- chess ---
prompt: '<g>1700 1-0 1. e4 e5 2. '
[raw]  repeated 8-grams 0%   16 legal moves, then Be1
d3 d6 3. Nc3 Nc6 4. Nf3 Nge7 5. Be2 Ng6 6. O-O Be7 7. h3 O-O 8. Nh2 f5 9. f3 f4 10. Be1 Be6 11. Bg4 Bxg4 12. fxg4 Nd4 13. Nd5 Nxe2+ 14. Qxe2
[adapted]  repeated 8-grams 0%   25 legal moves, then Nx
d3 Nc6 3. Nf3 d6 4. Be2 Nge7 5. O-O f5 6. Nc3 f4 7. Bd2 Ng6 8. Qe1 Be7 9. h3 O-O 10. Nh2 Be6 11. f3 Qd7 12. Ng4 Bxg4 13. fxg4 Nd4 14. Be3 Nx

--- self-knowledge ---
prompt: '<user>\nhow do you decide which experts to use?\n</user>\n<bot>\n'
[raw]  repeated 8-grams 2%
It holds the 96 experts to the card the 32, and that is deliberate. Some of my busiest experts hold the smallest gates - they are chosen con
[adapted]  repeated 8-grams 0%
It holds the 96 experts I touched most recently, as tensors ready to copy. An expert leaving the card goes there first, not to disk, so want
```

</details>
<!-- /auto:run-blocks -->


## What v5.1 adds beyond v5.0

- per-token hierarchical candidate retrieval now operates in `virtual_autograd`; each token chooses coarse groups from only its current causal hidden state, then scores the real router rows inside those groups with gradients enabled;
- periodic exact top-k recall audits fail safe on the same call that exposes a miss and keep the router exact for a bounded fallback window;
- `RoutingTape` records the hierarchical candidate graph as well as final top-k expert IDs, so activation replay cannot silently change sparse control flow;
- replay bypasses fresh index searches and stochastic Gumbel draws, preventing RNG drift during recomputation;
- the expert cache is now two-level: digest-verified expert files are prefetched in parallel into host RAM, optionally pinned on CUDA systems, then copied into the bounded device LRU;
- `pool.ram_cache`, `pool.prefetch_workers`, and `pool.pin_host` are real runtime controls rather than placeholders;
- expert updates invalidate both host and device cache entries, so no stale version can survive an optimizer commit;
- the v5 full-autograd UID/version barrier, external AdamW, growth rollback, archive/reactivation, checkpoint authority and all earlier scientific controls remain intact.

The hierarchical path is an acceleration mode, not a new authority boundary. Exact routing remains the fallback and the release does not claim the discrete top-k expert identity is differentiable.

## v5.1 validation

The full source regression suite passes **89/89 tests**, including per-token candidate selection, recall-failure fallback, digest-bound parallel host prefetch, candidate/top-k replay, and gradient flow through the virtual hierarchical router. Run `python tools/verify_virtual_paging.py` for the dense-reference UID-paging gradient check.

## What v5.0 adds beyond v4.2

- stable logical expert UIDs replace mutable GPU slot tensors as expert parameter identity;
- `GraphEpoch` pins every expert used by one graph to its SHA-256 version and forbids expert mutation before backward completes;
- `PagedExpertFunction` reloads the same UID/version in backward, recomputes the local expert, returns `dX` into the full shared graph, and stores expert `dW` outside `model.parameters()`;
- `ExternalExpertAdamW` owns `m`, `v`, and `step` by logical expert UID and updates only experts touched by the completed graph;
- `VirtualPagedPool` performs per-token full-pool causal routing while `pool.resident` becomes a cache-size knob rather than a semantic admission cap;
- `RoutingTape` records discrete top-k choices for exact recomputation;
- the primary reader and live-learning path use a graph-wide forward/backward/expert-step barrier when the v5 runtime is active;
- dense-reference tests force A → B → C → A through a one-entry cache and verify input and expert gradients match the all-resident computation;
- `pool.runtime: legacy_slots` preserves the v4.2 runtime for experiment reproduction.

These changes solve expert identity aliasing for one full-sequence autograd graph. They do **not** make the discrete top-k expert choice differentiable, and the model's independent recurrent-depth `bptt_window` remains a separate memory/gradient control.

## v5.0 validation

The v5 package retains the v4.2 test suite and adds virtual-paging gradient, mutation-barrier, routing-tape, external-AdamW, integrated sparse-dispatch, and tiny end-to-end create/load/train/save tests. The historical v5.0 regression target was **83 tests**; v5.1 expands the suite to **89/89 passing tests**. Run `python tools/verify_virtual_paging.py` for the dense-reference paging check. The release still does not bundle fabricated trained semantic weights or a trained process-reward model.

## What v4.2 adds beyond v4.1

- Ed25519 promotion authority separates the private signer from public verification; legacy HMAC receipts remain loadable but are no longer the preferred trust boundary;
- promotion receipts bind deterministic file-or-directory artifact digests, the exact qualification record hash, base generation, ledger head, key ID and nonce;
- candidate and expert-growth qualification can require paired task/world evidence with a deterministic bootstrap confidence lower bound instead of relying only on point estimates;
- episodic-memory retrieval now respects both `valid_from` and `valid_to`, supports historical `as_of` queries, and rejects contradiction/supersession references to nonexistent records;
- consolidation experiments can run inside an auto-rollback guard so a local experiment does not silently mutate the shared trunk unless explicitly committed;
- `SegmentedCausalStepper` gives a safe routing-fidelity path in which the paged expert card can evolve at every routing segment; `segment_tokens=1` gives token-by-token causal admission, with the explicit cost of detached/truncated gradients between segments;
- all v4.1 audited expert indexing, shadow-data sealing, CodeAct limits, verifier search controls, replay holdouts, run provenance and the 267-task functional regression suite are retained.

These are reliability, authority and experimental-control improvements. They are not evidence that an untrained v4 model has acquired new capabilities.

## v4.2 validation

The v4.2 source package is compile-checked and its regression suite passes **75/75 tests**. The deterministic **267-task** functional regression suite is unchanged as a task-level gate. The historical loss/sample numbers above remain results from the existing byte-model run; they are **not** relabeled as v4 results. No trained v4 semantic checkpoint or trained process-reward model is bundled.

## Motivation

Every language model you can actually own today is a model somebody else trained and then froze. You can fine-tune around the edges of it, but you cannot train one from scratch on your own hardware, and you cannot keep training it on what you do day to day - the moment you try, it forgets what it knew before. The result is that a personal model is always somebody else's model with a thin layer of you on top, and it stops learning the day it ships.

**mini-AGI** model has small enough GPU footprint that it is possible to train end-to-end on one consumer card, and it is built so that training never has to stop. It reads a stream of characters one chunk at a time, takes a gradient step on each, and the same path serves generation. There is no separate fine-tuning regime and no frozen base: reading and being trained are the same event.

Three constraints shape everything else in the design:

- **It has to fit on 8 GB.** Not with quantisation - training needs gradients and optimiser state, which is roughly three times the weights again. So the weights live on disk and only the experts the current forward uses are on the card.
- **It has to not forget.** A model that learns continually and overwrites itself is worse than one that does not learn at all.
- **It has to be able to read anything.** Legacy checkpoints use raw UTF-8 bytes. New v4 models use ByteLevel-BPE with a complete byte alphabet, so arbitrary input remains representable while common words and code fragments consume far fewer sequence positions.

The model can be trained and served on your hardware and can optionally continue learning from user-provided conversation text. Durable episodic memory is kept separately from online SGD so remembering an event does not require immediately rewriting model weights.

<!--**Watch this video for a detailed explanation**
[here will be the link to the video when its ready]()-->

## How the architecture works

Characters (bytes) does not pass through a fixed stack of layers as it would be in a traditional LLM. Instead, it passes through **two dense prelude blocks** and then through **one recurrent block applied up to 24 times**, each application choosing its own experts from a shared pool. The latent state between applications is never decoded - it is merged with the embedded input by an adapter each time round, so the loop cannot drift away from the text it is reading.

Three distinct blocks, up to 26 block-applications per character.

![mini-AGI's layers: an input embedding; two prelude layers of attention and feed-forward; one recurrent layer - the previous state h joined to the encoded text x, a linear adapter, attention, and a mixture of experts behind a router - applied up to 24 times with the same weights; a norm and two heads, the next character and the halt probability. Beside it, inside each attention layer: 8 heads of 64 numbers, rotary positions on queries and keys, the cache of keys and values they read, and the concat and projection back into the character's state](assets/layers.svg)

*Every box is a layer, every ⊕ a residual add, and the shaded groups are what repeats: the prelude layer twice, the recurrent layer up to 24 times with the same weights. On the right, the inside of each attention layer - its 8 heads, the cache they read, and where their output goes.*

- **Adaptive depth.** A halting head scores every character at every row, and the character stops as soon as another row would not change the answer. Easy characters take one row, hard ones take many. A character that has stopped is finished: its experts stop running, and the characters after it read its final state at the deeper rows. Training does exactly the same, with the PonderNet recipe inside it: every row a character runs is weighted by its halting probability, so the halting head learns through those weights.
- **Routing per block-application, not per character.** Each recurrent application a character runs, up to 24, picks its own top-8 experts from the 32 on the card, so one character touches far more of the pool than "top-8" suggests, and the same expert can be selected several times at different depths. What varies is *which* eight at each point.
- **No expert is assigned a subject.** There are no labels anywhere. Soft top-k routing distributes capability across the pool by itself, and a character can combine fragments from several experts. The cost is that capabilities share parameters and so *can* interfere.

![how the model processes one character](assets/shape.gif)

**This is the architecture assembling itself, one character at a time**, captured from the live model - nothing here is drawn by hand.

Each tile on the left is one expert; colour is expert identity and stays the same for the whole clip. A **row** is one application of the recurrent block, and the eight tiles in it are the eight experts that row actually ran. The stack grows downward as the model keeps going, and the amber line is where halting stopped it - **the grey rows below are computation the model declined to spend.**

The trace on the right is how many rows each character took. It moves constantly between 9 and 15 against a ceiling of 24, and the caret under the text shows which character is being read. The text is a held-out story, read one character at a time after its first 2,048, and every character is a forward of its own: it adds its requests to the story's vote and routes among the 32 experts the story so far has asked for most, each row picking its eight from those. Over these 160 characters the card does not change once - the 2,048 characters before them have already voted, and the story goes on asking for the same experts.

Positions are rotary and carry no learned parameters, which is why the context window can be extended by continued training rather than by re-initialising anything.

### ...and the same thing while it writes

![how the model generates text](assets/generate.gif)

The clip above is the model **reading** - every character is held-out text it is being shown. This one is the model **writing**: it was primed with 2,495 characters of held-out stories and then continued on its own, so the grey text is what it was given and **the green text is entirely its own**. Greedy decoding with the repetition guard - the `adapted` reading in the sample log - and no sampling anywhere: run it twice and you get the same sentence.

Two things are worth watching. The stack behaves the same way, because generating and reading are the same forward pass in this model - the only difference is whether the next character comes from a file or from the model's own argmax - and it costs about the same: 11.8 rows a character over the 140 characters written, against 12.1 over the 160 read from the same stories. And **the whole text chooses the experts**, the way it does in training. The prompt's forward admitted the 32 its characters asked for most; after that, every character added its own requests to the vote and routed among what the prompt and the reply so far had asked for. This reply never moved the vote far enough to change the card, so not one expert was loaded after the prompt; a reply that wanders from its prompt takes the card with it. Letting each character choose alone instead, as an earlier version did, made its predictions on the same held-out text a fifth worse, 1.35 nats a character against 1.11, and garbled its words. Nothing was written back: nothing is trained while it writes.

What it produced, continuing the story about a cherry tree that the prompt ends on:

> They would stay inside, but only for any more.
>
> One day, the cherry tree saw another cherry. It was very big, and it had long, shiny wings.

Spelled right and on the story it was given, but "only for any more" means nothing, and the other cherry has wings - a fair picture of where the model was at 648.6M characters, when the clip was recorded.

## How paging works

Every expert is a file on disk holding its weights and its Adam moments. Above disk sit a RAM cache and the card:

| | key | what it is |
|---|---|---|
| disk | — | every expert the model has; bounded by free space |
| RAM | `ram_cache` | the experts used most recently, least-recently-used evicted |
| VRAM | `resident` | the experts the current forward admitted - what a character may route through |

Selection still follows one rule: **the causal prefix chooses**. The v3-compatible path uses a bounded causal prefix vote to select the resident card and masks losses before the end of that prefix. v4 can additionally use a hierarchical expert index: a coarse centroid router selects promising expert groups, then exact router scores are computed only for experts inside those groups. Exact top-k routing over the admitted resident experts is unchanged.

The resident count is what VRAM can hold through backward and the optimiser step. Generation remains naturally causal because each newly generated token is another forward pass. Training still does **not** perform exact token-by-token expert-card replacement inside one backward graph; v4 does not claim that equivalence. The hierarchical index fixes the O(number-of-experts) admission bottleneck, not the remaining train/generation approximation.

Three rules the project holds to:

- **Every expert is stepped with its own moments.** Adam's moments belong to the expert, not to the slot of VRAM it happens to occupy. Stepping it with whatever its slot held would hand it the momentum of the expert before it, and training would carry on looking healthy while every swapped expert inherited a stranger's history. Only a step reads them, so they travel only to a step: an expert comes to the card with its weights alone, and just before each optimiser step every expert on the card that is not already holding its own moments gets them from RAM or disk. A reply loads experts at almost every character and steps none of them, so writing moves no moments at all.
- **An expert already on the card stays in the slot it is in.** Admission is a set, not a ranking. An expert the new forward also admits is never moved; a newcomer takes the slot of an expert this forward did not admit - an empty slot first, then the one admitted longest ago. The number of loads is exactly the number of admitted experts that were not already there, and consecutive windows of one subject ask for nearly the same experts: over fifteen minutes of real training a step loaded 2.4 experts on average.
- **An expert nothing trained is not written back.** An expert leaving the card is copied back to RAM - and later to disk - only if the optimiser stepped it while it was there. A reply loads experts at almost every character and trains none of them, so each is still exactly its copy, and writing it back would cost a disk write per load for nothing.

**What was on the card before changes the cost, never the choice.** Admission reads only the text and the weights, so the same text admits the same experts whatever was read before it; history decides only how many of them have to be loaded. And because the rows that admit experts are the rows that route every character, admission is learned by the ordinary gradient. When a character mixes its experts, the loss raises the router score of each one whose output helped more than the mixture as a whole and lowers the score of each that helped less. Only the rows of experts the window admitted receive that signal, and the next text whose states look like those asks more strongly for the experts that helped.

**While it trains, the router is taught to use the whole pool.** That gradient only reaches experts that get chosen, and left alone the router keeps choosing the same ones: reading held-out text on its own, it put just 59 of its 118 experts on a card across every subject. So the loss carries a balance term - the [Switch Transformer's](https://arxiv.org/abs/2101.03961), with one change. Every expert keeps a running share of the recent training forwards that admitted it, and every forward that trains pays, on the router's probability for each expert over the whole pool, in proportion to that share: `balance × (experts × Σ share × probability − 1)`, zero when use is even. Probability on a busy expert costs more than on an idle one, so the router's rows move toward the idle. Switch charges each expert by its share of the current batch; here the share is taken over the last thousand or so training forwards, because a training window is one text and a text should be free to want few experts - what has to be even is the use across texts. Only the router's rows learn from it, since the characters' states are detached, so it cannot bend what the trunk computes. And because it changes the router itself, reading and writing choose the way training does. In a 15-minute test at weight 0.01, the experts the router put on a card across held-out rose from 59 to 114, at a cost of 0.023 in held-out loss over those minutes. The exploration bonus it replaced acted only while training: over the same 15 minutes it spread the training work across 116 experts, while the model reading on its own still used 58. The weight sets how hard the term leans against learning the text on the experts in use - an expert no text admits gets no other gradient, so Adam moves its row at the usual pace whatever the weight - and the default is gentler, 0.003.

## How growth and pruning work

The pool grows when it is short of capacity and shrinks when parts of it stop being asked for.

New experts are added on speculation, at a small gate so they change almost nothing, and kept only if something goes on asking for them. A new expert is built by **recombination** - whole hidden units taken from several existing experts - because a clone of one parent is not novel enough to be worth routing to, and a random expert computes nothing worth routing to. What works is novelty assembled from trained parts. Its router row is its parents' rows averaged, weighted by how many units each gave, so it starts out scoring every character with the same weighted average of its parents' scores - and since nothing has used it yet, the balance term pulls its row up from the first training forward after its birth, until texts start admitting it.

Growth is refused unless every brake agrees:

- **room** - disk and VRAM can take it
- **used** - the capacity already added is being asked for
- **earning** - the previous cohort survived its trial
- **fits** - not too many experts are already inside their trial
- **honest** - train and held-out have not separated

**Dead means unaddressed.** Both the growth brake and the pruner read how long it has been since a forward last admitted an expert, and never its gate: being used is being alive. This is the single most useful finding in the repository: the gate is not merely uninformative here, it is anti-predictive. The smallest gates belong to the *busiest* experts - one that behaves as a sink, chosen constantly and contributing little per character, reads as dead on a gate test, while a high-gate expert nothing has asked for in a long time reads as alive.

A new expert is safe for a full survival window no matter what, so it cannot be judged before it has had a chance to be chosen. When the model grows an expert a new file appears; when it prunes one, that file is deleted. 

## How continual learning works

Training on a single stream, one subject at a time, is the classic recipe for catastrophic forgetting. The test here is deliberately the worst case: the model is switched cold onto **PG19** - 19th-century novels, a domain it has never read - and made to read **1,048,576 consecutive characters of it and nothing else**, at batch 1.

**The trunk learning rate is the mechanism.** The trunk - embeddings, attention, routers, the halting head - is the part every character passes through, and it carries **98% of the squared gradient norm**. Running it at 0.1x the experts' rate is the difference between a model that absorbs a new domain and one that is wrecked by it.

| configuration | PG19, the new domain | the 8 it already knew | learned per nat forgotten | retained vs chance |
|---|---|---|---|---|
| experts frozen, trunk LR = expert LR | -0.2909 | +1.2663 | 0.23 | 74.12% |
| swapping, trunk LR = expert LR | -0.3106 | +1.2628 | 0.25 | 74.25% |
| **swapping, trunk at 0.1x - what the run uses** | **-0.4216** | **+0.1297** | **3.25** | **97.30%** |
| *interleaved: PG19 added as a 9th lane* | *-0.3124* | *-0.0065* | *nothing forgotten* | *100.13%* |

*These arms were measured before expert selection moved to the rule under [How paging works](#how-paging-works), when the experts on the card were re-chosen before every chunk; the probe now runs the current rule.*

The fourth column is the exchange rate: nats gained on the new domain for every nat lost across the eight. **The mitigated configuration is 13x better at that trade than either unmitigated one** - and interleaved there is no trade at all.

![Reading a new domain under three configurations](assets/mitigations.png)

**This is the measurement the whole design rests on.** Panel A is what happened to the **eight subjects the model did not read** - zero means nothing was forgotten. Two configurations climb to +1.27 nats, which is the model losing most of what it knew. The third, at a trunk learning rate one tenth of the experts', reaches **+0.13 after more than a million characters of a single unfamiliar domain**.

Panel B: **what it learned while it was there**. The mitigated configuration is not trading plasticity for retention - it learns PG19 *faster* than either unmitigated arm, **-0.4216 nats against -0.3106 and -0.2909**, while forgetting ten times less. Slowing the trunk does not slow learning. It accelerates it, because the trunk stops being dragged around by every passage and the experts are free to specialise.

### This is a worst case, not a use case

A million consecutive characters of one subject is **32 passages back to back**. The run never does this: it reads a passage of 32,768 characters, moves to another subject, and comes back to the first about every 262,144 characters. It is the analogous to a person who does one thing for a solid week and a person who changes activity through the day. 

Nothing in normal use looks like the massed arm either. A conversation wanders, and a model reading your files reads whatever is there. Such regime only arises deliberately - a bot specialised on one subject and fed nothing else for a long stretch.

So, the last row here is the one that describes the actual system:

```
pg19         1.6773 -> 1.3649   -0.3124   (read)
reasoning    0.7076 -> 0.6846   -0.0230   (read)
code         0.6552 -> 0.6404   -0.0148   (read)
chat         0.7402 -> 0.7289   -0.0113   (read)
stories      0.5599 -> 0.5537   -0.0063   (read)
wikipedia    1.2549 -> 1.2493   -0.0056   (read)
arithmetic   0.6419 -> 0.6433   +0.0014   (read)
chess        0.4955 -> 0.4989   +0.0035   (read)
chat_hermes  1.1043 -> 1.1085   +0.0042   (withheld)
```

![PG19 read as one of eight interleaved subjects](assets/probe_interleaved.png)

**Add a new domain as a ninth lane and six of the nine improve.** PG19 falls by 0.31 nats and the eight the model already knew improve by 0.0065 on average - nothing moves more than +0.004 in the wrong direction, and the subject it was best at is untouched. Adding a domain to this model costs nothing: 100.13% retained is the eight coming out very slightly ahead of where they started. In the figure the eight are the flat band at zero and PG19 is the line leaving it - the same read that costs 0.13 nats when it is massed costs nothing when it is interleaved.

Two readings matter here:

- **The expert pool is not what prevents forgetting.** Freezing the experts on the card - removing the one property that makes the pool a pool - changes forgetting by 0.0035 nats, which is nothing, and it *learns the new domain slowest of all three*. In that arm 137 of 174 experts received no gradient at all and the model still collapsed. Preserving most of the weights is not sufficient; the trunk is where the damage happens.
- **The cost of learning is real, and it is small.** On genuinely new material the massed arm pays **0.13 nats across eight domains to gain 0.42 on a ninth** - a real exchange rate, and a favourable one. Interleaved, the exchange disappears.

![Every subject during a massed read, and how much of the pool was touched](assets/probe_massed.png)

**What that same read looks like from the inside.** This is the working configuration - trunk at 0.1x - during the 1,048,576-character PG19 read. On the left, every subject against where it started. PG19 drops away from the pack; the eight withheld subjects drift up together, and the ones that drift most are **chat, stories and reasoning** - the prose-like lanes, nearest to Victorian novels. Chess and arithmetic barely move, at +0.009 and +0.043. The damage lands where the representations overlap, which is what the routing story predicts.

The right panel is why the damage is bounded at all. Over the whole read only **44 of 174 experts received any gradient** - 75% of the model was structurally untouched, because routing never selected it. This is the pool doing exactly what a pool is for: confining an update to the part of the model that the text actually addressed.

**The learning rate is not scheduled.** A cosine schedule asserts that the run ends, which for a model that reads continually is false. Instead a controller watches held-out loss and moves the rate in both directions: clear improvement buys a little more, no evidence eases it down, and a confirmed jump in held-out steps it back up. 

<details>
<summary><b>Replicate this measurement yourself</b> - the probe, the results, and the weights it was measured on</summary>

The claim above is a measurement, and a measurement you cannot repeat is an assertion. The probe, all four arms and the figures ship in [`replication/`](replication/).

**Weights for the checkpoint every number above was measured on:**

| data read | held-out | experts | download |
|---|---|---|---|
| 428.2M characters | 0.7702 nats / 1.1112 bits/byte | 174 | [Volotat/mini-AGI-cl-replication-weights](https://huggingface.co/Volotat/mini-AGI-cl-replication-weights/tree/main/weights) |

Download the `weights/` folder into the repository root. Held-out there is the probe's own baseline - the eight training domains under `data/val`, 16 chunks each - which is what every figure above is measured against, and is evaluated on fewer chunks than the training run's own log.

**To run it:**

```bash
python3 -m corpora all                             # data/train and data/val
python3 -m corpora pg19 --split validation \
        --out data/cl/pg19                         # the new domain, 50 books

cp -a weights /tmp/w                               # a COPY - paging marks experts dirty
python3 replication/forgetting_probe.py --weights /tmp/w \
    --domain pg19 --read-root data/cl --steps 512 \
    --at 0,64,128,256,384,512 --eval-chunks 16 --chunk 2048 \
    --lr 2.08e-4 --held-out data/cl_val \
    --trunk-lr-mult 0.1 --out replication/results/mine.json

python3 replication/cl_summary.py                  # the table, control verdict first
python3 replication/plot_figures.py                # redraws the figures above
```

`--read-root` keeps the new domain outside `data/train` on purpose: put PG19 in the corpus and the training run would start reading it too. `data/cl_val` holds the eight existing held-out sets plus PG19 books that are never read, so "PG19 improved" cannot be memorisation. The PG19 **validation** split is used rather than the test split, so the 2.4496 BPB benchmark further down this page stays untouched.

Set `--trunk-lr-mult 1.0` for the unmitigated arm, add `--no-swap` to freeze the experts the first window admits, and pass `--rotate pg19,chess,code,stories,arithmetic,wikipedia,chat,reasoning` for the interleaved control.

**Read the control first.** `cl_summary.py` prints a verdict on it before anything else: every lane is read there, so forgetting is impossible by construction and any degradation is the instrument rather than the model. 

</details>


## Reading your own files

This is the shortest path to a model that knows something you care about.

```bash
python3 train.py read ~/notes                     # a dry read - nothing kept
python3 train.py read ~/src ~/docs --passes 3 --save
```

Point it at files or directories. For a legacy byte checkpoint, text is encoded directly as UTF-8 bytes. For a v4 semantic checkpoint, `train.py read` loads the tokenizer cryptographically bound to the checkpoint and uses that same token-ID mapping. Directories are walked, binaries are skipped by sampling their contents rather than trusting the extension, and each file is read from beginning to end because a document has an order.

It is the same path training uses: same chunking, same cache, same gradient step.

| flag | |
|---|---|
| `--passes N` | read the whole set N times |
| `--save` | keep what it learned; without it `weights/` is untouched |
| `--lr` | default 5e-5, below a training run: reading should adjust the model, not overwrite it |
| `--mix ""` | skip the before/after scoring |

Two defaults worth knowing. **Nothing is saved without `--save`**, so a read is a dry run until you decide otherwise. And it scores the held-out mixture before and after, then says plainly if reading your files cost the model ground elsewhere - the forgetting question measured per-read rather than assumed away.

## Benchmarks

The numbers below are for tracking purposes and move as the run continues. Held-out loss is reported with its standard error, and the size of the evaluation is what sets that error - a difference smaller than it is the instrument rather than a result.

There is a second variance underneath these figures. The same configuration run twice lands about 0.014 apart, because the expert dispatch is not deterministic on CUDA. **Treat about 0.03 as the threshold for a real difference**, not the error bar printed beside one score.

<!-- auto:benchmarks -->
**Where the model is** (1,180.9M characters read, 218 experts):

| | nats/char | bits/byte |
|---|---|---|
| **held-out, all 8 subjects** | **0.6456** ± 0.0288 | **0.9314** |
| train | 0.5299 | 0.7645 |

**Held-out loss per subject:**

| Subject | nats/char | bits/byte |
|---|---|---|
| `chess` | 0.423 | 0.610 |
| `stories` | 0.433 | 0.625 |
| `code` | 0.543 | 0.783 |
| `reasoning` | 0.555 | 0.801 |
| `arithmetic` | 0.620 | 0.894 |
| `chat` | 0.622 | 0.897 |
| `chat_hermes` | 0.903 | 1.303 |
| `wikipedia` | 1.066 | 1.538 |
<!-- /auto:benchmarks -->

### Data Scaling

<!-- auto:scaling -->
![Data scaling on PG19](assets/scaling.png)

Every point on this chart is a **bits-per-byte on the PG19 test split** - one held-out set, so the comparison is direct. This model scores **2.091 BPB** over the whole split (100 books, 41,289,001 bytes) at a context of 4,096, against its own mixture's 0.94. PG19 is out of distribution for it: it was trained on a corpus assembled for this project and has read no Victorian novels, so much of that gap is subject matter rather than capability.

**The results so far are promising.** The red line is the fitted power law on this model's own held-out, `L ∝ D^-0.235` with R² 0.98 over every point past the warmup - between Kaplan's 0.095 and Chinchilla's 0.28, and it has held for more than a decade of data. How steep it looks depends on where the fit starts, and the band on the chart spans that range rather than pretending to one number.

Read straight off that trend, on this model's own mixture. It has read 1.18B characters so far, in about 14 days of running. The days below assume the pace of the last 29 hours of it: 1,197 characters a second on the wall clock, held-out checks and rounds of samples included, because the reading waits for them.

| held-out | total data read | further reading | days from here at ~1,197 char/s |
|---|---|---|---|
| 0.93 BPB | 1.19B | +0.01B | 2 hours |
| 0.80 BPB | 2.25B | +1.07B | ~10 |
| **0.60 BPB** | 7.88B | +6.70B | **~65** |

The first two are hours to days of reading on one laptop GPU, and every one of them sits inside a single pass of the 7.88B-character corpus.

The right panel shows which subjects are still moving. reasoning, code, stories, chat are the steep ones; arithmetic and wikipedia have the shallowest slopes, which is the honest counterweight - the expensive domains are not the fastest ones.
<!-- /auto:scaling -->

## Running it

1. Make sure you have a CUDA-capable GPU with at least 8 GB of VRAM, and Python 3.10 or newer. The reference machine is an RTX 3070 Laptop GPU with 8 GB.
2. Clone the repository:
    ```bash
    git clone <repository-url>
    cd mini-AGI
    ```
3. Install the dependencies:
    ```bash
    pip install torch numpy pyyaml matplotlib      # the model, and its graphs
    pip install flask                              # serve.py
    pip install chess zstandard datasets           # building corpora
    pip install scipy                              # a few of the analysis tools
    ```
    PyTorch has to match your CUDA version - see [the PyTorch install page](https://pytorch.org/get-started/locally/). The reference environment is torch 2.6.0+cu124 with numpy 1.24.4. Only the first line is needed to train.
4. Build the corpus. One command downloads the four public datasets and generates the other four lanes:
    ```bash
    python3 -m corpora all                  # all eight subjects, a few GB
    python3 -m corpora all --limit 5000     # a small slice first, to try it
    python3 -m corpora all --full           # entire datasets: tens of GB, hours
    ```
    Lanes already on disk are left alone, so an interrupted build can simply be run again. Individual lanes are available too - `python3 -m corpora` lists them - or skip this entirely and point the model at your own files.
5. Start reading. The weights directory is created from `config.yaml` the first time, so there is nothing to set up:
    ```bash
    python3 train.py read data/train --save --weights-dir weights \
        --held-out data/val --sample-every 10
    ```
6. Serve it:
    ```bash
    python3 serve.py --port 8080            # then open http://127.0.0.1:8080
    ```

The run writes a sample log, redraws its graphs as it goes, and checkpoints every few minutes. It is meant to be left alone for days.

### Everything else

```bash
python3 -m minagi.store weights                    # what the model is right now
python3 -m corpora                                 # every corpus target
python3 -m corpora all --only wikipedia stories    # rebuild particular lanes
python3 -m corpora expand                          # .bin -> the text files read

python3 train.py read --help                       # every knob the reader has
python3 train.py stream --steps 140000 --lr 2e-4   # the packed-corpus path
python3 train.py ponder-probe --ckpt weights       # depth against difficulty
```

Every tool takes `--ckpt weights` - the directory is the model, and there are no `.pt` files to keep track of.

## Initialization

A fresh model starts small and grows into its shape. The context window begins at `model.context_start` and extends one character at a time, but only when the model is still getting something out of the far end of the window it already has. The expert pool begins at `pool.experts` and grows from there.

This means the first hours of a run look nothing like the rest of it. Loss falls fast, the pool churns, the window is short, and the learning-rate controller has not gathered enough evaluations to act. None of that is a problem to fix.

If a run diverges, it repairs itself: when held-out exceeds the best by more than `--revert-factor` (default 1.5x) the run reloads `weights/`, halves the learning rate, pulls the context back and continues. After `--max-reverts` it stops rather than thrash.

## Layout

```
minagi/          the model. no command lines here.
  config.py        reading config.yaml, which building and training both use
  precision.py     what the model computes in, and how moments are stored
  tokenizer.py     legacy bytes + checkpoint-bound ByteLevel-BPE for v4
  model.py         the transformer: RMSNorm, rotary positions, SwiGLU, flash attention
  decode.py        how a character is chosen, without a random number generator
  ingest.py        turning a pile of files into something to read
  pool.py          the expert pool, routing, growth and pruning
  expert_index.py  hierarchical candidate search for large expert pools
  paged.py         the same pool spread over disk, RAM and VRAM
  recur.py         latent recurrence; v4 can combine dense + sparse FFN paths
  consolidation.py distil validated sparse behaviour toward the stable trunk
  codeact.py       capability-scoped executable action VM
  verifier.py      step-wise process scoring and beam pruning interfaces
  dream.py         replay recorded discovery trees for policy evaluation
  shadow_live.py   quarantined serving-time learning intake
  stream.py        reading a corpus behind a KV cache, one chunk at a time
  store.py         the weights directory, which IS the model
  optim.py         how much of a gradient is signal
  plasticity.py    the learning rate, governed by held-out loss
  live.py          serving a model that is being trained underneath
  report.py        the model reading statistics off its own weights
  create.py        writing a fresh weights directory from config.yaml

train.py         read | stream | ponder-probe
serve.py         local web UI
config.yaml      the settings worth changing
corpora/         python3 -m corpora all - the whole corpus, downloaded and made
weights/         one file per expert. this directory is the model.
```

`weights/` is written on the first run and `data/` by `corpora`; neither is in
the repository. Everything else above is.

### The weights directory is the model

```
weights/
  manifest.json     what exists, its shape, and where it came from
  core.npz          embeddings, attention, norms, adapter, halting head
  routers.npz       the gate, and the router: one row per expert per site
  optim.npz         Adam moments for the trunk and the routers
  experts/          one file per expert: w1, w3, w2 and its own Adam moments
    e00000.npz ...
```

Training resumes from weights, optimiser state and step counts. Formal saves create immutable, hash-verified generations. `LAST` names the newest committed generation and `BEST` names the lowest-validation generation; they are separate on purpose. A mutable paging workspace is recovered from `LAST` after a crash, while content-addressed checkpoint objects prevent a later expert writeback from silently changing an older generation.

The directory is written on the first run.

## The model

There are now two model families. Existing v3 checkpoints remain byte-level with vocabulary 265 (256 bytes + 9 structural markers). New v4 checkpoints bind a ByteLevel-BPE tokenizer (the supplied profile targets a 16K vocabulary) while retaining full byte coverage. `profiles/v4-semantic-150m.yaml` increases the stable dense path to roughly the 150M-class before embeddings and adds sparse residual experts rather than making almost all learned capacity live in the expert warehouse. This is a **new model** and requires fresh training.

Quick start for the new path:

```bash
pip install -r requirements.txt -r requirements-v4.txt
python tools/train_tokenizer.py data/train --out artifacts/tokenizer.json --vocab 16384
python tools/create_v4.py --tokenizer artifacts/tokenizer.json --out weights-v4
python tools/pack_v4_corpus.py data/train --tokenizer artifacts/tokenizer.json --out data-v4
```

| | |
|---|---|
| body | RMSNorm, RoPE, SwiGLU, flash attention via `scaled_dot_product_attention` |
| depth | 3 distinct blocks, up to 26 block-applications per character |
| recurrence | one weight-shared block applied up to 24 times; the latent is never decoded |
| halting | PonderNet - each character halts independently, so hard ones get more depth |
| routing | top-8 experts per block-application, chosen per character |
| paging | 32 experts resident on the card; the rest live on disk |

<!-- auto:params -->
The parameter count moves, because the pool grows and prunes itself while training. `python3 -m minagi.store weights` prints what it is now. At step 577,407:

```
core        8.27M  embeddings, attention, norms, adapter, halting head
routers     0.11M  one row per expert at each call site, depth embedding, gates
experts   688.9M   219 x 3.15M each  (3 x 512 x 2048)
----------------------
total     697.3M
```

**VRAM is set by the card's 32 slots, not by the pool.** Only 32 experts are resident at a time - about 109M parameters of the 697M - which is why the pool can keep growing on an 8 GB card. Per byte the model activates about **344M** parameters - two prelude blocks, then attention and top-8 of the resident experts on each recurrent step, counted at the 12.8 steps training samples its depth around (a character may take up to 24) - so by the 6ND rule it costs the same per byte as a dense 344M byte-level transformer, not a 697M one. That is the figure the scaling chart in Benchmarks is drawn against.
<!-- /auto:params -->

## AI usage

This project was assisted by "Claude Opus" 5 and 5.5 models. The model did implemented most of code of this project, verified and debugged it when it was necessary. The model was searching for published papers related to the problems that the project were trying to solve, build tests and experiments, and help with brainstorming the complex problems that arose along the way. The animations, graphs and other media you see here are all done by Claude as well form the real data traces. While I myself provided main ideas, steering, intuition, rejections when thing went in a wrong direction, code monitoring and verification, as well as decisions and strong opinions of how everything should be wired together and work in principle. Documentation was written in tandem.  

## Acknowledgments

[PyTorch](https://pytorch.org/) does the arithmetic, [NumPy](https://numpy.org/) holds the weights on disk, and [Matplotlib](https://matplotlib.org/) draws every graph.

The parts the model is built out of:

[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) - Shazeer et al., 2017. The entire expert pool, and the load-balancing auxiliary loss.  
[Switch Transformers](https://arxiv.org/abs/2101.03961) - Fedus et al., 2021. The capacity-based batched dispatch, which is what lets the pool run as three matrix multiplies.  
[PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407) - Banino et al., 2021. The adaptive depth mechanism.  
[RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - Su et al., 2021. Why the context window can grow by continued training.  
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - Shazeer, 2020. SwiGLU.  
[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - Zhang & Sennrich, 2019.  
[FlashAttention](https://arxiv.org/abs/2205.14135) - Dao et al., 2022. Reached through PyTorch's `scaled_dot_product_attention`.  
[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) - Loshchilov & Hutter, 2017. AdamW.  
[Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) - Chen et al., 2016. Gradient checkpointing, which on 8 GB is not optional.  
[Block-Recurrent Transformers](https://arxiv.org/abs/2203.07852) - Hutchins et al., 2022. Carrying a recurrent state across blocks, which is the shape any continuity beyond the attention window has to take here.

[ZeRO-Offload](https://arxiv.org/abs/2101.06840) - Ren et al., 2021, and [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) - Rajbhandari et al., 2021. Training a model larger than the card it sits on is not a new capability.  
[Dynamic Mixture of Experts Against Severe Distribution Shifts](https://arxiv.org/abs/2511.18987) - Kim et al., 2025. Adds experts to a live MoE, and reports the failure this project spent a week fixing.

[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - Hoffmann et al., 2022. Chinchilla, and the ratio any efficiency claim has to be tested against.  
[The Pile](https://arxiv.org/abs/2101.00027) - Gao et al., 2020. Bits per UTF-8 byte, chosen there for invariance to tokenisation.  
[Transformer-XL](https://arxiv.org/abs/1901.02860) - Dai et al., 2019, and [Compressive Transformers](https://arxiv.org/abs/1911.05507) - Rae et al., 2019. The character-level benchmarks to aim at.  
[An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) - McCandlish et al., 2018. The gradient noise scale.  
[The AdEMAMix Optimizer](https://arxiv.org/abs/2409.03137) - Pagliardini et al., 2024. Implemented for the trunk and available, though at the paper's settings it hurt this model and it is not the default.  

The corpus: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5), [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) and [the Lichess open database](https://database.lichess.org/). Wikipedia and the source-code portion come from public dumps and public repositories.

## Citation

If you use this project in your research or work, please cite it as:

```bibtex
@software{Borsky_mini_AGI_2026,
  author = {Borsky, Alexey},
  month = {9},
  title = {{mini-AGI: A Continually Learning Byte-Level Language Model}},
  url = {https://github.com/volotat/mini-AGI},
  version = {1.0.0},
  year = {2026}
}
```
