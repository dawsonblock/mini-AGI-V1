# Learning & generalization report (3B.3)

Status: **strict-scored, evidence-graded.** Supersedes all
casefold-era learning claims. The question this campaign set out to
answer — *does controlled LoRA adaptation acquire new capability that
transfers to held-out task families?* — has a measured answer, and it
is not the one originally reported.

## What was actually learned

Under `exact-match-v2` (the corrected oracle):

| finding | evidence | status |
|---|---|---|
| Adapter can FIT a small verified subset | subset fit 1.000 at 2 and 6 families | **established** |
| Fit transfers to held-out families | L6 hidden 0.167 vs frozen 0.333; destroys 2/4 frozen-correct, gains 0 | **refuted at this scale** — adaptation does not transfer, it destroys |
| Verified replay beats frozen inference | L5b 0.417 vs L1 0.333; net +1, sign-test p = 1.0 | **promising, NOT established** — one net row in 12 cannot exclude chance |
| Replay during TRAINING preserves capability | L7 (inference-time replay over adapted weights) 0.083, destroys 4/4 | L7's mechanism failed; rehearsal-in-gradients is a DIFFERENT mechanism, measured in 3B.3 |
| Rank increases capacity usefully | ladder non-monotone under a confounded alpha/r sweep | **inconclusive** — CONFOUNDED, not refuted (RD-13) |
| Family composition dominates | spread across sets exceeds seed noise | **established** — largest measured lever |

## The honest summary

1. **Nothing in the neural path beat frozen inference reliably.**
   Every LoRA arm sits at or below the frozen baseline under strict
   scoring, and the only arm above it is retrieval, not weights.
2. **Destruction is the real measured effect.** Adapted models break
   frozen-correct rows at rate ≈0.6 on average (strict); L7 broke all
   four. Whatever preservation mechanism we add must be measured
   against that number.
3. **"Learning" was conflated with scoring slack.** The casefold oracle
   accepted wrong-case outputs on a case-transformation benchmark;
   ~14% of all scored rows flip under strict scoring. The corrected
   picture is smaller but true.
4. **The selector now knows what it doesn't know.** CONFOUNDED,
   INCONCLUSIVE, INSUFFICIENT-EVIDENCE, FAILED-IMPLEMENTATION and
   HIGH-RISK are distinct decision states — a noisy or confounded sweep
   can no longer be silently converted into a refusal, and a refusal
   requires measured contradiction beyond seed noise.

## What a positive claim would now require

- a matched multi-seed replication under the strict oracle
- net gained minus net destroyed > 0, surviving a paired family-level
  test, not a point estimate
- the candidate's adapter artifact digest bound into a verified
  receipt, independently confirmed on a fresh sealed benchmark

None of these is currently met. That is the correct output of this
campaign: a true negative with the measurement integrity to trust it.
