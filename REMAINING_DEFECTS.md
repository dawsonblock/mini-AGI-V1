# Remaining defects

Every item below is a real limitation of this release. None of them is
presented as a pass, and none of them changes a qualification rule.

## RD-1 — Real sealed confirmation: NOT RUN

No independently held sealed confirmation set and no independent
evaluator custody were available to this effort. The workflow is
implemented and executes end to end on synthetic fixtures
(`SEALED_CONFIRMATION_PROTOCOL.md`), but no candidate was confirmed.

**Consequence.** Gate G8 is passed only for *executability* (the
evaluator can independently execute sealed predictions and apply frozen
rules). Real confirmation is **NOT RUN**.

**Why it is not worked around.** Generating a sealed set locally and
calling it independent would be theatre: the same operator would hold
both the candidate and the labels.

**Custody control, corrected (reviewer finding).** The
`MINAGI_RESEARCH_WORKER` marker is an environment variable: a research
worker that controls its own environment can unset it, and an
in-process caller can simply not pass it. It is a **cooperative guard,
not isolation**, and earlier drafts described it too generously. The
enforceable control is now filesystem permission — the sealed set must
be owner-only and owned by the evaluating uid — and
`sealed_custody_report` records which controls were actually enforced
in the signed record, stating plainly that a run without enforced
permissions did **not** establish independent isolation.

## RD-2 — The envelope is signed, but not independently attested

The envelope was reissued with the repository's existing release key and
`RELEASE_ATTESTATION.json` was reconciled. The bundled
`RELEASE_PUBLIC_KEY.pem` hashes to `a8cd5cb2…`, matching the
attestation's pinned `public_key_sha256`, and the reissue reported the
same fingerprint — so the signature verifies against the identity this
repository already publishes, and no new key was minted.

**The limitation that remains:** the key lives on a development machine
(`~/.config/miniagi/release-ed25519.pem`) and is not an independently
witnessed third-party attestation. The signature establishes that the
envelope was produced by the holder of that key; it says nothing about
whether the scientific claims are correct.

**Note on a mid-flight correction.** An earlier draft of this release
reported the envelope as unsigned, on the mistaken belief that the
attestation's `public_key_sha256` (`a8cd5cb2…`) disagreed with the
bundled key. It does not: `public_key_sha256` is the hash of the **PEM
file bytes**, and that is exactly what the bundled key hashes to. The
earlier claim was wrong and has been corrected in every document that
carried it.

## RD-3 — The research question is unanswered

The executed positive control returned **REFUSED** (5 of 7 readiness
gates passed):

- the 95% fit gate was unmet (L6 train accuracy 0.767);
- cross-family transfer was negative (L6 0.167 vs frozen 0.417 on the
  held-out families).

**Consequence.** Gate G12 is passed in the sense that *no improvement
claim is made without recorded evidence* — and the recorded evidence is
negative. There is no evidence in this package that fixed LoRA improves
unseen-task performance beyond the non-neural baseline, and there is
weak evidence that at this scale it hurts.

**What was deliberately not done.** The failure was not compensated for
by adding dynamic LoRA, more agents, recursive training, a larger model,
or by relaxing the gate.

## RD-4 — The diagnostic is single-seed and small

One seed, 30 training examples, 60 LoRA steps, a 0.5B model. It is a
readiness diagnostic, not a research campaign. It cannot support any
claim about the intended generalization question, and it is not
presented as one.

## RD-5 — Delayed retention is inherited, not measured, on the sealed lane

The sealed holdout does not contain the seed-end delayed-retention
probes. The sealed gate records an explicit `inherited` entry with its
justification, and inheritance applies only to an UNDEFINED category —
a measured failure still blocks. This is documented rather than
silently skipped, but it is a genuine reduction in what the final
confirmation independently measures.

## RD-6 — CLOSED: enforced-lint gap (four unused imports, one of them mine)

**Closed.** The repository's enforced lint contract
(`select = ["E4","E7","E9","F"]`) was violated by four unused imports —
three pre-existing in `src-python/minagi/runtime/`, one mine
(`typing.Iterable`, orphaned when a dead helper was removed). Each was
verified unused *in its own file* before removal. Hygiene-only: removing
an unused import cannot change behaviour, and it was verified rather than
asserted — full suite 1225 passed / 0 failed and native CTest 28/28,
both re-run after the edits. `ruff check src-python/` is now clean.

## RD-7 — Campaign 3A archive provenance is only as good as its own manifests

The 3A archive was extracted, inventoried and checked for CRC, duplicate
paths, symlinks, path traversal and resource limits. Its provenance
rests on the manifests it ships with; no independent attestation of the
3A archive was available. The REFUSED decision and its evidence are
preserved unmodified.

## RD-8 — GPU and Linux-isolation tests: NOT RUN on this host

- **GPU tests: NOT RUN.** No CUDA device on this host; the campaign
  config runs with `require_cuda: false`.
- **18 environment-gated tests skipped**, all honest: NsJail/systemd
  boundaries (Linux only), cgroup v2, uid demotion (requires privilege),
  `/proc` marker scan, `NOTE_TRACK`, and `RLIMIT_AS` on macOS. No
  skipped test is counted as passing.
- The v16.4.10 record includes a privileged-container qualification
  (aarch64 Debian, systemd 252, cgroup v2) — that is the runtime's
  evidence, not re-run here.

## RD-9 — The safety suite is a probe set, not a proof

`SAFETY_AND_RETENTION_REPORT.md` is explicit: eight named categories
with objectively verifiable predicates. It does not establish general
safety and does not cover adversarial cases outside its probes.

## RD-10 — Diagnostic reproducibility after the corpus change

The Step 13 diagnostic was first executed against a 68-row corpus whose
security block held four smoke lures and whose retention block had no
delayed probes. The governed suite replaced the lures with ten probes
plus the retained lures, and the delayed probes were added — so that
first run is not reproducible from the current corpus, and it could
never have qualified (the delayed-retention gate would always have
refused).

**Resolved.** The diagnostic was re-run against the committed 80-row
corpus and `LEARNABILITY_TEST_REPORT.md` reports the re-run's numbers
(corpus digest `sha256:fe12aec4…`, plan digest `sha256:4830f2a1…`). The
superseded run is retained in `docs/science/LEARNABILITY_REPORT.md` and
marked as such. The sealed holdout digest is unchanged
(`sha256:11ea116394fdc1cfffb7d63153dec44c5a8802f344f65a425d7037c9094dcf63`)
and the generated campaign config is byte-identical, so the sealed
binding was not disturbed by the corpus change.

## RD-11 — CLOSED (non-issue): defect id SCI-304 is unused

The register runs SCI-301, SCI-302, SCI-303, SCI-305, SCI-306..
SCI-304 was never assigned. There is no missing document and no gap in
the register; the id is simply unallocated. Recorded here only so a
reader does not go looking for it. Closed as a non-issue.

## RD-12 — CLOSED: the qualifier could not qualify an L5b/L7 campaign

The qualifier resolved an arm's non-parametric state directory with a
rule that hardcoded `L5` as the only per-seed arm. L5b and L7 carry
per-seed replay state, so the Campaign 3B **primary** comparison could
never be qualified — every attempt returned `INVALID_EVIDENCE` with
"arm state dir missing".

Invisible until the qualifier was actually run against executed
evidence: the runner's own summary was unaffected, so nothing else
caught it. Found by the independent qualification run of 2026-10-13.

**Fixed** with an explicit arm → state-directory map that also records
that L7 shares L5b's directory by design, plus a sync test pinning the
map to `arms.py`. Verified against the real evidence: the qualifier now
completes the full chain and returns REFUSED with
`runner_decision_agreement: true`.

## RD-13 — PARTIALLY RESOLVED: fit experiment run, ablations still open

Step 4 has now been executed (`CAMPAIGN3B1_FIT_EXPERIMENT.md`): four
runs with matched negative controls. Findings — fixed-rank LoRA reaches
1.000 subset fit at both 2 and 6 families; at **2** families it
generalizes *positively* (hidden 0.417 → 0.583, control 0.000, so the
gain is learning); at **6** families it shows **no measurable benefit
over the shuffled-label control** (0.250 vs 0.250).

**Still open:** the 2-family and 6-family runs differ in example count
(10 vs 30) and steps (300 vs 600), so breadth is not isolated from data
volume. A 6-family run at 300 steps, or a 2-family run at 600, is the
obvious next ablation and it is cheap. One seed, one rank, one learning
rate, 12 hidden rows — direction well-supported, magnitudes provisional.

## RD-13b — ORIGINAL TEXT: the adapter's effect size is unmeasured

The Campaign 3B.1 interference diagnosis
(`CAMPAIGN3B1_INTERFERENCE_DIAGNOSIS.md`) establishes the *mechanism*
with every context control verified identical, but the *effect size*
rests on 4 held-out families × 3 rows × 1 seed. The −0.500 is 6 rows;
the strongest single signal (`pc-len` collapsing 3/3 → 0/3 while the
shuffled-label control retains 3/3) is 3 rows.

**Consequence.** Treat the mechanism as a well-supported hypothesis and
the magnitude as unknown until steps 4-7 run at a larger scale. Step 4
in particular — proving the adapter can fit a small verified subset to
≥95% — must precede any transfer claim, because the current run never
cleared that bar and the "generalization failure" may partly be an
under-fitting failure.

## RD-17 — OPEN: two declared train families are the same function

`pc-rev` and `pc-swap` are identical on every training word. All training
words are three letters long, and for a three-letter word "swap the first
and last characters" IS "reverse the word". The corpus declares six train
families but carries **five distinct functions**.

**Consequence:** every breadth count over the train families overstates
by one, and `pc-rev` / `pc-swap` are not independent evidence about each
other.

**Not fixed, deliberately.** v2 keeps the training block byte-identical
to v1 so v1- and v2-trained runs are comparable; fixing the collision
would silently change every training condition. Pinned by a test that
asserts exactly the known collision, so a new one fails the build and a
silent fix also fails it. A future corpus should use varied word lengths.

Found while interpreting the single-family diagnostic
(`SINGLE_FAMILY_DIAGNOSTIC.md`).

## RD-16 — CLOSED: breadth was confounded with family composition

The 3B.2 Stage 3 conclusion "breadth is the binding constraint" was an
artifact. The 2-family condition was always `pc-upper,pc-rev` while every
breadth-4/6 condition contained `pc-rot1`, so breadth and composition
moved together.

External review warned about exactly this ("family breadth is also
associated with which particular transformations are included") and
asked for composition to be balanced. That warning was correct.

**Closed** by `FAMILY_COMPOSITION_CORRECTION.md`: varying only which two
families spans 0.111–0.500, and at fixed breadth 4, excluding `pc-rot1`
gives 0.528 against 0.200 with it. Over 28 correct-label runs, presence
of `pc-rot1` separates perfectly (mean 0.181, 0/17 above frozen) from
absence (mean 0.474, 13/19 above). The published report now carries a
withdrawal notice at the top.

**Lesson recorded:** the original design changed breadth and composition
together. Any future breadth claim must vary family composition
explicitly or hold it fixed.

## RD-15 — UNRESOLVED: the review's test counts do not match this tree

The Campaign 3B.2 review reports **1,111 passed / 47 skipped** for the
Python regression. This tree reports **1,206 passed / 18 skipped** after
the 3B.2 repairs (1,191 / 18 before them). That is 51 more tests
collected and 29 fewer skips.

This is not explained by the 3B.2 changes and I am not claiming either
number is wrong — the review may have run a different commit, a
different test scope, or a different environment. The most plausible
cause is an environment without the `ml` extras: 59 tests live in files
that import `torch`/`transformers`/`peft` at module level, and a host
without them would collect fewer tests and report more skips. That is a
hypothesis, not a finding.

**Both numbers are reproducible from the same command**, which is the
point:

```sh
python3 -m pytest tests-python -q     # 1206 passed, 18 skipped here
```

**Needs:** the review's commit SHA, environment (torch present?) and
exact pytest invocation, so the two runs can be reconciled before either
count is quoted in a release document. Until then, quote this tree's
number and note it is host-specific.

## RD-14 — CLOSED: safety failures are now separated by cause in the runner

**Closed.** The runner now computes the same absolute-vs-adaptation-
induced classification from its per-arm per-category metrics
(`arm_safety_classes`), so the provisional summary and the authoritative
record agree on the distinction. Both still block promotion.

## Summary

| id | class | status |
|---|---|---|
| RD-1 | confirmation not executed | **NOT RUN** |
| RD-2 | signed envelope, not independently attested | **limitation declared** |
| RD-3 | research question unanswered | **negative result reported** |
| RD-4 | single-seed diagnostic | limitation |
| RD-5 | sealed delayed retention inherited | documented |
| RD-6 | enforced-lint gap | **CLOSED** |
| RD-7 | 3A archive provenance | limitation |
| RD-8 | GPU / Linux isolation not run here | **NOT RUN** |
| RD-9 | safety suite is a probe set | scope limit |
| RD-10 | first diagnostic not reproducible | re-run |
| RD-11 | unused defect id | **CLOSED** (non-issue) |
| RD-12 | qualifier could not qualify L5b/L7 | **CLOSED** |
| RD-13 | adapter effect size unmeasured | **step 4 executed**; breadth-vs-volume ablation open |
| RD-14 | runner reports a single violation count | **CLOSED** |
| RD-15 | review's test counts differ from this tree | **unresolved** — needs their env + commit |
| RD-16 | breadth confounded with family composition | **CLOSED** — conclusion withdrawn and replaced |
| RD-17 | pc-rev and pc-swap are the same function | **OPEN** — recorded, deliberately not fixed |
