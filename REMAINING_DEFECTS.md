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

## RD-6 — Pre-existing lint findings outside the enforced scope

`ruff` over the full tree reports 15 `F401` unused-import findings in
`src-python/minagi/runtime/*` (`worker_backend.py`,
`worker_executor.py`, and others). **None is in a file this work
changed**, and the repository's own release gate
(`scripts/release/host_gate.sh`) does not run ruff. They are recorded
here rather than fixed, because editing the worker backend for a lint
finding is not a change this campaign should make to the containment
surface.

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

## RD-11 — Minor: defect id SCI-304 is unused

The Campaign 3B register runs SCI-301, SCI-302, SCI-303, SCI-305.
SCI-304 was never assigned to a defect. Cosmetic; recorded so a reader
does not assume a missing document.

## Summary

| id | class | status |
|---|---|---|
| RD-1 | confirmation not executed | **NOT RUN** |
| RD-2 | signed envelope, not independently attested | **limitation declared** |
| RD-3 | research question unanswered | **negative result reported** |
| RD-4 | single-seed diagnostic | limitation |
| RD-5 | sealed delayed retention inherited | documented |
| RD-6 | pre-existing lint findings | recorded, not fixed |
| RD-7 | 3A archive provenance | limitation |
| RD-8 | GPU / Linux isolation not run here | **NOT RUN** |
| RD-9 | safety suite is a probe set | scope limit |
| RD-10 | first diagnostic not reproducible | re-run |
| RD-11 | unused defect id | cosmetic |
