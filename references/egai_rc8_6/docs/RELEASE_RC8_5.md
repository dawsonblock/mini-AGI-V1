# RC8.5 — Real Task Learning + Replay Qualification

RC8.5 connects previously separate research controls.

## Added
- Prefix-causal replay `step()` where the policy sees only historical prefix while the replay environment may reveal the recorded next transition.
- Replay integrity audits bound into ExperimentClosure and ExperimentResult.
- Negative-transfer measurement over future + retention cases.
- Qualification gates for negative transfer, replay grounded coverage, replay prefix violations, and future evaluation sample size.
- Sandbox skill lifecycle: active -> degraded -> retired based on accumulated negative evidence.
- Retired skills are automatically excluded from retrieval.
- Skill lifecycle events and health are included in the content-addressed skill snapshot.
- Runtime skill successes/failures are tracked separately from induction support.
- Episode traces are retained in the sequential report for replay auditing.

Replay is evidence about historical support only. It does not authorize promotion and imagined outcomes remain non-evidence until externally verified.
