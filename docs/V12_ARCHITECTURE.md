# mini-AGI v12.0 EGAI Research-Governed Architecture

v12 adds an **EGAI v2 research-governance layer** above the v11 production-converged substrate. It does not replace v11's production identity, signed transition ledger, target qualification, hardware qualification, rollback, serving snapshots, or CANARY/SHADOW supervision.

The architectural correction in v12 is explicit:

> **A component that discovers a potentially useful learning intervention is not allowed to make that intervention persistent.**

The Plasticity Proposer is therefore a proposal generator, not an authority.

## 1. Three authority planes

```text
COGNITIVE PLANE
    solve / retrieve / reason / simulate / verify / experiment
                    |
                    v
                experience
                    |
                    v
LEARNING PLANE
    evidence -> replay -> hypothesis -> abstraction -> falsification
    -> transfer -> plasticity proposal -> candidate build
                    |
                    v
              LearningProposal
                    |
              TRUST BOUNDARY
                    |
                    v
PROMOTION PLANE
    independent qualification -> ablation -> hard constraints
    -> signed authorization -> v11 production transition
```

### Cognitive plane

The cognitive plane can read promoted knowledge, evidence, and promoted skills. It may create actions and observations through the existing bounded/transactional action runtime. It has no promotion method.

### Learning plane

The learning plane may generate:

- beliefs and belief revisions;
- hypotheses and abstractions;
- procedural-skill candidates;
- retrieval/routing policy candidates;
- module composition candidates;
- adapter/module/consolidation proposals at higher escalation levels;
- grounded replay policies and model-imagination experiments.

It can stage candidates and proposals, but it owns no promotion signer and no production-transition capability.

### Promotion plane

The promotion plane is the only EGAI plane that can issue a persistence authorization. It consumes an independently produced `QualificationBundle`, applies hard constraints, records the decision in the Improvement Ledger, and—only on approval—issues a signed `PromotionAuthorization` that downstream stores can verify.

The signed production transition itself still belongs to the v11 production substrate.

## 2. The online intelligence loop

```text
Environment
    |
    v
Trusted evidence ingest
    |
    v
Immutable Evidence Ledger
    |-----------------------|
    v                       v
Belief model             Task model
    |                       |
    +-----------+-----------+
                v
        Cognitive workspace
                |
                v
        Frozen neural core
                |
                v
              action
                |
                v
              outcome
                |
                v
         causal diagnosis
                |
                v
           new evidence
```

v12 intentionally keeps this loop compatible with a byte-identical frozen foundation model.

## 3. Evidence is typed by origin

Raw evidence append is the only deliberate exception to the general "nothing persists without promotion" rule. A narrow `TrustedEvidenceIngestor` may append an immutable record that an event occurred. It may not convert an interpretation into authoritative knowledge.

Every `EvidenceRecord` carries an `origin_class`:

- `environment`
- `deterministic_tool`
- `external_source`
- `human`
- `model_inference`
- `grounded_replay`
- `simulation`

and an `evidence_class`:

- `observed`
- `derived`
- `simulated`

Simulation is structurally marked non-eligible as promotion evidence. Grounded replay is derived evidence backed by recorded outcomes. Model inference remains model inference even if repeated.

The `EvidenceLedger` is append-only, content-addressed, and hash-chained. Its public interface is read-only; the append capability is held by `TrustedEvidenceIngestor`.

## 4. Replay worlds and prefix observability

The replay architecture takes the DREAM-RSI history-as-simulator idea but hardens it for continual-learning research.

A grounded replay world contains only outcomes actually recorded online. The `GroundedReplayEngine` never invents an unseen child.

Replay policies receive a `PrefixReplayView`. That object contains only nodes revealed at that replay time. It deliberately contains no reference to the source world, which makes future-outcome access structurally unavailable rather than merely prohibited by prompt text.

```text
real experience
    |
    v
Experience graph
    |
    v
ReplayWorldCompiler
    |
    +--------------------+
    |                    |
    v                    v
Grounded replay       World-model imagination
recorded outcomes     predicted outcomes
promotion-eligible    NEVER direct evidence
    |                    |
    +---------+----------+
              v
           Dream engine
              |
              v
       learning proposals
```

## 5. The permanence ladder

v12 freezes the following escalation order:

| Level | Mechanism | Default v12 research status |
|---|---|---|
| L0 | working / latent state | enabled |
| L1 | evidence / episodic memory | enabled |
| L2 | belief update | enabled |
| L3 | procedural skill | enabled |
| L4 | retrieval / routing / policy modulation | enabled |
| L5 | existing-module composition | enabled |
| L6 | module or adapter adaptation | interface only for first experiment |
| L7 | new module | disabled for first experiment |
| L8 | shared neural consolidation | disabled for first experiment |
| L9 | architecture modification | disabled for first experiment |

The proposer implements one rule before utility optimization:

> **Use the least permanent feasible mechanism.**

A high-utility L6 adapter does not outrank a sufficient L3 procedure merely because its immediate benchmark gain is larger.

## 6. Evidence demand scales with permanence

`IndependentQualificationGate` enforces increasing minimum qualification-world counts and transfer breadth as the requested learning level increases.

The default reference schedule is intentionally conservative and configurable:

| Level | Min independent qualification worlds | Min transfer rings |
|---:|---:|---:|
| 0 | 1 | 0 |
| 1 | 2 | 0 |
| 2 | 4 | 1 |
| 3 | 8 | 2 |
| 4 | 12 | 3 |
| 5 | 20 | 4 |
| 6 | 32 | 5 |
| 7 | 48 | 5 |
| 8 | 96 | 6 |
| 9 | 192 | 7 |

These are engineering defaults, not empirical claims that these sample counts are sufficient in every domain.

Transfer rings are:

1. episode;
2. procedure;
3. reusable skill;
4. composition;
5. task family;
6. new domain;
7. extrapolation.

## 7. FTE is an objective, not an authority score

Forward Transfer Efficiency is represented as:

```text
           P(unseen, after) - P(unseen, before)
FTE = ------------------------------------------------
      experience + lambda*compute + mu*capacity_growth
```

Promotion is not `FTE > threshold`. It is constrained optimization. The reference gate rejects the candidate if any hard invariant fails, including:

- forgetting over the configured limit;
- any security regression;
- any unauthorized persistent write;
- incomplete provenance closure;
- unacceptable calibration regression;
- insufficient OOD performance;
- stale/reused qualification worlds where fresh one-shot worlds are required;
- insufficient ablation/causal attribution for L3+ changes.

A large forward-transfer gain cannot compensate for a security regression.

## 8. Skills are governed executable capabilities

A promoted skill is not a prompt fragment. `SkillManifest` binds:

- activation conditions;
- preconditions;
- contraindications;
- implementation digest;
- permissions;
- resource budget;
- termination conditions;
- verifier digest;
- supporting evidence;
- known failures;
- qualification receipt;
- rollback target.

The learning plane may register a candidate. `SkillRepository.promote()` requires a signed EGAI promotion authorization whose candidate digest and qualification digest match.

## 9. Improvement Ledger and meta-learning

Every promotion or rejection is written to the hash-chained `ImprovementLedger` as an `ImprovementRecord`.

Later, observed future consequences can be attached:

```text
learning decision
    |
    v
qualification decision
    |
    v
production observation horizon
    |
    v
FutureConsequence
```

`meta_learning_rows()` joins decisions with their later consequences. This is the dataset intended for eventual learning of the Plasticity Proposer. The proposer is not trained from immediate task reward by default.

## 10. First falsification experiment

The first experiment deliberately excludes L6-L9.

The base model digest must remain byte-identical at every checkpoint:

```text
0, 50, 100, 250, 500, 1000 experiences
```

The primary hypothesis is:

```text
P_future(E_1000) > P_future(E_0)
```

while:

```text
base_model_digest_1000 == base_model_digest_0
old_capability_drop <= configured tolerance
```

The four comparison conditions are:

1. frozen model only;
2. frozen + evidence retrieval;
3. frozen + promoted beliefs/skills/composition;
4. frozen + beliefs/skills + replay-optimized learning/routing policy.

This decomposition is designed to attribute improvement to memory, procedural abstraction, and replay/meta-policy optimization separately.

## 11. Relationship to v11 production authority

v12 deliberately does not create a second production-transition system.

EGAI's `PromotionAuthorization` says:

> this learning candidate passed the EGAI research qualification gate.

v11's production layer must still establish:

- production identity compatibility;
- target/matrix qualification;
- hardware qualification where required;
- signed monotonic transition authority;
- cache epoch invalidation;
- serving snapshot integrity;
- CANARY/SHADOW/EXACT operational health.

The two gates are complementary rather than interchangeable.

## 12. Research-source integration boundaries

v12 keeps source-derived ideas separated from implementation claims:

- DREAM-RSI motivates grounded replay worlds and policy experimentation over recorded history. v12 adds stronger prefix-observability and independent promotion boundaries.
- Process Reward Agents motivate inference-time stepwise verification around a frozen reasoner. PRA remains a cognitive-plane mechanism, not promotion authority.
- CodeAct motivates executable actions with control/data flow; v12 retains the existing restricted transactional action runtime rather than unrestricted Python.
- DeepSeek-V4.1-Flash motivates asymmetric persistence/recomputation of neural state and cheap-proposal/expensive-verification patterns; v12 does not claim a production DeepSeek CED/CSA2/FP4 reproduction.
- ASYMSPEC remains an optional cognitive-runtime optimization whose use is reflected in execution identity/evidence, not epistemic authority.

See `docs/research/RESEARCH_INTEGRATION_MAP.md` for exact source hashes and boundaries.

## 13. Frozen architectural invariants

1. **Plasticity proposal is not promotion authority.**
2. **Simulation is not observation.**
3. **Grounded replay cannot reveal future nodes before they become observable.**
4. **More permanence requires more independent evidence and broader transfer.**
5. **Security/provenance failures are hard constraints, not weighted penalties.**
6. **A skill is executable, permissioned, versioned, verifiable, and rollback-bound.**
7. **The first continual-learning claim must be demonstrated with an unchanged base-model digest.**
8. **EGAI promotion does not bypass v11 production qualification.**
9. **Meta-learning trains on future consequences of learning decisions, not only immediate reward.**
10. **No external-paper benchmark result is claimed without a separately bound evidence receipt.**
