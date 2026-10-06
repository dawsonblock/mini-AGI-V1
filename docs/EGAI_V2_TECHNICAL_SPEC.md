# EGAI v2 Technical Specification

Status: **implementation baseline for mini-AGI v12.0**

This document specifies interfaces and invariants for the research-governance layer. It is intentionally narrower than a full continual-intelligence implementation. The objective is to make the first hypothesis falsifiable without granting the learning system direct persistent-write authority.

## 1. Normative vocabulary

`MUST`, `MUST NOT`, `SHOULD`, and `MAY` are used normatively.

- **Evidence**: an immutable record that an event, source, deterministic result, inference, replay result, or simulation occurred.
- **Candidate**: a non-authoritative artifact staged for evaluation.
- **Proposal**: a request to make a candidate or interpretation persistent.
- **Qualification**: an independent evaluation against fresh/hidden tests and hard constraints.
- **Promotion**: authorization for a qualified candidate to enter a persistent authoritative store.
- **Production transition**: v11-level deployment transition; separate from EGAI promotion.

## 2. Authority model

### 2.1 Cognitive plane

MAY read evidence and promoted skills. MAY run reasoning, retrieval, verification, experiments, and actions under the existing capability system. MUST NOT issue EGAI promotion authorization.

### 2.2 Learning plane

MAY stage candidates and `LearningProposal` objects. MUST NOT possess a promotion signer. MUST NOT directly mutate promoted belief/skill/module stores.

### 2.3 Promotion plane

MUST consume a `QualificationBundle`. MUST fail closed on hard-constraint violations. MAY issue a signed `PromotionAuthorization` only after an `APPROVE` decision.

## 3. Core schemas

Machine-readable reference schemas are under `schemas/egai-v2/`.

### 3.1 EvidenceRecord

Required logical fields:

```text
EvidenceRecord
  evidence_id
  origin_class
  evidence_class
  producer
  payload_digest
  provenance_digest
  production_identity_digest
  observed_at
  valid_from / valid_until
  confidence
  parent_digests
  source_locator
  summary
```

Invariant: simulation origin MUST be classified as simulated and MUST NOT be promotion-evidence eligible.

### 3.2 Belief

A belief is an interpretation of evidence, not raw evidence. It may exist as a candidate without qualification, but a `promoted` belief MUST carry a qualification digest. Belief records bind supporting and contradicting evidence, confidence, validity interval, and production identity. `BeliefRepository` allows learning-plane candidate registration but requires signed promotion authorization for persistence in the promoted store.

### 3.3 Hypothesis

A hypothesis binds a claim digest to supporting evidence, explicit falsification tests, predicted transfer rings, and status (`open`, `falsified`, or `supported`). A hypothesis is a learning object; support does not itself authorize persistence of a belief, skill, adapter, or model change.

### 3.4 LearningProposal

```text
LearningProposal
  proposal_id
  action
  level (L0..L9)
  target
  evidence_digests[]
  rationale_digest
  proposer_id
  production_identity_digest
  expected_forward_transfer
  expected_interference
  expected_compute_cost
  expected_capacity_growth
  risk
```

A proposal contains no method that mutates the target system.

### 3.5 SkillManifest

```text
SkillManifest
  skill_id / version / name
  implementation_digest
  activation_conditions[]
  preconditions[]
  contraindications[]
  permissions[]
  resource_budget{}
  termination_conditions[]
  verifier_digest
  supporting_evidence[]
  known_failures[]
  qualification_digest
  rollback_target
```

### 3.6 QualificationBundle

```text
QualificationBundle
  proposal_digest
  candidate_digest
  production_identity_digest
  evaluator_id
  qualification_worlds
  transfer_rings_passed[]
  fresh_one_shot_worlds
  hidden_until_evaluation
  metrics
  suite_digest
```

### 3.7 ImprovementRecord

Records the proposal/candidate/qualification/decision linkage, learning level, origin evidence, affected components, metrics, production identity, and rollback target.

A later `FutureConsequence` links actual subsequent outcomes to the improvement record.

## 4. State machines

### 4.1 Candidate lifecycle

```text
DISCOVERED
   |
   v
PROPOSED
   |
   v
BUILT
   |
   v
SANDBOXED
   |
   v
QUALIFIED -----------------------+
   |                              |
   v                              v
REJECTED                      APPROVED
                                  |
                                  v
                         SIGNED AUTHORIZATION
                                  |
                                  v
                         V11 PRODUCTION GATE
                                  |
                         +--------+--------+
                         |                 |
                         v                 v
                      REJECT           DEPLOY
```

No transition from `PROPOSED`, `BUILT`, or `SANDBOXED` directly to `DEPLOY` is legal.

### 4.2 Evidence lifecycle

```text
external event/result
      |
      v
trusted ingest
      |
      v
immutable EvidenceRecord
      |
      +--> model interpretation --> candidate belief/procedure
      |                              (requires promotion)
      |
      +--> grounded replay --> derived evidence
      |
      +--> simulation --> simulated record only
```

## 5. Replay protocol

A replay policy sees a `PrefixReplayView` containing only currently revealed nodes. The implementation MUST NOT hand the policy the source world object.

For a selected visible leaf/root, the grounded replay engine MAY reveal only a recorded child. It MUST NOT synthesize a new child.

Model-based imagination MUST produce `SimulationRecord`, not `EvidenceRecord` with observed/derived class.

## 6. Plasticity selection protocol

The reference proposer takes candidate options and applies:

```text
1. discard infeasible options
2. discard options below the utility floor
3. find minimum permanence level among remaining options
4. choose maximum utility only within that level
5. emit LearningProposal
```

This enforces the least-permanent-capable-mechanism rule.

## 7. Independent qualification protocol

The gate evaluates the candidate against hard constraints.

Default metrics:

- forward transfer delta;
- forgetting;
- OOD delta;
- calibration regression;
- number of security regressions;
- number of unauthorized writes;
- provenance closure fraction;
- ablation attribution;
- compute delta;
- capacity growth.

Reference hard failures include:

```text
security_regressions != 0
unauthorized_writes != 0
provenance_closure < 1.0
forgetting > configured tolerance
calibration_regression > configured tolerance
OOD delta below configured minimum
fresh one-shot world requirement violated
hidden-until-evaluation requirement violated
insufficient world count or transfer breadth for level
missing positive ablation attribution for L3+
```

FTE is reported but cannot override hard failures.

## 8. Qualification pool partition

The intended evaluator maintains distinct pools:

```text
Dream-train worlds       reusable by the learner
Replay-validation worlds evaluation only / limited reuse
Qualification worlds     fresh one-shot resources
Audit worlds             never visible during development
```

The existing v10/v11 RSI firewall remains the lower-level mechanism for exposure accounting. EGAI v2 treats fresh-one-shot and hidden-until-evaluation as qualification facts that must be attested in the bundle.

## 9. Promotion authorization

An approval produces a signed authorization body:

```text
schema = mini-agi-egai-promotion-authorization-v2
authorization_id
proposal_digest
candidate_digest
qualification_digest
decision_digest
verdict = approve
issued_at
```

A persistent skill store MUST verify the signature and candidate digest before promotion.

## 10. Production handoff

An EGAI approval is necessary but not sufficient for production deployment of neural/runtime changes.

The v11 gate remains responsible for:

- production identity;
- target profile and matrix gate;
- hardware certificate when acceleration is requested;
- signed production transition;
- rollback;
- cache epoch;
- runtime health supervision.

## 11. First experiment protocol

Foundation weights remain frozen. L6-L9 are not enabled.

Required checkpoint schedule:

```text
0, 50, 100, 250, 500, 1000
```

At every checkpoint record at least:

- future-task success;
- old-task success;
- calibration error;
- tokens/task;
- tool calls/task;
- wall time/task;
- skill reuse rate;
- repeated-error rate;
- exact base-model digest.

Primary pass condition in the reference protocol:

```text
future success improves
AND old-task drop <= tolerance
AND base-model digest is unchanged at all checkpoints
```

This is a research gate, not a claim that 1,000 episodes is universally sufficient to establish continual intelligence.

## 12. Failure modes that MUST fail closed

- qualification identity does not match proposal identity;
- candidate digest mismatch;
- stale/insufficient qualification cohort;
- future replay node selected before it is visible;
- simulation passed as observed evidence;
- unsigned/mismatched skill promotion authorization;
- foundation digest changes during frozen-foundation experiment;
- Improvement Ledger chain/hash corruption;
- evidence payload hash mismatch;
- security or unauthorized-write regression.

## 13. Explicit non-goals for v12

v12 does not claim:

- proof of continual intelligence;
- autonomous safe self-modification;
- trained Plasticity Proposer;
- production world-model imagination;
- production L6-L9 neural adaptation;
- external-paper benchmark reproduction;
- production DeepSeek CED/CSA2/FP4 equivalence;
- Apple-Silicon acceleration qualification unless a target-bound certificate is created on that hardware.
