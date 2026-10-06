# v14.1 Transactional Control Plane

v14.1-alpha1 narrows the next release around a single invariant: persistent production state must be reachable only through a complete, content-bound, auditable authority chain.

## Canonical artifact chain

`LearningProposal -> CandidateManifest -> BuildManifest -> EvaluationBundle -> QualificationRecord -> PromotionAuthorization -> RuntimeManifest -> ActivationReceipt`

Artifact bodies are immutable CAS objects. SQLite stores authoritative bindings and the active runtime pointer. A candidate cannot be evaluated without its registered build; a qualification cannot exist without the bound evaluation; a promotion cannot exist unless the qualification decision is `PROMOTE`; and activation requires a signed promotion authorization for the same candidate.

## Trust boundaries

- Candidate builders can register/build but cannot promote.
- Evaluators can record evaluation artifacts but cannot mint promotion authority.
- Qualification records carry the policy and authority generation under which they were produced.
- The promotion authority holds the promotion signing key and refuses rejected, stale-generation, or mismatched qualifications.
- The persistence gateway validates the signed authorization before changing the runtime head.

## Immutable state

Beliefs and skills are revisioned rather than overwritten. Current state is represented by an atomic head pointer. Negative skill evidence is retained, and previous belief revisions remain queryable after supersession.

## Auditability

Every privileged lifecycle mutation emits a hash-chained audit event. `GovernanceDB.verify_audit_chain()` recomputes the chain and fails closed on mutation.

## Scope

This alpha hardens authority and persistence. It does not enable L7-L10 production mutation, claim real-weight Metal/CUDA qualification, or claim autonomous recursive self-improvement.
