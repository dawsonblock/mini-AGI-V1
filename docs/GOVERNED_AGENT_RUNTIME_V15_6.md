# Governed Agent Policy Runtime — v15.6

## Purpose

v15.6 closes the behavioral gap left by v15.5. Retrieval and SkillIR artifacts were already physically measured and bound into the served artifact root; this release makes the higher agent layer consume those exact bytes. No new mutation or promotion authority is introduced.

## Execution boundary

`StateEpoch lease -> native QW3 state verification -> physical policy re-hash -> SkillIR route OR governed QW3 fallback -> execution receipt`

Before every task the runtime fails closed unless the physical skill/retrieval files still hash to the roots in the served manifest and the existing serving contract reports the exact leased epoch, manifest and aggregate artifact root.

A matching `SkillIR` executes deterministically in the agent layer. If there is no matching skill, the bound retrieval policy ranks evidence-bearing episodic records and adds only the selected records to the fallback QW3 request. The runtime records the selected record/evidence digests in an immutable execution receipt.

## Sealed learning proof

The v15.6 regression uses the existing fresh-task vault and frozen A0/A1 gate. Hidden reverse-text tasks are sealed before evaluation. A0 returns input unchanged. A1 executes the physically bound qualified reverse SkillIR through `GovernedAgentRuntime`. The candidate passes with mean paired delta 1.0 and a positive 95% bootstrap lower bound in the deterministic test fixture.

This establishes that a learned procedural artifact can change behavior while remaining bound to the same governed serving identity. It does not establish general transfer, neural weight learning, or autonomous RSI.

## Nonclaims

- no real model-weight improvement result;
- no CUDA/GPU qualification in this environment;
- no general tool-planning learner;
- retrieval is deterministic lexical/family scoring, not learned dense retrieval;
- SkillIR remains deliberately constrained;
- no 10k–100k continual campaign;
- no independent reproduction.
