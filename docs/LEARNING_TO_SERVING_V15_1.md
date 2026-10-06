# v15.1 Learning-to-Serving Closure

## Scope

This release closes one deliberately narrow continual-learning path: L4 procedural skill learning. It does not train foundation weights. The goal is to prove that qualified experience can create a new content-addressed served artifact and that the native serving boundary refuses stale or mismatched state.

## Runtime chain

1. `VerifiedTrajectoryV145` carries evidence and verification receipt digests.
2. `TrajectorySkillInducerV145` remains proposal-only and contradiction-blocking.
3. `HeldOutTransferQualifierV145` must move a candidate to `transfer_qualified`.
4. `SkillIRCompiler` compiles only a conservative allow-list of deterministic operations. Unknown operation signatures fail closed.
5. Hidden fresh tasks are sealed before the candidate evaluation. `FreshTaskPairedEvaluator` consumes each task exactly once and executes A0 and A1 against the same reveal.
6. `FrozenBaselineGateV145` requires a positive paired lower bound, retention floor, and zero security regressions.
7. A passing result creates a content-addressed `SkillPolicyBundle`. Its digest becomes the new served `skill_policy_root`.
8. `ServedArtifactManifest.artifact_root` changes because it closes over `skill_policy_root`.
9. QW3 exposes the loaded epoch, manifest digest and artifact root at `/v1/runtime/state` and rejects primary generation requests unless all three request identities match.

## Authority boundary

`GovernedSkillLearningLoop` does not authorize promotion and does not mark a StateEpoch SERVABLE. It only prepares immutable artifacts. Production persistence still requires the existing candidate/build/evaluation/qualification/promotion/atomic activation/StateEpoch chain.

## Fixed Integration1 defect

Integration1 represented `epoch_digest` and `runtime_manifest_digest` as bare hex values. Governance objects use canonical `sha256:<hex>` digests, so a real `StateEpochLeaseV145` could not satisfy the bridge contract. v15.1 uses the governance representation end-to-end while keeping physical artifact roots as raw SHA-256 hex values for the native serving envelope.

## Non-claims

This release does not demonstrate:

- LoRA or base-weight continual training;
- broad semantic skill synthesis;
- real-weight CUDA or Metal qualification;
- 10k–100k episode stability;
- autonomous architecture modification;
- independent third-party reproduction.

The next material milestone is L7 adapter learning: verified trajectories -> training corpus -> isolated LoRA candidate -> sealed A0/A1 + forgetting/security battery -> exact adapter-set root -> governed QW3 loading and serving.
