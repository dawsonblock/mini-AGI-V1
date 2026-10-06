# v15.2 Adapter Qualification and Serving Identity Closure

v15.2 adds an adapter-candidate path without creating a new promotion authority.
It is a build/qualification layer over the v14.1-alpha5 authority trunk and the
v15 governed-serving contract.

## Closed path

```text
VerifiedTrajectory
  -> VerifiedAdapterDataset
  -> AdapterTrainingPlan
  -> isolated trainer output directory
  -> exact per-file SHA-256 manifest
  -> raw adapter bytes published into ImmutableCAS
  -> AdapterSetBundle
  -> sealed fresh-task A0/A1 experiment
  -> paired transfer lower-bound gate
  -> retention floor
  -> zero security regressions
  -> QualifiedAdapterSetCandidate
  -> adapter_set_root change in ServedArtifactManifest
  -> StateEpoch qualification/promotion remains external
  -> governed QW3 request identity includes adapter_set_root
```

## Important invariants

1. Only trajectories carrying both evidence digests and verification receipts can
   become adapter training examples.
2. A training plan binds the foundation model, dataset, trainer identity,
   backend, hyperparameters, seed, and optional parent adapter-set root.
3. Adapter artifacts are a sorted manifest of exact bytes. Symlinks and path
   traversal are rejected.
4. Qualification republishes every adapter payload file into ImmutableCAS and
   verifies byte length and SHA-256 against the build manifest. A modified file
   is rejected even if the higher-level manifest object is unchanged.
5. The sealed experiment is bound to one concrete adapter artifact and one
   candidate adapter-set root.
6. The existing frozen-baseline gate requires positive paired transfer,
   required rings, minimum retention, and zero security regressions.
7. A passing adapter can change only the `adapter_set_root` candidate field.
   It cannot promote itself or make a StateEpoch SERVABLE.
8. The qualified adapter foundation digest must match the foundation digest of
   the served manifest.
9. Governed QW3 now reports and independently checks `adapter_set_root` in
   addition to the StateEpoch, manifest digest, and aggregate artifact root.

## MLX-LM path

`MlxLoraCandidateRunner` can materialize the verified dataset as `train.jsonl`
and invoke the MLX-LM LoRA entry point with the bound training parameters. The
runner is build authority only. Its output must still pass artifact hashing,
CAS publication, sealed A0/A1 qualification, and the existing promotion chain.

## Explicit non-claim

The native QW3 engine in this release does **not** yet apply LoRA matrices to its
custom inference kernels. v15.2 closes adapter *artifact qualification and
serving identity*, not native adapter execution. A real in-weight serving
release must implement adapter application in QW3 (or a separately measured
backend), prove that the reported adapter-set root corresponds to the bytes
actually applied during inference, and rerun sealed A0/A1 evaluation on that
backend.
