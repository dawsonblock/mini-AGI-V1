# v15.8 Campaign1 — preregistered real-weight neural campaigns

v15.8 turns the v15.7 A0/A1 evaluator into a reproducible experiment campaign. It still does not claim a real-weight gain from this build environment.

## New invariant

Before any hidden task is consumed, the campaign freezes and content-addresses:

- exact A0 and A1 governed runtime identities;
- endpoints and StateEpoch identities;
- adapter/native-adapter roots;
- the ordered fresh-task commitments;
- transfer/retention/bootstrap gate parameters;
- scorer identity;
- retention policy identity;
- security policy identity.

`PreregisteredNeuralCampaignRunner` refuses to consume the task battery unless that exact plan already exists in immutable CAS. Runtime arm substitution, task-battery reordering/substitution, or gate changes after preregistration are rejected.

## Operational launch recipe

`QW3LaunchSpec` emits the native QW3 governed-serving arguments directly from a `ServedArtifactManifest`. Paths remain deployment inputs rather than trust anchors: QW3 must independently measure the deployed bytes and match the authorized roots before serving.

## Result bundle

A completed run emits a content-addressed `NeuralCampaignResultBundle` binding:

- preregistered plan digest;
- sealed neural experiment digest;
- baseline report digest;
- all paired task result digests;
- all fresh-task consumption receipts;
- all A0/A1 runtime evidence digests;
- candidate adapter/native-adapter roots;
- PASS/BLOCK decision.

`promotion_ready=True` means only that the preregistered experiment gate returned PASS. The campaign runner has no promotion authority and does not mutate production state.

## Remaining proof

A real Qwen model and qualified adapter must still be run through this campaign on sealed unseen, retention, and security batteries. The release tests use deterministic model doubles to validate campaign immutability and evidence closure, not neural capability.
