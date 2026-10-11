# Threat Model — v16.4.5

Trust ordering (unchanged): independent research/training and
qualification → trusted runtime supervisor (authorization, admission,
durable audit, activation) → restricted model worker (bounded
inference). The supervisor is more trusted than the worker.

## Adversary classes

1. **Compromised model worker** — controls worker stdout/stdin bytes,
   worker memory, and worker code execution. Must not execute code in
   the supervisor, fabricate authority, read signing credentials, or
   write the authority store.
2. **Tampered staged artifacts** — attacker modifies files between
   authorization and load, or presents a snapshot descriptor pointing
   at a different tree.
3. **Revoked-but-resident model** — a promotion decision revoked after
   authorization but before commit, while already serving, or across a
   cold restart.
4. **Release tampering** — modified governed files, unmanifested
   additions, swapped manifest/key/signature, rolled-back metadata.

## Controls and their boundaries

| Threat | Control | Boundary |
|---|---|---|
| Worker crafts pickled payload | No pickle anywhere on the channel; framed JSON v2, closed schema, bounded sizes, verified digests | IPC only — serialization cannot substitute for OS isolation |
| Worker exfiltrates credentials | Explicit env construction; KEY/TOKEN/SECRET/PASSWORD/CREDENTIAL names dropped; private 0700 scratch | Name-based filtering is a denylist for known credential names — nonstandard variable names are not caught; production deployments should additionally drop the whole env and whitelist |
| Worker consumes unbounded resources | POSIX rlimits (CPU, memory, processes, files, file size) where supported | **macOS: RLIMIT_AS is not enforced by the kernel** — memory limits are requested but not guaranteed; process-group termination still applies. Platform qualification is Linux-first; macOS claims are limited to what tests demonstrate |
| Worker escapes via filesystem | Private scratch; read-only approved artifact access is a deployment property (the isolation policy provides the mechanism; sandbox-exec profile only when explicitly configured) | No filesystem sandbox is applied by default on macOS — do not claim sandbox equivalence without explicit configuration and testing |
| Tampered staged bytes | Descriptor transport re-measures every byte in the receiving process and compares to authorized digests | Covers the measured tree; the base model path is resolved under the same staging rules |
| Revoked promotion restored | Full-chain re-verification under current revocation evidence; fresh epoch-bound grant; commit-time recheck; route withdrawal + quarantine for serving models; revoked predecessors retired before fallback | Requires valid, current, non-rolled-back revocation evidence — `RevocationStore.latest_valid(require=True)` fails closed when evidence is mandatory |
| Revocation race authorize→commit | Grant binds `revocation_epoch`; commit refuses on any advance; explicit revoked-decision and revoked-key checks underneath | Same-epoch replacement is impossible through `publish()` (write-once epochs) |
| Release tampering | Manifest over `git ls-files`∩disk, Ed25519 signature, pinned key fingerprint in `verify_release.py`, attestation reconciliation (exit 5 on drift), fresh-extraction verification | The pinned fingerprint is only as trustworthy as the verifier copy carrying it — for high assurance use `--trusted-key` from an independent channel |
| Signing-key loss | Key lives outside the repo; `reissue_release.py` refuses to mint one silently | If the pinned key is lost, the release line must be re-pinned explicitly — by design there is no quiet path |

## Explicitly out of scope for v16.4.5

- Full sandbox equivalence on macOS (documented above).
- GPU/Colab real-model platform qualification — v16.4.6 target.
- The worker's own network egress policy beyond the environment and
  launch controls described (a deployment-level property).
