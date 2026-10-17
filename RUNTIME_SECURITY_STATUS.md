# Runtime security status

Status: **boundary-hardened where this host can enforce it; honestly
declared where it cannot.** Last reviewed for Campaign 3B.3.

## What is enforced

| control | mechanism | status |
|---|---|---|
| Sealed set access | POSIX `0600`, owned by evaluating uid | enforced, tested |
| Custody independence | storage-owner uid ≠ evaluator-owner uid | enforced, tested (RD-19) |
| Confirmation mode | `production` fails closed without verified artifacts + custody; `synthetic` labelled `production_valid: false` | enforced, tested |
| Worker launch | refuses if no verified cgroup established | fail-closed |
| Worker teardown | `cgroup.events populated=0` + unit inactive + invocation-identity match required before `REMOVED_VERIFIED` | fail-closed |
| `MINAGI_RESEARCH_WORKER` | env marker | cooperative guard only — NOT isolation |

## `ContainmentState` semantics (fail-closed)

- `REMOVED_VERIFIED` is asserted only when the cgroup is gone **and**
  the unit is inactive **and** the invocation identity matches —
  `populated=1` wins over a plausible-looking inactive state.
- Malformed or missing `cgroup.events` → `UNKNOWN`, which blocks
  teardown confirmation rather than guessing.
- Nothing transitions to a terminal-safe state on unverified evidence.

## Not exercised on this host (declared, not hidden)

- NsJail / systemd isolation chain — Linux-only; reachable only via
  `minagi.runtime.service`, which is `SERVICE-ENTRYPOINT` in the
  control-plane map.
- `minagi.runtime.service` deployment path is `NOT VERIFIED` — nothing
  in this tree proves production runs it rather than the `kvcontinual`
  gateway (`macos/run_gateway.sh`) or the native `qw3` binary
  (`docker/Dockerfile.cuda`). Hardening the isolation files is
  necessary; pinning that production invokes them is a separate,
  still-open, deployment-provenance task.
- cgroup v2, uid demotion, `/proc` marker scan — 18 environment-gated
  tests skip honestly on this host and are not counted as passing.

## What runtime security must NOT do

While scientific work proceeds, these are hard constraints:

- no weakening of the fail-closed transitions above to make a run
  "succeed"
- no treating the cooperative env marker as if it were isolation
- no quoting a synthetic confirmation record as a real one
- no promotion claim that rests on an unverified deployment path
