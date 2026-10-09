# Threat model — v16.4.3 Hardened Runtime

Scope: the supervised model-serving runtime (`minagi.runtime`,
`minagi.security`, `minagi.v161` admission/launch). Scientific
claims, campaign evidence, and the learning controller are out of
scope — they have independent gates.

## Assets

- Qualified model/tokenizer/adapter bytes and their measured digests.
- Admission authority: role-separated Ed25519 keys, grants,
  revocation snapshots, promotion/qualification digests.
- Live serving state: which loaded instance may receive inference.
- The durable event log and grant ledger (integrity + ordering).
- Service availability (bounded I/O, worker pool).

## Trust boundaries

1. **Client → service.** Research/operator clients connect over
   AF_UNIX to `research.sock`/`operator.sock`. Authentication is the
   authenticated peer uid (Linux `SO_PEERCRED`, BSD `getpeereid`);
   where credentials cannot be extracted the connection is refused.
   The role comes from the uid→principal map, never from request
   fields. Filesystem permissions on the sockets are defense in
   depth, not the authorization check.
2. **Service → storage.** `state/authority.sqlite` is the single
   authority for grants, events, outcomes, audit, and the pointer.
   `snapshots/`, `receipts/`, `revocations/`, `quarantine/` are
   service-owned (`secure_dir`: ownership+mode+no-symlinks). All
   client strings that could reach a path are refused — the service
   generates opaque ids and chooses every destination itself.
3. **Service → keys.** Signing keys live outside the repository and
   outside the service's artifact root; the service holds only the
   roles it needs (`runtime`, `admission`, `revocation`,
   `migration`). No operation exports key material.
4. **Process boundary.** No research path bypasses the supervisor by
   importing a class: grant reservation, staging, commit, and
   routing all require the authority store under the service-owned
   root, and a `MeasuredSnapshot` carries no authorization.

## Adversaries

- **Malicious research client** — submits crafted requests, hostile
  `campaign_id`/`seed`, oversized or truncated payloads, replays
  grants, races concurrent requests. Defeated by bounded reads,
  deadlines, per-op authorization, opaque ids, atomic reservation.
- **Malicious operator** — can run lifecycle ops but cannot mint
  grants, move paths, sign as `runtime`/`admission`, or promote
  itself; every decision is audit-recorded.
- **Same-uid sibling process** — inherits the uid→principal mapping
  of the client it impersonates; cannot exceed that principal's
  role. The supervisor's own uid is required for lifecycle ops.
- **Filesystem attacker without service uid** — cannot write the
  service-owned roots (mode + ownership verified at open); a
  tampered store fails `PRAGMA integrity_check` / digest checks and
  fails closed.
- **Filesystem attacker with storage write but no keys** — can
  corrupt availability (detected, fails closed) but cannot forge
  grant reservations, event signatures, or checkpoints: every
  authority-bearing event must verify under the required role's key.
- **Rollback/replay attacker** — presents superseded revocation
  epochs, stale grants, or historical records as live state;
  recovery requires fresh admission + health probes before serving.

## Explicit non-goals (accepted risks)

- A **service-uid compromise** is total — mitigation is deployment
  (dedicated unprivileged account, systemd/launchd restrictions,
  namespace isolation), documented not coded.
- **Memory-only enforcement** of snapshot immutability: a same-uid
  adversary can rewrite staged bytes between check and load;
  mitigated by re-verification at `stage()`, not eliminated.
- **Tail truncation** of the event log by someone with both storage
  write and no detection channel: mitigated by the signed checkpoint
  anchor (`anchor_checkpoint`) and store `integrity_check`, but a
  fully offline anchor outside the host remains stronger.
- **macOS peer credentials**: CPython exposes no `SO_PEERCRED`/
  `getpeereid` — the service fails closed; secure cross-uid service
  on macOS awaits a platform transport (OPS register).
- **GPU/driver/firmware identity**: recorded as environment evidence
  in the backend manifest; exact immutability is impractical, so
  compatibility gates apply rather than digest equality.

## Fail-closed invariants

- No grant ⇒ no staging; no commit ⇒ no routing; no live verified
  handle ⇒ `UNAVAILABLE`, never phantom `SERVING`.
- Unavailable/corrupt authority store ⇒ every authorization raises.
- Unknown uid, unknown role, or revoked principal ⇒ refused and
  audited.
- Any journal inconsistency (digest, signature, grammar, anchor) ⇒
  recovery refuses; human inspection required.
