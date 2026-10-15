# Sealed evaluator custody

Status: **implemented, fail-closed, honestly bounded.** This records
what the custody model actually enforces — and what it cannot enforce
on this host.

## The trust boundary

A sealed confirmation is only independent if the evaluator and the
research side are held by **different operators**. Three layers exist:

| layer | enforcement | honest status |
|---|---|---|
| cooperative guard | `MINAGI_RESEARCH_WORKER` env marker | **cooperative, not isolation** — the worker controls its own env |
| filesystem custody | sealed set must be `0600`, owned by the *evaluating* uid, and that uid must differ from the research storage owner | enforceable, now checked |
| OS isolation | NsJail / systemd cgroup confinement, distinct uid | Linux-only; not exercised on this macOS host |

## What changed (RD-19)

- `sealed_custody_report()` previously accepted "sealed file is 0600
  and owned by euid" as independent isolation — true even when the
  sealed store and research store share one uid, i.e. exactly when
  custody is not independent. It now checks **distinct storage-owner /
  evaluator-owner identity**, and reports which controls were actually
  enforced in the signed record.
- `run_confirmation()` gained explicit `production` vs `synthetic`
  modes. **Production fails closed**: missing artifact verification or
  missing independent custody → `NOT_QUALIFIED`, never a soft pass.
  **Synthetic** records are labelled `production_valid: false` and can
  never be quoted as real confirmation.

## What this host can and cannot claim

- **Can claim:** the custody *logic* is fail-closed and tested
  (`test_sealed_custody_enforcement.py`, `test_confirmation_fail_closed.py`,
  `test_confirmation_access_denial.py` — 34 tests).
- **Cannot claim:** a real sealed confirmation has run. Same-uid host,
  no independent evaluator operator, synthetic fixtures only. RD-1
  stays **NOT RUN** — correctly.
- **Cannot claim (yet):** that production deployments invoke this
  entrypoint. `minagi.runtime.service` is declared with `NOT VERIFIED`
  status in the control-plane manifest; nothing here proves production
  runs it rather than the `kvcontinual` gateway or the native binary.

## The bar for a real confirmation

1. Sealed set owned by a uid distinct from the research tree's owner.
2. Evaluator runs under that uid, on OS-level isolation (not just the
   env marker).
3. Candidate artifacts verified against the research receipt BEFORE
   execution — digest match, not name match.
4. The confirmation record is signed by the evaluator's key, not the
   research key.

Until all four hold, every confirmation record in this tree is
synthetic and labelled so.
