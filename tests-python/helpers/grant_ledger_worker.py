"""Child-process entry point for the grant-ledger concurrency test
(v16.4.5 portability fix).

The child target must live in an importable module — a function
defined in the test module cannot be imported by multiprocessing
'spawn' children when the test directory is not on sys.path
(importlib mode). Spawn children reconstruct sys.path from the
parent's preparation data, so this module resolves there.
"""
from minagi.runtime.authority_store import (AuthorityStore,
                                            GrantConsumed)
from minagi.security.admission_grants import AdmissionGrantV1


def proc_reserve(db_path: str, grant_value: dict, aid: str,
                 at: int, queue):
    """One reservation attempt from a fresh process. Reports exactly
    one outcome string on the queue: 'ok', 'refused', or 'error:...'."""
    store = AuthorityStore(db_path)
    try:
        store.reserve_grant(AdmissionGrantV1.from_value(grant_value),
                            activation_id=aid, at=at)
        queue.put("ok")
    except GrantConsumed:
        queue.put("refused")
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        queue.put(f"error:{type(exc).__name__}:{exc}")
