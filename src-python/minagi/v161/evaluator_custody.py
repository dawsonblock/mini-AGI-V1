"""Evaluator key-custody checks (v4.0.3-r2 / Phase 3.3).

The authority registry proves WHICH key signed; it cannot prove WHERE
the key lives. Diagnostics 9 requires the research agent and the
evaluator run under genuinely separate credentials — separate OS
identities, storage permissions and protected keys — not merely
separate Python objects.

This module enforces the checkable half: a signing-key file the
research agent could read or write is not an evaluator key. The other
half (distinct OS users, sealed storage, HSM custody) is deployment
policy — declared in
``configs/capability_acquisition/evaluator_policy.yaml`` and verified
out-of-band, because no in-process check can attest another user's
permissions.

Checks performed:

* the key file exists and is a regular file;
* POSIX mode is owner-read/write only (``0o600`` or stricter — group
  and other must carry no bits);
* the file is owned by the CALLING process's uid — a key belonging to
  another user is not this evaluator's custody;
* the key does not live inside the research workspace (a key co-
  resident with run artifacts is reachable by the agent being
  evaluated).
"""
from __future__ import annotations

import os
import stat
from pathlib import Path


def check_signing_key_custody(key_path, *, workspace_root=None) -> dict:
    """Verify the evaluator's signing key is under evaluator custody.

    Returns ``{"custody_ok": bool, "failures": [...]}``. Every check
    that can be verified in-process fails loudly; deployment-level
    separation remains the operator's responsibility."""
    key_path = Path(key_path)
    failures: list[str] = []
    if not key_path.is_file():
        failures.append(f"signing key {key_path} does not exist")
        return {"custody_ok": False, "failures": failures}
    st = key_path.stat()
    if not stat.S_ISREG(st.st_mode):
        failures.append(f"signing key {key_path} is not a regular file")
    mode = stat.S_IMODE(st.st_mode)
    if mode & 0o077:
        failures.append(
            f"signing key mode {mode:o} grants group/other access — "
            "an evaluator private key must be owner-only (0o600)")
    try:
        if st.st_uid != os.getuid():
            failures.append(
                f"signing key owned by uid {st.st_uid}, not the "
                f"calling uid {os.getuid()} — the evaluator must own "
                "its key")
    except (AttributeError, OSError):
        pass   # non-POSIX platform — ownership check not applicable
    if workspace_root is not None:
        try:
            key_path.resolve().relative_to(
                Path(workspace_root).resolve())
            failures.append(
                f"signing key {key_path} lives inside the research "
                "workspace — co-resident keys are reachable by the "
                "agent being evaluated")
        except ValueError:
            pass
    return {"custody_ok": not failures, "failures": failures}


def assert_signing_key_custody(key_path, *, workspace_root=None):
    """Raise unless the evaluator's signing key passes custody checks —
    the load-time guard for evaluator tooling."""
    res = check_signing_key_custody(key_path,
                                    workspace_root=workspace_root)
    if not res["custody_ok"]:
        raise PermissionError(
            "evaluator signing-key custody check failed: "
            + "; ".join(res["failures"]))
    return res
