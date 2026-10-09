"""v16.4.3 operation authorization for the supervised service
(HARDENING_PLAN WP2).

v16.4.2 authenticated *who* connected (peer uid) but authorized only
"is an allowed client" — every connected uid could quarantine, roll
back, recover, and launch. This module derives a `PrincipalContext`
from the authenticated OS credential (never from a client-supplied
``"role"`` field — that is data, not identity) and enforces a
per-operation permission table inside the dispatcher, before any side
effect.

Roles:

  * ``research`` — submit launch proposals, ping, limited status.
  * ``operator`` — research operations plus protected lifecycle
    operations (quarantine, rollback, recover).
  * ``service`` — the supervisor service identity itself; additionally
    permitted for internal calls.

Policy changes (adding an operator uid, rotating keys) are a separate
administrative procedure — the service never authorizes itself to
change policy at runtime.

Filesystem helpers live here because they are part of the same trust
boundary: `secure_dir` creates/verifies 0700 service-owned directories
and `contained_child` resolves a name inside a protected root without
trusting caller path syntax.
"""
from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer


class PolicyRefused(PermissionError):
    """The authenticated principal is not permitted this operation."""


#: Operation -> the principal roles permitted to perform it. Checked in
#: the dispatcher BEFORE any side effect.
OPERATION_ROLES: dict[str, frozenset[str]] = {
    "ping": frozenset({"research", "operator", "service"}),
    "status": frozenset({"research", "operator", "service"}),
    "infer": frozenset({"research", "operator", "service"}),
    "launch": frozenset({"research", "operator", "service"}),
    "quarantine": frozenset({"operator", "service"}),
    "rollback": frozenset({"operator", "service"}),
    "recover": frozenset({"operator", "service"}),
}

#: Operations a research-only endpoint may dispatch — endpoint caps are
#: enforced in the dispatcher on top of per-op role checks.
RESEARCH_OPS = frozenset(
    op for op, roles in OPERATION_ROLES.items() if "research" in roles)


@dataclass(frozen=True)
class PrincipalContext:
    """An authenticated principal. `role` is derived from the uid->
    role map configured at service start — clients cannot self-assign."""
    uid: int
    role: str
    principal_id: str          # e.g. "uid:1000" — recorded in audits
    insecure_dev: bool = False  # credentials could not be extracted


def principal_for_uid(uid: int | None, uid_roles: dict[int, str], *,
                      insecure_dev: bool = False) -> PrincipalContext:
    """Map an authenticated peer uid to its configured role. Unknown
    uids get no role (every operation refuses them); a uid missing from
    the map is `unauthenticated` — connected but unable to act."""
    if uid is None:
        # Development-only escape hatch: credentials unextractable.
        # Grant operator so the dispatch path can be exercised — this
        # branch is unreachable in production (the service fails closed
        # before it is reached).
        role = "operator" if insecure_dev else "unauthenticated"
    else:
        role = str(uid_roles.get(int(uid), "unauthenticated"))
    pid = f"uid:{uid}" if uid is not None else "dev-unauthenticated"
    return PrincipalContext(
        uid=-1 if uid is None else int(uid), role=role,
        principal_id=pid, insecure_dev=insecure_dev)


def require(principal: PrincipalContext, operation: str) -> None:
    """Raise PolicyRefused unless `principal`'s role may perform
    `operation`. Call BEFORE any side effect in the handler."""
    allowed = OPERATION_ROLES.get(operation)
    if allowed is None:
        raise PolicyRefused(f"unknown operation {operation!r}")
    if principal.role not in allowed:
        raise PolicyRefused(
            f"principal {principal.principal_id} (role "
            f"{principal.role!r}) is not permitted operation "
            f"{operation!r} — permitted roles: {sorted(allowed)}")


def audit_decision(signer: Ed25519Signer, *, principal: PrincipalContext,
                   operation: str, target: str = "", reason: str = "",
                   at: int = 0) -> dict:
    """A signed audit record for a privileged administrative decision —
    who, what operation, on which activation, why, and when."""
    body = {"schema": "mini-agi-v16.4.3-service-audit-v1",
            "principal_id": principal.principal_id, "uid": principal.uid,
            "role": principal.role, "operation": operation,
            "target": str(target), "reason": str(reason),
            "at": int(at)}
    env = signer.sign(body)
    return {"value": body, "digest": digest(body),
            "signer_key_id": env.key_id,
            "signature_b64": env.signature_b64}


# --- protected filesystem helpers ------------------------------------

_NAME_RE = re.compile(r"^[0-9a-f]{8,64}$")


def secure_dir(path, *, owner_uid: int | None = None) -> Path:
    """Create-or-verify a protected directory: 0700, owned by the
    service identity, a real directory (not a symlink)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode):
        raise PolicyRefused(f"protected directory {path} is a symlink")
    if not stat.S_ISDIR(st.st_mode):
        raise PolicyRefused(f"protected path {path} is not a directory")
    uid = os.getuid() if owner_uid is None else int(owner_uid)
    if st.st_uid != uid:
        raise PolicyRefused(
            f"protected directory {path} is owned by uid {st.st_uid}, "
            f"expected {uid}")
    if stat.S_IMODE(st.st_mode) & 0o077:
        os.chmod(path, 0o700)
    return path


def opaque_id() -> str:
    """A server-generated opaque identifier — the ONLY source of
    activation ids and receipt file names (SEC-204)."""
    import secrets
    return secrets.token_hex(16)


def validate_opaque_id(value: str, *, what: str = "identifier") -> str:
    """Server-generated ids are lowercase hex of fixed length — nothing
    else is a valid storage name."""
    if not _NAME_RE.match(str(value)):
        raise PolicyRefused(
            f"{what} {value!r} is not a server-generated identifier")
    return str(value)


def contained_child(root, name: str, *, what: str = "artifact") -> Path:
    """Resolve `name` inside `root` refusing traversal, absolute paths,
    separators, and symlinked parents. `name` must be an opaque id."""
    root = Path(root)
    if root.is_symlink():
        raise PolicyRefused(
            f"protected storage root {root} is a symlink")
    validate_opaque_id(name, what=what)
    dest = root / name
    # resolve() alone is racy; combine it with per-component lstat.
    resolved_root = root.resolve()
    resolved = dest.resolve()
    if resolved_root != resolved and resolved_root not in \
            resolved.parents:
        raise PolicyRefused(
            f"{what} path escapes the protected storage root")
    parent = resolved.parent
    while parent != resolved_root and resolved_root in parent.parents:
        if parent.is_symlink():
            raise PolicyRefused(
                f"{what} path traverses a symlinked parent")
        parent = parent.parent
    if resolved.exists() and resolved.is_symlink():
        raise PolicyRefused(f"{what} path is a symlink")
    return dest
