"""Evaluator-authority resolution (v4.0.3-r1 / Phase 1).

An "authorized independent evaluator" is an ``evaluation``-role key in
the campaign trust root — not a flag, not a process name, and not the
same key under a different label. This module is the narrow surface
the controller and the matrix analyzer use to establish that
authorization:

* ``load_trust_root`` builds the ``AuthorityRegistry`` + verifier pair
  from a TRUST_ROOT.json;
* ``evaluation_verifier`` restricts a verifier to keys that are
  *currently* authorized for the evaluation role — a signature that
  verifies under it is already role-authorized (revoked, expired and
  not-yet-valid keys never register);
* ``authorization_failure`` returns the precise reason an identity may
  not sign evaluation evidence, so denials are auditable rather than a
  bare boolean.

Custody is operational, not cryptographic: the registry proves a key
holds the evaluation role; it cannot prove the private key lives on a
different machine than the research agent's. Deployments must hold the
evaluation key under a separate OS identity (see
SEALED_EVALUATOR_CUSTODY.md). What this module guarantees is that when
custody IS separated, a self-issued receipt can never validate.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from egai.common.crypto import Ed25519Verifier

from .authority import AuthorityRegistry

EVALUATOR_ROLE = "evaluation"


def load_trust_root(path) -> tuple[AuthorityRegistry, Ed25519Verifier]:
    """TRUST_ROOT.json -> (registry, verifier covering every active
    identity). Both are returned so callers can check the role AND the
    signature rather than only key presence."""
    reg = AuthorityRegistry.load(Path(path))
    return reg, reg.verifier()


def authorization_failure(registry: AuthorityRegistry, key_id: str, *,
                          now: datetime | None = None) -> str:
    """'' iff ``key_id`` may sign evaluation evidence at ``now``.

    Distinguishes the denial reasons — unregistered, wrong role,
    revoked, expired or not-yet-valid — so an audit log records *why*
    a signature carried no evaluation authority.
    """
    rec = registry.record_for(key_id)
    if rec is None:
        return "unregistered signing identity"
    if rec.role != EVALUATOR_ROLE:
        return (f"key registered for role {rec.role!r}, not "
                f"{EVALUATOR_ROLE!r}")
    return rec.invalid_reason(now or datetime.now().astimezone())


def evaluation_verifier(registry: AuthorityRegistry, *,
                        now: datetime | None = None) -> Ed25519Verifier:
    """A verifier registering ONLY identities currently authorized for
    the evaluation role. A signature that verifies under it carries
    evaluation authority; any other signature cannot verify at all."""
    at = now or datetime.now().astimezone()
    v = Ed25519Verifier()
    for rec in getattr(registry, "_by_key", {}).values():
        if rec.role == EVALUATOR_ROLE and not rec.invalid_reason(at):
            v.register(rec.key_id, rec.public_bytes())
    return v


def is_authorized_evaluator(registry: AuthorityRegistry, key_id: str, *,
                            now: datetime | None = None) -> bool:
    return not authorization_failure(registry, key_id, now=now)
