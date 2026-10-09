"""v16.4.2 activation state machine (UPGRADE_PLAN §3.5).

Production activation is a transaction over filesystem state, model
processes, and traffic routing. The states make the transaction
explicit so a failure at any point has a defined outcome:

    REQUESTED → AUTHORIZED → STAGED → PREPARED → READY → COMMITTED
              → ACTIVE

Terminal failure states: ABORTED (never served / cleaned up) and
QUARANTINED (was or might have been live, then withdrawn). There is no
transition out of a terminal state — recovery acts on the journal, not
on resuming dead activations.
"""
from __future__ import annotations

from enum import Enum


class ActivationState(str, Enum):
    REQUESTED = "REQUESTED"
    AUTHORIZED = "AUTHORIZED"
    STAGED = "STAGED"
    PREPARED = "PREPARED"      # loaded but unavailable to requests
    READY = "READY"            # health + inference probes passed
    COMMITTED = "COMMITTED"    # durable activation intent recorded
    ACTIVE = "ACTIVE"          # traffic routed
    ABORTED = "ABORTED"        # terminal: never served
    QUARANTINED = "QUARANTINED"  # terminal: withdrawn after commit

    @property
    def terminal(self) -> bool:
        return self in _TERMINAL


_TERMINAL = frozenset(
    {ActivationState.ABORTED, ActivationState.QUARANTINED})

# Legal forward transitions. Anything else is refused — the lifecycle
# is a protocol, not a suggestion.
TRANSITIONS: dict[ActivationState, frozenset[ActivationState]] = {
    ActivationState.REQUESTED: frozenset(
        {ActivationState.AUTHORIZED, ActivationState.ABORTED}),
    ActivationState.AUTHORIZED: frozenset(
        {ActivationState.STAGED, ActivationState.ABORTED}),
    ActivationState.STAGED: frozenset(
        {ActivationState.PREPARED, ActivationState.ABORTED}),
    ActivationState.PREPARED: frozenset(
        {ActivationState.READY, ActivationState.ABORTED}),
    ActivationState.READY: frozenset(
        {ActivationState.COMMITTED, ActivationState.ABORTED}),
    ActivationState.COMMITTED: frozenset(
        {ActivationState.ACTIVE, ActivationState.ABORTED,
         ActivationState.QUARANTINED}),
    ActivationState.ACTIVE: frozenset(
        {ActivationState.QUARANTINED, ActivationState.ABORTED}),
    ActivationState.ABORTED: frozenset(),
    ActivationState.QUARANTINED: frozenset(),
}


class IllegalTransition(RuntimeError):
    """A state transition that is not part of the activation protocol."""


def check_transition(src: ActivationState, dst: ActivationState) -> None:
    """Raise IllegalTransition unless src -> dst is a protocol move."""
    if dst not in TRANSITIONS[src]:
        raise IllegalTransition(
            f"activation may not move {src.value} -> {dst.value} "
            f"(allowed: {sorted(s.value for s in TRANSITIONS[src])})")
