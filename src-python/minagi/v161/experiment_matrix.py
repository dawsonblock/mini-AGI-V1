"""Experiment completeness matrix (V4-15).

A multi-seed experiment is only a comparison when the FULL preregistered
matrix executed: every declared arm, every seed, every required
evaluation cell, under a single model revision, scorer version and
task-family definition. Before this module the qualifier caught missing
cells per-campaign, but there was no standalone, tested declaration of
what "complete" means — so partially-executed matrices could be
reported as finished comparisons.

`MatrixSpec` declares the requirement; `audit_matrix` evaluates observed
cells against it and returns an explicit status:

    COMPLETE      — every required cell present and verified
    INCOMPLETE    — required cells absent (not a failure of evidence,
                    a failure of coverage: qualification may not claim
                    the full matrix)
    INCONSISTENT  — cells exist but disagree on a binding identity
                    (model digest, tokenizer digest, evaluator digest,
                    environment digest, task-family digest)

INCOMPLETE and INCONSISTENT are terminal for qualification claims: the
experiment may continue executing, but nothing downstream may treat the
matrix as a finished comparison.

V4-30 — a cell that exists but cannot name its binding identities is
not verified evidence. ``MatrixSpec.required_identity_fields`` names
the identity fields every present cell must carry non-empty (default:
all five). A present cell with an absent required field is recorded as
an identity violation and the audit cannot return COMPLETE — empty
digests never yield a qualified comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

COMPLETE = "COMPLETE"
INCOMPLETE = "INCOMPLETE"
INCONSISTENT = "INCONSISTENT"


@dataclass(frozen=True)
class CellIdentity:
    """The binding identities every executed cell must agree on."""
    model_digest: str = ""
    tokenizer_digest: str = ""
    evaluator_digest: str = ""
    environment_digest: str = ""
    dataset_digest: str = ""


@dataclass(frozen=True)
class MatrixSpec:
    """The preregistered shape of a multi-seed experiment."""
    campaign_id: str
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    required_cells: tuple[str, ...] = ()
    #: extra required cells per arm, e.g. {"L1": ("L1_delayed",),
    #: "L6": ("L6_delayed",)} — evaluated only for that arm.
    per_arm_cells: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: identity values every cell must share; empty string = unbound
    #: (checked for internal consistency only, not against a value).
    expected_identity: CellIdentity = CellIdentity()
    #: identity fields every PRESENT cell must carry non-empty. An
    #: absent required digest is an identity violation — the cell's
    #: evidence cannot be bound, so the matrix is not a verified
    #: comparison. Defaults to all five fields (strict by default);
    #: narrow it only for deliberately partial audits.
    required_identity_fields: tuple[str, ...] = (
        "model_digest", "tokenizer_digest", "evaluator_digest",
        "environment_digest", "dataset_digest")


def required_cells(spec: MatrixSpec, arm: str) -> tuple[str, ...]:
    extra = tuple(spec.per_arm_cells.get(arm, ()))
    return (arm,) + tuple(spec.required_cells) + extra


def audit_matrix(
        spec: MatrixSpec,
        cells: Mapping[tuple[int, str], object]) -> dict:
    """Audit observed cells against the spec.

    `cells` maps (seed, cell_name) -> an object exposing the identity
    fields named on CellIdentity via attributes or a mapping with the
    same keys (a verified receipt or its plain dict). Missing keys are
    treated as unbound.

    Returns a report dict: status, per-seed present/missing, identity
    violations, and the complete deficit list."""
    missing: list[dict] = []
    present: dict[int, list[str]] = {s: [] for s in spec.seeds}
    observed_identities: list[tuple[int, str, CellIdentity]] = []

    def _id_of(cell) -> CellIdentity:
        def g(name):
            if isinstance(cell, Mapping):
                return str(cell.get(name, ""))
            return str(getattr(cell, name, ""))
        # EvidenceReceiptV3 names its bound dataset identity
        # ``evaluation_dataset_digest`` — the same identity under a
        # receipt-specific field name; accept it as the dataset digest.
        dataset = g("dataset_digest") or g("evaluation_dataset_digest")
        return CellIdentity(g("model_digest"), g("tokenizer_digest"),
                            g("evaluator_digest"), g("environment_digest"),
                            dataset)

    for seed in spec.seeds:
        for arm in spec.arms:
            for cell_name in required_cells(spec, arm):
                cell = cells.get((seed, cell_name))
                if cell is None:
                    missing.append({"seed": seed, "cell": cell_name})
                else:
                    present[seed].append(cell_name)
                    observed_identities.append(
                        (seed, cell_name, _id_of(cell)))

    violations: list[dict] = []
    # V4-30: a present cell with an absent required identity field is a
    # violation — its evidence cannot be bound to the experiment.
    required = set(spec.required_identity_fields) | {
        f for f in CellIdentity.__dataclass_fields__
        if getattr(spec.expected_identity, f)}
    for fname in sorted(required):
        if fname not in CellIdentity.__dataclass_fields__:
            raise ValueError(
                f"unknown required identity field: {fname}")
        for seed, cell_name, cid in observed_identities:
            if not getattr(cid, fname):
                violations.append({"field": fname, "seed": seed,
                                   "cell": cell_name, "absent": True})

    for fname in CellIdentity.__dataclass_fields__:
        vals = {getattr(c, fname) for _, _, c in observed_identities
                if getattr(c, fname)}
        bound = getattr(spec.expected_identity, fname)
        if bound:
            violations.extend(
                {"field": fname, "expected": bound, "found": v}
                for v in vals if v != bound)
        if len(vals) > 1:
            violations.append({"field": fname,
                               "conflict": sorted(vals)})

    if violations:
        status = INCONSISTENT
    elif missing:
        status = INCOMPLETE
    else:
        status = COMPLETE

    return {
        "schema": "mini-agi-v4.0-experiment-matrix-audit-v1",
        "campaign_id": spec.campaign_id,
        "status": status,
        "seeds_required": list(spec.seeds),
        "arms_required": list(spec.arms),
        "n_cells_required": sum(
            len(required_cells(spec, a)) * len(spec.seeds)
            for a in spec.arms),
        "n_cells_present": sum(len(v) for v in present.values()),
        "missing": missing,
        "identity_violations": violations,
        "qualified_comparison": status == COMPLETE,
    }
