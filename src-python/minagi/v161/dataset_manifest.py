from __future__ import annotations
from dataclasses import dataclass
from egai.common.canonical import digest, validate_digest


@dataclass(frozen=True, order=True)
class DatasetMember:
    sample_id: str
    task_family_id: str
    content_digest: str
    source_id: str = ""
    generator_id: str = ""
    schema: str = "mini-agi-v16.1-dataset-member-v1"

    def __post_init__(self) -> None:
        if not self.sample_id or not self.task_family_id:
            raise ValueError("sample_id and task_family_id required")
        validate_digest(self.content_digest)


@dataclass(frozen=True)
class DatasetMembershipManifest:
    name: str
    members: tuple[DatasetMember, ...]
    schema: str = "mini-agi-v16.1-dataset-membership-v1"

    def __post_init__(self) -> None:
        if not self.name or not self.members:
            raise ValueError("manifest name and members required")
        members = tuple(sorted(self.members, key=lambda x: (x.sample_id, x.content_digest)))
        if len({x.sample_id for x in members}) != len(members):
            raise ValueError("duplicate sample_id in partition")
        object.__setattr__(self, "members", members)

    @property
    def digest(self) -> str:
        return digest(self)

    @property
    def sample_ids(self) -> frozenset[str]:
        return frozenset(x.sample_id for x in self.members)

    @property
    def family_ids(self) -> frozenset[str]:
        return frozenset(x.task_family_id for x in self.members)


@dataclass(frozen=True)
class DatasetPartitionSet:
    train: DatasetMembershipManifest
    validation: DatasetMembershipManifest
    hidden: DatasetMembershipManifest
    retention: DatasetMembershipManifest | None = None
    security: DatasetMembershipManifest | None = None
    require_family_disjoint_hidden: bool = True
    schema: str = "mini-agi-v16.1-dataset-partition-set-v1"

    def __post_init__(self) -> None:
        self.assert_disjoint()

    @property
    def digest(self) -> str:
        return digest(self)

    def assert_disjoint(self) -> None:
        named = [("train", self.train), ("validation", self.validation), ("hidden", self.hidden)]
        if self.retention is not None:
            named.append(("retention", self.retention))
        if self.security is not None:
            named.append(("security", self.security))
        # Hidden data must never share exact samples with any other partition.
        for name, part in named:
            if name == "hidden":
                continue
            overlap = part.sample_ids & self.hidden.sample_ids
            if overlap:
                raise PermissionError(f"hidden sample leakage from {name}: {sorted(overlap)[:5]}")
            if self.require_family_disjoint_hidden:
                fam = part.family_ids & self.hidden.family_ids
                if fam:
                    raise PermissionError(f"hidden task-family leakage from {name}: {sorted(fam)[:5]}")
        # V4-09: every evaluation partition must also be sample-disjoint
        # from every other non-hidden partition — a retention/security
        # row leaking into train (or a validation row used for tuning)
        # silently inflates the corresponding measurement.
        others = [(n, p) for n, p in named if n != "hidden"]
        for i, (n1, p1) in enumerate(others):
            for n2, p2 in others[i + 1:]:
                overlap = p1.sample_ids & p2.sample_ids
                if overlap:
                    raise PermissionError(
                        f"partition sample leakage {n1}/{n2}: "
                        f"{sorted(overlap)[:5]}")

    def proof(self) -> dict:
        self.assert_disjoint()
        return {
            "schema": "mini-agi-v16.1-disjointness-proof-v1",
            "partition_set_digest": self.digest,
            "train_digest": self.train.digest,
            "validation_digest": self.validation.digest,
            "hidden_digest": self.hidden.digest,
            "hidden_exact_overlap": 0,
            "hidden_family_overlap": 0 if self.require_family_disjoint_hidden else None,
            "verified": True,
        }


@dataclass(frozen=True)
class DatasetPartitionSetV2(DatasetPartitionSet):
    """v16.6 partition set — adds the evaluator-sealed FINAL HOLDOUT.

    The holdout partition is authority-controlled: its rows never reach
    the worker's training, model-selection, prompt-tuning, or feedback
    paths (the runner binds only its digest). It must be disjoint — by
    sample id AND family — from every other partition.

    This is a NEW schema (different dataclass shape + schema string), so
    V1 partition digests used by executed campaigns are byte-stable.
    """
    final_holdout: DatasetMembershipManifest | None = None
    schema: str = "mini-agi-v16.6-dataset-partition-set-v1"

    def assert_disjoint(self) -> None:
        super().assert_disjoint()
        if self.final_holdout is None:
            return
        for name, part in (("train", self.train),
                           ("validation", self.validation),
                           ("hidden", self.hidden),
                           ("retention", self.retention),
                           ("security", self.security)):
            if part is None:
                continue
            if part.sample_ids & self.final_holdout.sample_ids:
                raise PermissionError(
                    f"final holdout sample leakage into {name}")
            if part.family_ids & self.final_holdout.family_ids:
                raise PermissionError(
                    f"final holdout family leakage into {name}")

    def proof(self) -> dict:
        p = super().proof()
        p["schema"] = "mini-agi-v16.6-disjointness-proof-v1"
        p["final_holdout_digest"] = (self.final_holdout.digest
                                    if self.final_holdout else None)
        return p


@dataclass(frozen=True)
class DatasetPartitionSetV3(DatasetPartitionSetV2):
    """v4.0 partition set — adds the COMPOSITIONAL holdout (split E).

    Compositional rows combine operations learned from TRAINED families
    on unseen inputs, so they may share families with train/validation
    but must be sample-disjoint from every other partition and
    family-disjoint from hidden/final_holdout (the same independence
    guarantee hidden carries).

    NEW schema — V1/V2 partition digests used by executed campaigns stay
    byte-stable.
    """
    compositional: DatasetMembershipManifest | None = None
    schema: str = "mini-agi-v4.0-dataset-partition-set-v1"

    def assert_disjoint(self) -> None:
        super().assert_disjoint()
        if self.compositional is None:
            return
        for name, part in (("train", self.train),
                           ("validation", self.validation),
                           ("hidden", self.hidden),
                           ("retention", self.retention),
                           ("security", self.security),
                           ("final_holdout", self.final_holdout)):
            if part is None:
                continue
            if part.sample_ids & self.compositional.sample_ids:
                raise PermissionError(
                    f"compositional sample leakage from {name}")
        for name, part in (("hidden", self.hidden),
                           ("final_holdout", self.final_holdout)):
            if part is None:
                continue
            if part.family_ids & self.compositional.family_ids:
                raise PermissionError(
                    f"compositional family leakage from {name}")

    def proof(self) -> dict:
        p = super().proof()
        p["schema"] = "mini-agi-v4.0-disjointness-proof-v1"
        p["compositional_digest"] = (self.compositional.digest
                                    if self.compositional else None)
        return p


@dataclass(frozen=True)
class DatasetPartitionSetV4(DatasetPartitionSetV3):
    """v4.1 partition set — adds the WITHIN-FAMILY CONFIRMATION holdout.

    ``within_holdout`` carries TRAINED families on unseen inputs (like
    validation) but is reserved for independent confirmation: the
    development process may tune on ``validation`` yet must never use
    ``within_holdout`` for model or intervention selection. Its
    families are required to be a subset of the train families — a
    'within-family' holdout over untrained families measures transfer,
    not generalization (the campaign3b defect).

    Constraints: sample-disjoint from every partition; family-disjoint
    from hidden/compositional/final_holdout; family subset of train.

    ``rehearsal`` (optional) is a TRAIN-SIDE partition: verified-correct
    pairs of capabilities other than the acquisition target that a
    rehearsal arm may interleave during training. Its families must be
    disjoint from train, validation, within_holdout, hidden,
    compositional and final_holdout — rehearsing a measured family (or
    the target itself) would contaminate the measurement.

    NEW schema — V1/V2/V3 partition digests used by executed campaigns
    stay byte-stable.
    """
    within_holdout: DatasetMembershipManifest | None = None
    rehearsal: DatasetMembershipManifest | None = None
    schema: str = "mini-agi-v4.1-dataset-partition-set-v1"

    def assert_disjoint(self) -> None:
        super().assert_disjoint()
        if self.rehearsal is not None:
            for name, part in (
                    ("train", self.train),
                    ("validation", self.validation),
                    ("hidden", self.hidden),
                    ("retention", self.retention),
                    ("security", self.security),
                    ("compositional", self.compositional),
                    ("final_holdout", self.final_holdout),
                    ("within_holdout", self.within_holdout)):
                if part is None:
                    continue
                if part.sample_ids & self.rehearsal.sample_ids:
                    raise PermissionError(
                        f"rehearsal sample leakage into {name}")
            measured = [self.train, self.validation, self.hidden,
                        self.compositional, self.final_holdout,
                        self.within_holdout]
            fam_overlap = self.rehearsal.family_ids & frozenset().union(
                *(p.family_ids for p in measured if p is not None))
            if fam_overlap:
                raise PermissionError(
                    f"rehearsal family leakage into measured surfaces: "
                    f"{sorted(fam_overlap)[:5]} — rehearsing a measured "
                    "capability contaminates the measurement")
        if self.within_holdout is None:
            return
        for name, part in (("train", self.train),
                           ("validation", self.validation),
                           ("hidden", self.hidden),
                           ("retention", self.retention),
                           ("security", self.security),
                           ("compositional", self.compositional),
                           ("final_holdout", self.final_holdout)):
            if part is None:
                continue
            if part.sample_ids & self.within_holdout.sample_ids:
                raise PermissionError(
                    f"within_holdout sample leakage into {name}")
        for name, part in (("hidden", self.hidden),
                           ("compositional", self.compositional),
                           ("final_holdout", self.final_holdout)):
            if part is None:
                continue
            if part.family_ids & self.within_holdout.family_ids:
                raise PermissionError(
                    f"within_holdout family leakage into {name}")
        not_within = self.within_holdout.family_ids - self.train.family_ids
        if not_within:
            raise PermissionError(
                f"within_holdout carries untrained families "
                f"{sorted(not_within)[:5]} — that is transfer, not a "
                "within-family confirmation")

    def proof(self) -> dict:
        p = super().proof()
        p["schema"] = "mini-agi-v4.1-disjointness-proof-v1"
        p["within_holdout_digest"] = (self.within_holdout.digest
                                      if self.within_holdout else None)
        p["rehearsal_digest"] = (self.rehearsal.digest
                                 if self.rehearsal else None)
        return p
