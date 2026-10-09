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
        if self.train.sample_ids & self.validation.sample_ids:
            raise PermissionError("train/validation sample overlap")

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
