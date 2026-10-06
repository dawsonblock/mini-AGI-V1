from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Iterable
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
