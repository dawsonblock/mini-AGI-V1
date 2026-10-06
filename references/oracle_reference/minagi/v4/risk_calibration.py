from __future__ import annotations

"""Persistable exact-oracle calibration for the RC10 risk router."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence
import json

from .oracle import OracleComparison
from .rc10 import RiskFeatures, RiskRouter


@dataclass(frozen=True)
class RiskTolerance:
    max_state_relative_l2: float = 0.05
    max_state_angle_degrees: float = 3.0
    max_logit_kl: float = 0.02
    require_top1_agreement: bool = True

    def unacceptable(self, comparison: OracleComparison) -> bool:
        if comparison.state.relative_l2 > self.max_state_relative_l2:
            return True
        if comparison.state.angle_degrees > self.max_state_angle_degrees:
            return True
        if comparison.logit_kl is not None and comparison.logit_kl > self.max_logit_kl:
            return True
        if self.require_top1_agreement and comparison.top1_agree is False:
            return True
        return False


@dataclass(frozen=True)
class RiskObservation:
    features: RiskFeatures
    unacceptable: bool

    @classmethod
    def from_oracle(cls, features: RiskFeatures, comparison: OracleComparison,
                    tolerance: RiskTolerance = RiskTolerance()):
        return cls(features, tolerance.unacceptable(comparison))


@dataclass(frozen=True)
class RiskModelArtifact:
    weights: tuple[float, ...]
    thresholds: tuple[float, float, float]
    tolerance: RiskTolerance
    observation_count: int
    schema: str = "mini-agi-rc10-risk-v1"

    def save(self, path: str | Path) -> None:
        doc = asdict(self)
        Path(path).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path):
        doc = json.loads(Path(path).read_text())
        if doc.get("schema") != "mini-agi-rc10-risk-v1":
            raise ValueError("unsupported risk-model schema")
        return cls(
            tuple(float(x) for x in doc["weights"]),
            tuple(float(x) for x in doc["thresholds"]),
            RiskTolerance(**doc["tolerance"]),
            int(doc["observation_count"]),
            doc["schema"],
        )

    def router(self) -> RiskRouter:
        return RiskRouter(self.weights, self.thresholds)


def fit_risk_model(
    observations: Sequence[RiskObservation],
    *,
    tolerance: RiskTolerance = RiskTolerance(),
    thresholds: tuple[float, float, float] = (0.15, 0.35, 0.65),
    lr: float = 0.1,
    steps: int = 800,
) -> RiskModelArtifact:
    if len(observations) < 4:
        raise ValueError("at least four observations are required")
    labels = [1 if x.unacceptable else 0 for x in observations]
    if len(set(labels)) < 2:
        raise ValueError("calibration requires both acceptable and unacceptable examples")
    router = RiskRouter(thresholds=thresholds)
    router.fit([x.features for x in observations], labels, lr=lr, steps=steps)
    return RiskModelArtifact(tuple(float(x) for x in router.weights), thresholds,
                             tolerance, len(observations))
