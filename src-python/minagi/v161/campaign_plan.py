from __future__ import annotations
from dataclasses import dataclass
from egai.common.canonical import digest, validate_digest


@dataclass(frozen=True)
class ColabCampaignPlanV161:
    campaign_id: str
    model_id: str
    model_revision: str
    dataset_partition_digest: str
    scorer_artifact_digest: str
    retention_artifact_digest: str
    security_artifact_digest: str
    seeds: tuple[int, ...]
    minimum_forward_transfer: float = 0.0
    minimum_retention: float = 0.95
    require_zero_security_regressions: bool = True
    schema: str = "mini-agi-v16.1-colab-campaign-plan-v1"

    def __post_init__(self):
        if not self.campaign_id or not self.model_id or not self.model_revision:
            raise ValueError("campaign_id/model_id/model_revision required")
        for x in (self.dataset_partition_digest, self.scorer_artifact_digest,
                  self.retention_artifact_digest, self.security_artifact_digest):
            validate_digest(x)
        if not self.seeds or any(int(x) < 0 for x in self.seeds):
            raise ValueError("non-negative preregistered seeds required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("duplicate campaign seeds")
        if not 0 <= self.minimum_retention <= 1:
            raise ValueError("minimum_retention must be in [0,1]")

    @property
    def digest(self) -> str:
        return digest(self)
