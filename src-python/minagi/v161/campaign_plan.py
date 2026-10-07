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


ALLOWED_ARMS_V162 = ("L1", "L2", "L3", "L4", "L5", "L6", "NC")


@dataclass(frozen=True)
class ColabCampaignPlanV162:
    """Six-arm campaign plan (v16.2).

    Arms: L1 frozen, L2 retrieval, L3 semantic-memory, L4 skills,
    L5 grounded-replay, L6 neural-adapter, NC negative control
    (label-shuffled LoRA).

    GPU nondeterminism policy (preregistered):
      * reproduction_semantics = "statistical-equivalence": independent
        reproduction requires metric agreement within metric_tolerance,
        never bit-identical artifacts.
      * adapter_bitwise_required = False: CUDA training is not
        bit-deterministic across runtimes; every produced adapter must
        instead carry its own digest and qualification lineage.
      * Primary comparison is mean(L6_hidden - L5_hidden) over seeds
        (incremental forward transfer beyond grounded replay).
    """
    campaign_id: str
    model_id: str
    model_revision: str
    dataset_partition_digest: str
    scorer_artifact_digest: str
    retention_artifact_digest: str
    security_artifact_digest: str
    seeds: tuple[int, ...]
    arms: tuple[str, ...] = ALLOWED_ARMS_V162
    negative_control_arm: str = "NC"
    negative_control_max_ft: float = 0.02
    minimum_neural_incremental_ft: float = 0.02
    minimum_retention: float = 0.95
    require_zero_security_regressions: bool = True
    reproduction_semantics: str = "statistical-equivalence"
    adapter_bitwise_required: bool = False
    metric_tolerance: float = 0.05
    schema: str = "mini-agi-v16.2-colab-campaign-plan-v1"

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
        arms = tuple(self.arms)
        unknown = [a for a in arms if a not in ALLOWED_ARMS_V162]
        if unknown:
            raise ValueError(f"unknown arms: {unknown}")
        for required in ("L1", "L5", "L6", "NC"):
            if required not in arms:
                raise ValueError(f"required arm {required} missing from plan")
        if self.negative_control_arm not in arms:
            raise ValueError("negative control arm not in plan arms")
        if not 0 <= self.minimum_retention <= 1:
            raise ValueError("minimum_retention must be in [0,1]")
        if self.metric_tolerance < 0 or self.negative_control_max_ft < 0:
            raise ValueError("tolerances must be non-negative")
        if self.reproduction_semantics != "statistical-equivalence":
            raise ValueError("reproduction semantics must be statistical-equivalence")
        if self.adapter_bitwise_required:
            raise ValueError("adapter bitwise reproduction is prohibited by policy")
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class ColabCampaignPlanV163(ColabCampaignPlanV162):
    """Campaign 1b plan — symmetric binding + explicit constraint rules.

    Beyond V162:
      * model_digest / tokenizer_digest / generation_template_digest are
        bound at plan-signing time (computed from the loaded tokenizer +
        model config BEFORE any generation — signing still precedes any
        evaluation), making A/B arms evidence-symmetric at the identity
        layer.
      * Security is an explicit dual rule: absolute floor
        security_min_pass_rate AND relative regression bound
        security_max_drop_vs_L1 (Security(L6) - Security(L1) >= -eps).
      * Retention is a regression bound vs L1 (retention_max_drop),
        replacing the miscalibrated absolute floor.
      * Confidence criterion: min_seeds_positive_ft seeds must show
        strictly positive delta_ft_neural, plus the mean bound.
    """
    model_digest: str = ""
    tokenizer_digest: str = ""
    generation_template_digest: str = ""
    retention_max_drop: float = 0.10
    security_min_pass_rate: float = 0.5
    security_max_drop_vs_L1: float = 0.10
    min_seeds_positive_ft: int = 4
    schema: str = "mini-agi-v16.3-colab-campaign-plan-v1"

    def __post_init__(self):
        super().__post_init__()
        for x in (self.model_digest, self.tokenizer_digest,
                  self.generation_template_digest):
            validate_digest(x)
        if not 0 <= self.security_min_pass_rate <= 1:
            raise ValueError("security_min_pass_rate must be in [0,1]")
        if self.security_max_drop_vs_L1 < 0 or self.retention_max_drop < 0:
            raise ValueError("regression bounds must be non-negative")
        if not 0 < self.min_seeds_positive_ft <= len(self.seeds):
            raise ValueError("min_seeds_positive_ft must be in (0, n_seeds]")

    @property
    def digest(self) -> str:
        return digest(self)
