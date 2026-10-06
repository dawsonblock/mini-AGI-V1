"""mini-AGI v4: controlled continual learning + RC10 coherent hybrid memory.

The v4 contract is intentionally conservative:
* canonical memory is authoritative and model-independent;
* neural caches are disposable compiled artifacts;
* the foundation model is frozen through the v1.0 research milestone;
* fast learning is reversible;
* permanent changes require qualification and promotion.
"""
from .baseline import BaselineManifest
from .generation import EffectiveModelGeneration, CacheCompatibility
from .memory import CanonicalMemoryStore, CanonicalMemoryRecord
from .rc10 import (TransitionOrientation, AffineSegmentSummary, compose_summaries,
                   summary_from_steps, RC10BlockCache, RecurrentVariant, BoundaryPayload, RiskFeatures, RiskRouter,
                   RepairDecision)
from .fast_weights import FastWeightSession
from .adapter_bank import SkillAdapterManifest, GradientFreeSkillRouter, SkillAdapterBank
from .replay import DriftClock, ReplayItem, AdaptiveReplayScheduler
from .qualification import (CandidateState, CandidateArtifact, QualificationResult,
                            QualificationPlane)
from .admission import MemoryAction, AdmissionSignals, AdmissionDecision, MemoryAdmissionController
from .cache_registry import CompiledCacheArtifact, NeuralCacheRegistry
from .frozen import FrozenBaseGuard, FrozenBaseViolation
from .runtime import ControlledLearningRuntime
from .assembler import RC10Assembler, AssemblyOp, PlannedOp, BlockAssemblyPlan
from .workspace import V4Workspace
from .pretrained_runtime import V4PretrainedRuntime, V4RuntimeConfig
from .gdn_reference import GDNInputs, affine_step, delta_step, scan as gdn_scan, compile_summary as compile_gdn_summary, zero_state_for
from .conv_boundary import ConvBoundaryState, capture_boundary, causal_depthwise_conv
from .pic import FAPICSegment, RelocatedFAPIC, relocate as relocate_fa_pic, append_segments as append_fa_pic
from .oracle import TensorDivergence, OracleComparison, tensor_divergence, compare as compare_oracle
from .qwen35_rc10 import CapturedGDNLayer, CompiledQwen35Layer, captured_steps, compile_layer as compile_qwen35_layer, compile_recurrent_variant as compile_qwen35_variant, slice_captured_layer
from .reference_executor import SegmentExecution, ReferenceExecutionResult, execute_segment as execute_rc10_segment, qualify as qualify_rc10_reference
from .rc10_compiler import CompilePolicy, compile_qwen35_block
from .risk_calibration import RiskTolerance, RiskObservation, RiskModelArtifact, fit_risk_model

from .hf_hybrid_capture import (
    HybridLayerKind, LayerDescriptor, HybridModelLayout, CapturedFullAttentionLayer,
    ForwardCapture, discover_hybrid_layout, fla_l2norm, capture_gdn_from_hidden,
    HFHybridTraceCollector, extract_full_attention_pic,
)
from .full_model_oracle import (
    AssemblyCaseKind, TokenBlock, AssemblyCase, ExactModelResult, ModelDivergence,
    QualificationRecord as FullModelQualificationRecord, assemble as assemble_token_blocks,
    standard_cases as standard_rc10_cases, ExactFullModelOracle, compare_model_results,
    FullModelQualificationHarness,
)
from .model_probe import ModelProbeReport, probe_model
from .trace_bundle import TraceBundleManifest, save_trace_bundle, load_trace_bundle

__all__ = [
    "BaselineManifest", "EffectiveModelGeneration", "CacheCompatibility",
    "CanonicalMemoryStore", "CanonicalMemoryRecord",
    "TransitionOrientation", "AffineSegmentSummary", "compose_summaries",
    "summary_from_steps", "RC10BlockCache", "RecurrentVariant", "BoundaryPayload", "RiskFeatures", "RiskRouter",
    "RepairDecision", "FastWeightSession", "SkillAdapterManifest",
    "GradientFreeSkillRouter", "SkillAdapterBank", "DriftClock", "ReplayItem",
    "AdaptiveReplayScheduler", "CandidateState", "CandidateArtifact",
    "QualificationResult", "QualificationPlane", "MemoryAction", "AdmissionSignals",
    "AdmissionDecision", "MemoryAdmissionController", "CompiledCacheArtifact",
    "NeuralCacheRegistry", "FrozenBaseGuard", "FrozenBaseViolation",
    "ControlledLearningRuntime", "RC10Assembler", "AssemblyOp", "PlannedOp",
    "BlockAssemblyPlan", "V4Workspace", "V4PretrainedRuntime", "V4RuntimeConfig",
    "GDNInputs", "affine_step", "delta_step", "gdn_scan", "compile_gdn_summary",
    "zero_state_for", "ConvBoundaryState", "capture_boundary", "causal_depthwise_conv",
    "FAPICSegment", "RelocatedFAPIC", "relocate_fa_pic", "append_fa_pic",
    "TensorDivergence", "OracleComparison", "tensor_divergence", "compare_oracle",
    "CapturedGDNLayer", "CompiledQwen35Layer", "captured_steps",
    "compile_qwen35_layer", "compile_qwen35_variant", "slice_captured_layer", "SegmentExecution",
    "ReferenceExecutionResult", "execute_rc10_segment", "qualify_rc10_reference",
    "CompilePolicy", "compile_qwen35_block", "RiskTolerance",
    "RiskObservation", "RiskModelArtifact", "fit_risk_model",
    "HybridLayerKind", "LayerDescriptor", "HybridModelLayout",
    "CapturedFullAttentionLayer", "ForwardCapture", "discover_hybrid_layout",
    "fla_l2norm", "capture_gdn_from_hidden", "HFHybridTraceCollector",
    "extract_full_attention_pic", "AssemblyCaseKind", "TokenBlock",
    "AssemblyCase", "ExactModelResult", "ModelDivergence",
    "FullModelQualificationRecord", "assemble_token_blocks",
    "standard_rc10_cases", "ExactFullModelOracle", "compare_model_results",
    "FullModelQualificationHarness", "ModelProbeReport", "probe_model",
    "TraceBundleManifest", "save_trace_bundle", "load_trace_bundle",
]

from .rc10_execution import (CanonicalRC10Block, ReplayPlaneResult, RC10ReplayBackend, ExecutionPath, BlockExecutionRecord, RC10ExecutionResult, RC10StatePlaneExecutor)
from .hf_boundary_replay import HFExactBoundaryReplayBackend
from .rc10_artifact import (RC10BlockArtifactManifest, save_rc10_block_artifact, load_rc10_block_artifact)

__all__ += ["CanonicalRC10Block", "ReplayPlaneResult", "RC10ReplayBackend", "ExecutionPath", "BlockExecutionRecord", "RC10ExecutionResult", "RC10StatePlaneExecutor", "HFExactBoundaryReplayBackend", "RC10BlockArtifactManifest", "save_rc10_block_artifact", "load_rc10_block_artifact"]

from .rc10_shadow import AttentionPlaneDivergence, RC10ShadowResult, RC10ShadowHarness
__all__ += ["AttentionPlaneDivergence", "RC10ShadowResult", "RC10ShadowHarness"]
