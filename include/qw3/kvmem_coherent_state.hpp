#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace qw3 {

// Execution-cache identity. Recurrent/attention execution state is valid only
// for the exact model configuration that produced it. Persistent semantic
// memory is intentionally outside this identity domain.
struct KvMemCacheIdentity {
    std::string base_model_digest;
    std::string adapter_set_digest;
    std::string tokenizer_digest;
    std::string layer_layout_digest;
    std::string position_scheme;
    std::string recurrence_impl;

    bool operator==(const KvMemCacheIdentity& rhs) const;
};

enum class KvMemCacheCompatibility : uint8_t {
    ExactCompatible = 0,
    ReplayRequired = 1,
    Invalid = 2,
};

// Fail-closed compatibility. Adapter changes require replay; structural
// recurrence/layout/position changes invalidate cached execution state.
KvMemCacheCompatibility compare_cache_identity(
    const KvMemCacheIdentity& cached,
    const KvMemCacheIdentity& requested);

// Dense reference representation of a segment-level affine recurrence:
// S_out = T * S_in + Z. This is deliberately a host/reference primitive used
// to validate ordering and composition semantics. Production Gated DeltaNet
// should use the model-specific structured representation rather than storing
// a naive dense T for every block.
struct KvMemAffineSummary {
    uint32_t dim = 0;
    std::vector<double> T; // row-major dim x dim
    std::vector<double> Z; // vector-state reference, dim

    static KvMemAffineSummary identity(uint32_t dim);
    void validate() const;
    std::vector<double> apply(const std::vector<double>& state) const;
    KvMemAffineSummary then(const KvMemAffineSummary& next) const;
};

struct KvMemLayerAffineSummary {
    uint32_t layer = 0;
    KvMemAffineSummary affine;
};

struct KvMemHybridBlockSummary {
    uint32_t block_id = 0;
    KvMemCacheIdentity identity;
    std::vector<KvMemLayerAffineSummary> layers; // strictly increasing layer IDs
};

struct KvMemHybridComposition {
    std::vector<uint32_t> selected_block_ids;
    std::vector<KvMemLayerAffineSummary> layers;
};

// Compose A,D,F,... in exactly that order. Every block must have the exact
// requested cache identity and exact same recurrent layer set. Any mismatch
// fails closed; missing recurrent layers are never silently intersected away.
KvMemHybridComposition compose_hybrid_blocks(
    const std::vector<KvMemHybridBlockSummary>& blocks,
    const KvMemCacheIdentity& expected_identity);

struct KvMemCoherenceMetrics {
    double state_rel_l2 = 0.0;
    double hidden_rel_l2 = 0.0;
    double logit_kl = 0.0;
    double routing_disagreement = 0.0;
    double transition_condition = 1.0;
    double retrieval_discontinuity = 0.0;
};

struct KvMemCoherenceThresholds {
    double max_state_rel_l2 = 0.02;
    double max_hidden_rel_l2 = 0.02;
    double max_logit_kl = 0.02;
    double max_routing_disagreement = 0.02;
    double max_transition_condition = 1.0e5;
    double max_retrieval_discontinuity = 0.5;
};

bool coherence_acceptable(
    const KvMemCoherenceMetrics& metrics,
    const KvMemCoherenceThresholds& thresholds = {});

enum class KvMemReconstructionAction : uint8_t {
    Compose = 0,
    SeamReplay = 1,
    SuffixReplay = 2,
    ExactSelectedReplay = 3,
};

struct KvMemReconstructionDecision {
    KvMemReconstructionAction action = KvMemReconstructionAction::ExactSelectedReplay;
    uint32_t replay_tokens = 0;
};

// Runtime policy only. Actual seam/suffix replay is performed by the native
// Qwen executor; existing selected-replay remains the exact oracle/fallback.
class KvMemAdaptiveRepairPolicy {
public:
    KvMemAdaptiveRepairPolicy();
    KvMemAdaptiveRepairPolicy(
        std::vector<uint32_t> seam_sizes,
        std::vector<uint32_t> suffix_sizes,
        KvMemCoherenceThresholds thresholds = {});

    KvMemReconstructionDecision decide(
        const std::vector<KvMemCoherenceMetrics>& seam_trials,
        const std::vector<KvMemCoherenceMetrics>& suffix_trials) const;

    const std::vector<uint32_t>& seam_sizes() const { return seam_sizes_; }
    const std::vector<uint32_t>& suffix_sizes() const { return suffix_sizes_; }

private:
    std::vector<uint32_t> seam_sizes_;
    std::vector<uint32_t> suffix_sizes_;
    KvMemCoherenceThresholds thresholds_;
};

} // namespace qw3
