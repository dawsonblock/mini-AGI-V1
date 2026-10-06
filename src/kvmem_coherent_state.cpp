#include "qw3/kvmem_coherent_state.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace qw3 {

bool KvMemCacheIdentity::operator==(const KvMemCacheIdentity& rhs) const {
    return base_model_digest == rhs.base_model_digest &&
           adapter_set_digest == rhs.adapter_set_digest &&
           tokenizer_digest == rhs.tokenizer_digest &&
           layer_layout_digest == rhs.layer_layout_digest &&
           position_scheme == rhs.position_scheme &&
           recurrence_impl == rhs.recurrence_impl;
}

KvMemCacheCompatibility compare_cache_identity(
    const KvMemCacheIdentity& cached,
    const KvMemCacheIdentity& requested) {
    if (cached == requested) return KvMemCacheCompatibility::ExactCompatible;

    // A different base model, tokenizer, topology, position convention or
    // recurrent implementation changes the interpretation/dimensions of state.
    if (cached.base_model_digest != requested.base_model_digest ||
        cached.tokenizer_digest != requested.tokenizer_digest ||
        cached.layer_layout_digest != requested.layer_layout_digest ||
        cached.position_scheme != requested.position_scheme ||
        cached.recurrence_impl != requested.recurrence_impl) {
        return KvMemCacheCompatibility::Invalid;
    }

    // Adapter deltas can change hidden activations, Q/K/V, gates and recurrence
    // operators. Rebuild from source tokens unless a separately qualified
    // cross-adapter compatibility mechanism exists.
    if (cached.adapter_set_digest != requested.adapter_set_digest) {
        return KvMemCacheCompatibility::ReplayRequired;
    }
    return KvMemCacheCompatibility::Invalid;
}

KvMemAffineSummary KvMemAffineSummary::identity(uint32_t n) {
    KvMemAffineSummary out;
    out.dim = n;
    out.T.assign(static_cast<size_t>(n) * n, 0.0);
    out.Z.assign(n, 0.0);
    for (uint32_t i = 0; i < n; ++i) out.T[static_cast<size_t>(i) * n + i] = 1.0;
    return out;
}

void KvMemAffineSummary::validate() const {
    if (dim == 0) throw std::invalid_argument("affine summary dim must be > 0");
    const size_t n = static_cast<size_t>(dim);
    if (T.size() != n * n) throw std::invalid_argument("affine T size mismatch");
    if (Z.size() != n) throw std::invalid_argument("affine Z size mismatch");
    for (double v : T) if (!std::isfinite(v)) throw std::invalid_argument("affine T is non-finite");
    for (double v : Z) if (!std::isfinite(v)) throw std::invalid_argument("affine Z is non-finite");
}

std::vector<double> KvMemAffineSummary::apply(const std::vector<double>& state) const {
    validate();
    if (state.size() != dim) throw std::invalid_argument("state dimension mismatch");
    std::vector<double> out(dim, 0.0);
    for (uint32_t r = 0; r < dim; ++r) {
        double acc = Z[r];
        for (uint32_t c = 0; c < dim; ++c) {
            acc += T[static_cast<size_t>(r) * dim + c] * state[c];
        }
        out[r] = acc;
    }
    return out;
}

KvMemAffineSummary KvMemAffineSummary::then(const KvMemAffineSummary& next) const {
    validate();
    next.validate();
    if (dim != next.dim) throw std::invalid_argument("affine dimensions differ");

    KvMemAffineSummary out;
    out.dim = dim;
    const size_t n = dim;
    out.T.assign(n * n, 0.0);
    out.Z.assign(n, 0.0);

    // A then B: T = T_B*T_A, Z = T_B*Z_A + Z_B.
    for (size_t r = 0; r < n; ++r) {
        for (size_t c = 0; c < n; ++c) {
            double acc = 0.0;
            for (size_t k = 0; k < n; ++k) {
                acc += next.T[r * n + k] * T[k * n + c];
            }
            out.T[r * n + c] = acc;
        }
        double z = next.Z[r];
        for (size_t k = 0; k < n; ++k) z += next.T[r * n + k] * Z[k];
        out.Z[r] = z;
    }
    return out;
}

static void validate_layer_shape(const std::vector<KvMemLayerAffineSummary>& layers) {
    if (layers.empty()) throw std::invalid_argument("recurrent layer summary cannot be empty");
    uint32_t prev = 0;
    bool first = true;
    for (const auto& layer : layers) {
        layer.affine.validate();
        if (!first && layer.layer <= prev) {
            throw std::invalid_argument("recurrent layer IDs must be strictly increasing and unique");
        }
        first = false;
        prev = layer.layer;
    }
}

KvMemHybridComposition compose_hybrid_blocks(
    const std::vector<KvMemHybridBlockSummary>& blocks,
    const KvMemCacheIdentity& expected_identity) {
    if (blocks.empty()) throw std::invalid_argument("cannot compose zero blocks");

    const auto& first = blocks.front();
    if (!(first.identity == expected_identity)) throw std::invalid_argument("cache identity mismatch");
    validate_layer_shape(first.layers);

    KvMemHybridComposition out;
    out.layers = first.layers;
    out.selected_block_ids.reserve(blocks.size());
    out.selected_block_ids.push_back(first.block_id);

    for (size_t bi = 1; bi < blocks.size(); ++bi) {
        const auto& block = blocks[bi];
        if (!(block.identity == expected_identity)) throw std::invalid_argument("cache identity mismatch");
        validate_layer_shape(block.layers);
        if (block.layers.size() != out.layers.size()) {
            throw std::invalid_argument("recurrent layer set mismatch");
        }
        for (size_t li = 0; li < out.layers.size(); ++li) {
            if (block.layers[li].layer != out.layers[li].layer) {
                throw std::invalid_argument("recurrent layer set mismatch");
            }
            out.layers[li].affine = out.layers[li].affine.then(block.layers[li].affine);
        }
        out.selected_block_ids.push_back(block.block_id);
    }
    return out;
}

bool coherence_acceptable(
    const KvMemCoherenceMetrics& m,
    const KvMemCoherenceThresholds& t) {
    const double vals[] = {
        m.state_rel_l2, m.hidden_rel_l2, m.logit_kl,
        m.routing_disagreement, m.transition_condition,
        m.retrieval_discontinuity,
    };
    for (double v : vals) if (!std::isfinite(v)) return false;
    return m.state_rel_l2 <= t.max_state_rel_l2 &&
           m.hidden_rel_l2 <= t.max_hidden_rel_l2 &&
           m.logit_kl <= t.max_logit_kl &&
           m.routing_disagreement <= t.max_routing_disagreement &&
           m.transition_condition <= t.max_transition_condition &&
           m.retrieval_discontinuity <= t.max_retrieval_discontinuity;
}

KvMemAdaptiveRepairPolicy::KvMemAdaptiveRepairPolicy()
    : seam_sizes_{8, 16, 32, 64, 128}, suffix_sizes_{256, 512, 1024, 2048} {}

KvMemAdaptiveRepairPolicy::KvMemAdaptiveRepairPolicy(
    std::vector<uint32_t> seam_sizes,
    std::vector<uint32_t> suffix_sizes,
    KvMemCoherenceThresholds thresholds)
    : seam_sizes_(std::move(seam_sizes)),
      suffix_sizes_(std::move(suffix_sizes)),
      thresholds_(thresholds) {
    if (seam_sizes_.empty()) throw std::invalid_argument("seam sizes cannot be empty");
    if (suffix_sizes_.empty()) throw std::invalid_argument("suffix sizes cannot be empty");
    if (!std::is_sorted(seam_sizes_.begin(), seam_sizes_.end()) ||
        !std::is_sorted(suffix_sizes_.begin(), suffix_sizes_.end())) {
        throw std::invalid_argument("repair sizes must be sorted ascending");
    }
}

KvMemReconstructionDecision KvMemAdaptiveRepairPolicy::decide(
    const std::vector<KvMemCoherenceMetrics>& seam_trials,
    const std::vector<KvMemCoherenceMetrics>& suffix_trials) const {
    const size_t seam_n = std::min(seam_trials.size(), seam_sizes_.size());
    for (size_t i = 0; i < seam_n; ++i) {
        if (coherence_acceptable(seam_trials[i], thresholds_)) {
            return {KvMemReconstructionAction::SeamReplay, seam_sizes_[i]};
        }
    }
    const size_t suffix_n = std::min(suffix_trials.size(), suffix_sizes_.size());
    for (size_t i = 0; i < suffix_n; ++i) {
        if (coherence_acceptable(suffix_trials[i], thresholds_)) {
            return {KvMemReconstructionAction::SuffixReplay, suffix_sizes_[i]};
        }
    }
    return {KvMemReconstructionAction::ExactSelectedReplay, 0};
}

} // namespace qw3
