#include "qw3/gdn_summary_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace qw3 {
namespace {

bool same_block(const GdnBlockAffineSummary& a,
                const GdnBlockAffineSummary& b) {
    return a.version == b.version &&
           a.layer_index == b.layer_index &&
           a.position_start == b.position_start &&
           a.token_count == b.token_count &&
           a.num_v_heads == b.num_v_heads &&
           a.head_dim == b.head_dim &&
           a.model_id == b.model_id &&
           a.T == b.T && a.Z == b.Z &&
           a.conv_kernel_size == b.conv_kernel_size &&
           a.conv_dim == b.conv_dim &&
           a.conv_tail_rows == b.conv_tail_rows &&
           a.conv_tail == b.conv_tail;
}

size_t state_elems(uint32_t heads, uint32_t dim) {
    if (heads == 0 || dim == 0) {
        throw std::invalid_argument("GDN runtime dimensions must be non-zero");
    }
    const uint64_t n = static_cast<uint64_t>(heads) * dim * dim;
    if (n > std::numeric_limits<size_t>::max()) {
        throw std::invalid_argument("GDN runtime state size overflow");
    }
    return static_cast<size_t>(n);
}

} // namespace

void gdn_apply_summary_to_state_column_major(
    const GdnBlockAffineSummary& summary,
    std::vector<float>& state_column_major) {
    summary.validate();
    const uint32_t H = summary.num_v_heads;
    const uint32_t D = summary.head_dim;
    const size_t per = static_cast<size_t>(D) * D;
    if (state_column_major.size() != state_elems(H, D)) {
        throw std::invalid_argument("GDN runtime state shape mismatch");
    }

    std::vector<float> out(state_column_major.size(), 0.0f);
    for (uint32_t h = 0; h < H; ++h) {
        const size_t hb = static_cast<size_t>(h) * per;
        for (uint32_t col = 0; col < D; ++col) {
            for (uint32_t row = 0; row < D; ++row) {
                double acc = static_cast<double>(
                    summary.Z[hb + static_cast<size_t>(row) * D + col]);
                for (uint32_t k = 0; k < D; ++k) {
                    const double t = static_cast<double>(
                        summary.T[hb + static_cast<size_t>(row) * D + k]);
                    // Native recurrent state is column-major within each head.
                    const double s = static_cast<double>(
                        state_column_major[hb + static_cast<size_t>(col) * D + k]);
                    acc += t * s;
                }
                out[hb + static_cast<size_t>(col) * D + row] =
                    static_cast<float>(acc);
            }
        }
    }
    state_column_major.swap(out);
}

GdnSummaryStore GdnSummaryStore::from_summaries(
    const std::vector<GdnBlockAffineSummary>& summaries) {
    if (summaries.empty()) {
        throw std::invalid_argument("GDN summary store cannot be empty");
    }
    GdnSummaryStore out;
    for (const auto& s : summaries) out.add(s);
    return out;
}

GdnSummaryStore GdnSummaryStore::load_directory(
    const std::string& directory) {
    if (directory.empty()) {
        throw std::invalid_argument("GDN summary directory is empty");
    }
    const std::filesystem::path root(directory);
    if (!std::filesystem::exists(root) || !std::filesystem::is_directory(root)) {
        throw std::runtime_error("GDN summary directory does not exist: " + directory);
    }

    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".qgds") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::runtime_error("GDN summary directory contains no .qgds files: " + directory);
    }

    GdnSummaryStore out;
    for (const auto& path : files) {
        for (const auto& s : read_gdn_summary_archive(path.string())) {
            out.add(s);
        }
    }
    if (out.empty()) {
        throw std::runtime_error("GDN summary directory produced an empty index: " + directory);
    }
    return out;
}

void GdnSummaryStore::add(const GdnBlockAffineSummary& summary) {
    summary.validate();
    if (summary.version == 1) contains_legacy_v1_ = true;
    if (!summary.model_id.empty()) {
        if (model_id_.empty()) model_id_ = summary.model_id;
        else if (model_id_ != summary.model_id) {
            throw std::runtime_error(
                "GDN summary store mixes different model identities");
        }
    }

    auto& vec = by_layer_[summary.layer_index][summary.position_start];
    for (const auto& existing : vec) {
        if (existing.token_count != summary.token_count) continue;
        if (!same_block(existing, summary)) {
            throw std::runtime_error(
                "conflicting duplicate GDN summary block at layer " +
                std::to_string(summary.layer_index) + " position " +
                std::to_string(summary.position_start));
        }
        return; // exact duplicate across capture/archive files
    }
    vec.push_back(summary);
    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
        return a.token_count > b.token_count;
    });
    ++block_count_;
}

void GdnSummaryStore::require_model(const std::string& expected_model_id,
                                    bool allow_legacy_v1) const {
    if (expected_model_id.empty()) {
        throw std::invalid_argument("running GDN model identity is empty");
    }
    if (contains_legacy_v1_ && !allow_legacy_v1) {
        throw std::runtime_error(
            "GDN summary store contains identity-less RC10.7 v1 blocks; "
            "set QW3_GDN_SUMMARY_ALLOW_LEGACY=1 only for controlled qualification");
    }
    if (model_id_.empty()) {
        if (contains_legacy_v1_ && allow_legacy_v1) return;
        throw std::runtime_error(
            "GDN summary store has no model identity");
    }
    if (model_id_ != expected_model_id) {
        throw std::runtime_error(
            "GDN summary model mismatch: archive='" + model_id_ +
            "' runtime='" + expected_model_id + "'");
    }
}

std::vector<const GdnBlockAffineSummary*> GdnSummaryStore::resolve_span(
    uint32_t layer_index, const GdnSummarySpan& span) const {
    if (span.token_count == 0) {
        throw std::invalid_argument("GDN summary span token_count must be non-zero");
    }
    const auto lit = by_layer_.find(layer_index);
    if (lit == by_layer_.end()) {
        throw std::runtime_error("no GDN summaries for layer " +
                                 std::to_string(layer_index));
    }
    uint64_t cursor = span.position_start;
    uint64_t remaining = span.token_count;
    std::vector<const GdnBlockAffineSummary*> out;
    while (remaining > 0) {
        const auto pit = lit->second.find(cursor);
        if (pit == lit->second.end()) {
            throw std::runtime_error(
                "missing GDN summary coverage at layer " +
                std::to_string(layer_index) + " position " +
                std::to_string(cursor));
        }
        const GdnBlockAffineSummary* chosen = nullptr;
        for (const auto& candidate : pit->second) {
            if (candidate.token_count <= remaining) {
                chosen = &candidate;
                break;
            }
        }
        if (!chosen) {
            throw std::runtime_error(
                "GDN summary block overruns requested span at layer " +
                std::to_string(layer_index) + " position " +
                std::to_string(cursor));
        }
        out.push_back(chosen);
        cursor += chosen->token_count;
        remaining -= chosen->token_count;
    }
    return out;
}

bool GdnSummaryStore::covers(
    uint32_t layer_index,
    const std::vector<GdnSummarySpan>& spans) const {
    try {
        for (const auto& span : spans) (void)resolve_span(layer_index, span);
        return !spans.empty();
    } catch (...) {
        return false;
    }
}

GdnSummaryRuntimeResult GdnSummaryStore::reconstruct_zero_state(
    uint32_t layer_index,
    const std::vector<GdnSummarySpan>& spans,
    GdnSeamPolicy seam_policy) const {
    if (spans.empty()) {
        throw std::invalid_argument("GDN runtime requires at least one selected span");
    }

    GdnSummaryRuntimeStats stats;
    stats.requested_spans = static_cast<uint32_t>(spans.size());
    stats.starts_at_zero = spans.front().position_start == 0;
    uint64_t expected = spans.front().position_start;
    for (size_t i = 0; i < spans.size(); ++i) {
        const auto& span = spans[i];
        if (span.token_count == 0) {
            throw std::invalid_argument("GDN runtime span has zero tokens");
        }
        if (i > 0 && span.position_start != expected) ++stats.discontinuity_seams;
        if (span.position_start > std::numeric_limits<uint64_t>::max() - span.token_count) {
            throw std::invalid_argument("GDN runtime span position overflow");
        }
        expected = span.position_start + span.token_count;
    }
    stats.full_model_exact = stats.starts_at_zero && stats.discontinuity_seams == 0;
    if (seam_policy == GdnSeamPolicy::FullModelExact && !stats.full_model_exact) {
        throw std::runtime_error(
            "GDN full-model-exact reconstruction rejected a non-prefix/non-contiguous "
            "selection; RC10.8 has no convolution seam repair yet");
    }

    uint32_t H = 0, D = 0;
    std::vector<float> state;
    bool initialized = false;
    for (const auto& span : spans) {
        const auto blocks = resolve_span(layer_index, span);
        for (const auto* block : blocks) {
            if (!initialized) {
                H = block->num_v_heads;
                D = block->head_dim;
                state.assign(state_elems(H, D), 0.0f);
                initialized = true;
            } else if (block->num_v_heads != H || block->head_dim != D) {
                throw std::runtime_error("GDN summary geometry changes within one layer");
            }
            gdn_apply_summary_to_state_column_major(*block, state);
            ++stats.archive_blocks_used;
        }
    }

    GdnSummaryRuntimeResult out;
    out.layer_index = layer_index;
    out.num_v_heads = H;
    out.head_dim = D;
    out.state_column_major = std::move(state);
    out.stats = stats;
    return out;
}

} // namespace qw3
