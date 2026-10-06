#pragma once

#include "qw3/gdn_segment_io.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace qw3 {

// One source span selected into a KVMem logical window. Spans are expressed in
// original model-token coordinates, not compacted window coordinates.
struct GdnSummarySpan {
    uint64_t position_start = 0;
    uint32_t token_count = 0;
};

enum class GdnSeamPolicy : uint8_t {
    // Full-model exactness from a zero recurrent/conv origin. This requires a
    // gap-free source prefix beginning at position 0. It is intentionally
    // strict because RC10.7 summaries contain post-convolution prepared inputs.
    FullModelExact = 0,
    // Compose the captured prepared-input operators exactly even when source
    // spans have gaps. This is algebraically exact for the stored operators,
    // but the first conv-kernel-width-1 tokens after each gap retain their
    // original source boundary context. It is therefore experimental.
    PreparedInputApprox = 1,
};

struct GdnSummaryRuntimeStats {
    uint32_t requested_spans = 0;
    uint32_t archive_blocks_used = 0;
    uint32_t discontinuity_seams = 0;
    bool starts_at_zero = false;
    bool full_model_exact = false;
};

struct GdnSummaryRuntimeResult {
    uint32_t layer_index = 0;
    uint32_t num_v_heads = 0;
    uint32_t head_dim = 0;
    // Native CUDA/Metal recurrent-state layout: head-major, then matrix
    // column-major within each head [H, col, row].
    std::vector<float> state_column_major;
    GdnSummaryRuntimeStats stats;
};

// Applies one canonical row-major block summary to a native column-major
// recurrent state. This is the CPU correctness path used by qualification and
// by the optional live executor bridge. Accumulation is fp64, output fp32.
void gdn_apply_summary_to_state_column_major(
    const GdnBlockAffineSummary& summary,
    std::vector<float>& state_column_major);

// Immutable, validated index over one or more .qgds archives. Duplicate blocks
// are allowed only when byte-equivalent; conflicting duplicates fail closed.
class GdnSummaryStore {
public:
    static GdnSummaryStore load_directory(const std::string& directory);
    static GdnSummaryStore from_summaries(
        const std::vector<GdnBlockAffineSummary>& summaries);

    bool empty() const { return block_count_ == 0; }
    size_t block_count() const { return block_count_; }
    const std::string& model_id() const { return model_id_; }
    bool contains_legacy_v1() const { return contains_legacy_v1_; }

    // Ensure this store belongs to the running checkpoint. Legacy RC10.7
    // summaries have no model_id and are rejected unless explicitly allowed.
    void require_model(const std::string& expected_model_id,
                       bool allow_legacy_v1 = false) const;

    // Reconstruct a recurrent state from zero by applying the selected source
    // spans in the order provided. A span may be covered by several archive
    // blocks; coverage must be exact with no holes/overrun.
    GdnSummaryRuntimeResult reconstruct_zero_state(
        uint32_t layer_index,
        const std::vector<GdnSummarySpan>& spans,
        GdnSeamPolicy seam_policy) const;

    // True only when every requested span has exact summary coverage.
    bool covers(uint32_t layer_index,
                const std::vector<GdnSummarySpan>& spans) const;

private:
    struct Key {
        uint32_t layer = 0;
        uint64_t position = 0;
        uint32_t tokens = 0;
        bool operator<(const Key& o) const {
            if (layer != o.layer) return layer < o.layer;
            if (position != o.position) return position < o.position;
            return tokens < o.tokens;
        }
    };

    // layer -> position -> candidate blocks, longest token_count first.
    std::map<uint32_t, std::map<uint64_t, std::vector<GdnBlockAffineSummary>>> by_layer_;
    std::string model_id_;
    bool contains_legacy_v1_ = false;
    size_t block_count_ = 0;

    void add(const GdnBlockAffineSummary& summary);
    std::vector<const GdnBlockAffineSummary*> resolve_span(
        uint32_t layer_index, const GdnSummarySpan& span) const;
};

} // namespace qw3
