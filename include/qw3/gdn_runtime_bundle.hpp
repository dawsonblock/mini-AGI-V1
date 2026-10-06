#pragma once

#include "qw3/gdn_segment_io.hpp"
#include "qw3/kvmem_coherent_state.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace qw3 {

// Identity-bound, multi-layer runtime package for recurrent block summaries.
// The identity domain is deliberately stronger than the model_id carried by
// individual .qgds archives: adapters, tokenizer/layout, positional scheme,
// and recurrence implementation must all match before execution-state reuse.
struct GdnRuntimeBundle {
    uint32_t version = 2;
    std::string model_id;
    KvMemCacheIdentity identity;
    std::vector<GdnBlockAffineSummary> summaries;

    void validate() const;
    std::vector<uint32_t> layer_indices() const;
    uint32_t block_count() const;
};

struct GdnRuntimeSelectedBlock {
    uint32_t block_id = 0;       // canonical ordinal in the bundle
    uint64_t position_start = 0; // source token coordinate
    uint32_t token_count = 0;
};

struct GdnRuntimeLayerState {
    uint32_t layer_index = 0;
    uint32_t num_v_heads = 0;
    uint32_t head_dim = 0;
    // Native CUDA/Metal recurrent-state storage [head,column,row].
    std::vector<float> state_column_major;
    uint32_t conv_kernel_size = 0;
    uint32_t conv_dim = 0;
    // Native QW3 causal-convolution ring layout [channel, kernel-1].
    std::vector<float> conv_state_channel_major;
};

struct GdnRuntimeReconstruction {
    std::vector<uint32_t> selected_block_ids;
    std::vector<GdnRuntimeLayerState> layers;
    bool starts_at_zero = false;
    bool contiguous = false;
    bool conv_state_available = false;
    bool full_model_boundary_exact = false;
    bool hidden_replay_required = true;
    bool seam_repair_required = true; // compatibility alias of hidden_replay_required
};

struct GdnSeamReplayWindow {
    uint32_t downstream_block_id = 0;
    uint32_t compact_token_start = 0;
    uint32_t token_count = 0;
    uint64_t source_position_start = 0;
};

struct GdnSeamReplayPlan {
    std::vector<uint32_t> selected_block_ids;
    std::vector<GdnSeamReplayWindow> windows;
    uint32_t selected_tokens = 0;
    uint32_t discontinuities = 0;
    uint32_t conv_repair_tokens = 0;
    uint32_t hidden_repair_tokens = 0;
    uint32_t replay_tokens = 0;
    double replay_fraction = 0.0;
    bool full_model_boundary_exact = false;
    bool qualification_shortcut_eligible = false;
    bool parity_gate_required = true;
};

GdnRuntimeBundle build_gdn_runtime_bundle(
    const KvMemCacheIdentity& identity,
    const std::vector<GdnBlockAffineSummary>& summaries);

void write_gdn_runtime_bundle(const std::string& path,
                              const GdnRuntimeBundle& bundle);
GdnRuntimeBundle read_gdn_runtime_bundle(const std::string& path);

// Returns canonical block metadata. All recurrent layers are required to have
// exactly the same block boundaries; validation fails closed otherwise.
std::vector<GdnRuntimeSelectedBlock> gdn_runtime_blocks(
    const GdnRuntimeBundle& bundle);

// Reconstruct all recurrent layers from zero for the selected chronological
// block ordinals. Exact cache identity is mandatory. Non-contiguous selections
// are algebraically valid for the stored prepared-input affine operators but
// remain unsafe for full-model serving until Conv1D/hidden seam repair runs;
// seam_repair_required therefore stays true unless a gap-free prefix from zero
// is selected.
GdnRuntimeReconstruction reconstruct_gdn_runtime_bundle_cpu(
    const GdnRuntimeBundle& bundle,
    const std::vector<uint32_t>& selected_block_ids,
    const KvMemCacheIdentity& expected_identity);

// Build a bounded hidden-state replay plan around every sparse seam. This does
// not claim fixed-window replay is exact: any sparse plan remains parity-gated
// against exact selected replay and must fall back when the gate fails.
GdnSeamReplayPlan plan_gdn_seam_replay(
    const GdnRuntimeBundle& bundle,
    const std::vector<uint32_t>& selected_block_ids,
    uint32_t hidden_repair_tokens,
    double max_replay_fraction = 0.25);

} // namespace qw3
