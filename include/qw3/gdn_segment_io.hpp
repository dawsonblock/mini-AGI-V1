#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qw3 {

// Versioned interchange for model-integrated Qwen Gated-DeltaNet captures.
// q/k are post-convolution, post-SiLU, post-L2-normalization; v is
// post-convolution/post-SiLU. decay and beta are already preprocessed.
struct GdnPreparedCapture {
    // v2 adds the raw pre-convolution recurrent projection rows needed to
    // reconstruct the causal Conv1D ring state at a selected-context boundary.
    // v1 remains readable for RC10.7/RC10.8 compatibility.
    uint32_t version = 2;
    uint32_t layer_index = 0;
    uint64_t position_start = 0;
    uint32_t T = 0;
    uint32_t num_k_heads = 0;
    uint32_t num_v_heads = 0;
    uint32_t head_dim = 0;
    std::string model_id;
    std::vector<float> q;      // [T, num_k_heads, head_dim]
    std::vector<float> k;      // [T, num_k_heads, head_dim]
    std::vector<float> v;      // [T, num_v_heads, head_dim]
    std::vector<float> decay;  // [T, num_v_heads]
    std::vector<float> beta;   // [T, num_v_heads]
    uint32_t conv_kernel_size = 0;
    uint32_t conv_dim = 0;
    std::vector<float> preconv; // [T, conv_dim], projection before causal Conv1D

    void validate() const;
};

struct GdnBlockAffineSummary {
    // v2 adds model identity. v3 additionally carries the trailing raw
    // pre-convolution projection rows required to reconstruct the causal
    // Conv1D ring state at the end of a selected block chain.
    uint32_t version = 3;
    uint32_t layer_index = 0;
    uint64_t position_start = 0;
    uint32_t token_count = 0;
    uint32_t num_v_heads = 0;
    uint32_t head_dim = 0;
    // Canonical logical row-major matrices, head-major:
    // [num_v_heads, head_dim, head_dim]. This is intentionally independent
    // of CUDA/Metal column-major recurrent-state storage.
    std::vector<float> T;
    std::vector<float> Z;
    std::string model_id;
    std::string producer;
    uint32_t conv_kernel_size = 0;
    uint32_t conv_dim = 0;
    uint32_t conv_tail_rows = 0;
    // Row-major [conv_tail_rows, conv_dim]. For v3 this is the final
    // min(token_count, conv_kernel_size-1) raw projection rows in the block.
    std::vector<float> conv_tail;

    void validate() const;
};

void write_gdn_capture(const std::string& path, const GdnPreparedCapture& capture);
GdnPreparedCapture read_gdn_capture(const std::string& path);

void write_gdn_summary_archive(
    const std::string& path,
    const std::vector<GdnBlockAffineSummary>& summaries);
std::vector<GdnBlockAffineSummary> read_gdn_summary_archive(const std::string& path);

// Correctness oracle: compose each capture block in fp64, then serialize f32.
std::vector<GdnBlockAffineSummary> gdn_summarize_capture_reference(
    const GdnPreparedCapture& capture,
    uint32_t block_tokens);

} // namespace qw3
