#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qw3 {

// Correctness-first Apple Metal implementation of the QW3 Gated-DeltaNet
// recurrent update. This is deliberately a qualification kernel, not yet the
// optimized production KVMem path. Inputs use the same logical layouts as the
// CUDA reference in src/gated_delta_net.cu.
struct GdnMetalInputs {
    uint32_t T = 0;
    uint32_t num_k_heads = 0;
    uint32_t num_v_heads = 0;
    uint32_t head_dim = 0;

    // Raw projections/gates.
    // q,k: [T, num_k_heads, head_dim]
    // v:   [T, num_v_heads, head_dim]
    // alpha_raw,beta_raw: [T, num_v_heads]
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> alpha_raw;
    std::vector<float> beta_raw;
    std::vector<float> dt_bias; // [num_v_heads]
    std::vector<float> ssm_a;   // [num_v_heads]

    // Column-major-within-head state, identical to CUDA storage:
    // [num_v_heads, head_dim columns, head_dim rows].
    std::vector<float> state;

    void validate() const;
};

struct GdnMetalResult {
    std::vector<float> state;
    std::vector<float> out;   // [T, num_v_heads, head_dim]
    std::vector<float> decay; // preprocessed g [T, num_v_heads]
    std::vector<float> beta;  // sigmoid(beta_raw) [T, num_v_heads]
    std::string device_name;
};

// True only when compiled for Apple with Metal enabled and a Metal device can
// be created. Non-Apple builds expose a fail-closed stub.
bool gdn_metal_available();

// Runs gate preprocessing and the recurrent GDN update on Apple Metal.
// Throws std::runtime_error when Metal is unavailable or execution fails.
GdnMetalResult gdn_metal_run(const GdnMetalInputs& inputs);

} // namespace qw3

namespace qw3 {

// Prepared inputs used for block affine summarization. Unlike GdnMetalInputs,
// gates are already transformed to decay/beta and no recurrent initial state
// or readout q is needed for (T_C,Z_C) construction.
struct GdnMetalPreparedInputs {
    uint32_t T = 0;
    uint32_t num_k_heads = 0;
    uint32_t num_v_heads = 0;
    uint32_t head_dim = 0;
    std::vector<float> k;     // [T, num_k_heads, head_dim]
    std::vector<float> v;     // [T, num_v_heads, head_dim]
    std::vector<float> decay; // [T, num_v_heads]
    std::vector<float> beta;  // [T, num_v_heads]
    void validate() const;
};

struct GdnMetalAffineSummary {
    uint32_t num_v_heads = 0;
    uint32_t head_dim = 0;
    // Canonical logical row-major matrices, head-major.
    std::vector<float> T;
    std::vector<float> Z;
    std::string device_name;
};

// Correctness-first native Metal construction of per-head affine summaries.
// Non-Apple builds fail closed through the stub.
GdnMetalAffineSummary gdn_metal_summarize(const GdnMetalPreparedInputs& inputs);

} // namespace qw3

namespace qw3 {

// Apply a chronological chain of canonical row-major affine block summaries
// to native recurrent state [head,column,row]. T/Z layout is
// [block,head,row,column]. This is the RC10.8 Metal summary-consumption path.
struct GdnMetalSummaryChainInputs {
    uint32_t block_count = 0;
    uint32_t num_v_heads = 0;
    uint32_t head_dim = 0;
    std::vector<float> T;
    std::vector<float> Z;
    std::vector<float> state;
    void validate() const;
};

struct GdnMetalSummaryChainResult {
    std::vector<float> state;
    std::string device_name;
};

GdnMetalSummaryChainResult gdn_metal_apply_summary_chain(
    const GdnMetalSummaryChainInputs& inputs);

} // namespace qw3
