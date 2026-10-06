#pragma once

#include <cstdint>
#include <vector>

namespace qw3 {

// Host/reference implementation of the exact state update used by
// src/gated_delta_net.cu after alpha/beta preprocessing. This is intentionally
// dense and correctness-oriented; it is not a production storage format.
//
// For one value head, with matrix state S [d,d]:
//   kv    = S^T k
//   delta = beta * (v - g * kv)
//   S'    = g * S + k delta^T
//         = g(I - beta k k^T) S + beta k v^T
//         = T S + Z
//
// This provides the exact per-token affine form required for HYPIC-style
// segment composition when k/v/g/beta are held fixed.
struct GdnReferenceToken {
    std::vector<double> k;
    std::vector<double> v;
    double g = 1.0;       // preprocessed decay exp(log_g)
    double beta = 0.0;    // preprocessed sigmoid(beta_raw)
};

struct GdnMatrixAffineSummary {
    uint32_t dim = 0;
    std::vector<double> T; // row-major [d,d]
    std::vector<double> Z; // row-major [d,d]

    static GdnMatrixAffineSummary identity(uint32_t dim);
    void validate() const;
    std::vector<double> apply(const std::vector<double>& state) const;
    GdnMatrixAffineSummary then(const GdnMatrixAffineSummary& next) const;
};

// Numerically stable equivalents of the CUDA preprocessing stage.
double gdn_sigmoid(double x);
double gdn_softplus(double x);
double gdn_decay_from_raw(double alpha_raw, double dt_bias, double ssm_a);

GdnMatrixAffineSummary gdn_token_affine(const GdnReferenceToken& token);
GdnMatrixAffineSummary gdn_compose_tokens(const std::vector<GdnReferenceToken>& tokens);

// Direct token-by-token oracle matching the CUDA update algebra.
std::vector<double> gdn_replay_reference(
    const std::vector<GdnReferenceToken>& tokens,
    const std::vector<double>& initial_state,
    uint32_t dim);

// Optional state readout S^T q * scale, matching the recurrent kernel's
// post-update output convention for one value head.
std::vector<double> gdn_readout_reference(
    const std::vector<double>& state,
    const std::vector<double>& q,
    uint32_t dim,
    double scale = 1.0);

struct GdnReferenceError {
    double max_abs = 0.0;
    double rel_l2 = 0.0;
};

GdnReferenceError gdn_compare_state(
    const std::vector<double>& a,
    const std::vector<double>& b);

} // namespace qw3
