#include "qw3/gdn_metal.hpp"

#include <cmath>
#include <stdexcept>

namespace qw3 {

void GdnMetalInputs::validate() const {
    if (T == 0 || num_k_heads == 0 || num_v_heads == 0 || head_dim == 0) {
        throw std::invalid_argument("GDN Metal dimensions must be non-zero");
    }
    const size_t kd = static_cast<size_t>(T) * num_k_heads * head_dim;
    const size_t vd = static_cast<size_t>(T) * num_v_heads * head_dim;
    const size_t gd = static_cast<size_t>(T) * num_v_heads;
    const size_t sd = static_cast<size_t>(num_v_heads) * head_dim * head_dim;
    if (q.size() != kd || k.size() != kd || v.size() != vd ||
        alpha_raw.size() != gd || beta_raw.size() != gd ||
        dt_bias.size() != num_v_heads || ssm_a.size() != num_v_heads ||
        state.size() != sd) {
        throw std::invalid_argument("GDN Metal input shape mismatch");
    }
    auto finite = [](const std::vector<float>& x) {
        for (float v : x) if (!std::isfinite(v)) return false;
        return true;
    };
    if (!finite(q) || !finite(k) || !finite(v) || !finite(alpha_raw) ||
        !finite(beta_raw) || !finite(dt_bias) || !finite(ssm_a) || !finite(state)) {
        throw std::invalid_argument("GDN Metal inputs must be finite");
    }
}

bool gdn_metal_available() { return false; }

GdnMetalResult gdn_metal_run(const GdnMetalInputs& inputs) {
    inputs.validate();
    throw std::runtime_error("Apple Metal GDN is unavailable on this platform/build");
}


void GdnMetalPreparedInputs::validate() const {
    if (T == 0 || num_k_heads == 0 || num_v_heads == 0 || head_dim == 0 ||
        num_v_heads % num_k_heads != 0) {
        throw std::invalid_argument("GDN Metal prepared dimensions are invalid");
    }
    const size_t kd = static_cast<size_t>(T) * num_k_heads * head_dim;
    const size_t vd = static_cast<size_t>(T) * num_v_heads * head_dim;
    const size_t gd = static_cast<size_t>(T) * num_v_heads;
    if (k.size() != kd || v.size() != vd || decay.size() != gd || beta.size() != gd) {
        throw std::invalid_argument("GDN Metal prepared input shape mismatch");
    }
    auto finite = [](const std::vector<float>& x) {
        for (float v : x) if (!std::isfinite(v)) return false;
        return true;
    };
    if (!finite(k) || !finite(v) || !finite(decay) || !finite(beta)) {
        throw std::invalid_argument("GDN Metal prepared inputs must be finite");
    }
}

GdnMetalAffineSummary gdn_metal_summarize(const GdnMetalPreparedInputs& inputs) {
    inputs.validate();
    throw std::runtime_error("Apple Metal GDN summary is unavailable on this platform/build");
}

} // namespace qw3

namespace qw3 {

void GdnMetalSummaryChainInputs::validate() const {
    if (block_count == 0 || num_v_heads == 0 || head_dim == 0) {
        throw std::invalid_argument("GDN Metal summary-chain dimensions must be non-zero");
    }
    const uint64_t per = static_cast<uint64_t>(num_v_heads) * head_dim * head_dim;
    const uint64_t all = static_cast<uint64_t>(block_count) * per;
    if (per > static_cast<uint64_t>(SIZE_MAX) || all > static_cast<uint64_t>(SIZE_MAX) ||
        T.size() != static_cast<size_t>(all) || Z.size() != static_cast<size_t>(all) ||
        state.size() != static_cast<size_t>(per)) {
        throw std::invalid_argument("GDN Metal summary-chain shape mismatch");
    }
    for (float v : T) if (!std::isfinite(v)) throw std::invalid_argument("GDN summary T contains non-finite value");
    for (float v : Z) if (!std::isfinite(v)) throw std::invalid_argument("GDN summary Z contains non-finite value");
    for (float v : state) if (!std::isfinite(v)) throw std::invalid_argument("GDN summary state contains non-finite value");
}

GdnMetalSummaryChainResult gdn_metal_apply_summary_chain(
    const GdnMetalSummaryChainInputs& inputs) {
    inputs.validate();
    throw std::runtime_error("Apple Metal GDN summary-chain application is unavailable on this platform/build");
}

} // namespace qw3
