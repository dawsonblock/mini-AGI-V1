#include "qw3/gdn_reference.hpp"
#include "qw3/gdn_segment_io.hpp"
#include "qw3/gdn_summary_runtime.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

qw3::GdnPreparedCapture make_capture() {
    qw3::GdnPreparedCapture c;
    c.layer_index = 5;
    c.position_start = 0;
    c.T = 12;
    c.num_k_heads = 1;
    c.num_v_heads = 2;
    c.head_dim = 8;
    c.model_id = "unit:qwen-runtime";
    c.q.resize(static_cast<size_t>(c.T) * c.num_k_heads * c.head_dim);
    c.k.resize(c.q.size());
    c.v.resize(static_cast<size_t>(c.T) * c.num_v_heads * c.head_dim);
    c.decay.resize(static_cast<size_t>(c.T) * c.num_v_heads);
    c.beta.resize(c.decay.size());
    c.conv_kernel_size = 4;
    c.conv_dim = 2u * c.num_k_heads * c.head_dim + c.num_v_heads * c.head_dim;
    c.preconv.resize(static_cast<size_t>(c.T) * c.conv_dim);
    for (size_t i = 0; i < c.q.size(); ++i) {
        c.q[i] = 0.01f * std::sin(float(i + 1));
        c.k[i] = 0.025f * std::cos(float(i + 2));
    }
    for (size_t i = 0; i < c.v.size(); ++i) {
        c.v[i] = 0.035f * std::sin(float(i + 5) * 0.2f);
    }
    for (size_t i = 0; i < c.decay.size(); ++i) {
        c.decay[i] = 0.92f + 0.001f * float(i % 20);
        c.beta[i] = 0.15f + 0.01f * float(i % 6);
    }
    for (size_t i = 0; i < c.preconv.size(); ++i) {
        c.preconv[i] = 0.017f * std::sin(float(i + c.layer_index + 11) * 0.13f);
    }
    c.validate();
    return c;
}

std::vector<float> direct_zero_state(const qw3::GdnPreparedCapture& c,
                                     uint32_t begin,
                                     uint32_t count) {
    const uint32_t D = c.head_dim;
    const size_t per = static_cast<size_t>(D) * D;
    std::vector<float> native(static_cast<size_t>(c.num_v_heads) * per, 0.0f);
    for (uint32_t vh = 0; vh < c.num_v_heads; ++vh) {
        std::vector<qw3::GdnReferenceToken> toks;
        for (uint32_t t = begin; t < begin + count; ++t) {
            qw3::GdnReferenceToken tok;
            tok.k.resize(D); tok.v.resize(D);
            const uint32_t kh = vh % c.num_k_heads;
            const size_t kb = (static_cast<size_t>(t) * c.num_k_heads + kh) * D;
            const size_t vb = (static_cast<size_t>(t) * c.num_v_heads + vh) * D;
            for (uint32_t d = 0; d < D; ++d) {
                tok.k[d] = c.k[kb + d];
                tok.v[d] = c.v[vb + d];
            }
            tok.g = c.decay[static_cast<size_t>(t) * c.num_v_heads + vh];
            tok.beta = c.beta[static_cast<size_t>(t) * c.num_v_heads + vh];
            toks.push_back(std::move(tok));
        }
        std::vector<double> zero(per, 0.0);
        const auto logical = qw3::gdn_replay_reference(toks, zero, D);
        const size_t hb = static_cast<size_t>(vh) * per;
        for (uint32_t col = 0; col < D; ++col) {
            for (uint32_t row = 0; row < D; ++row) {
                native[hb + static_cast<size_t>(col) * D + row] =
                    static_cast<float>(logical[static_cast<size_t>(row) * D + col]);
            }
        }
    }
    return native;
}

} // namespace

int main() {
    const auto c = make_capture();
    const auto summaries = qw3::gdn_summarize_capture_reference(c, 4);
    assert(summaries.size() == 3);
    for (const auto& s : summaries) {
        assert(s.version == 3);
        assert(s.conv_kernel_size == 4);
        assert(s.conv_tail_rows == 3);
        assert(s.model_id == c.model_id);
    }

    const auto store = qw3::GdnSummaryStore::from_summaries(summaries);
    assert(store.block_count() == 3);
    store.require_model(c.model_id);

    // A gap-free zero-origin prefix is full-model exact for the captured
    // post-convolution prepared inputs.
    const std::vector<qw3::GdnSummarySpan> prefix{{0, 4}, {4, 8}};
    const auto r = store.reconstruct_zero_state(
        c.layer_index, prefix, qw3::GdnSeamPolicy::FullModelExact);
    assert(r.stats.full_model_exact);
    assert(r.stats.discontinuity_seams == 0);
    assert(r.stats.archive_blocks_used == 3);
    const auto direct = direct_zero_state(c, 0, 12);
    assert(direct.size() == r.state_column_major.size());
    double worst = 0.0;
    for (size_t i = 0; i < direct.size(); ++i) {
        worst = std::max(worst, std::abs(double(direct[i]) - r.state_column_major[i]));
    }
    // Stored summaries are f32 and runtime accumulation converts back to f32.
    assert(worst < 2e-5);

    // Missing-source seams are never silently labeled exact.
    const std::vector<qw3::GdnSummarySpan> gapped{{0, 4}, {8, 4}};
    bool strict_rejected = false;
    try {
        (void)store.reconstruct_zero_state(
            c.layer_index, gapped, qw3::GdnSeamPolicy::FullModelExact);
    } catch (const std::runtime_error&) {
        strict_rejected = true;
    }
    assert(strict_rejected);
    const auto approx = store.reconstruct_zero_state(
        c.layer_index, gapped, qw3::GdnSeamPolicy::PreparedInputApprox);
    assert(!approx.stats.full_model_exact);
    assert(approx.stats.discontinuity_seams == 1);
    assert(approx.stats.archive_blocks_used == 2);

    bool mismatch_rejected = false;
    try { store.require_model("other:model"); }
    catch (const std::runtime_error&) { mismatch_rejected = true; }
    assert(mismatch_rejected);

    // Coverage must fail when a requested position was never summarized.
    assert(store.covers(c.layer_index, {{0, 8}}));
    assert(!store.covers(c.layer_index, {{2, 4}}));
    return 0;
}
