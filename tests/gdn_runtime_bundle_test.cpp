#include "qw3/gdn_reference.hpp"
#include "qw3/gdn_runtime_bundle.hpp"
#include "qw3/gdn_segment_io.hpp"
#include "qw3/gdn_summary_runtime.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

qw3::GdnPreparedCapture make_capture(uint32_t layer) {
    qw3::GdnPreparedCapture c;
    c.layer_index = layer;
    c.position_start = 0;
    c.T = 12;
    c.num_k_heads = 1;
    c.num_v_heads = 2;
    c.head_dim = 8;
    c.model_id = "unit:qwen-bundle";
    c.q.resize(static_cast<size_t>(c.T) * c.num_k_heads * c.head_dim);
    c.k.resize(c.q.size());
    c.v.resize(static_cast<size_t>(c.T) * c.num_v_heads * c.head_dim);
    c.decay.resize(static_cast<size_t>(c.T) * c.num_v_heads);
    c.beta.resize(c.decay.size());
    c.conv_kernel_size = 4;
    c.conv_dim = 2u * c.num_k_heads * c.head_dim + c.num_v_heads * c.head_dim;
    c.preconv.resize(static_cast<size_t>(c.T) * c.conv_dim);
    for (size_t i = 0; i < c.q.size(); ++i) {
        c.q[i] = 0.01f * std::sin(float(i + layer + 1));
        c.k[i] = 0.02f * std::cos(float(i + layer + 2));
    }
    for (size_t i = 0; i < c.v.size(); ++i) {
        c.v[i] = 0.03f * std::sin(float(i + 2 * layer + 5) * 0.2f);
    }
    for (size_t i = 0; i < c.decay.size(); ++i) {
        c.decay[i] = 0.91f + 0.001f * float(i % 15);
        c.beta[i] = 0.11f + 0.01f * float(i % 7);
    }
    for (size_t i = 0; i < c.preconv.size(); ++i) {
        c.preconv[i] = 0.017f * std::sin(float(i + c.layer_index + 11) * 0.13f);
    }
    c.validate();
    return c;
}

qw3::KvMemCacheIdentity ident() {
    return {"base-sha", "adapter-sha", "tokenizer-sha", "layout-sha",
            "rope-v1", "gdn-affine-v2"};
}

} // namespace

int main() {
    std::vector<qw3::GdnBlockAffineSummary> all;
    for (uint32_t layer : {3u, 9u}) {
        auto v = qw3::gdn_summarize_capture_reference(make_capture(layer), 4);
        all.insert(all.end(), v.begin(), v.end());
    }
    const auto bundle = qw3::build_gdn_runtime_bundle(ident(), all);
    assert(bundle.block_count() == 3);
    assert((bundle.layer_indices() == std::vector<uint32_t>{3, 9}));

    const auto path = std::filesystem::temp_directory_path() / "qw3-gdn-runtime-bundle-test.qgdb";
    qw3::write_gdn_runtime_bundle(path.string(), bundle);
    const auto roundtrip = qw3::read_gdn_runtime_bundle(path.string());
    assert(roundtrip.version == 2);
    assert(roundtrip.identity == ident());
    assert(roundtrip.summaries.size() == 6);

    const auto r = qw3::reconstruct_gdn_runtime_bundle_cpu(roundtrip, {0, 2}, ident());
    assert(r.layers.size() == 2);
    assert(r.conv_state_available);
    assert(r.seam_repair_required);
    assert(r.hidden_replay_required);
    assert(!r.full_model_boundary_exact);
    assert(!r.contiguous);
    assert(r.starts_at_zero);

    // Independently apply the same selected block summaries to each layer and
    // compare native [head,column,row] state exactly within f32 roundoff.
    for (const auto& layer_state : r.layers) {
        std::vector<qw3::GdnBlockAffineSummary> selected;
        for (const auto& s : roundtrip.summaries) {
            if (s.layer_index == layer_state.layer_index &&
                (s.position_start == 0 || s.position_start == 8)) {
                selected.push_back(s);
            }
        }
        assert(selected.size() == 2);
        std::sort(selected.begin(), selected.end(), [](const auto& a, const auto& b) {
            return a.position_start < b.position_start;
        });
        std::vector<float> expected(layer_state.state_column_major.size(), 0.0f);
        for (const auto& s : selected) qw3::gdn_apply_summary_to_state_column_major(s, expected);
        double worst = 0.0;
        for (size_t i = 0; i < expected.size(); ++i) {
            worst = std::max(worst, std::abs(double(expected[i]) - layer_state.state_column_major[i]));
        }
        assert(worst < 1e-6);
        assert(layer_state.conv_kernel_size == 4);
        assert(layer_state.conv_state_channel_major.size() ==
               static_cast<size_t>(layer_state.conv_dim) * 3);
        // Sparse chain ends at block 2 (positions 8..11), so the physical
        // Conv1D ring must contain the final three pre-convolution rows 9..11.
        const auto source = make_capture(layer_state.layer_index);
        for (uint32_t c = 0; c < source.conv_dim; ++c) {
            for (uint32_t rr = 0; rr < 3; ++rr) {
                const float expected_conv = source.preconv[
                    static_cast<size_t>(9 + rr) * source.conv_dim + c];
                assert(std::abs(expected_conv - layer_state.conv_state_channel_major[
                    static_cast<size_t>(c) * 3 + rr]) < 1e-7f);
            }
        }
    }

    // A gap-free prefix from zero reconstructs both recurrent and Conv1D state
    // as an exact executable hybrid boundary.
    const auto exact = qw3::reconstruct_gdn_runtime_bundle_cpu(roundtrip, {0, 1}, ident());
    assert(exact.conv_state_available);
    assert(exact.full_model_boundary_exact);
    assert(!exact.hidden_replay_required);
    assert(!exact.seam_repair_required);

    const auto sparse_plan = qw3::plan_gdn_seam_replay(roundtrip, {0, 2}, 4, 0.6);
    assert(sparse_plan.discontinuities == 1);
    assert(sparse_plan.conv_repair_tokens == 3);
    assert(sparse_plan.replay_tokens == 4);
    assert(sparse_plan.parity_gate_required);
    assert(sparse_plan.qualification_shortcut_eligible);
    const auto exact_plan = qw3::plan_gdn_seam_replay(roundtrip, {0, 1}, 4, 0.6);
    assert(exact_plan.full_model_boundary_exact);
    assert(exact_plan.replay_tokens == 0);
    assert(!exact_plan.parity_gate_required);

    bool too_short = false;
    try { (void)qw3::plan_gdn_seam_replay(roundtrip, {0, 2}, 2, 0.6); }
    catch (const std::invalid_argument&) { too_short = true; }
    assert(too_short);

    auto wrong = ident();
    wrong.adapter_set_digest = "different-adapter";
    bool mismatch = false;
    try { (void)qw3::reconstruct_gdn_runtime_bundle_cpu(roundtrip, {0}, wrong); }
    catch (const std::runtime_error&) { mismatch = true; }
    assert(mismatch);

    bool reordered = false;
    try { (void)qw3::reconstruct_gdn_runtime_bundle_cpu(roundtrip, {2, 0}, ident()); }
    catch (const std::invalid_argument&) { reordered = true; }
    assert(reordered);

    // Byte corruption must be detected by the bundle checksum.
    {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        assert(f);
        f.seekg(32);
        char c = 0;
        f.read(&c, 1);
        f.seekp(32);
        c ^= 0x5a;
        f.write(&c, 1);
    }
    bool corrupt = false;
    try { (void)qw3::read_gdn_runtime_bundle(path.string()); }
    catch (const std::runtime_error&) { corrupt = true; }
    assert(corrupt);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return 0;
}
