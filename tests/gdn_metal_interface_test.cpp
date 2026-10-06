#include "qw3/gdn_metal.hpp"
#include "qw3/gdn_reference.hpp"
#include "qw3/gdn_segment_io.hpp"
#include "qw3/gdn_summary_runtime.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
#ifndef __APPLE__
    assert(!qw3::gdn_metal_available());
    qw3::GdnMetalPreparedInputs prepared;
    prepared.T = 1; prepared.num_k_heads = 1; prepared.num_v_heads = 1; prepared.head_dim = 16;
    prepared.k.assign(16, 0.01f); prepared.v.assign(16, 0.02f);
    prepared.decay.assign(1, 0.95f); prepared.beta.assign(1, 0.2f);
    bool summary_failed_closed = false;
    try { (void)qw3::gdn_metal_summarize(prepared); }
    catch (const std::runtime_error&) { summary_failed_closed = true; }
    assert(summary_failed_closed);
    qw3::GdnMetalSummaryChainInputs chain;
    chain.block_count = 2; chain.num_v_heads = 1; chain.head_dim = 16;
    chain.T.assign(2 * 16 * 16, 0.0f); chain.Z.assign(2 * 16 * 16, 0.0f);
    chain.state.assign(16 * 16, 0.0f);
    for (uint32_t b = 0; b < 2; ++b) for (uint32_t d = 0; d < 16; ++d)
        chain.T[static_cast<size_t>(b) * 16 * 16 + d * 16 + d] = 1.0f;
    bool chain_failed_closed = false;
    try { (void)qw3::gdn_metal_apply_summary_chain(chain); }
    catch (const std::runtime_error&) { chain_failed_closed = true; }
    assert(chain_failed_closed);
    std::cout << "Metal unavailable on non-Apple host (expected)\n";
    return 0;
#else
    if (!qw3::gdn_metal_available()) {
        std::cout << "Metal device unavailable; runtime parity test skipped\n";
        return 0;
    }

    constexpr uint32_t T = 6;
    constexpr uint32_t D = 16;
    constexpr uint32_t KH = 1;
    constexpr uint32_t VH = 1;

    qw3::GdnMetalInputs in;
    in.T = T;
    in.num_k_heads = KH;
    in.num_v_heads = VH;
    in.head_dim = D;
    in.q.resize(T * KH * D);
    in.k.resize(T * KH * D);
    in.v.resize(T * VH * D);
    in.alpha_raw.resize(T * VH);
    in.beta_raw.resize(T * VH);
    in.dt_bias.assign(VH, -0.25f);
    in.ssm_a.assign(VH, -0.9f);
    in.state.resize(VH * D * D);

    for (size_t i = 0; i < in.q.size(); ++i) in.q[i] = 0.025f * std::sin(float(i + 1));
    for (size_t i = 0; i < in.k.size(); ++i) in.k[i] = 0.020f * std::cos(float(i + 3));
    for (size_t i = 0; i < in.v.size(); ++i) in.v[i] = 0.030f * std::sin(float(i + 7) * 0.5f);
    for (size_t i = 0; i < in.alpha_raw.size(); ++i) in.alpha_raw[i] = -0.4f + 0.01f * float(i);
    for (size_t i = 0; i < in.beta_raw.size(); ++i) in.beta_raw[i] = 0.2f - 0.015f * float(i);
    for (size_t i = 0; i < in.state.size(); ++i) in.state[i] = 0.005f * std::sin(float(i) * 0.1f);

    auto metal = qw3::gdn_metal_run(in);
    // Validate the compensated softplus across underflow/rounding/large-value
    // boundaries using the independent double-precision gate oracle.
    auto edge = in;
    edge.dt_bias.assign(VH, 0.0f);
    edge.alpha_raw = {-30.0f, -19.0f, -10.0f, 0.0f, 19.0f, 30.0f};
    edge.beta_raw = {-30.0f, -19.0f, -10.0f, 0.0f, 19.0f, 30.0f};
    const auto edge_metal = qw3::gdn_metal_run(edge);
    for (uint32_t i = 0; i < T; ++i) {
        const double expected_decay = qw3::gdn_decay_from_raw(edge.alpha_raw[i], 0.0, edge.ssm_a[0]);
        const double expected_beta = qw3::gdn_sigmoid(edge.beta_raw[i]);
        assert(std::abs(edge_metal.decay[i] - expected_decay) < 1e-6);
        assert(std::abs(edge_metal.beta[i] - expected_beta) < 1e-6);
    }
    assert(metal.state.size() == in.state.size());
    assert(metal.out.size() == T * VH * D);

    // Metal/CUDA state storage is column-major within each head, while the
    // fp64 oracle exposes the logical S[row,col] matrix in row-major form.
    std::vector<double> cpu_state(D * D, 0.0);
    for (uint32_t col = 0; col < D; ++col) {
        for (uint32_t row = 0; row < D; ++row) {
            cpu_state[row * D + col] = in.state[col * D + row];
        }
    }
    std::vector<double> cpu_out;
    cpu_out.reserve(T * D);
    for (uint32_t t = 0; t < T; ++t) {
        qw3::GdnReferenceToken tok;
        tok.k.resize(D);
        tok.v.resize(D);
        for (uint32_t d = 0; d < D; ++d) {
            tok.k[d] = in.k[(t * KH) * D + d];
            tok.v[d] = in.v[(t * VH) * D + d];
        }
        tok.g = qw3::gdn_decay_from_raw(in.alpha_raw[t], in.dt_bias[0], in.ssm_a[0]);
        tok.beta = qw3::gdn_sigmoid(in.beta_raw[t]);
        cpu_state = qw3::gdn_replay_reference({tok}, cpu_state, D);
        std::vector<double> q(D);
        for (uint32_t d = 0; d < D; ++d) q[d] = in.q[(t * KH) * D + d];
        auto y = qw3::gdn_readout_reference(cpu_state, q, D, 1.0 / std::sqrt(double(D)));
        cpu_out.insert(cpu_out.end(), y.begin(), y.end());
    }

    std::vector<double> metal_state_logical(D * D, 0.0);
    for (uint32_t col = 0; col < D; ++col) {
        for (uint32_t row = 0; row < D; ++row) {
            metal_state_logical[row * D + col] = metal.state[col * D + row];
        }
    }

    double state_max = 0.0;
    double out_max = 0.0;
    for (size_t i = 0; i < cpu_state.size(); ++i) state_max = std::max(state_max, std::abs(cpu_state[i] - metal_state_logical[i]));
    for (size_t i = 0; i < cpu_out.size(); ++i) out_max = std::max(out_max, std::abs(cpu_out[i] - metal.out[i]));

    std::cout << "Metal device: " << metal.device_name << "\n";
    std::cout << "state_max_abs=" << state_max << " out_max_abs=" << out_max << "\n";
    assert(state_max < 2e-4);
    assert(out_max < 2e-4);

    // RC10.7: qualify direct Metal (T_C,Z_C) construction against the same
    // fp64 affine oracle, using the prepared gates returned by the Metal run.
    qw3::GdnMetalPreparedInputs prepared;
    prepared.T = T;
    prepared.num_k_heads = KH;
    prepared.num_v_heads = VH;
    prepared.head_dim = D;
    prepared.k = in.k;
    prepared.v = in.v;
    prepared.decay = metal.decay;
    prepared.beta = metal.beta;
    auto ms = qw3::gdn_metal_summarize(prepared);
    std::vector<qw3::GdnReferenceToken> toks;
    for (uint32_t t = 0; t < T; ++t) {
        qw3::GdnReferenceToken tok;
        tok.k.resize(D); tok.v.resize(D);
        for (uint32_t d = 0; d < D; ++d) {
            tok.k[d] = in.k[t * D + d];
            tok.v[d] = in.v[t * D + d];
        }
        tok.g = metal.decay[t];
        tok.beta = metal.beta[t];
        toks.push_back(std::move(tok));
    }
    const auto rs = qw3::gdn_compose_tokens(toks);
    double t_max = 0.0, z_max = 0.0;
    for (size_t i = 0; i < rs.T.size(); ++i) {
        t_max = std::max(t_max, std::abs(rs.T[i] - ms.T[i]));
        z_max = std::max(z_max, std::abs(rs.Z[i] - ms.Z[i]));
    }
    std::cout << "summary_T_max_abs=" << t_max << " summary_Z_max_abs=" << z_max << "\n";
    assert(t_max < 5e-4);
    assert(z_max < 5e-4);

    // RC10.8: apply a two-block affine chain directly on Metal and compare
    // the native column-major recurrent state with the CPU correctness path.
    qw3::GdnMetalSummaryChainInputs chain;
    chain.block_count = 2; chain.num_v_heads = VH; chain.head_dim = D;
    chain.T.reserve(2 * ms.T.size()); chain.Z.reserve(2 * ms.Z.size());
    chain.T.insert(chain.T.end(), ms.T.begin(), ms.T.end());
    chain.T.insert(chain.T.end(), ms.T.begin(), ms.T.end());
    chain.Z.insert(chain.Z.end(), ms.Z.begin(), ms.Z.end());
    chain.Z.insert(chain.Z.end(), ms.Z.begin(), ms.Z.end());
    chain.state.assign(VH * D * D, 0.0f);
    auto chain_metal = qw3::gdn_metal_apply_summary_chain(chain);
    qw3::GdnBlockAffineSummary block;
    block.version = 2; block.layer_index = 0; block.position_start = 0;
    block.token_count = T; block.num_v_heads = VH; block.head_dim = D;
    block.model_id = "metal-test"; block.producer = "metal-test"; block.T = ms.T; block.Z = ms.Z;
    std::vector<float> chain_cpu(VH * D * D, 0.0f);
    qw3::gdn_apply_summary_to_state_column_major(block, chain_cpu);
    block.position_start = T;
    qw3::gdn_apply_summary_to_state_column_major(block, chain_cpu);
    double chain_max = 0.0;
    for (size_t i = 0; i < chain_cpu.size(); ++i)
        chain_max = std::max(chain_max, std::abs(double(chain_cpu[i]) - chain_metal.state[i]));
    std::cout << "summary_chain_state_max_abs=" << chain_max << "\n";
    assert(chain_max < 1e-3);
    return 0;
#endif
}
