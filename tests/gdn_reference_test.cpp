#include "qw3/gdn_reference.hpp"

#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

using namespace qw3;

static std::vector<GdnReferenceToken> random_tokens(uint32_t dim, uint32_t count) {
    std::mt19937_64 rng(0xC0105ULL);
    std::normal_distribution<double> normal(0.0, 0.15);
    std::uniform_real_distribution<double> decay(0.80, 0.999);
    std::uniform_real_distribution<double> beta(0.05, 0.95);
    std::vector<GdnReferenceToken> out;
    for (uint32_t t = 0; t < count; ++t) {
        GdnReferenceToken tok;
        tok.k.resize(dim);
        tok.v.resize(dim);
        for (uint32_t i = 0; i < dim; ++i) {
            tok.k[i] = normal(rng);
            tok.v[i] = normal(rng);
        }
        tok.g = decay(rng);
        tok.beta = beta(rng);
        out.push_back(std::move(tok));
    }
    return out;
}

int main() {
    constexpr uint32_t dim = 8;
    const auto tokens = random_tokens(dim, 17);

    std::vector<double> s0(static_cast<size_t>(dim) * dim, 0.0);
    for (size_t i = 0; i < s0.size(); ++i) s0[i] = std::sin(static_cast<double>(i) * 0.17) * 0.1;

    // Model-specific affine composition must match the direct CUDA algebra.
    const auto direct = gdn_replay_reference(tokens, s0, dim);
    const auto summary = gdn_compose_tokens(tokens);
    const auto composed = summary.apply(s0);
    const auto err = gdn_compare_state(composed, direct);
    assert(err.max_abs < 1e-12);
    assert(err.rel_l2 < 1e-12);

    // Z_C must equal zero-start replay exactly (within fp64 roundoff).
    std::vector<double> zero(static_cast<size_t>(dim) * dim, 0.0);
    const auto zero_replay = gdn_replay_reference(tokens, zero, dim);
    const auto zerr = gdn_compare_state(summary.Z, zero_replay);
    assert(zerr.max_abs < 1e-12);
    assert(zerr.rel_l2 < 1e-12);

    // Split segment composition A then B must equal a single summary.
    std::vector<GdnReferenceToken> a(tokens.begin(), tokens.begin() + 7);
    std::vector<GdnReferenceToken> b(tokens.begin() + 7, tokens.end());
    const auto ab = gdn_compose_tokens(a).then(gdn_compose_tokens(b));
    const auto split = ab.apply(s0);
    const auto split_err = gdn_compare_state(split, direct);
    assert(split_err.max_abs < 1e-12);

    // Preprocessing should remain numerically stable at extreme raw gates.
    assert(gdn_sigmoid(1000.0) == 1.0);
    assert(gdn_sigmoid(-1000.0) == 0.0);
    assert(std::isfinite(gdn_softplus(1000.0)));
    assert(std::isfinite(gdn_softplus(-1000.0)));
    const double g = gdn_decay_from_raw(0.25, -0.1, -0.5);
    assert(g > 0.0 && g <= 1.0);

    // Fail closed on incompatible token dimensions.
    bool threw = false;
    try {
        auto bad = tokens;
        bad.back().v.pop_back();
        (void)gdn_compose_tokens(bad);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // Readout convention S^T q should have one value per state column.
    std::vector<double> q(dim, 0.25);
    const auto readout = gdn_readout_reference(direct, q, dim, 0.5);
    assert(readout.size() == dim);
    for (double x : readout) assert(std::isfinite(x));

    return 0;
}
