#include "qw3/gdn_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qw3 {

namespace {

void validate_matrix_size(const std::vector<double>& x, uint32_t dim, const char* name) {
    const size_t n = static_cast<size_t>(dim);
    if (dim == 0 || x.size() != n * n) {
        throw std::invalid_argument(std::string(name) + " matrix size mismatch");
    }
    for (double v : x) {
        if (!std::isfinite(v)) throw std::invalid_argument(std::string(name) + " is non-finite");
    }
}

void validate_token(const GdnReferenceToken& token) {
    if (token.k.empty() || token.k.size() != token.v.size()) {
        throw std::invalid_argument("GDN token k/v dimensions differ or are empty");
    }
    if (!std::isfinite(token.g) || token.g < 0.0) {
        throw std::invalid_argument("GDN token decay g must be finite and non-negative");
    }
    if (!std::isfinite(token.beta) || token.beta < 0.0 || token.beta > 1.0) {
        throw std::invalid_argument("GDN token beta must be finite and in [0,1]");
    }
    for (double x : token.k) if (!std::isfinite(x)) throw std::invalid_argument("GDN k is non-finite");
    for (double x : token.v) if (!std::isfinite(x)) throw std::invalid_argument("GDN v is non-finite");
}

} // namespace

GdnMatrixAffineSummary GdnMatrixAffineSummary::identity(uint32_t n) {
    if (n == 0) throw std::invalid_argument("GDN affine dimension must be > 0");
    GdnMatrixAffineSummary out;
    out.dim = n;
    out.T.assign(static_cast<size_t>(n) * n, 0.0);
    out.Z.assign(static_cast<size_t>(n) * n, 0.0);
    for (uint32_t i = 0; i < n; ++i) out.T[static_cast<size_t>(i) * n + i] = 1.0;
    return out;
}

void GdnMatrixAffineSummary::validate() const {
    validate_matrix_size(T, dim, "GDN T");
    validate_matrix_size(Z, dim, "GDN Z");
}

std::vector<double> GdnMatrixAffineSummary::apply(const std::vector<double>& state) const {
    validate();
    validate_matrix_size(state, dim, "GDN state");
    const size_t n = dim;
    std::vector<double> out(n * n, 0.0);
    // Matrix state: out = T * state + Z.
    for (size_t r = 0; r < n; ++r) {
        for (size_t c = 0; c < n; ++c) {
            double acc = Z[r * n + c];
            for (size_t k = 0; k < n; ++k) {
                acc += T[r * n + k] * state[k * n + c];
            }
            out[r * n + c] = acc;
        }
    }
    return out;
}

GdnMatrixAffineSummary GdnMatrixAffineSummary::then(const GdnMatrixAffineSummary& next) const {
    validate();
    next.validate();
    if (dim != next.dim) throw std::invalid_argument("GDN affine dimensions differ");
    const size_t n = dim;
    GdnMatrixAffineSummary out;
    out.dim = dim;
    out.T.assign(n * n, 0.0);
    out.Z.assign(n * n, 0.0);

    // A then B: T = T_B*T_A, Z = T_B*Z_A + Z_B.
    for (size_t r = 0; r < n; ++r) {
        for (size_t c = 0; c < n; ++c) {
            double t = 0.0;
            for (size_t k = 0; k < n; ++k) t += next.T[r * n + k] * T[k * n + c];
            out.T[r * n + c] = t;

            double z = next.Z[r * n + c];
            for (size_t k = 0; k < n; ++k) z += next.T[r * n + k] * Z[k * n + c];
            out.Z[r * n + c] = z;
        }
    }
    return out;
}

double gdn_sigmoid(double x) {
    if (!std::isfinite(x)) throw std::invalid_argument("sigmoid input must be finite");
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(x);
    return z / (1.0 + z);
}

double gdn_softplus(double x) {
    if (!std::isfinite(x)) throw std::invalid_argument("softplus input must be finite");
    if (x > 40.0) return x;
    if (x < -40.0) return std::exp(x);
    return std::log1p(std::exp(x));
}

double gdn_decay_from_raw(double alpha_raw, double dt_bias, double ssm_a) {
    if (!std::isfinite(alpha_raw) || !std::isfinite(dt_bias) || !std::isfinite(ssm_a)) {
        throw std::invalid_argument("GDN raw gate inputs must be finite");
    }
    return std::exp(gdn_softplus(alpha_raw + dt_bias) * ssm_a);
}

GdnMatrixAffineSummary gdn_token_affine(const GdnReferenceToken& token) {
    validate_token(token);
    const uint32_t dim = static_cast<uint32_t>(token.k.size());
    const size_t n = dim;
    auto out = GdnMatrixAffineSummary::identity(dim);

    // T = g * (I - beta * k k^T)
    for (size_t r = 0; r < n; ++r) {
        for (size_t c = 0; c < n; ++c) {
            const double identity = (r == c) ? 1.0 : 0.0;
            out.T[r * n + c] = token.g * (identity - token.beta * token.k[r] * token.k[c]);
            out.Z[r * n + c] = token.beta * token.k[r] * token.v[c];
        }
    }
    return out;
}

GdnMatrixAffineSummary gdn_compose_tokens(const std::vector<GdnReferenceToken>& tokens) {
    if (tokens.empty()) throw std::invalid_argument("cannot compose zero GDN tokens");
    validate_token(tokens.front());
    auto out = GdnMatrixAffineSummary::identity(static_cast<uint32_t>(tokens.front().k.size()));
    for (const auto& token : tokens) {
        validate_token(token);
        if (token.k.size() != out.dim) throw std::invalid_argument("GDN token dimensions differ");
        out = out.then(gdn_token_affine(token));
    }
    return out;
}

std::vector<double> gdn_replay_reference(
    const std::vector<GdnReferenceToken>& tokens,
    const std::vector<double>& initial_state,
    uint32_t dim) {
    validate_matrix_size(initial_state, dim, "GDN initial state");
    std::vector<double> state = initial_state;
    const size_t n = dim;
    for (const auto& token : tokens) {
        validate_token(token);
        if (token.k.size() != dim) throw std::invalid_argument("GDN token dimension mismatch");

        // CUDA algebra, column by column:
        // kv[col] = k^T state[:,col]
        // delta    = beta * (v[col] - g*kv[col])
        // state'[:,col] = g*state[:,col] + k*delta
        std::vector<double> next(n * n, 0.0);
        for (size_t c = 0; c < n; ++c) {
            double kv = 0.0;
            for (size_t r = 0; r < n; ++r) kv += state[r * n + c] * token.k[r];
            const double delta = (token.v[c] - token.g * kv) * token.beta;
            for (size_t r = 0; r < n; ++r) {
                next[r * n + c] = token.g * state[r * n + c] + token.k[r] * delta;
            }
        }
        state.swap(next);
    }
    return state;
}

std::vector<double> gdn_readout_reference(
    const std::vector<double>& state,
    const std::vector<double>& q,
    uint32_t dim,
    double scale) {
    validate_matrix_size(state, dim, "GDN state");
    if (q.size() != dim) throw std::invalid_argument("GDN q dimension mismatch");
    if (!std::isfinite(scale)) throw std::invalid_argument("GDN scale must be finite");
    for (double x : q) if (!std::isfinite(x)) throw std::invalid_argument("GDN q is non-finite");

    const size_t n = dim;
    std::vector<double> out(n, 0.0);
    for (size_t c = 0; c < n; ++c) {
        double acc = 0.0;
        for (size_t r = 0; r < n; ++r) acc += state[r * n + c] * q[r];
        out[c] = acc * scale;
    }
    return out;
}

GdnReferenceError gdn_compare_state(
    const std::vector<double>& a,
    const std::vector<double>& b) {
    if (a.size() != b.size() || a.empty()) throw std::invalid_argument("state comparison size mismatch");
    double max_abs = 0.0;
    long double diff2 = 0.0;
    long double ref2 = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
        }
        const double d = a[i] - b[i];
        max_abs = std::max(max_abs, std::fabs(d));
        diff2 += static_cast<long double>(d) * d;
        ref2 += static_cast<long double>(b[i]) * b[i];
    }
    const double denom = std::sqrt(static_cast<double>(ref2));
    const double rel = std::sqrt(static_cast<double>(diff2)) / std::max(denom, 1e-30);
    return {max_abs, rel};
}

} // namespace qw3
