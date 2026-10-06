#include <metal_stdlib>
using namespace metal;

struct GdnDims {
    uint T;
    uint num_k_heads;
    uint num_v_heads;
    uint head_dim;
    uint qkv_row_stride;
    uint v_row_stride;
    uint gb_row_stride;
    uint out_row_stride;
};

inline float stable_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return exp(x);
    // Metal does not provide log1p. Correct the rounding in 1+y so small
    // positive softplus values do not disappear when the sum rounds to 1.
    const float y = exp(x);
    const float u = 1.0f + y;
    if (u == 1.0f) return y;
    return log(u) * (y / (u - 1.0f));
}

kernel void qw3_gdn_prepare(
    device float* alpha [[buffer(0)]],
    device float* beta [[buffer(1)]],
    const device float* dt_bias [[buffer(2)]],
    const device float* ssm_a [[buffer(3)]],
    constant GdnDims& dims [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {

    const uint count = dims.T * dims.num_v_heads;
    if (gid >= count) return;
    const uint t = gid / dims.num_v_heads;
    const uint vh = gid - t * dims.num_v_heads;
    const uint idx = t * dims.gb_row_stride + vh;

    const float raw_a = alpha[idx];
    const float log_g = stable_softplus(raw_a + dt_bias[vh]) * ssm_a[vh];
    alpha[idx] = exp(log_g);

    const float raw_b = beta[idx];
    beta[idx] = 1.0f / (1.0f + exp(-raw_b));
}

// Correctness-first mapping: one Metal thread owns one complete state column.
// This mirrors the CUDA recurrence algebra exactly but intentionally does not
// attempt the CUDA warp-shuffle optimization. It is the qualification bridge
// for Metal-vs-CPU parity before a tiled/SIMD-group production kernel exists.
kernel void qw3_gdn_recurrent(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    const device float* decay [[buffer(3)]],
    const device float* beta [[buffer(4)]],
    device float* state [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant GdnDims& dims [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {

    const uint columns = dims.num_v_heads * dims.head_dim;
    if (gid >= columns) return;

    const uint vh = gid / dims.head_dim;
    const uint col = gid - vh * dims.head_dim;
    const uint kh = vh % dims.num_k_heads;
    const uint D = dims.head_dim;
    const ulong head_state_base = (ulong)vh * D * D;
    const ulong column_base = head_state_base + (ulong)col * D;

    for (uint t = 0; t < dims.T; ++t) {
        const ulong q_base = (ulong)t * dims.qkv_row_stride + (ulong)kh * D;
        const ulong k_base = q_base;
        const ulong v_base = (ulong)t * dims.v_row_stride + (ulong)vh * D;
        const ulong gb = (ulong)t * dims.gb_row_stride + vh;

        float kv_col = 0.0f;
        for (uint row = 0; row < D; ++row) {
            kv_col += state[column_base + row] * k[k_base + row];
        }

        const float g = decay[gb];
        const float b = beta[gb];
        const float delta_col = (v[v_base + col] - g * kv_col) * b;

        float attn_col = 0.0f;
        for (uint row = 0; row < D; ++row) {
            const ulong sidx = column_base + row;
            const float s = g * state[sidx] + k[k_base + row] * delta_col;
            state[sidx] = s;
            attn_col += s * q[q_base + row];
        }
        out[(ulong)t * dims.out_row_stride + (ulong)vh * D + col] = attn_col * scale;
    }
}

// Correctness-first block affine summary builder. One thread owns one value
// head and composes every token into canonical logical row-major T and Z.
// This deliberately prioritizes auditability over throughput. The recurrence
// uses the rank-1 form so composition is O(T*d^2), not O(T*d^3).
kernel void qw3_gdn_summarize(
    const device float* k [[buffer(0)]],
    const device float* v [[buffer(1)]],
    const device float* decay [[buffer(2)]],
    const device float* beta [[buffer(3)]],
    device float* summary_t [[buffer(4)]],
    device float* summary_z [[buffer(5)]],
    constant GdnDims& dims [[buffer(6)]],
    uint vh [[thread_position_in_grid]]) {

    if (vh >= dims.num_v_heads) return;
    const uint D = dims.head_dim;
    const uint kh = vh % dims.num_k_heads;
    const ulong head_base = (ulong)vh * D * D;

    // Host initializes T=I,Z=0. Compose one token at a time:
    //   T' = g(I-beta*k*k^T)T
    //   Z' = g(I-beta*k*k^T)Z + beta*k*v^T
    for (uint t = 0; t < dims.T; ++t) {
        const ulong k_base = (ulong)t * dims.qkv_row_stride + (ulong)kh * D;
        const ulong v_base = (ulong)t * dims.v_row_stride + (ulong)vh * D;
        const ulong gb = (ulong)t * dims.gb_row_stride + vh;
        const float g = decay[gb];
        const float b = beta[gb];

        // Updating one column at a time is safe in place: each column's dot
        // product is fully evaluated before any row in that column is changed.
        for (uint col = 0; col < D; ++col) {
            float kt = 0.0f;
            float kz = 0.0f;
            for (uint row = 0; row < D; ++row) {
                const float kval = k[k_base + row];
                const ulong idx = head_base + (ulong)row * D + col;
                kt += kval * summary_t[idx];
                kz += kval * summary_z[idx];
            }
            for (uint row = 0; row < D; ++row) {
                const float kval = k[k_base + row];
                const ulong idx = head_base + (ulong)row * D + col;
                summary_t[idx] = g * (summary_t[idx] - b * kval * kt);
                summary_z[idx] = g * (summary_z[idx] - b * kval * kz)
                               + b * kval * v[v_base + col];
            }
        }
    }
}

struct GdnSummaryChainDims {
    uint block_count;
    uint num_v_heads;
    uint head_dim;
};

// One thread owns one native recurrent-state column. T/Z are canonical
// row-major [block,head,row,col], while state is [head,col,row].
kernel void qw3_gdn_apply_summary_chain(
    const device float* summary_t [[buffer(0)]],
    const device float* summary_z [[buffer(1)]],
    device float* state [[buffer(2)]],
    constant GdnSummaryChainDims& dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {

    const uint D = dims.head_dim;
    const uint columns = dims.num_v_heads * D;
    if (gid >= columns || D > 128) return;
    const uint vh = gid / D;
    const uint col = gid - vh * D;
    const ulong per_head = (ulong)D * D;
    const ulong state_base = (ulong)vh * per_head + (ulong)col * D;

    float current[128];
    float next[128];
    for (uint row = 0; row < D; ++row) current[row] = state[state_base + row];

    const ulong per_block = (ulong)dims.num_v_heads * per_head;
    for (uint block = 0; block < dims.block_count; ++block) {
        const ulong head_base = (ulong)block * per_block + (ulong)vh * per_head;
        for (uint row = 0; row < D; ++row) {
            float acc = summary_z[head_base + (ulong)row * D + col];
            for (uint k = 0; k < D; ++k) {
                acc += summary_t[head_base + (ulong)row * D + k] * current[k];
            }
            next[row] = acc;
        }
        for (uint row = 0; row < D; ++row) current[row] = next[row];
    }
    for (uint row = 0; row < D; ++row) state[state_base + row] = current[row];
}
