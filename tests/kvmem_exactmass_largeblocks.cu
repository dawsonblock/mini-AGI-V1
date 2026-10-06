// ExactMass block-count scalability regression.
//
// The historical kernel staged one float per logical block in dynamic shared
// memory and therefore rejected roughly >12K blocks on a 48 KiB limit. This
// test deliberately exceeds that boundary and verifies a globally normalized
// uniform distribution still sums to one.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace qw3::ported {
bool launch_block_attn_score_exactmass(float *score, const float *q_layer,
                                       const float *kraw, uint32_t M,
                                       uint32_t total_tokens, uint32_t n_blocks,
                                       uint32_t block_tokens, uint32_t n_heads,
                                       uint32_t n_kv_heads, uint32_t head_dim,
                                       float scale, float head_w,
                                       uint32_t group_mean,
                                       cudaStream_t stream);
}

#define CHECK(call) do {                                                   \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err_));                            \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

int main() {
    constexpr uint32_t n_blocks = 13001;  // intentionally above old ~12K cap
    constexpr uint32_t block_tokens = 1;
    constexpr uint32_t total_tokens = n_blocks;
    constexpr uint32_t M = 1;
    constexpr uint32_t n_heads = 1;
    constexpr uint32_t n_kv_heads = 1;
    constexpr uint32_t head_dim = 8;

    std::vector<float> q(M * n_heads * head_dim, 0.0f);
    std::vector<float> k(static_cast<size_t>(total_tokens) * n_kv_heads * head_dim,
                         0.0f);
    std::vector<float> score(n_blocks, 0.0f);

    float *d_q = nullptr;
    float *d_k = nullptr;
    float *d_score = nullptr;
    CHECK(cudaMalloc(&d_q, q.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_k, k.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_score, score.size() * sizeof(float)));
    CHECK(cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_score, 0, score.size() * sizeof(float)));

    const bool ok = qw3::ported::launch_block_attn_score_exactmass(
        d_score, d_q, d_k, M, total_tokens, n_blocks, block_tokens,
        n_heads, n_kv_heads, head_dim, 1.0f, 1.0f,
        /*group_mean=*/0, /*stream=*/0);
    if (!ok) {
        std::fprintf(stderr, "ExactMass launcher rejected >12K blocks\n");
        return 1;
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(score.data(), d_score, score.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));

    double sum = 0.0;
    double max_err = 0.0;
    const double expected = 1.0 / static_cast<double>(n_blocks);
    for (float v : score) {
        sum += v;
        max_err = std::max(max_err, std::fabs(static_cast<double>(v) - expected));
    }
    std::printf("[kvmem-exactmass-largeblocks] blocks=%u sum=%.9f max_err=%.3e\n",
                n_blocks, sum, max_err);
    if (std::fabs(sum - 1.0) > 5e-5 || max_err > 5e-6) {
        std::fprintf(stderr, "FAIL: non-normalized ExactMass result\n");
        return 1;
    }

    CHECK(cudaFree(d_q));
    CHECK(cudaFree(d_k));
    CHECK(cudaFree(d_score));
    return 0;
}
